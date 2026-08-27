/* zbtree.c -- Order-statistic T-tree used as the large-encoding backend of
 * Redis sorted sets (ZSETs), replacing the previous skiplist.
 *
 * A T-tree (Lehman & Carey, 1986) is an AVL-balanced binary tree whose nodes
 * each hold a small sorted array of elements instead of a single key. It keeps
 * the cache friendliness of a B-tree's packed leaves while being a plain
 * balanced binary tree, so there is a single node type and no separator keys.
 *
 * Design goals (see the ZSET encoding notes in server.h):
 *   - Elements are ordered by (score, member) exactly like the old skiplist.
 *   - Each member is a single heap object (zbtElem) with an embedded SDS, so
 *     the ZSET dict can keep mapping member -> zbtElem* for O(1) score lookup.
 *   - Nodes are packed arrays of zbtElem pointers, threaded in sorted order
 *     (prev/next) so range scans walk node to node without re-descending.
 *   - Each node carries its subtree element count, giving O(log N) rank and
 *     rank-based access (ZRANK / ZRANGE by index / ZREMRANGEBYRANK).
 *
 * The tree balances on node height (AVL), not on element occupancy, so every
 * root-to-leaf path visits O(log N) nodes and every operation stays O(log N)
 * comparisons plus an O(fanout) array shift.
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "server.h"

/* Fanout of a node's element array. A node may temporarily hold one extra slot
 * (hence the "+1" sized array) while an insertion is being resolved. Internal
 * nodes (two children) are kept at >= MIN occupancy; leaf and half-leaf nodes
 * are allowed to hold fewer, and empty ones are spliced out.
 *
 * 68 keeps the whole node on jemalloc's 640-byte size class: 88 bytes of
 * header plus (68+1)*8 = 552 bytes of pointers is 640, so the fence fields
 * below are free and the fanout is a touch higher than the old 64. */
#define ZBT_NODE_MAX 68
#define ZBT_NODE_MIN (ZBT_NODE_MAX/2)

/* Length of the member prefix cached in the descent line. 8 bytes fills the
 * hole next to the two fence scores without growing line 0. */
#define ZBT_PFX 8

/* T-tree node. One type for the whole tree: a leaf has no children, a
 * half-leaf has exactly one (which AVL forces to be a leaf), an internal node
 * has both. Elements are kept sorted by (score, member).
 *
 * The first cache line holds everything a root-to-leaf descent reads: child
 * pointers, the two fence keys (the node's min = elems[0] and max =
 * elems[count-1], cached as score + an 8-byte zero-padded member prefix) and
 * the order-statistic aggregates. A descent therefore compares against the
 * fences without dereferencing the scattered elems[] at all, except in the one
 * boundary node it lands in (and in the rare prefix-tie fallback). Mutation and
 * iteration fields sit on later lines. */
struct zbtNode {
    /* ---- line 0: descent-read fields (60 bytes) ---- */
    struct zbtNode *left, *right;      /* 16 */
    double   min_score, max_score;     /* 16  fences: elems[0], elems[count-1] */
    char     min_pfx[ZBT_PFX];         /*  8  first bytes of the min member */
    char     max_pfx[ZBT_PFX];         /*  8  first bytes of the max member */
    uint32_t size;                     /*  4  elements in this whole subtree */
    uint32_t lsize;                    /*  4  elements in the left subtree */
    uint16_t count;                    /*  2  used slots in elems[] */
    uint8_t  height;                   /*  1  AVL height (leaf == 1, NULL == 0) */
    uint8_t  fence_flags;              /*  1  bit0/bit1: min/max prefix decisive */
    /* ---- line 1+: mutation / iteration only ---- */
    struct zbtNode *parent;
    struct zbtNode *prev, *next;       /* in-order predecessor/successor node */
    zbtElem *elems[ZBT_NODE_MAX + 1];
};

/* A subtree size is uint32_t, so one tree caps at ~4.3B elements; the public
 * rank API already narrows to unsigned int (zbtDeleteRangeByRank), so this is
 * not a new limit. Keep line 0 within a single 64-byte cache line. */
static_assert(offsetof(struct zbtNode, parent) <= 64,
              "zbtNode descent fields must fit one cache line");

/* Two prefix bits in fence_flags. */
#define ZBT_FENCE_MIN_FULL 1
#define ZBT_FENCE_MAX_FULL 2

/*-----------------------------------------------------------------------------
 * Element allocation
 *----------------------------------------------------------------------------*/

/* The member offset stored in zbtElem.data[0] is bounded by the largest sds
 * header, so a single byte is always enough to hold it. */
static_assert(sizeof(zbtElem) + 1 + sizeof(struct sdshdr64) <= UINT8_MAX,
              "zbtElem member offset must fit in a byte");

/* Allocate an element with the member SDS embedded in the same allocation
 * (single block: zbtElem header + sds header + data). The member is copied
 * from 'buf', which does not have to be an sds: callers holding plain bytes
 * (listpack entries, integer members) can build an element without first
 * materializing a temporary sds. */
zbtElem *zbtCreateElemBuf(double score, const char *buf, size_t len) {
    char sds_type = sdsReqType(len);
    size_t sds_hdr_len = sdsHdrSize(sds_type);
    size_t hdr = sizeof(zbtElem) + 1;  /* score + the offset byte itself */
    size_t sds_buf_size = sds_hdr_len + len + 1;
    size_t total = hdr + sds_buf_size;

    zbtElem *e = zmalloc(total);
    e->score = score;
    size_t sds_offset = hdr + sds_hdr_len;
    zbtSetOffset(e, (uint8_t)sds_offset);

    char *dst = (char *)e + hdr;
    sds emb = sdsnewplacement(dst, sds_buf_size, sds_type, buf, len);
    serverAssert(emb == (sds)((char *)e + sds_offset));
    return e;
}

/* Same as zbtCreateElemBuf(), for callers that already hold an sds. The caller
 * keeps ownership of 'ele' (it is copied). */
zbtElem *zbtCreateElem(double score, sds ele) {
    return zbtCreateElemBuf(score, ele, sdslen(ele));
}

/* Free a detached element that is not owned by any tree. Used by callers that
 * allocate an element with zbtCreateElem() but fail before ownership is
 * transferred to a tree (e.g. duplicate detection on RDB load). */
void zbtFreeElem(zbtElem *e) {
    zfree(e);  /* embedded sds is part of the allocation, no separate free */
}

/* Compare {score, ele} with element 'e'. Returns 1 (bigger), 0 (equal),
 * -1 (smaller). NULL is treated as +infinity. Ordering: score, then member. */
int zbtCompare(double score, sds ele, const zbtElem *e) {
    if (e == NULL) return -1;
    if (score < e->score) return -1;
    if (score > e->score) return 1;
    return sdscmp(ele, zbtGetEle(e));
}

/* dict keyFromStoredKey callback: recover the member SDS from a stored
 * zbtElem*. */
const void *zbtGetEleForDict(const void *elem) {
    return zbtGetEle((const zbtElem *)elem);
}

/*-----------------------------------------------------------------------------
 * Node allocation / tree lifecycle
 *----------------------------------------------------------------------------*/

static int zbtNH(const zbtNode *n) { return n ? n->height : 0; }
static unsigned long zbtNS(const zbtNode *n) { return n ? n->size : 0; }

/* Recompute a node's cached aggregates (subtree size, left-subtree size and AVL
 * height) from its children and its own element count. Children must already be
 * consistent. Fences are maintained separately (zbtSetFence), since they change
 * only when elems[0]/elems[count-1] change, not on rotations. */
static void zbtFixNode(zbtNode *n) {
    unsigned long ls = zbtNS(n->left);
    n->lsize = (uint32_t)ls;
    n->size = (uint32_t)(n->count + ls + zbtNS(n->right));
    int lh = zbtNH(n->left), rh = zbtNH(n->right);
    n->height = (uint8_t)(1 + (lh > rh ? lh : rh));
}

/* Fill an 8-byte zero-padded prefix from member 'm' and report whether the
 * prefix is *decisive*: the member is at most ZBT_PFX bytes and contains no NUL,
 * so the padded form injectively encodes it. When either side of a comparison
 * is not decisive and the prefixes tie, the caller must fall back to sdscmp. */
static inline int zbtPfxFill(char *dst, sds m) {
    size_t l = sdslen(m);
    size_t c = l < ZBT_PFX ? l : ZBT_PFX;
    memset(dst, 0, ZBT_PFX);
    memcpy(dst, m, c);
    int has_nul = 0;
    for (size_t i = 0; i < c; i++) if (m[i] == 0) { has_nul = 1; break; }
    return (l <= ZBT_PFX) && !has_nul;
}

/* Refresh a node's fence keys from its current first/last elements. Must be
 * called at every site that changes elems[0] or elems[count-1]; a stale fence
 * silently corrupts descents. Requires count >= 1. */
static void zbtSetFence(zbtNode *n) {
    zbtElem *lo = n->elems[0];
    zbtElem *hi = n->elems[n->count - 1];
    n->min_score = lo->score;
    n->max_score = hi->score;
    int minf = zbtPfxFill(n->min_pfx, zbtGetEle(lo));
    int maxf = zbtPfxFill(n->max_pfx, zbtGetEle(hi));
    n->fence_flags = (minf ? ZBT_FENCE_MIN_FULL : 0) |
                     (maxf ? ZBT_FENCE_MAX_FULL : 0);
}

/* Search-key context for a descent: the (score, member) plus a precomputed
 * 8-byte prefix and whether it is decisive. Built once per descent so each
 * level's fence comparison is branch-light and never re-scans the member. */
typedef struct {
    double score;
    sds ele;
    char pfx[ZBT_PFX];
    int full;
} zbtKey;

static inline void zbtKeyInit(zbtKey *k, double score, sds ele) {
    k->score = score;
    k->ele = ele;
    k->full = zbtPfxFill(k->pfx, ele);
}

/* Compare the key against a node's min (elems[0]) fence. Returns <0/0/>0 like
 * zbtCompare, but dereferences elems[] only in the rare prefix-tie fallback. */
static inline int zbtKeyCmpMin(const zbtKey *k, const zbtNode *n) {
    if (k->score < n->min_score) return -1;
    if (k->score > n->min_score) return 1;
    int c = memcmp(k->pfx, n->min_pfx, ZBT_PFX);
    if (c) return c < 0 ? -1 : 1;
    if (k->full && (n->fence_flags & ZBT_FENCE_MIN_FULL)) return 0;
    return sdscmp(k->ele, zbtGetEle(n->elems[0]));
}

/* Compare the key against a node's max (elems[count-1]) fence. */
static inline int zbtKeyCmpMax(const zbtKey *k, const zbtNode *n) {
    if (k->score < n->max_score) return -1;
    if (k->score > n->max_score) return 1;
    int c = memcmp(k->pfx, n->max_pfx, ZBT_PFX);
    if (c) return c < 0 ? -1 : 1;
    if (k->full && (n->fence_flags & ZBT_FENCE_MAX_FULL)) return 0;
    return sdscmp(k->ele, zbtGetEle(n->elems[n->count - 1]));
}

static zbtNode *zbtNewNode(zbtree *t) {
    size_t usable;
    zbtNode *n = zmalloc_usable(sizeof(*n), &usable);
    n->parent = n->left = n->right = NULL;
    n->prev = n->next = NULL;
    n->height = 1;
    n->size = 0;
    n->lsize = 0;
    n->count = 0;
    n->fence_flags = 0;
    t->alloc_size += usable;
    return n;
}

static void zbtFreeNodeShallow(zbtree *t, zbtNode *n) {
    size_t usable;
    zfree_usable(n, &usable);
    t->alloc_size -= usable;
}

zbtree *zbtCreate(void) {
    size_t usable;
    zbtree *t = zmalloc_usable(sizeof(*t), &usable);
    t->length = 0;
    t->alloc_size = usable;
    t->root = NULL;
    t->head = t->tail = NULL;
    t->defrag_resume = NULL;
    t->defrag_resume_score = 0;
    return t;
}

static void zbtFreeSubtree(zbtree *t, zbtNode *n) {
    if (n == NULL) return;
    zbtFreeSubtree(t, n->left);
    zbtFreeSubtree(t, n->right);
    /* No per-element alloc accounting: the whole tree (and its counter) is
     * being torn down, so zbtFree zeroes alloc_size once at the end instead. */
    for (uint32_t i = 0; i < n->count; i++)
        zbtFreeElem(n->elems[i]);
    zbtFreeNodeShallow(t, n);
}

void zbtFree(zbtree *t) {
    zbtFreeSubtree(t, t->root);
    t->alloc_size = 0;
    if (t->defrag_resume) sdsfree(t->defrag_resume);
    zfree(t);
}

size_t zbtAllocSize(const zbtree *t) { return t->alloc_size; }

/*-----------------------------------------------------------------------------
 * Navigation helpers
 *----------------------------------------------------------------------------*/

/* Binary search for (score,ele) inside a single node. Sets *found and returns
 * the matching index (found) or the sorted insertion position. */
static int zbtNodeSearch(const zbtNode *n, double score, sds ele, int *found) {
    int lo = 0, hi = (int)n->count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int c = zbtCompare(score, ele, n->elems[mid]);
        if (c == 0) { *found = 1; return mid; }
        if (c < 0) hi = mid - 1; else lo = mid + 1;
    }
    *found = 0;
    return lo;
}

/* Descend to the node that bounds (score,ele): the node whose min <= key <= max
 * if one exists, otherwise the node where the search fell off the end (the spot
 * a new element would attach to). Sets *pidx to the position inside the node and
 * *pfound to whether the element is actually present. Returns NULL only for an
 * empty tree. */
static zbtNode *zbtFindNode(zbtree *t, double score, sds ele,
                            int *pidx, int *pfound) {
    zbtKey k;
    zbtKeyInit(&k, score, ele);
    zbtNode *n = t->root;
    while (n) {
        __builtin_prefetch(n->left);
        __builtin_prefetch(n->right);
        if (zbtKeyCmpMin(&k, n) < 0) {
            if (n->left) { n = n->left; continue; }
            *pidx = 0; *pfound = 0;
            return n;
        }
        if (zbtKeyCmpMax(&k, n) > 0) {
            if (n->right) { n = n->right; continue; }
            *pidx = (int)n->count; *pfound = 0;
            return n;
        }
        *pidx = zbtNodeSearch(n, score, ele, pfound);
        return n;
    }
    *pidx = 0; *pfound = 0;
    return NULL;
}

/* Insert element 'e' into node 'n' at sorted position 'idx' (array shift). */
static void zbtNodeInsertAt(zbtNode *n, int idx, zbtElem *e) {
    memmove(&n->elems[idx + 1], &n->elems[idx],
            ((int)n->count - idx) * sizeof(zbtElem *));
    n->elems[idx] = e;
    n->count++;
    zbtSetFence(n); /* elems[0] and/or elems[count-1] may have changed */
}

/*-----------------------------------------------------------------------------
 * Balancing (AVL) and aggregate maintenance
 *----------------------------------------------------------------------------*/

/* Rotate the subtree rooted at x to the left; x's right child y becomes the new
 * root and is returned. Parent/child links inside the subtree and the two
 * nodes' aggregates are fixed; linking the returned node to x's old parent is
 * the caller's job. */
static zbtNode *zbtRotateLeft(zbtNode *x) {
    zbtNode *y = x->right;
    zbtNode *t2 = y->left;
    y->left = x;
    x->parent = y;
    x->right = t2;
    if (t2) t2->parent = x;
    zbtFixNode(x);
    zbtFixNode(y);
    return y;
}

/* Mirror image of zbtRotateLeft(). */
static zbtNode *zbtRotateRight(zbtNode *x) {
    zbtNode *y = x->left;
    zbtNode *t2 = y->right;
    y->right = x;
    x->parent = y;
    x->left = t2;
    if (t2) t2->parent = x;
    zbtFixNode(x);
    zbtFixNode(y);
    return y;
}

/* Walk from 'n' to the root, refreshing subtree sizes/heights and applying AVL
 * rotations wherever a node became unbalanced. Rotations never change in-order
 * position, so prev/next threads are untouched here; only structural add/remove
 * paths adjust them. */
static void zbtRebalance(zbtree *t, zbtNode *n) {
    while (n) {
        zbtFixNode(n);
        int bf = zbtNH(n->left) - zbtNH(n->right);
        zbtNode *parent = n->parent;
        zbtNode *sub = n;

        if (bf > 1) {
            if (zbtNH(n->left->left) < zbtNH(n->left->right)) {
                zbtNode *nl = zbtRotateLeft(n->left);
                nl->parent = n;
                n->left = nl;
            }
            sub = zbtRotateRight(n);
        } else if (bf < -1) {
            if (zbtNH(n->right->right) < zbtNH(n->right->left)) {
                zbtNode *nr = zbtRotateRight(n->right);
                nr->parent = n;
                n->right = nr;
            }
            sub = zbtRotateLeft(n);
        }

        if (sub != n) {
            sub->parent = parent;
            if (parent) {
                if (parent->left == n) parent->left = sub;
                else parent->right = sub;
            } else {
                t->root = sub;
            }
        }
        n = parent;
    }
}

/* Apply a +1/-1 element-count delta from node 'n' up to the root, adjusting
 * each ancestor's cached size (and its lsize when we ascend from its left
 * child). Valid only on the non-structural insert/delete paths, where no node
 * is added, removed or rotated, so heights and balance factors are provably
 * unchanged and the cold sibling read that zbtFixNode does per level is
 * unnecessary. 'n' itself gains/loses the element in its own array, so only its
 * size (not lsize) moves. */
static void zbtApplySizeDelta(zbtNode *n, long delta) {
    zbtNode *child = NULL;
    while (n) {
        n->size = (uint32_t)((long)n->size + delta);
        if (child != NULL && n->left == child)
            n->lsize = (uint32_t)((long)n->lsize + delta);
        child = n;
        n = n->parent;
    }
}

/*-----------------------------------------------------------------------------
 * Threading (sorted node chain) helpers
 *----------------------------------------------------------------------------*/

/* Splice 'nn' into the in-order thread as the immediate predecessor of 'pos'. */
static void zbtThreadBefore(zbtree *t, zbtNode *pos, zbtNode *nn) {
    nn->next = pos;
    nn->prev = pos->prev;
    if (pos->prev) pos->prev->next = nn; else t->head = nn;
    pos->prev = nn;
}

/* Splice 'nn' into the in-order thread as the immediate successor of 'pos'. */
static void zbtThreadAfter(zbtree *t, zbtNode *pos, zbtNode *nn) {
    nn->prev = pos;
    nn->next = pos->next;
    if (pos->next) pos->next->prev = nn; else t->tail = nn;
    pos->next = nn;
}

/* Remove 'n' from the in-order thread, repairing head/tail. */
static void zbtUnthread(zbtree *t, zbtNode *n) {
    if (n->prev) n->prev->next = n->next; else t->head = n->next;
    if (n->next) n->next->prev = n->prev; else t->tail = n->prev;
}

/* Attach 'nn' as the (currently empty) left child of 'p'. 'nn' becomes p's
 * in-order predecessor node. */
static void zbtAttachLeft(zbtree *t, zbtNode *p, zbtNode *nn) {
    p->left = nn;
    nn->parent = p;
    zbtThreadBefore(t, p, nn);
}

/* Attach 'nn' as the (currently empty) right child of 'p'. 'nn' becomes p's
 * in-order successor node. */
static void zbtAttachRight(zbtree *t, zbtNode *p, zbtNode *nn) {
    p->right = nn;
    nn->parent = p;
    zbtThreadAfter(t, p, nn);
}

/*-----------------------------------------------------------------------------
 * Insertion
 *----------------------------------------------------------------------------*/

/* Insert an already-allocated element. The caller must guarantee the member is
 * not already present. Ownership of 'e' transfers to the tree. */
void zbtInsertElem(zbtree *t, zbtElem *e) {
    double score = e->score;
    sds ele = zbtGetEle(e);

    t->length++;
    t->alloc_size += zmalloc_usable_size(e);

    if (t->root == NULL) {
        zbtNode *nn = zbtNewNode(t);
        nn->elems[0] = e;
        nn->count = 1;
        zbtSetFence(nn);
        zbtFixNode(nn);
        t->root = t->head = t->tail = nn;
        return;
    }

    zbtNode *n = t->root;
    for (;;) {
        if (zbtCompare(score, ele, n->elems[0]) < 0) {
            /* Key is below this node's minimum. */
            if (n->left) { n = n->left; continue; }
            if (n->count < ZBT_NODE_MAX) {
                zbtNodeInsertAt(n, 0, e);
                zbtApplySizeDelta(n, +1); /* non-structural: size only */
            } else {
                zbtNode *nn = zbtNewNode(t);
                nn->elems[0] = e; nn->count = 1;
                zbtSetFence(nn);
                zbtAttachLeft(t, n, nn);
                zbtRebalance(t, nn);
            }
            return;
        }
        if (zbtCompare(score, ele, n->elems[n->count - 1]) > 0) {
            /* Key is above this node's maximum. */
            if (n->right) { n = n->right; continue; }
            if (n->count < ZBT_NODE_MAX) {
                zbtNodeInsertAt(n, (int)n->count, e);
                zbtApplySizeDelta(n, +1); /* non-structural: size only */
            } else {
                zbtNode *nn = zbtNewNode(t);
                nn->elems[0] = e; nn->count = 1;
                zbtSetFence(nn);
                zbtAttachRight(t, n, nn);
                zbtRebalance(t, nn);
            }
            return;
        }

        /* Bounding node: min < key < max (key is distinct, so strictly). */
        int found;
        int idx = zbtNodeSearch(n, score, ele, &found);
        serverAssert(!found);
        if (n->count < ZBT_NODE_MAX) {
            zbtNodeInsertAt(n, idx, e);
            zbtApplySizeDelta(n, +1); /* non-structural: size only */
            return;
        }

        /* Node is full: insert, then push the displaced minimum down into the
         * greatest-lower-bound node (rightmost of the left subtree), creating a
         * new node there if that one is full too. */
        zbtNodeInsertAt(n, idx, e);
        zbtElem *m = n->elems[0];
        memmove(&n->elems[0], &n->elems[1], (n->count - 1) * sizeof(zbtElem *));
        n->count--;
        zbtSetFence(n); /* stripped the minimum: elems[0] changed */

        if (n->left == NULL) {
            zbtNode *nn = zbtNewNode(t);
            nn->elems[0] = m; nn->count = 1;
            zbtSetFence(nn);
            zbtAttachLeft(t, n, nn);
            zbtRebalance(t, nn);
        } else {
            zbtNode *d = n->left;
            while (d->right) d = d->right;
            if (d->count < ZBT_NODE_MAX) {
                zbtNodeInsertAt(d, (int)d->count, m);
                zbtRebalance(t, d);
            } else {
                zbtNode *nn = zbtNewNode(t);
                nn->elems[0] = m; nn->count = 1;
                zbtSetFence(nn);
                zbtAttachRight(t, d, nn);
                zbtRebalance(t, nn);
            }
        }
        return;
    }
}

zbtElem *zbtInsert(zbtree *t, double score, sds ele) {
    zbtElem *e = zbtCreateElem(score, ele);
    zbtInsertElem(t, e);
    return e;
}

/* Build a balanced, packed T-tree over 'elems[0..n)' in O(n). The elements must
 * already be strictly ascending by (score, member) and ownership of each one
 * transfers to the tree. 't' must be freshly created and empty. Cheaper than n
 * independent zbtInsert() calls (used by RDB load, COPY and listpack->tree
 * conversion, where the source order is already known). */
void zbtBuildFromSorted(zbtree *t, zbtElem **elems, unsigned long n) {
    if (n == 0) return;
    serverAssert(t->length == 0 && t->root == NULL);

    /* Chunk the elements into nodes, distributed as evenly as possible so no
     * node exceeds ZBT_NODE_MAX and multi-node builds stay well filled. */
    unsigned long nnodes = (n + ZBT_NODE_MAX - 1) / ZBT_NODE_MAX;
    unsigned long base = n / nnodes;
    unsigned long rem = n % nnodes;

    zbtNode **nodes = zmalloc(sizeof(zbtNode *) * nnodes);
    unsigned long pos = 0;
    zbtNode *prev = NULL;
    for (unsigned long i = 0; i < nnodes; i++) {
        zbtNode *nd = zbtNewNode(t);
        unsigned long cnt = base + (i < rem ? 1 : 0);
        serverAssert(cnt >= 1 && cnt <= ZBT_NODE_MAX);
        memcpy(nd->elems, &elems[pos], cnt * sizeof(zbtElem *));
        nd->count = (uint16_t)cnt;
        zbtSetFence(nd);
        pos += cnt;
        nd->prev = prev;
        if (prev) prev->next = nd; else t->head = nd;
        prev = nd;
        nodes[i] = nd;
    }
    prev->next = NULL;
    t->tail = prev;

    /* Build a balanced BST over the (already sorted) node array with an
     * explicit stack, fixing each node's aggregates bottom-up. */
    typedef struct { long lo, hi; zbtNode *parent; int side; } Frame;
    Frame *stack = zmalloc(sizeof(Frame) * (nnodes + 1));
    int sp = 0;
    stack[sp++] = (Frame){0, (long)nnodes - 1, NULL, 0};
    /* First pass: link parents/children by repeatedly splitting at the middle.
     * A second bottom-up pass fixes aggregates once the shape is known. */
    zbtNode **order = zmalloc(sizeof(zbtNode *) * nnodes);
    unsigned long onum = 0;
    while (sp > 0) {
        Frame f = stack[--sp];
        if (f.lo > f.hi) continue;
        long mid = (f.lo + f.hi) / 2;
        zbtNode *nd = nodes[mid];
        nd->parent = f.parent;
        nd->left = nd->right = NULL;
        if (f.parent) {
            if (f.side < 0) f.parent->left = nd;
            else f.parent->right = nd;
        } else {
            t->root = nd;
        }
        order[onum++] = nd;
        stack[sp++] = (Frame){f.lo, mid - 1, nd, -1};
        stack[sp++] = (Frame){mid + 1, f.hi, nd, +1};
    }
    /* Aggregates: process nodes deepest-first. 'order' is a pre-order listing,
     * so iterating it in reverse fixes children before parents. */
    for (long i = (long)onum - 1; i >= 0; i--)
        zbtFixNode(order[i]);

    zfree(order);
    zfree(stack);
    zfree(nodes);

    t->length = n;
    for (unsigned long i = 0; i < n; i++)
        t->alloc_size += zmalloc_usable_size(elems[i]);
}

/*-----------------------------------------------------------------------------
 * Deletion
 *----------------------------------------------------------------------------*/

/* Restore structural health of a leaf or half-leaf node 'x' after it lost an
 * element: drop it if empty, or fold a small leaf child into a half-leaf, then
 * rebalance. */
static void zbtCleanupNode(zbtree *t, zbtNode *x) {
    if (x->count == 0) {
        /* Splice the empty node out, promoting its single child (if any). AVL
         * guarantees a node with one child has that child as a leaf, so an
         * empty node is a leaf or half-leaf and this is a clean splice. */
        zbtNode *child = x->left ? x->left : x->right;
        zbtNode *parent = x->parent;
        if (child) child->parent = parent;
        if (parent) {
            if (parent->left == x) parent->left = child;
            else parent->right = child;
        } else {
            t->root = child;
        }
        zbtUnthread(t, x);
        zbtFreeNodeShallow(t, x);
        if (parent) zbtRebalance(t, parent);
        else if (child) zbtRebalance(t, child);
        return;
    }

    /* Half-leaf: its lone child is a leaf (AVL). Merge them when they fit in a
     * single node so occupancy does not decay under repeated deletes. */
    int haveL = x->left != NULL, haveR = x->right != NULL;
    if (haveL != haveR) {
        zbtNode *c = x->left ? x->left : x->right;
        if (c->count + x->count <= ZBT_NODE_MAX) {
            if (c == x->left) {
                memmove(&x->elems[c->count], &x->elems[0],
                        x->count * sizeof(zbtElem *));
                memcpy(&x->elems[0], c->elems, c->count * sizeof(zbtElem *));
                x->left = NULL;
            } else {
                memcpy(&x->elems[x->count], c->elems,
                       c->count * sizeof(zbtElem *));
                x->right = NULL;
            }
            x->count += c->count;
            zbtSetFence(x); /* merged in a child: min or max moved */
            zbtUnthread(t, c);
            zbtFreeNodeShallow(t, c);
        }
    }
    zbtRebalance(t, x);
}

/* Remove element 'e' from its node's array (by identity) and free it. Updates
 * length and alloc accounting. Returns the node it was removed from. */
static zbtNode *zbtDetachElem(zbtree *t, zbtElem *e) {
    double score = e->score;
    sds ele = zbtGetEle(e);
    int idx, found;
    zbtNode *n = zbtFindNode(t, score, ele, &idx, &found);
    serverAssert(n && found && n->elems[idx] == e);

    memmove(&n->elems[idx], &n->elems[idx + 1],
            ((int)n->count - idx - 1) * sizeof(zbtElem *));
    n->count--;
    if (n->count > 0) zbtSetFence(n); /* a boundary element may have gone */
    t->length--;
    t->alloc_size -= zmalloc_usable_size(e);
    zbtFreeElem(e);
    return n;
}

/* Restore invariants after 'n' lost exactly one element. */
static void zbtFixupAfterDelete(zbtree *t, zbtNode *n) {
    if (n->left && n->right) {
        /* Internal nodes are kept at >= MIN by pulling up their greatest lower
         * bound (max of the rightmost node in the left subtree). */
        if (n->count < ZBT_NODE_MIN) {
            zbtNode *d = n->left;
            while (d->right) d = d->right;
            zbtNodeInsertAt(n, 0, d->elems[d->count - 1]); /* refreshes n's fence */
            d->count--;
            if (d->count > 0) zbtSetFence(d); /* pulled up d's max */
            zbtCleanupNode(t, d);   /* also rebalances up through n */
        } else {
            zbtApplySizeDelta(n, -1); /* non-structural: size only */
        }
    } else {
        zbtCleanupNode(t, n);
    }
}

/* Remove element 'e' from the tree and free it. The caller is responsible for
 * removing it from the ZSET dict first (the dict has no key destructor). */
void zbtDeleteElem(zbtree *t, zbtElem *e) {
    zbtNode *n = zbtDetachElem(t, e);
    zbtFixupAfterDelete(t, n);
}

/* Move an existing element to reflect a new score. The element object is reused
 * so the ZSET dict entry does not need updating. */
void zbtUpdateScore(zbtree *t, zbtElem *e, double newscore) {
    /* The member is unchanged, so the dict mapping member -> elem stays valid
     * regardless of which path runs. */
    double score = e->score;
    sds ele = zbtGetEle(e);
    int idx, found;
    zbtNode *n = zbtFindNode(t, score, ele, &idx, &found);
    serverAssert(n && found && n->elems[idx] == e);

    /* In-node fast path: if the new key still sorts strictly after everything
     * below this node (prev node's max) and before everything above it (next
     * node's min), the element stays in this very node. Then only elems[] and
     * the fence move -- counts, subtree sizes, heights and the prev/next thread
     * are all unchanged, so there is no rebalance and no re-descent. This is the
     * common ZINCRBY case, where a small delta rarely leaves the ~fanout-wide
     * node. (A move to an adjacent node falls through to the general path.) */
    int stays = 1;
    if (n->prev &&
        zbtCompare(newscore, ele, n->prev->elems[n->prev->count - 1]) <= 0)
        stays = 0;
    if (stays && n->next &&
        zbtCompare(newscore, ele, n->next->elems[0]) >= 0)
        stays = 0;
    if (stays) {
        memmove(&n->elems[idx], &n->elems[idx + 1],
                ((int)n->count - idx - 1) * sizeof(zbtElem *));
        n->count--;
        e->score = newscore;
        int nfound;
        int nidx = zbtNodeSearch(n, newscore, ele, &nfound);
        serverAssert(!nfound);
        zbtNodeInsertAt(n, nidx, e); /* restores count, refreshes fence */
        return;
    }

    /* General path: detach from the current node and reinsert from the root. */
    memmove(&n->elems[idx], &n->elems[idx + 1],
            ((int)n->count - idx - 1) * sizeof(zbtElem *));
    n->count--;
    if (n->count > 0) zbtSetFence(n); /* a boundary element may have gone */
    t->length--;
    t->alloc_size -= zmalloc_usable_size(e);
    zbtFixupAfterDelete(t, n);

    e->score = newscore;
    zbtInsertElem(t, e);
}

/*-----------------------------------------------------------------------------
 * Rank and rank-based access
 *----------------------------------------------------------------------------*/

/* 1-based rank of (score,ele), or 0 when the element does not exist. */
unsigned long zbtGetRank(zbtree *t, double score, sds ele) {
    zbtKey k;
    zbtKeyInit(&k, score, ele);
    unsigned long rank = 0;
    zbtNode *n = t->root;
    while (n) {
        __builtin_prefetch(n->left);
        __builtin_prefetch(n->right);
        if (zbtKeyCmpMin(&k, n) < 0) {
            n = n->left;
            continue;
        }
        if (zbtKeyCmpMax(&k, n) > 0) {
            rank += n->lsize + n->count;
            n = n->right;
            continue;
        }
        int found;
        int idx = zbtNodeSearch(n, score, ele, &found);
        if (!found) return 0;
        return rank + n->lsize + (unsigned long)idx + 1;
    }
    return 0;
}

/* 1-based rank of element 'e' (which must exist). */
unsigned long zbtRankByElem(zbtree *t, zbtElem *e) {
    unsigned long rank = zbtGetRank(t, e->score, zbtGetEle(e));
    serverAssert(rank != 0);
    return rank;
}

/* Return the element at the given 1-based rank, or NULL if out of range. When
 * 'it' is not NULL it is positioned at the returned element. */
zbtElem *zbtElemByRank(zbtree *t, unsigned long rank, zbtIter *it) {
    if (rank < 1 || rank > t->length) return NULL;
    unsigned long r = rank - 1; /* 0-based */
    zbtNode *n = t->root;
    while (n) {
        __builtin_prefetch(n->left);
        __builtin_prefetch(n->right);
        unsigned long ls = n->lsize;
        if (r < ls) {
            n = n->left;
        } else if (r < ls + n->count) {
            int idx = (int)(r - ls);
            if (it) { it->leaf = n; it->idx = idx; }
            return n->elems[idx];
        } else {
            r -= ls + n->count;
            n = n->right;
        }
    }
    return NULL; /* unreachable given the range check above */
}

/*-----------------------------------------------------------------------------
 * Iteration
 *----------------------------------------------------------------------------*/

zbtElem *zbtFirst(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtNode *n = t->head;
    if (it) { it->leaf = n; it->idx = 0; }
    return n->elems[0];
}

zbtElem *zbtLast(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtNode *n = t->tail;
    if (it) { it->leaf = n; it->idx = (int)n->count - 1; }
    return n->elems[n->count - 1];
}

/* How far ahead the iterator prefetches element payloads. Elements are scattered
 * on the heap, so touching elems[idx +/- D] a few steps early hides the load
 * latency that dominates a linear ZRANGE scan. */
#define ZBT_ITER_PREFETCH 4

zbtElem *zbtIterNext(zbtIter *it) {
    zbtNode *n = it->leaf;
    if (!n) return NULL;
    it->idx++;
    if (it->idx >= (int)n->count) {
        n = n->next;
        it->leaf = n;
        it->idx = 0;
        if (!n) return NULL;
        /* A threaded node always has >= 1 element, so no empty-node check. */
    }
    if (it->idx + ZBT_ITER_PREFETCH < (int)n->count)
        __builtin_prefetch(n->elems[it->idx + ZBT_ITER_PREFETCH]);
    return n->elems[it->idx];
}

zbtElem *zbtIterPrev(zbtIter *it) {
    zbtNode *n = it->leaf;
    if (!n) return NULL;
    it->idx--;
    if (it->idx < 0) {
        n = n->prev;
        it->leaf = n;
        if (!n) return NULL;
        it->idx = (int)n->count - 1;
    }
    if (it->idx - ZBT_ITER_PREFETCH >= 0)
        __builtin_prefetch(n->elems[it->idx - ZBT_ITER_PREFETCH]);
    return n->elems[it->idx];
}

/* Position 'it' exactly on the element matching (score,ele). Returns 1 if
 * found. */
static int zbtSeek(zbtree *t, double score, sds ele, zbtIter *it) {
    int idx, found;
    zbtNode *n = zbtFindNode(t, score, ele, &idx, &found);
    it->leaf = n;
    it->idx = idx;
    return found && n != NULL;
}

/* Element-based next/prev (O(log N)). Used where holding an iterator is
 * inconvenient (e.g. the module API cursor). */
zbtElem *zbtNext(zbtree *t, zbtElem *e) {
    zbtIter it;
    if (!zbtSeek(t, e->score, zbtGetEle(e), &it)) return NULL;
    return zbtIterNext(&it);
}

zbtElem *zbtPrev(zbtree *t, zbtElem *e) {
    zbtIter it;
    if (!zbtSeek(t, e->score, zbtGetEle(e), &it)) return NULL;
    return zbtIterPrev(&it);
}

/*-----------------------------------------------------------------------------
 * Range queries
 *----------------------------------------------------------------------------*/

/* Range bound predicates, inlined at the seek call sites so the innermost
 * descent loop carries no indirect call. The lex side keeps using
 * zslLexValueGteMin/LteMax so bound semantics stay bit-identical to the
 * skiplist; the score side is a two-line comparison. */
static inline int zbtGteLower(const zbtElem *e, const void *range, int lex) {
    if (lex)
        return zslLexValueGteMin(zbtGetEle((zbtElem *)e), (zlexrangespec *)range);
    const zrangespec *r = range;
    return r->minex ? (e->score > r->min) : (e->score >= r->min);
}

static inline int zbtLteUpper(const zbtElem *e, const void *range, int lex) {
    if (lex)
        return zslLexValueLteMax(zbtGetEle((zbtElem *)e), (zlexrangespec *)range);
    const zrangespec *r = range;
    return r->maxex ? (e->score < r->max) : (e->score <= r->max);
}

/* Fence-aware node-level variants used during the seek descent: the score side
 * reads the cached fence score and never touches elems[]; the lex side still
 * needs the member, so it dereferences the fence element (the pure-lex case the
 * layout cannot help). */
static inline int zbtNodeMaxGteLower(const zbtNode *n, const void *range, int lex) {
    if (!lex) {
        const zrangespec *r = range;
        return r->minex ? (n->max_score > r->min) : (n->max_score >= r->min);
    }
    return zslLexValueGteMin(zbtGetEle(n->elems[n->count - 1]), (zlexrangespec *)range);
}
static inline int zbtNodeMinLteUpper(const zbtNode *n, const void *range, int lex) {
    if (!lex) {
        const zrangespec *r = range;
        return r->maxex ? (n->min_score < r->max) : (n->min_score <= r->max);
    }
    return zslLexValueLteMax(zbtGetEle(n->elems[0]), (zlexrangespec *)range);
}

/* Single-descent lower-bound seek: position 'it' on the first element that
 * satisfies the range's lower bound (GteMin) and return it, with its 1-based
 * rank via *out_rank. Returns NULL (and leaves *out_rank untouched) when no
 * element qualifies. The accumulator 'acc' is the number of elements strictly
 * to the left of the current node's subtree, so the rank comes for free, and
 * the boundary node is bisected rather than scanned. */
static zbtElem *zbtSeekLowerBound(zbtree *t, const void *range, int lex,
                                  zbtIter *it, unsigned long *out_rank) {
    zbtNode *node = t->root;
    unsigned long acc = 0;
    zbtNode *bn = NULL; int bi = 0; unsigned long brank = 0;
    while (node) {
        __builtin_prefetch(node->left);
        __builtin_prefetch(node->right);
        unsigned long ls = node->lsize;
        if (!zbtNodeMaxGteLower(node, range, lex)) {
            /* Even the node's max is below the bound: skip node and left. */
            acc += ls + node->count;
            node = node->right;
        } else {
            /* The node's max qualifies, so the first qualifier lives here or in
             * the left subtree. Bisect for the first qualifying slot. */
            int lo = 0, hi = (int)node->count; /* first i with GteLower true */
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (!zbtGteLower(node->elems[mid], range, lex)) lo = mid + 1;
                else hi = mid;
            }
            int i = lo; /* < count, since the max qualifies */
            if (i > 0) {
                /* Slots [0,i) are below the bound, so the whole left subtree is
                 * too: the boundary is exactly here. */
                if (it) { it->leaf = node; it->idx = i; }
                if (out_rank) *out_rank = acc + ls + (unsigned long)i + 1;
                return node->elems[i];
            }
            /* elems[0] qualifies; a smaller qualifier may still sit in the left
             * subtree, so remember this slot and keep going left. */
            bn = node; bi = 0; brank = acc + ls + 1;
            node = node->left;
        }
    }
    if (bn) {
        if (it) { it->leaf = bn; it->idx = bi; }
        if (out_rank) *out_rank = brank;
        return bn->elems[bi];
    }
    return NULL;
}

/* Single-descent upper-bound seek: position 'it' on the last element that
 * satisfies the range's upper bound (LteMax) and return it with its 1-based
 * rank. Mirror image of zbtSeekLowerBound. */
static zbtElem *zbtSeekUpperBound(zbtree *t, const void *range, int lex,
                                  zbtIter *it, unsigned long *out_rank) {
    zbtNode *node = t->root;
    unsigned long acc = 0;
    zbtNode *bn = NULL; int bi = 0; unsigned long brank = 0;
    while (node) {
        __builtin_prefetch(node->left);
        __builtin_prefetch(node->right);
        unsigned long ls = node->lsize;
        if (!zbtNodeMinLteUpper(node, range, lex)) {
            /* Even the node's min exceeds the bound: only the left subtree can
             * hold qualifying elements. */
            node = node->left;
        } else {
            /* elems[0] qualifies: bisect for the first slot that exceeds the
             * bound; the one before it is the last qualifier in this node. */
            int lo = 0, hi = (int)node->count; /* first i with LteUpper false */
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (zbtLteUpper(node->elems[mid], range, lex)) lo = mid + 1;
                else hi = mid;
            }
            int j = lo - 1; /* >= 0, since elems[0] qualifies */
            if (j < (int)node->count - 1) {
                if (it) { it->leaf = node; it->idx = j; }
                if (out_rank) *out_rank = acc + ls + (unsigned long)j + 1;
                return node->elems[j];
            }
            /* The whole node qualifies; a larger qualifier may sit in the right
             * subtree, so remember the node's max and keep going right. */
            bn = node; bi = (int)node->count - 1;
            brank = acc + ls + node->count;
            acc += ls + node->count;
            node = node->right;
        }
    }
    if (bn) {
        if (it) { it->leaf = bn; it->idx = bi; }
        if (out_rank) *out_rank = brank;
        return bn->elems[bi];
    }
    return NULL;
}

/* Shared body for zbtNthIn{,Lex}Range: n >= 0 counts forward from the first
 * in-range element, n < 0 counts back from the last. One lower-bound seek
 * serves the forward cases and one upper-bound seek the reverse; a small offset
 * is walked along the prev/next thread, a large one taken by rank. The final
 * bound check on the returned element is mandatory, not an optimization: it is
 * what keeps the "in range or NULL" contract for the module cursor API
 * (RM_ZsetFirstInScoreRange), which assigns the result straight through. */
static zbtElem *zbtNthCommon(zbtree *t, const void *range, int lex, long n,
                             unsigned long *out_rank, zbtIter *it) {
    if (t->length == 0) return NULL;

    if (n >= 0) {
        zbtIter lit;
        unsigned long r0;
        zbtElem *lb = zbtSeekLowerBound(t, range, lex, &lit, &r0);
        if (!lb) return NULL;

        zbtElem *e;
        unsigned long rank;
        if (n == 0) {
            e = lb;
            rank = r0;
            if (it) *it = lit;
        } else {
            unsigned long target = r0 + (unsigned long)n;
            if (target > t->length) return NULL;
            if ((unsigned long)n <= 2 * ZBT_NODE_MAX) {
                zbtIter s = lit;
                e = NULL;
                for (long k = 0; k < n; k++) { e = zbtIterNext(&s); }
                if (!e) return NULL;
                if (it) *it = s;
            } else {
                e = zbtElemByRank(t, target, it);
                if (!e) return NULL;
            }
            rank = target;
        }
        if (!zbtLteUpper(e, range, lex)) return NULL; /* past the far bound */
        if (out_rank) *out_rank = rank;
        return e;
    } else {
        zbtIter uit;
        unsigned long r1;
        zbtElem *ub = zbtSeekUpperBound(t, range, lex, &uit, &r1);
        if (!ub) return NULL;

        zbtElem *e;
        unsigned long rank;
        if (n == -1) {
            e = ub;
            rank = r1;
            if (it) *it = uit;
        } else {
            unsigned long back = (unsigned long)(-n - 1); /* steps back from ub */
            if (r1 <= back) return NULL; /* target rank < 1 */
            unsigned long target = r1 - back;
            if (back <= 2 * ZBT_NODE_MAX) {
                zbtIter s = uit;
                e = NULL;
                for (unsigned long k = 0; k < back; k++) { e = zbtIterPrev(&s); }
                if (!e) return NULL;
                if (it) *it = s;
            } else {
                e = zbtElemByRank(t, target, it);
                if (!e) return NULL;
            }
            rank = target;
        }
        if (!zbtGteLower(e, range, lex)) return NULL; /* past the near bound */
        if (out_rank) *out_rank = rank;
        return e;
    }
}

zbtElem *zbtNthInRange(zbtree *t, zrangespec *range, long n,
                       unsigned long *out_rank, zbtIter *it) {
    return zbtNthCommon(t, range, 0, n, out_rank, it);
}

zbtElem *zbtNthInLexRange(zbtree *t, zlexrangespec *range, long n,
                          unsigned long *out_rank, zbtIter *it) {
    return zbtNthCommon(t, range, 1, n, out_rank, it);
}

/*-----------------------------------------------------------------------------
 * Range deletion (also removes the members from the ZSET dict)
 *----------------------------------------------------------------------------*/

/* Splice an emptied node out of the tree, or - if it is internal and thus can't
 * be spliced without orphaning a subtree - refill it with its in-order
 * predecessor so it stays valid with a single element. Rebalances to the root,
 * so all aggregates above are recomputed. */
static void zbtRemoveEmptyNode(zbtree *t, zbtNode *n) {
    if (n->left && n->right) {
        /* Internal node: pull up the greatest lower bound (the max of the
         * rightmost node in the left subtree). That element sits below the
         * deleted window, so relocating it is safe and leaves n with count 1,
         * which satisfies the >= 1 per-node invariant (min-occupancy is not
         * enforced, so one pull-up is enough - no need to reach ZBT_NODE_MIN). */
        zbtNode *donor = n->left;
        while (donor->right) donor = donor->right;
        n->elems[0] = donor->elems[donor->count - 1];
        n->count = 1;
        zbtSetFence(n);
        donor->count--;
        if (donor->count > 0) zbtSetFence(donor);
        zbtCleanupNode(t, donor); /* fixes donor + rebalances/aggregates to root */
    } else {
        /* Leaf or half-leaf: zbtCleanupNode's empty branch splices it cleanly. */
        zbtCleanupNode(t, n);
    }
}

/* Delete every element whose 1-based rank falls in [first, last] (inclusive),
 * removing each member from the companion dict 'd' as well. Rather than peeling
 * one element per descent, each iteration locates the node holding rank 'first'
 * once and strips the whole contiguous slice of it that lies in the window with
 * a single memmove, then repairs the tree just once for that node. That is
 * O(M/fanout) descents instead of O(M). 'first' stays fixed while 'last' shrinks
 * by the slice size, since every removal shifts the tail down toward 'first'. */
static unsigned long zbtDeleteRankRange(zbtree *t, unsigned long first,
                                        unsigned long last, dict *d) {
    unsigned long removed = 0;
    if (last > t->length) last = t->length;
    while (first <= last) {
        zbtIter it;
        zbtElem *e = zbtElemByRank(t, first, &it);
        if (!e) break;
        zbtNode *n = it.leaf;
        int idx = it.idx;

        /* Slice [idx, idx+k) is the run of this node's elements inside the
         * window (bounded by the node's end and the remaining window). */
        unsigned long want = last - first + 1;
        int avail = (int)n->count - idx;
        int k = (want < (unsigned long)avail) ? (int)want : avail;

        for (int j = idx; j < idx + k; j++) {
            zbtElem *victim = n->elems[j];
            dictDelete(d, zbtGetEle(victim));
            t->alloc_size -= zmalloc_usable_size(victim);
            zbtFreeElem(victim);
        }
        memmove(&n->elems[idx], &n->elems[idx + k],
                ((int)n->count - idx - k) * sizeof(zbtElem *));
        n->count -= k;
        t->length -= k;
        removed += k;
        last -= k;

        if (n->count == 0) {
            zbtRemoveEmptyNode(t, n);          /* structural: recomputes to root */
        } else {
            zbtSetFence(n);                    /* a boundary element may have gone */
            zbtApplySizeDelta(n, -(long)k);    /* non-structural: size only */
        }
    }
    return removed;
}

/* Resolve a range to the inclusive rank window [*first,*last] (1-based) of the
 * elements it selects, via one lower-bound and one upper-bound seek. Returns 0
 * for an empty selection (in which case first and last are unspecified). */
static int zbtRangeRankWindow(zbtree *t, const void *range, int lex,
                              unsigned long *first, unsigned long *last) {
    unsigned long r0, r1;
    zbtElem *lb = zbtSeekLowerBound(t, range, lex, NULL, &r0);
    zbtElem *ub = zbtSeekUpperBound(t, range, lex, NULL, &r1);
    unsigned long before = lb ? r0 - 1 : t->length; /* elements below the range */
    unsigned long upto = ub ? r1 : 0;               /* elements up to the far bound */
    if (before >= upto) return 0;
    *first = before + 1;
    *last = upto;
    return 1;
}

unsigned long zbtDeleteRangeByScore(zbtree *t, zrangespec *range, dict *d) {
    if (t->length == 0) return 0;
    unsigned long first, last;
    if (!zbtRangeRankWindow(t, range, 0, &first, &last)) return 0;
    return zbtDeleteRankRange(t, first, last, d);
}

unsigned long zbtDeleteRangeByLex(zbtree *t, zlexrangespec *range, dict *d) {
    if (t->length == 0) return 0;
    unsigned long first, last;
    if (!zbtRangeRankWindow(t, range, 1, &first, &last)) return 0;
    return zbtDeleteRankRange(t, first, last, d);
}

/* Delete elements whose 1-based rank is in [start, end] (inclusive). */
unsigned long zbtDeleteRangeByRank(zbtree *t, unsigned int start,
                                   unsigned int end, dict *d) {
    if (t->length == 0 || start > end) return 0;
    return zbtDeleteRankRange(t, start, end, d);
}

/*-----------------------------------------------------------------------------
 * Active defragmentation support
 *----------------------------------------------------------------------------*/

/* Replace element 'olde' with the (content-identical) relocated 'newe' in its
 * node slot. The key is unchanged, so no aggregate or ordering shifts. */
void zbtReplaceElem(zbtree *t, zbtElem *olde, zbtElem *newe) {
    int idx, found;
    zbtNode *n = zbtFindNode(t, newe->score, zbtGetEle(newe), &idx, &found);
    serverAssert(n && found && n->elems[idx] == olde);
    /* Same score and member, so the fence keys are byte-identical: no refresh. */
    serverAssert(newe->score == olde->score &&
                 sdscmp(zbtGetEle(newe), zbtGetEle(olde)) == 0);
    n->elems[idx] = newe;
}

/* Relocate every node in the subtree via the defrag allocator, fixing parent
 * and child back-pointers. The in-order thread is rebuilt afterwards. */
static zbtNode *zbtDefragNode(zbtNode *n, void *(*fn)(void *)) {
    zbtNode *nn = fn(n);
    if (nn) n = nn;
    if (n->left) {
        zbtNode *c = zbtDefragNode(n->left, fn);
        n->left = c; c->parent = n;
    }
    if (n->right) {
        zbtNode *c = zbtDefragNode(n->right, fn);
        n->right = c; c->parent = n;
    }
    return n;
}

/* Rebuild the in-order thread (prev/next) and head/tail by walking the tree in
 * order after a bulk relocation. */
static void zbtRethread(zbtree *t, zbtNode *n, zbtNode **prev) {
    if (n == NULL) return;
    zbtRethread(t, n->left, prev);
    n->prev = *prev;
    if (*prev) (*prev)->next = n; else t->head = n;
    *prev = n;
    zbtRethread(t, n->right, prev);
}

/* Relocate every tree node using the provided defrag allocator, then rebuild
 * the sorted node thread and head/tail. Element objects are handled separately
 * by the caller (via the ZSET dict scan + zbtReplaceElem). */
void zbtDefragNodes(zbtree *t, void *(*fn)(void *)) {
    if (t->root == NULL) return;
    t->root = zbtDefragNode(t->root, fn);
    t->root->parent = NULL;
    zbtNode *prev = NULL;
    zbtRethread(t, t->root, &prev);
    if (prev) prev->next = NULL;
    t->tail = prev;
}

/* Relocate a single node (if the allocator moves it) and repair every external
 * reference: the parent child slot (or the root), the two child back-pointers
 * and the thread/head/tail links. Returns the current (possibly new) node. */
static zbtNode *zbtDefragRelocNode(zbtree *t, zbtNode *n, void *(*fn)(void *)) {
    zbtNode *nn = fn(n);
    if (!nn) return n;
    if (nn->parent) {
        zbtNode *p = nn->parent;
        if (p->left == n) p->left = nn; else p->right = nn;
    } else {
        t->root = nn;
    }
    if (nn->left) nn->left->parent = nn;
    if (nn->right) nn->right->parent = nn;
    if (nn->prev) nn->prev->next = nn; else t->head = nn;
    if (nn->next) nn->next->prev = nn; else t->tail = nn;
    return nn;
}

/* Incremental variant of zbtDefragNodes(): relocate up to 'budget' nodes,
 * walking them in sorted order via the thread. Each relocation repairs all
 * external references immediately, so the tree stays valid between slices. The
 * resume position is stored as the (score, member) key of the first element of
 * the next node to process, so it survives inserts/deletes between calls.
 * Returns 1 if more work remains (bookmark saved) or 0 when done. */
int zbtDefragNodesIncremental(zbtree *t, void *(*fn)(void *), unsigned int budget) {
    if (t->root == NULL) {
        if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
        return 0;
    }

    zbtNode *n;
    if (t->defrag_resume) {
        int idx, found;
        n = zbtFindNode(t, t->defrag_resume_score, t->defrag_resume, &idx, &found);
    } else {
        n = t->head;
    }

    unsigned int work = 0;
    while (n) {
        zbtNode *next = n->next; /* successor object survives the relocation */
        n = zbtDefragRelocNode(t, n, fn);
        work++;

        if (!next) {
            if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
            return 0;
        }
        if (work >= budget) {
            zbtElem *e0 = next->elems[0];
            sds m = sdsdup(zbtGetEle(e0));
            if (t->defrag_resume) sdsfree(t->defrag_resume);
            t->defrag_resume = m;
            t->defrag_resume_score = e0->score;
            return 1;
        }
        n = next;
    }

    if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
    return 0;
}

/*-----------------------------------------------------------------------------
 * Debugging / test verification
 *----------------------------------------------------------------------------*/

#ifdef REDIS_TEST
#include <assert.h>
#include "testhelp.h"

/* Recursively verify one subtree, returning its element count. 'lo'/'hi' bound
 * the keys allowed in this subtree (NULL == unbounded). */
static unsigned long zbtVerifyNode(zbtree *t, zbtNode *n, zbtElem *lo,
                                   zbtElem *hi, int *height) {
    if (n == NULL) { *height = 0; return 0; }

    serverAssert(n->count >= 1 && n->count <= ZBT_NODE_MAX);

    /* Elements strictly ascending within the node. */
    for (uint32_t i = 1; i < n->count; i++)
        serverAssert(zbtCompare(n->elems[i - 1]->score,
                                zbtGetEle(n->elems[i - 1]), n->elems[i]) < 0);

    /* Node range respects the bounds inherited from ancestors. */
    if (lo) serverAssert(zbtCompare(lo->score, zbtGetEle(lo), n->elems[0]) < 0);
    if (hi) serverAssert(zbtCompare(hi->score, zbtGetEle(hi),
                                    n->elems[n->count - 1]) > 0);

    int lh, rh;
    unsigned long ls = zbtVerifyNode(t, n->left, lo, n->elems[0], &lh);
    unsigned long rs = zbtVerifyNode(t, n->right, n->elems[n->count - 1], hi, &rh);

    /* AVL balance and cached height/size/lsize. */
    int bf = lh - rh;
    serverAssert(bf >= -1 && bf <= 1);
    serverAssert(n->height == 1 + (lh > rh ? lh : rh));
    unsigned long total = ls + rs + n->count;
    serverAssert(n->size == total);
    serverAssert(n->lsize == ls);

    /* Fence keys must mirror the actual first/last elements: scores exact, and
     * a rebuilt prefix/flag pair byte-identical to what zbtSetFence would set.
     * A recomputed comparison against the real min/max must agree, catching any
     * missed zbtSetFence immediately. */
    {
        zbtElem *lo = n->elems[0], *hi = n->elems[n->count - 1];
        serverAssert(n->min_score == lo->score && n->max_score == hi->score);
        char mp[ZBT_PFX], xp[ZBT_PFX];
        int mf = zbtPfxFill(mp, zbtGetEle(lo));
        int xf = zbtPfxFill(xp, zbtGetEle(hi));
        serverAssert(memcmp(mp, n->min_pfx, ZBT_PFX) == 0);
        serverAssert(memcmp(xp, n->max_pfx, ZBT_PFX) == 0);
        serverAssert(((n->fence_flags & ZBT_FENCE_MIN_FULL) != 0) == (mf != 0));
        serverAssert(((n->fence_flags & ZBT_FENCE_MAX_FULL) != 0) == (xf != 0));
        zbtKey k;
        zbtKeyInit(&k, lo->score, zbtGetEle(lo));
        serverAssert(zbtKeyCmpMin(&k, n) == 0);
        zbtKeyInit(&k, hi->score, zbtGetEle(hi));
        serverAssert(zbtKeyCmpMax(&k, n) == 0);
    }

    /* Child back-pointers. */
    if (n->left) serverAssert(n->left->parent == n);
    if (n->right) serverAssert(n->right->parent == n);

    *height = n->height;
    return total;
}

/* Panics if any structural invariant is violated. */
void zbtDebugVerify(zbtree *t) {
    int height;
    unsigned long total = zbtVerifyNode(t, t->root, NULL, NULL, &height);
    serverAssert(total == t->length);
    if (t->root) serverAssert(t->root->parent == NULL);

    /* The sorted thread must match an in-order traversal and head/tail. */
    unsigned long chain = 0;
    zbtNode *n = t->head;
    zbtNode *pn = NULL;
    zbtElem *prev_elem = NULL;
    while (n) {
        serverAssert(n->prev == pn);
        serverAssert(n->count >= 1);
        for (uint32_t i = 0; i < n->count; i++) {
            if (prev_elem) {
                zbtElem *cur = n->elems[i];
                serverAssert(zbtCompare(prev_elem->score, zbtGetEle(prev_elem), cur) < 0);
            }
            prev_elem = n->elems[i];
            chain++;
        }
        pn = n;
        n = n->next;
    }
    serverAssert(chain == t->length);
    serverAssert(t->tail == pn);
    if (t->length == 0)
        serverAssert(t->root == NULL && t->head == NULL && t->tail == NULL);
}

/* Test allocator that always relocates the block, to exercise the pointer
 * fix-ups in the incremental node-defrag path. */
static void *zbtTestReloc(void *ptr) {
    size_t sz = zmalloc_usable_size(ptr);
    void *n = zmalloc(sz);
    memcpy(n, ptr, sz);
    zfree(ptr);
    return n;
}

/*-----------------------------------------------------------------------------
 * Differential range-seek harness (Phase 0 correctness net)
 *
 * zbtNthInRange / zbtNthInLexRange have no direct C-test coverage, yet Phase 1
 * rewrites the descent/rank arithmetic underneath them. This harness pins their
 * exact contract before that rewrite: for many random ranges it compares the
 * returned element, its 1-based rank and the seeded iterator position against a
 * brute-force scan of the in-order element sequence. The brute force reuses the
 * public bound predicates (zslValueGteMin/LteMax, zslLexValueGteMin/LteMax) so
 * it validates the *traversal and counting*, which is what changes, not the
 * comparison semantics, which do not.
 *----------------------------------------------------------------------------*/

/* Materialize the in-order element sequence via the prev/next thread. */
static unsigned long zbtDiffSeq(zbtree *t, zbtElem **seq) {
    zbtIter it;
    unsigned long c = 0;
    for (zbtElem *e = zbtFirst(t, &it); e != NULL; e = zbtIterNext(&it))
        seq[c++] = e;
    return c;
}

/* Battery of offsets around a [firstRank,lastRank] window (1-based, inclusive;
 * empty when firstRank > lastRank). For each offset n, derive the expected
 * element/rank exactly as zbtNthGeneric does and cross-check the public entry
 * point (lex==1 selects zbtNthInLexRange). */
static void zbtDiffCheckWindow(zbtree *t, int lex, void *range, zbtElem **seq,
                               unsigned long len, unsigned long firstRank,
                               unsigned long lastRank) {
    int valid = (firstRank >= 1 && firstRank <= lastRank);
    long span = (long)lastRank - (long)firstRank; /* < 0 when empty */
    long ns[13];
    int nn = 0;
    ns[nn++] = 0;
    ns[nn++] = 1;
    ns[nn++] = 2;
    ns[nn++] = -1;
    ns[nn++] = -2;
    ns[nn++] = span;             /* last element in the window (n >= 0) */
    ns[nn++] = span + 1;         /* one past the far end */
    ns[nn++] = -(span + 1);      /* first element in the window (n < 0) */
    ns[nn++] = -(span + 2);      /* one past the near end */
    ns[nn++] = (long)len;        /* far past the end */
    ns[nn++] = -(long)len - 1;   /* far past the front */
    ns[nn++] = (span > 1) ? span / 2 : 0;
    ns[nn++] = (long)len / 2;

    for (int k = 0; k < nn; k++) {
        long n = ns[k];
        long target = (n >= 0) ? (long)firstRank + n : (long)lastRank + 1 + n;
        int ok = valid && target >= (long)firstRank && target <= (long)lastRank;
        zbtElem *exp = ok ? seq[target - 1] : NULL;

        unsigned long got_rank = 12345; /* sentinel: must stay untouched on NULL */
        zbtIter it;
        it.leaf = NULL;
        it.idx = -1;
        zbtElem *got = lex ?
            zbtNthInLexRange(t, (zlexrangespec *)range, n, &got_rank, &it) :
            zbtNthInRange(t, (zrangespec *)range, n, &got_rank, &it);

        serverAssert(got == exp);
        if (got != NULL) {
            serverAssert(got_rank == (unsigned long)target);
            serverAssert(it.leaf != NULL && it.leaf->elems[it.idx] == got);
        }
    }
}

static void zbtDiffTestScoreRanges(void) {
    zbtree *t = zbtCreate();
    const int N = 3000;
    for (int i = 0; i < N; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "m:%06d", i);
        sds e = sdsnew(buf);
        zbtInsert(t, (double)(rand() % 50), e); /* dup scores exercise the boundary node */
        sdsfree(e);
    }
    zbtDebugVerify(t);

    zbtElem **seq = zmalloc(sizeof(zbtElem *) * t->length);
    unsigned long len = zbtDiffSeq(t, seq);
    serverAssert(len == t->length);

    for (int trial = 0; trial < 4000; trial++) {
        double lo = (double)(rand() % 60) - 5;
        double hi = (double)(rand() % 60) - 5;
        int pick = rand() % 8;
        if (pick == 0) lo = -1.0 / 0.0;   /* -inf */
        else if (pick == 1) hi = 1.0 / 0.0; /* +inf */
        else if (pick == 2) hi = lo;        /* single-value window */
        if (lo > hi) { double tmp = lo; lo = hi; hi = tmp; }

        zrangespec r;
        r.min = lo;
        r.max = hi;
        r.minex = rand() & 1;
        r.maxex = rand() & 1;

        unsigned long first = 0;
        while (first < len && !zslValueGteMin(seq[first]->score, &r)) first++;
        unsigned long last = len;
        while (last > 0 && !zslValueLteMax(seq[last - 1]->score, &r)) last--;

        zbtDiffCheckWindow(t, 0, &r, seq, len, first + 1, last);
    }

    zfree(seq);
    zbtFree(t);
}

/* Random member-ish token: three random lowercase letters (varies lex order)
 * plus a counter that guarantees uniqueness so zbtInsert never sees a dup. */
static sds zbtDiffToken(int uniq) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%c%c%c%05d",
             'a' + rand() % 6, 'a' + rand() % 6, 'a' + rand() % 6, uniq);
    return sdsnew(buf);
}

static void zbtDiffTestLexRanges(void) {
    /* The test entrypoint runs before createSharedObjects(), so stand up the
     * two lex sentinels the +/- bounds are pointer-compared against. */
    if (shared.minstring == NULL) {
        shared.minstring = sdsnew("minstring");
        shared.maxstring = sdsnew("maxstring");
    }

    zbtree *t = zbtCreate();
    const int N = 3000;
    for (int i = 0; i < N; i++) {
        sds e = zbtDiffToken(i);
        zbtInsert(t, 0.0, e); /* all equal scores: pure lex ordering */
        sdsfree(e);
    }
    zbtDebugVerify(t);

    zbtElem **seq = zmalloc(sizeof(zbtElem *) * t->length);
    unsigned long len = zbtDiffSeq(t, seq);
    serverAssert(len == t->length);

    for (int trial = 0; trial < 4000; trial++) {
        zlexrangespec lr;
        int mp = rand() % 4;
        if (mp == 0) { lr.min = shared.minstring; lr.minex = 1; }
        else { lr.min = zbtDiffToken(rand() % (N + 200)); lr.minex = rand() & 1; }
        int xp = rand() % 4;
        if (xp == 0) { lr.max = shared.maxstring; lr.maxex = 1; }
        else { lr.max = zbtDiffToken(rand() % (N + 200)); lr.maxex = rand() & 1; }

        unsigned long first = 0;
        while (first < len && !zslLexValueGteMin(zbtGetEle(seq[first]), &lr)) first++;
        unsigned long last = len;
        while (last > 0 && !zslLexValueLteMax(zbtGetEle(seq[last - 1]), &lr)) last--;

        zbtDiffCheckWindow(t, 1, &lr, seq, len, first + 1, last);

        if (lr.min != shared.minstring && lr.min != shared.maxstring) sdsfree(lr.min);
        if (lr.max != shared.minstring && lr.max != shared.maxstring) sdsfree(lr.max);
    }

    zfree(seq);
    zbtFree(t);
}

int zbtreeTest(int argc, char **argv, int flags) {
    UNUSED(argc);
    UNUSED(argv);
    UNUSED(flags);

    printf("Testing T-tree operations with structure verification\n");

    const int N = 2000;
    zbtree *t = zbtCreate();

    typedef struct {
        double score;
        sds ele;
        zbtElem *elem;
        int deleted;
    } Inserted;

    Inserted *elements = zmalloc(sizeof(Inserted) * N);
    srand(12345);

    for (int i = 0; i < N; i++) {
        double score = (double)(rand() % 137);
        char buf[32];
        snprintf(buf, sizeof(buf), "elem:%d", i);
        sds ele = sdsnew(buf);
        zbtElem *e = zbtInsert(t, score, ele);
        elements[i].score = score;
        elements[i].ele = ele;
        elements[i].elem = e;
        elements[i].deleted = 0;

        if (i % 97 == 0) zbtDebugVerify(t);

        unsigned long rank = zbtGetRank(t, score, ele);
        assert(rank != 0);
        assert(zbtElemByRank(t, rank, NULL) == e);
        assert(zbtRankByElem(t, e) == rank);
    }
    zbtDebugVerify(t);
    test_cond("Insert N elements", t->length == (unsigned long)N);

    /* Full in-order scan is sorted and matches length. */
    {
        zbtIter it;
        zbtElem *e = zbtFirst(t, &it);
        unsigned long c = 0;
        zbtElem *prev = NULL;
        while (e) {
            if (prev) assert(zbtCompare(prev->score, zbtGetEle(prev), e) < 0);
            prev = e;
            c++;
            e = zbtIterNext(&it);
        }
        test_cond("Forward scan sorted and complete", c == (unsigned long)N);
    }

    /* Reverse scan is sorted and complete. */
    {
        zbtIter it;
        zbtElem *e = zbtLast(t, &it);
        unsigned long c = 0;
        zbtElem *prev = NULL;
        while (e) {
            if (prev) assert(zbtCompare(e->score, zbtGetEle(e), prev) < 0);
            prev = e;
            c++;
            e = zbtIterPrev(&it);
        }
        test_cond("Reverse scan sorted and complete", c == (unsigned long)N);
    }

    /* Delete half in random order. */
    for (int i = 0; i < N; i += 2) {
        zbtElem *e = elements[i].elem;
        assert(zbtGetRank(t, e->score, zbtGetEle(e)) != 0);
        zbtDeleteElem(t, e);
        elements[i].deleted = 1;
        if (i % 101 == 0) zbtDebugVerify(t);
    }
    zbtDebugVerify(t);
    test_cond("Delete half", t->length == (unsigned long)N / 2);

    /* Update scores of the survivors. */
    for (int i = 1; i < N; i += 2) {
        zbtElem *e = elements[i].elem;
        zbtUpdateScore(t, e, (double)(rand() % 300));
    }
    zbtDebugVerify(t);

    /* Delete the rest. */
    for (int i = 1; i < N; i += 2) {
        zbtDeleteElem(t, elements[i].elem);
        if (i % 103 == 0) zbtDebugVerify(t);
    }
    zbtDebugVerify(t);
    test_cond("Delete rest", t->length == 0);

    for (int i = 0; i < N; i++) sdsfree(elements[i].ele);
    zfree(elements);
    zbtFree(t);

    /* --- Bottom-up bulk build and batched range deletion --- */
    /* Cover boundary sizes around node fan-out multiples. */
    static const int sizes[] = {1, ZBT_NODE_MAX, ZBT_NODE_MAX + 1,
                                ZBT_NODE_MAX * ZBT_NODE_MAX + 3, 5000};
    for (int trial = 0; trial < (int)(sizeof(sizes) / sizeof(sizes[0])); trial++) {
        int M = sizes[trial];
        dict *d = dictCreate(&zsetDictType);
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "bm:%08d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, s);
            sdsfree(s);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        for (int i = 0; i < M; i++)
            serverAssert(dictAdd(d, arr[i], NULL) == DICT_OK);
        zfree(arr);
        zbtDebugVerify(bt);
        serverAssert(bt->length == (unsigned long)M);
        serverAssert(zbtElemByRank(bt, 1, NULL)->score == 0);
        serverAssert(zbtElemByRank(bt, M, NULL)->score == (double)(M - 1));

        if (M >= 10) {
            /* Remove a middle window and confirm dict/tree stay in sync. */
            unsigned long lo = M / 4 + 1, hi = M / 2;
            unsigned long want = hi - lo + 1;
            unsigned long got = zbtDeleteRangeByRank(bt, lo, hi, d);
            zbtDebugVerify(bt);
            serverAssert(got == want);
            serverAssert(bt->length == (unsigned long)M - want);
            serverAssert(dictSize(d) == bt->length);
            /* Score suffix removal. */
            zrangespec rs = {.min = (double)(M * 3 / 4), .max = 1.0 / 0.0,
                             .minex = 0, .maxex = 0};
            zbtDeleteRangeByScore(bt, &rs, d);
            zbtDebugVerify(bt);
            serverAssert(dictSize(d) == bt->length);
        }
        /* Remove everything that is left. */
        zbtDeleteRangeByRank(bt, 1, bt->length, d);
        zbtDebugVerify(bt);
        serverAssert(bt->length == 0 && dictSize(d) == 0);
        dictRelease(d);
        zbtFree(bt);
    }
    test_cond("Bulk build + range delete", 1);

    /* --- Random-window range deletion, checking rank consistency --- */
    {
        const int M = 20000;
        dict *d = dictCreate(&zsetDictType);
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "fz:%08d", i);
            sds sd = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, sd);
            sdsfree(sd);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        for (int i = 0; i < M; i++)
            serverAssert(dictAdd(d, arr[i], NULL) == DICT_OK);
        zfree(arr);

        unsigned long seed = 12345;
        while (bt->length > 200) {
            seed = seed * 1103515245 + 12345;
            unsigned long span = 1 + (seed >> 16) % (ZBT_NODE_MAX * 2);
            seed = seed * 1103515245 + 12345;
            unsigned long lo = 1 + (seed >> 16) % bt->length;
            unsigned long hi = lo + span;
            if (hi > bt->length) hi = bt->length;
            unsigned long before = bt->length;
            unsigned long got = zbtDeleteRangeByRank(bt, lo, hi, d);
            zbtDebugVerify(bt);
            serverAssert(got == hi - lo + 1);
            serverAssert(bt->length == before - got);
            serverAssert(dictSize(d) == bt->length);
            /* Ranks stay dense and ordered after every window removal. */
            serverAssert(zbtRankByElem(bt, zbtElemByRank(bt, 1, NULL)) == 1);
            serverAssert(zbtRankByElem(bt, zbtElemByRank(bt, bt->length, NULL))
                         == bt->length);
        }
        zbtDeleteRangeByRank(bt, 1, bt->length, d);
        serverAssert(bt->length == 0 && dictSize(d) == 0);
        dictRelease(d);
        zbtFree(bt);
    }
    test_cond("Random-window range delete keeps ranks consistent", 1);

    /* --- Sorted insertion stays balanced, both directions --- */
    for (int desc = 0; desc < 2; desc++) {
        const int M = 20000;
        zbtree *at = zbtCreate();
        for (int k = 0; k < M; k++) {
            int i = desc ? M - 1 - k : k;   /* descending inserts prepend */
            char buf[32];
            snprintf(buf, sizeof(buf), "as:%08d", i);
            sds sd = sdsnew(buf);
            zbtInsert(at, (double)i, sd);
            sdsfree(sd);
        }
        zbtDebugVerify(at);
        serverAssert(at->length == (unsigned long)M);

        /* AVL keeps the height logarithmic: for M nodes of >= 1 element the
         * tree has at most M nodes, and AVL height <= 1.4404*log2(nodes+2). */
        double maxh = 1.4404 * (log2((double)at->length + 2.0)) + 1.0;
        serverAssert((double)at->root->height <= maxh);

        /* Deleting from the packed end keeps the invariant holding. */
        for (int k = 0; k < 3 * ZBT_NODE_MAX; k++) {
            unsigned long rank = desc ? 1 : at->length;
            zbtElem *e = zbtElemByRank(at, rank, NULL);
            zbtDeleteElem(at, e);
            zbtDebugVerify(at);
        }
        serverAssert(at->length == (unsigned long)M - 3 * ZBT_NODE_MAX);
        zbtFree(at);
    }
    test_cond("Sorted insert stays balanced both directions", 1);

    /* --- Incremental node defragmentation --- */
    {
        const int M = 4000;
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "dm:%08d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, s);
            sdsfree(s);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        zfree(arr);
        zbtDebugVerify(bt);

        int steps = 0;
        while (zbtDefragNodesIncremental(bt, zbtTestReloc, 16)) {
            zbtDebugVerify(bt); /* tree must stay valid after every slice */
            serverAssert(++steps < 100000);
        }
        zbtDebugVerify(bt);

        zbtIter it;
        zbtElem *e = zbtFirst(bt, &it);
        unsigned long c = 0;
        zbtElem *prev = NULL;
        while (e) {
            if (prev)
                serverAssert(zbtCompare(prev->score, zbtGetEle(prev), e) < 0);
            prev = e;
            c++;
            e = zbtIterNext(&it);
        }
        serverAssert(c == (unsigned long)M);
        zbtFree(bt);
        test_cond("Incremental node defrag", 1);
    }

    /* --- Differential range-seek harness (Phase 0 correctness net) --- */
    zbtDiffTestScoreRanges();
    zbtDiffTestLexRanges();
    test_cond("Range seek matches brute force (score + lex)", 1);

    return 0;
}
#endif
