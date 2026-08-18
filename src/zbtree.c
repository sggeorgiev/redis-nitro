/* zbtree.c -- Order-statistic B+ tree used as the large-encoding backend of
 * Redis sorted sets (ZSETs), replacing the previous skiplist.
 *
 * Design goals (see the ZSET encoding notes in server.h):
 *   - Elements are ordered by (score, member) exactly like the old skiplist.
 *   - Each member is a single heap object (zbtElem) with an embedded SDS, so
 *     the ZSET dict can keep mapping member -> zbtElem* for O(1) score lookup.
 *   - Leaves are packed arrays of zbtElem pointers, doubly linked so range
 *     scans walk contiguous memory instead of chasing skiplist pointers.
 *   - Internal nodes carry per-child subtree sizes, giving O(log N) rank and
 *     rank-based access (ZRANK / ZRANGE by index / ZREMRANGEBYRANK).
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "server.h"

/* Fanout of the tree. Nodes are allowed to temporarily hold one extra slot
 * (hence the "+1" sized arrays) before they are split. */
#define ZBT_LEAF_MAX   64
#define ZBT_LEAF_MIN   (ZBT_LEAF_MAX/2)
#define ZBT_INNER_MAX  64
#define ZBT_INNER_MIN  (ZBT_INNER_MAX/2)

/* Common node header. Both leaf and inner nodes start with it so that a
 * zbtNode* can be inspected polymorphically. */
struct zbtNode {
    struct zbtNode *parent;
    uint32_t count;   /* used slots: elems (leaf) or children (inner) */
    uint32_t isleaf;
};

typedef struct zbtLeaf {
    zbtNode n;
    struct zbtLeaf *prev, *next;         /* sibling leaves (sorted order) */
    zbtElem *elems[ZBT_LEAF_MAX + 1];
} zbtLeaf;

typedef struct zbtInner {
    zbtNode n;
    struct zbtNode *child[ZBT_INNER_MAX + 1];
    unsigned long csize[ZBT_INNER_MAX + 1]; /* subtree element count of child[i] */
    zbtElem *sep[ZBT_INNER_MAX + 1];        /* minimum element of child[i] */
    /* Cache of sep[i]->score kept in a contiguous array, so descents compare
     * doubles from (at most) eight cache lines instead of dereferencing one
     * random heap pointer per probe. Invariant: sepscore[i] == sep[i]->score
     * for every used slot.
     *
     * Almost every writer keeps this trivially: an element's score is not
     * modified while it is linked in a tree, either because the element is
     * detached first (the reinserting path of zbtUpdateScore()) or because it
     * is not in a tree at all (ZUNION aggregation, RDB load) or is replaced by
     * a content-identical copy (defrag). The one exception is the in-place
     * fast path of zbtUpdateScore(), which rewrites the score of a linked
     * element; when that element is the minimum of its leaf it is also some
     * ancestors' separator, so that path must call zbtRefreshLeafSep(). */
    double sepscore[ZBT_INNER_MAX + 1];
} zbtInner;

/*-----------------------------------------------------------------------------
 * Element allocation
 *----------------------------------------------------------------------------*/

/* Allocate an element with the member SDS embedded in the same allocation
 * (single block: zbtElem header + sds header + data). The member is copied
 * from 'buf', which does not have to be an sds: callers holding plain bytes
 * (listpack entries, integer members) can build an element without first
 * materializing a temporary sds. */
zbtElem *zbtCreateElemBuf(double score, const char *buf, size_t len) {
    char sds_type = sdsReqType(len);
    size_t sds_hdr_len = sdsHdrSize(sds_type);
    size_t hdr = sizeof(zbtElem);
    size_t sds_buf_size = sds_hdr_len + len + 1;
    size_t total = hdr + sds_buf_size;

    zbtElem *e = zmalloc(total);
    e->score = score;
    size_t sds_offset = hdr + sds_hdr_len;
    e->sdsoffset = (uint16_t)sds_offset;

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

static zbtLeaf *zbtNewLeaf(zbtree *t) {
    size_t usable;
    zbtLeaf *lf = zmalloc_usable(sizeof(*lf), &usable);
    lf->n.parent = NULL;
    lf->n.count = 0;
    lf->n.isleaf = 1;
    lf->prev = lf->next = NULL;
    t->alloc_size += usable;
    return lf;
}

static zbtInner *zbtNewInner(zbtree *t) {
    size_t usable;
    zbtInner *in = zmalloc_usable(sizeof(*in), &usable);
    in->n.parent = NULL;
    in->n.count = 0;
    in->n.isleaf = 0;
    t->alloc_size += usable;
    return in;
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
    t->defrag_resume = NULL;
    t->defrag_resume_score = 0;
    zbtLeaf *lf = zbtNewLeaf(t);
    t->root = (zbtNode *)lf;
    t->head = t->tail = (zbtNode *)lf;
    return t;
}

static void zbtFreeSubtree(zbtree *t, zbtNode *n) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        for (uint32_t i = 0; i < n->count; i++) {
            t->alloc_size -= zmalloc_usable_size(lf->elems[i]);
            zbtFreeElem(lf->elems[i]);
        }
    } else {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < n->count; i++)
            zbtFreeSubtree(t, in->child[i]);
    }
    zbtFreeNodeShallow(t, n);
}

void zbtFree(zbtree *t) {
    zbtFreeSubtree(t, t->root);
    if (t->defrag_resume) sdsfree(t->defrag_resume);
    zfree(t);
}

size_t zbtAllocSize(const zbtree *t) { return t->alloc_size; }

/*-----------------------------------------------------------------------------
 * Navigation helpers
 *----------------------------------------------------------------------------*/

/* Minimum element of a subtree rooted at 'n' (assumes non-empty). An inner node
 * records the minimum of every child in sep[], so sep[0] is already the minimum
 * of the whole subtree and there is no need to descend to the leftmost leaf.
 * Callers must therefore keep sep[] consistent bottom-up, which is what
 * zbtUpdateToRoot() and the split/merge paths do. */
static zbtElem *zbtNodeMin(zbtNode *n) {
    if (n->isleaf) return ((zbtLeaf *)n)->elems[0];
    return ((zbtInner *)n)->sep[0];
}

/* Number of elements contained in the subtree rooted at 'n'. */
static unsigned long zbtSubtreeSize(zbtNode *n) {
    if (n->isleaf) return n->count;
    zbtInner *in = (zbtInner *)n;
    unsigned long s = 0;
    for (uint32_t i = 0; i < in->n.count; i++) s += in->csize[i];
    return s;
}

/* Index of child 'c' inside inner node 'p'. */
static int zbtChildIdx(zbtInner *p, zbtNode *c) {
    for (uint32_t i = 0; i < p->n.count; i++)
        if (p->child[i] == c) return (int)i;
    serverPanic("zbtree: child not found in parent");
}

/* Set separator slot 'i' of 'in', keeping the sepscore cache in sync. Every
 * single-slot sep[] write must go through here (bulk memmove/memcpy sites
 * mirror the sepscore range explicitly). */
static inline void zbtSetSep(zbtInner *in, int i, zbtElem *e) {
    in->sep[i] = e;
    in->sepscore[i] = e->score;
}

/* First index j in [1, count) such that sep[j] > (score,ele), or count if no
 * such separator exists. sep[0] is never examined: it bounds the subtree from
 * below and slot 0 is the fallback child. Scores are compared against the
 * contiguous sepscore cache; sep[j] is dereferenced only to break score ties
 * on the member. */
static int zbtSepUpperBound(zbtInner *in, double score, sds ele) {
    int lo = 1, hi = (int)in->n.count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        double ms = in->sepscore[mid];
        int gt; /* sep[mid] > (score,ele) ? */
        if (score < ms) gt = 1;
        else if (score > ms) gt = 0;
        else gt = sdscmp(ele, zbtGetEle(in->sep[mid])) < 0;
        if (gt) hi = mid; else lo = mid + 1;
    }
    return lo;
}

/* Choose the child of inner node 'in' whose key range contains (score,ele). */
static int zbtInnerChildIdx(zbtInner *in, double score, sds ele) {
    return zbtSepUpperBound(in, score, ele) - 1;
}

/* Descend from the root to the leaf that would contain (score,ele). */
static zbtLeaf *zbtFindLeaf(zbtree *t, double score, sds ele) {
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        n = in->child[zbtInnerChildIdx(in, score, ele)];
    }
    return (zbtLeaf *)n;
}

/* Locate (score,ele) inside a leaf. Sets *found and returns the index where
 * the element is (if found) or where it should be inserted. Lower-bound
 * binary search; members are unique, so equality can return immediately. */
static int zbtLeafSearch(zbtLeaf *lf, double score, sds ele, int *found) {
    int lo = 0, hi = (int)lf->n.count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int c = zbtCompare(score, ele, lf->elems[mid]);
        if (c == 0) { *found = 1; return mid; }
        if (c > 0) lo = mid + 1; else hi = mid;
    }
    *found = 0;
    return lo;
}

/* Refresh csize/sep for every ancestor of 'n' up to the root. Used after an
 * insertion, deletion or in-place element replacement changed a subtree.
 *
 * The caller has already made 'n' itself consistent, and 'n' is the only child
 * whose recorded size can be stale. It is stale by a fixed amount, and every
 * ancestor's total is off by that same amount, so the subtree is sized once at
 * the bottom and the difference is propagated upwards rather than re-summing
 * csize[] at every level. The walk stops at the first level that turns out to
 * be unchanged, because then no ancestor above it can change either: that makes
 * sibling borrows and defrag replacements O(1) instead of O(height). */
static void zbtUpdateToRoot(zbtree *t, zbtNode *n) {
    UNUSED(t);
    long delta = 0;
    int have_delta = 0;

    while (n->parent) {
        zbtInner *p = (zbtInner *)n->parent;
        int idx = zbtChildIdx(p, n);
        unsigned long oldsize = p->csize[idx];
        unsigned long newsize;

        if (have_delta) {
            newsize = (unsigned long)((long)oldsize + delta);
        } else {
            /* O(1) when the walk starts at a leaf, which is the common case;
             * only the split/merge paths start at an inner node. */
            newsize = zbtSubtreeSize(n);
            delta = (long)newsize - (long)oldsize;
            have_delta = 1;
        }

        zbtElem *newsep = zbtNodeMin(n);
        if (newsize == oldsize && newsep == p->sep[idx]) return;
        p->csize[idx] = newsize;
        zbtSetSep(p, idx, newsep);
        n = (zbtNode *)p;
    }
}

/* Re-publish the minimum of leaf 'lf' into its ancestors' separator slots.
 *
 * Needed when the leaf's minimum element is unchanged as a pointer but its
 * score was rewritten in place, which zbtUpdateToRoot() deliberately treats as
 * "nothing changed" (it compares separators by pointer, and the element count
 * did not move either). Only slot 0 of a parent can propagate further up: a
 * node's own minimum is its first child's minimum, so once the refreshed slot
 * is not slot 0 no higher ancestor is affected. */
static void zbtRefreshLeafSep(zbtree *t, zbtLeaf *lf) {
    UNUSED(t);
    zbtNode *n = (zbtNode *)lf;
    while (n->parent) {
        zbtInner *p = (zbtInner *)n->parent;
        int ci = zbtChildIdx(p, n);
        zbtElem *min = zbtNodeMin(n);
        if (p->sep[ci] == min && p->sepscore[ci] == min->score) return;
        zbtSetSep(p, ci, min);
        if (ci != 0) return;
        n = (zbtNode *)p;
    }
}

/*-----------------------------------------------------------------------------
 * Insertion
 *----------------------------------------------------------------------------*/

static void zbtSplitInner(zbtree *t, zbtInner *in);

/* Insert 'right' as a new child immediately after 'left' in their parent.
 * If 'p' is NULL, 'left' is the current root and a new root is created. */
static void zbtInsertChild(zbtree *t, zbtInner *p, zbtNode *left, zbtNode *right) {
    if (p == NULL) {
        zbtInner *root = zbtNewInner(t);
        root->n.count = 2;
        root->child[0] = left;  left->parent = (zbtNode *)root;
        root->child[1] = right; right->parent = (zbtNode *)root;
        root->csize[0] = zbtSubtreeSize(left);  zbtSetSep(root, 0, zbtNodeMin(left));
        root->csize[1] = zbtSubtreeSize(right); zbtSetSep(root, 1, zbtNodeMin(right));
        t->root = (zbtNode *)root;
        return;
    }

    int li = zbtChildIdx(p, left);
    int at = li + 1;
    int tail = (int)p->n.count - at;
    memmove(&p->child[at + 1], &p->child[at], tail * sizeof(zbtNode *));
    memmove(&p->csize[at + 1], &p->csize[at], tail * sizeof(unsigned long));
    memmove(&p->sep[at + 1], &p->sep[at], tail * sizeof(zbtElem *));
    memmove(&p->sepscore[at + 1], &p->sepscore[at], tail * sizeof(double));
    p->child[at] = right;
    right->parent = (zbtNode *)p;
    p->n.count++;

    p->csize[li] = zbtSubtreeSize(left);  zbtSetSep(p, li, zbtNodeMin(left));
    p->csize[at] = zbtSubtreeSize(right); zbtSetSep(p, at, zbtNodeMin(right));

    if (p->n.count > ZBT_INNER_MAX)
        zbtSplitInner(t, p);
    else
        zbtUpdateToRoot(t, (zbtNode *)p);
}

static void zbtSplitInner(zbtree *t, zbtInner *in) {
    zbtInner *r = zbtNewInner(t);
    int total = (int)in->n.count; /* == ZBT_INNER_MAX + 1 */
    int keep = total / 2;
    int move = total - keep;
    memcpy(r->child, &in->child[keep], move * sizeof(zbtNode *));
    memcpy(r->csize, &in->csize[keep], move * sizeof(unsigned long));
    memcpy(r->sep, &in->sep[keep], move * sizeof(zbtElem *));
    memcpy(r->sepscore, &in->sepscore[keep], move * sizeof(double));
    r->n.count = move;
    in->n.count = keep;
    for (int i = 0; i < move; i++) r->child[i]->parent = (zbtNode *)r;
    zbtInsertChild(t, (zbtInner *)in->n.parent, (zbtNode *)in, (zbtNode *)r);
}

static void zbtSplitLeaf(zbtree *t, zbtLeaf *lf) {
    zbtLeaf *r = zbtNewLeaf(t);
    int total = (int)lf->n.count; /* == ZBT_LEAF_MAX + 1 */
    int keep = total / 2;
    int move = total - keep;
    memcpy(r->elems, &lf->elems[keep], move * sizeof(zbtElem *));
    r->n.count = move;
    lf->n.count = keep;

    r->next = lf->next;
    r->prev = lf;
    if (lf->next) lf->next->prev = r;
    else t->tail = (zbtNode *)r;
    lf->next = r;

    zbtInsertChild(t, (zbtInner *)lf->n.parent, (zbtNode *)lf, (zbtNode *)r);
}

/* Place 'e' at slot 'idx' of leaf 'lf', which must be where it belongs in
 * sorted order, and account for it. Splits the leaf if that pushed it over the
 * fan-out, otherwise republishes the leaf upwards. */
static void zbtInsertIntoLeaf(zbtree *t, zbtLeaf *lf, int idx, zbtElem *e) {
    memmove(&lf->elems[idx + 1], &lf->elems[idx],
            ((int)lf->n.count - idx) * sizeof(zbtElem *));
    lf->elems[idx] = e;
    lf->n.count++;
    t->length++;
    t->alloc_size += zmalloc_usable_size(e);

    if (lf->n.count > ZBT_LEAF_MAX)
        zbtSplitLeaf(t, lf);
    else
        zbtUpdateToRoot(t, (zbtNode *)lf);
}

/* Insert an already-allocated element. The caller must guarantee the member
 * is not already present. Ownership of 'e' transfers to the tree. */
void zbtInsertElem(zbtree *t, zbtElem *e) {
    double score = e->score;
    sds ele = zbtGetEle(e);

    /* Appending past the current maximum is the shape a growing sorted set
     * takes when scores are timestamps or monotonic counters, and t->tail
     * already names the leaf it belongs in. One comparison against that leaf's
     * last element replaces the whole descent; the memmove below degenerates
     * to nothing, since the slot is the leaf's end. */
    zbtLeaf *tl = (zbtLeaf *)t->tail;
    if (tl->n.count > 0 &&
        zbtCompare(score, ele, tl->elems[tl->n.count - 1]) > 0)
    {
        zbtInsertIntoLeaf(t, tl, (int)tl->n.count, e);
        return;
    }

    zbtLeaf *lf = zbtFindLeaf(t, score, ele);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    serverAssert(!found);
    zbtInsertIntoLeaf(t, lf, idx, e);
}

zbtElem *zbtInsert(zbtree *t, double score, sds ele) {
    zbtElem *e = zbtCreateElem(score, ele);
    zbtInsertElem(t, e);
    return e;
}

/* qsort() comparator ordering an array of zbtElem* by (score, member). */
int zbtElemPtrCompare(const void *a, const void *b) {
    zbtElem *ea = *(zbtElem *const *)a;
    zbtElem *eb = *(zbtElem *const *)b;
    return zbtCompare(ea->score, zbtGetEle(ea), eb);
}

/* Put 'elems[0..n)' into the strictly ascending order zbtBuildFromSorted()
 * requires. Already ascending input is left alone and fully descending input
 * is reversed, both of which are common (a range walk, or an RDB zset, which
 * Redis writes in descending order); anything else is sorted. Members must be
 * unique, which makes the resulting order strict. */
void zbtSortElems(zbtElem **elems, unsigned long n) {
    if (n < 2) return;

    int ascending = 1, descending = 1;
    for (unsigned long i = 1; i < n; i++) {
        int c = zbtCompare(elems[i - 1]->score, zbtGetEle(elems[i - 1]), elems[i]);
        if (c >= 0) ascending = 0;
        if (c <= 0) descending = 0;
        if (!ascending && !descending) break;
    }

    if (ascending) return;
    if (descending) {
        for (unsigned long i = 0, j = n - 1; i < j; i++, j--) {
            zbtElem *tmp = elems[i];
            elems[i] = elems[j];
            elems[j] = tmp;
        }
        return;
    }
    qsort(elems, n, sizeof(zbtElem *), zbtElemPtrCompare);
}

/* Build a packed, balanced tree over 'elems[0..n)' in O(n). The elements must
 * already be strictly ascending by (score, member) and ownership of each one
 * transfers to the tree. 't' must be freshly created and empty. This is much
 * cheaper than n independent zbtInsert() calls (used by RDB load, COPY and
 * listpack->tree conversion, where the source order is already known). */
void zbtBuildFromSorted(zbtree *t, zbtElem **elems, unsigned long n) {
    if (n == 0) return;
    serverAssert(t->length == 0);

    /* Discard the placeholder empty root leaf created by zbtCreate(). */
    zbtFreeNodeShallow(t, t->root);
    t->root = t->head = t->tail = NULL;

    /* Build the leaf level. Distribute elements as evenly as possible so that
     * every non-root leaf holds at least ZBT_LEAF_MIN elements (required by the
     * structural invariants). With nleaves = ceil(n / ZBT_LEAF_MAX) the even
     * split guarantees this for more than one leaf. */
    unsigned long nleaves = (n + ZBT_LEAF_MAX - 1) / ZBT_LEAF_MAX;
    unsigned long base = n / nleaves;
    unsigned long rem = n % nleaves;

    zbtNode **level = zmalloc(sizeof(zbtNode *) * nleaves);
    unsigned long pos = 0;
    zbtLeaf *prev = NULL;
    for (unsigned long i = 0; i < nleaves; i++) {
        zbtLeaf *lf = zbtNewLeaf(t);
        unsigned long cnt = base + (i < rem ? 1 : 0);
        memcpy(lf->elems, &elems[pos], cnt * sizeof(zbtElem *));
        lf->n.count = (uint32_t)cnt;
        pos += cnt;
        lf->prev = prev;
        if (prev) prev->next = lf;
        else t->head = (zbtNode *)lf;
        prev = lf;
        level[i] = (zbtNode *)lf;
    }
    prev->next = NULL;
    t->tail = (zbtNode *)prev;

    /* Build inner levels bottom-up until a single root remains. */
    unsigned long count = nleaves;
    while (count > 1) {
        unsigned long nparents = (count + ZBT_INNER_MAX - 1) / ZBT_INNER_MAX;
        unsigned long pbase = count / nparents;
        unsigned long prem = count % nparents;
        zbtNode **parents = zmalloc(sizeof(zbtNode *) * nparents);
        unsigned long ci = 0;
        for (unsigned long p = 0; p < nparents; p++) {
            zbtInner *in = zbtNewInner(t);
            unsigned long nch = pbase + (p < prem ? 1 : 0);
            in->n.count = (uint32_t)nch;
            for (unsigned long k = 0; k < nch; k++) {
                zbtNode *c = level[ci++];
                in->child[k] = c;
                in->csize[k] = zbtSubtreeSize(c);
                zbtSetSep(in, (int)k, zbtNodeMin(c));
                c->parent = (zbtNode *)in;
            }
            parents[p] = (zbtNode *)in;
        }
        zfree(level);
        level = parents;
        count = nparents;
    }

    t->root = level[0];
    t->root->parent = NULL;
    zfree(level);

    t->length = n;
    for (unsigned long i = 0; i < n; i++)
        t->alloc_size += zmalloc_usable_size(elems[i]);
}

/*-----------------------------------------------------------------------------
 * Deletion
 *----------------------------------------------------------------------------*/

static void zbtRebalanceInner(zbtree *t, zbtInner *in);

/* Remove child at position 'pos' from inner node 'p' (does not free it). */
static void zbtRemoveChild(zbtInner *p, int pos) {
    int tail = (int)p->n.count - pos - 1;
    memmove(&p->child[pos], &p->child[pos + 1], tail * sizeof(zbtNode *));
    memmove(&p->csize[pos], &p->csize[pos + 1], tail * sizeof(unsigned long));
    memmove(&p->sep[pos], &p->sep[pos + 1], tail * sizeof(zbtElem *));
    memmove(&p->sepscore[pos], &p->sepscore[pos + 1], tail * sizeof(double));
    p->n.count--;
}

/* Called when inner node 'p' became the single-child root, or a subtree
 * shrank: fix up parent slots or collapse the root as needed. */
static void zbtFixupInnerAfterShrink(zbtree *t, zbtInner *p, int slot) {
    p->csize[slot] = zbtSubtreeSize(p->child[slot]);
    zbtSetSep(p, slot, zbtNodeMin(p->child[slot]));
    if (p->n.parent && p->n.count < ZBT_INNER_MIN) {
        zbtRebalanceInner(t, p);
    } else if (!p->n.parent && p->n.count == 1) {
        zbtNode *c = p->child[0];
        c->parent = NULL;
        t->root = c;
        zbtFreeNodeShallow(t, (zbtNode *)p);
    } else {
        zbtUpdateToRoot(t, (zbtNode *)p);
    }
}

/* Restore the occupancy invariant of the underfull inner node 'in'. Unlike a
 * leaf, an inner node is only ever short by a single child: children are
 * dropped one at a time (zbtRemoveChild) and each removal is followed
 * immediately by zbtFixupInnerAfterShrink(), so borrowing one child is always
 * enough. */
static void zbtRebalanceInner(zbtree *t, zbtInner *in) {
    zbtInner *p = (zbtInner *)in->n.parent;
    int idx = zbtChildIdx(p, (zbtNode *)in);
    serverAssert(in->n.count == ZBT_INNER_MIN - 1);

    /* Borrow from left sibling. */
    if (idx > 0) {
        zbtInner *L = (zbtInner *)p->child[idx - 1];
        if (L->n.count > ZBT_INNER_MIN) {
            memmove(&in->child[1], &in->child[0], in->n.count * sizeof(zbtNode *));
            memmove(&in->csize[1], &in->csize[0], in->n.count * sizeof(unsigned long));
            memmove(&in->sep[1], &in->sep[0], in->n.count * sizeof(zbtElem *));
            memmove(&in->sepscore[1], &in->sepscore[0], in->n.count * sizeof(double));
            int last = (int)L->n.count - 1;
            in->child[0] = L->child[last];
            in->csize[0] = L->csize[last];
            in->sep[0] = L->sep[last];
            in->sepscore[0] = L->sepscore[last];
            in->child[0]->parent = (zbtNode *)in;
            in->n.count++;
            L->n.count--;
            p->csize[idx - 1] = zbtSubtreeSize((zbtNode *)L); zbtSetSep(p, idx - 1, zbtNodeMin((zbtNode *)L));
            p->csize[idx] = zbtSubtreeSize((zbtNode *)in);   zbtSetSep(p, idx, zbtNodeMin((zbtNode *)in));
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }
    /* Borrow from right sibling. */
    if (idx < (int)p->n.count - 1) {
        zbtInner *R = (zbtInner *)p->child[idx + 1];
        if (R->n.count > ZBT_INNER_MIN) {
            in->child[in->n.count] = R->child[0];
            in->csize[in->n.count] = R->csize[0];
            in->sep[in->n.count] = R->sep[0];
            in->sepscore[in->n.count] = R->sepscore[0];
            in->child[in->n.count]->parent = (zbtNode *)in;
            in->n.count++;
            zbtRemoveChild(R, 0);
            p->csize[idx] = zbtSubtreeSize((zbtNode *)in);   zbtSetSep(p, idx, zbtNodeMin((zbtNode *)in));
            p->csize[idx + 1] = zbtSubtreeSize((zbtNode *)R); zbtSetSep(p, idx + 1, zbtNodeMin((zbtNode *)R));
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }

    /* Merge with a sibling. */
    zbtInner *a, *b;
    int ai;
    if (idx > 0) { a = (zbtInner *)p->child[idx - 1]; b = in; ai = idx - 1; }
    else { a = in; b = (zbtInner *)p->child[idx + 1]; ai = idx; }
    for (uint32_t i = 0; i < b->n.count; i++) {
        a->child[a->n.count] = b->child[i];
        a->csize[a->n.count] = b->csize[i];
        a->sep[a->n.count] = b->sep[i];
        a->sepscore[a->n.count] = b->sepscore[i];
        b->child[i]->parent = (zbtNode *)a;
        a->n.count++;
    }
    zbtRemoveChild(p, ai + 1);
    zbtFreeNodeShallow(t, (zbtNode *)b);
    zbtFixupInnerAfterShrink(t, p, ai);
}

/* Restore the occupancy invariant of the underfull leaf 'lf'.
 *
 * A leaf can be short by more than one element: the range-delete path removes a
 * whole slice in one go. Redistribution therefore moves as many elements as it
 * takes to leave both sides at ZBT_LEAF_MIN or above, which is possible exactly
 * when the pair holds 2*ZBT_LEAF_MIN elements between them; otherwise the two
 * leaves are merged (their combined count is then below 2*ZBT_LEAF_MIN ==
 * ZBT_LEAF_MAX, so the survivor always fits). Borrowing a single element, as an
 * only-ever-short-by-one tree could, would silently leave 'lf' underfull. */
static void zbtRebalanceLeaf(zbtree *t, zbtLeaf *lf) {
    zbtInner *p = (zbtInner *)lf->n.parent;
    int idx = zbtChildIdx(p, (zbtNode *)lf);

    /* Redistribute with the left sibling: move its tail into the front of
     * 'lf' so that the two end up evenly filled. */
    if (idx > 0) {
        zbtLeaf *L = (zbtLeaf *)p->child[idx - 1];
        int total = (int)L->n.count + (int)lf->n.count;
        if (total >= 2 * ZBT_LEAF_MIN) {
            int lkeep = total / 2;
            int move = (int)L->n.count - lkeep;
            serverAssert(move > 0);
            memmove(&lf->elems[move], &lf->elems[0],
                    lf->n.count * sizeof(zbtElem *));
            memcpy(&lf->elems[0], &L->elems[lkeep], move * sizeof(zbtElem *));
            L->n.count = (uint32_t)lkeep;
            lf->n.count += (uint32_t)move;
            p->csize[idx - 1] = L->n.count; zbtSetSep(p, idx - 1, zbtNodeMin((zbtNode *)L));
            p->csize[idx] = lf->n.count;    zbtSetSep(p, idx, zbtNodeMin((zbtNode *)lf));
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }
    /* Redistribute with the right sibling: move its head onto the end of
     * 'lf'. */
    if (idx < (int)p->n.count - 1) {
        zbtLeaf *R = (zbtLeaf *)p->child[idx + 1];
        int total = (int)lf->n.count + (int)R->n.count;
        if (total >= 2 * ZBT_LEAF_MIN) {
            int move = total / 2 - (int)lf->n.count;
            serverAssert(move > 0);
            memcpy(&lf->elems[lf->n.count], &R->elems[0],
                   move * sizeof(zbtElem *));
            memmove(&R->elems[0], &R->elems[move],
                    ((int)R->n.count - move) * sizeof(zbtElem *));
            lf->n.count += (uint32_t)move;
            R->n.count -= (uint32_t)move;
            p->csize[idx] = lf->n.count;     zbtSetSep(p, idx, zbtNodeMin((zbtNode *)lf));
            p->csize[idx + 1] = R->n.count;  zbtSetSep(p, idx + 1, zbtNodeMin((zbtNode *)R));
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }

    /* Merge with a sibling: neither side could spare elements, so the two
     * together hold less than 2*ZBT_LEAF_MIN == ZBT_LEAF_MAX of them. */
    zbtLeaf *a, *b;
    int ai;
    if (idx > 0) { a = (zbtLeaf *)p->child[idx - 1]; b = lf; ai = idx - 1; }
    else { a = lf; b = (zbtLeaf *)p->child[idx + 1]; ai = idx; }
    serverAssert(a->n.count + b->n.count <= ZBT_LEAF_MAX);
    memcpy(&a->elems[a->n.count], b->elems, b->n.count * sizeof(zbtElem *));
    a->n.count += b->n.count;
    a->next = b->next;
    if (b->next) b->next->prev = a;
    else t->tail = (zbtNode *)a;
    zbtRemoveChild(p, ai + 1);
    zbtFreeNodeShallow(t, (zbtNode *)b);
    zbtFixupInnerAfterShrink(t, p, ai);
}

/* Remove element 'e' from the tree and free it. The caller is responsible for
 * removing it from the ZSET dict first (the dict has no key destructor). */
void zbtDeleteElem(zbtree *t, zbtElem *e) {
    double score = e->score;
    sds ele = zbtGetEle(e);
    zbtLeaf *lf = zbtFindLeaf(t, score, ele);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    serverAssert(found && lf->elems[idx] == e);

    memmove(&lf->elems[idx], &lf->elems[idx + 1],
            ((int)lf->n.count - idx - 1) * sizeof(zbtElem *));
    lf->n.count--;
    t->length--;
    t->alloc_size -= zmalloc_usable_size(e);
    zbtFreeElem(e);

    if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, lf);
    else
        zbtUpdateToRoot(t, (zbtNode *)lf);
}

/* Last element of the leaf preceding 'lf', or NULL when 'lf' starts the tree.
 * Every leaf except a lone root leaf is at least half full, so a sibling in the
 * chain always has an element to report. */
static zbtElem *zbtElemBeforeLeaf(zbtLeaf *lf) {
    zbtLeaf *pv = lf->prev;
    return pv ? pv->elems[pv->n.count - 1] : NULL;
}

/* First element of the leaf following 'lf', or NULL when 'lf' ends the tree. */
static zbtElem *zbtElemAfterLeaf(zbtLeaf *lf) {
    zbtLeaf *nx = lf->next;
    return nx ? nx->elems[0] : NULL;
}

/* Move an existing element to reflect a new score. The element object is
 * reused so the ZSET dict entry does not need updating.
 *
 * A score change usually leaves the element in the leaf it already occupies:
 * ZINCRBY on a leaderboard moves it by a small delta, and it has to overtake a
 * whole leaf's worth of members to land anywhere else. Handling that in place
 * avoids the delete-and-reinsert pair of descents along with the
 * rebalance-then-split churn that can come with them. */
void zbtUpdateScore(zbtree *t, zbtElem *e, double newscore) {
    /* The dict maps member -> elem and the member is unchanged, so whatever
     * happens below, the dict entry stays valid. */
    double score = e->score;
    sds ele = zbtGetEle(e);
    zbtLeaf *lf = zbtFindLeaf(t, score, ele);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    serverAssert(found && lf->elems[idx] == e);

    /* The element stays in this leaf as long as the new score keeps it within
     * the key span the leaf owns, which the elements flanking the leaf in the
     * sibling chain delimit. Staying inside that span is what makes the update
     * invisible to the rest of the tree: the leaf's minimum may change, and
     * with it this leaf's separator, but it still orders strictly between the
     * neighbouring subtrees, so no descent is misdirected. */
    zbtElem *lpred = zbtElemBeforeLeaf(lf);
    zbtElem *lsucc = zbtElemAfterLeaf(lf);
    if ((lpred == NULL || zbtCompare(newscore, ele, lpred) > 0) &&
        (lsucc == NULL || zbtCompare(newscore, ele, lsucc) < 0))
    {
        int nidx = idx;
        zbtElem *pred = idx > 0 ? lf->elems[idx - 1] : NULL;
        zbtElem *succ = idx + 1 < (int)lf->n.count ? lf->elems[idx + 1] : NULL;

        if ((pred == NULL || zbtCompare(newscore, ele, pred) > 0) &&
            (succ == NULL || zbtCompare(newscore, ele, succ) < 0))
        {
            /* Its neighbours still bracket it: the slot does not change. */
            e->score = newscore;
        } else {
            /* Re-seat it within the leaf. The element count is unchanged, so
             * nothing above the leaf has to be resized. */
            lf->n.count--;
            memmove(&lf->elems[idx], &lf->elems[idx + 1],
                    ((int)lf->n.count - idx) * sizeof(zbtElem *));
            e->score = newscore;
            nidx = zbtLeafSearch(lf, newscore, ele, &found);
            serverAssert(!found);
            memmove(&lf->elems[nidx + 1], &lf->elems[nidx],
                    ((int)lf->n.count - nidx) * sizeof(zbtElem *));
            lf->elems[nidx] = e;
            lf->n.count++;
        }

        /* Whenever the leaf's minimum is involved, it is also the separator
         * some ancestors recorded, and its cached score has just gone stale. */
        if (idx == 0 || nidx == 0) zbtRefreshLeafSep(t, lf);
        return;
    }

    /* Slow path: detach the element (which repairs every separator that
     * referenced it) and reinsert it at its new position. */
    memmove(&lf->elems[idx], &lf->elems[idx + 1],
            ((int)lf->n.count - idx - 1) * sizeof(zbtElem *));
    lf->n.count--;
    t->length--;
    t->alloc_size -= zmalloc_usable_size(e);
    if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, lf);
    else
        zbtUpdateToRoot(t, (zbtNode *)lf);

    e->score = newscore;
    zbtInsertElem(t, e);
}

/*-----------------------------------------------------------------------------
 * Rank and rank-based access
 *----------------------------------------------------------------------------*/

/* 1-based rank of element 'e'. */
unsigned long zbtRankByElem(zbtree *t, zbtElem *e) {
    double score = e->score;
    sds ele = zbtGetEle(e);
    unsigned long rank = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int ci = zbtInnerChildIdx(in, score, ele);
        for (int i = 0; i < ci; i++) rank += in->csize[i];
        n = in->child[ci];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    serverAssert(found);
    return rank + (unsigned long)idx + 1;
}

/* 1-based rank of (score,ele), or 0 when the element does not exist. */
unsigned long zbtGetRank(zbtree *t, double score, sds ele) {
    unsigned long rank = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int ci = zbtInnerChildIdx(in, score, ele);
        for (int i = 0; i < ci; i++) rank += in->csize[i];
        n = in->child[ci];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    if (!found) return 0;
    return rank + (unsigned long)idx + 1;
}

/* Return the element at the given 1-based rank, or NULL if out of range.
 * When 'it' is not NULL it is positioned at the returned element. */
zbtElem *zbtElemByRank(zbtree *t, unsigned long rank, zbtIter *it) {
    if (rank < 1 || rank > t->length) return NULL;
    unsigned long r = rank - 1; /* 0-based */
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        uint32_t i = 0;
        while (i < in->n.count && r >= in->csize[i]) { r -= in->csize[i]; i++; }
        n = in->child[i];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = (int)r; }
    return lf->elems[r];
}

/*-----------------------------------------------------------------------------
 * Iteration
 *----------------------------------------------------------------------------*/

zbtElem *zbtFirst(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf = (zbtLeaf *)t->head;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = 0; }
    return lf->elems[0];
}

zbtElem *zbtLast(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf = (zbtLeaf *)t->tail;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = (int)lf->n.count - 1; }
    return lf->elems[lf->n.count - 1];
}

zbtElem *zbtIterNext(zbtIter *it) {
    if (!it->leaf) return NULL;
    zbtLeaf *lf = (zbtLeaf *)it->leaf;
    it->idx++;
    if (it->idx >= (int)lf->n.count) {
        it->leaf = (zbtNode *)lf->next;
        it->idx = 0;
        if (!it->leaf) return NULL;
        lf = (zbtLeaf *)it->leaf;
        if (lf->n.count == 0) return NULL;
    }
    return lf->elems[it->idx];
}

zbtElem *zbtIterPrev(zbtIter *it) {
    if (!it->leaf) return NULL;
    zbtLeaf *lf = (zbtLeaf *)it->leaf;
    it->idx--;
    if (it->idx < 0) {
        it->leaf = (zbtNode *)lf->prev;
        if (!it->leaf) return NULL;
        lf = (zbtLeaf *)it->leaf;
        it->idx = (int)lf->n.count - 1;
        if (it->idx < 0) return NULL;
    }
    return lf->elems[it->idx];
}

/* Position 'it' exactly on the element matching (score,ele). Returns 1 if
 * found. */
static int zbtSeek(zbtree *t, double score, sds ele, zbtIter *it) {
    zbtLeaf *lf = zbtFindLeaf(t, score, ele);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found);
    it->leaf = (zbtNode *)lf;
    it->idx = idx;
    return found;
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

/* The two partition functions below split the sorted order into a passing
 * prefix and a failing suffix under a monotonic predicate (true for a prefix
 * of the order, then false), in one root-to-leaf descent. They return the
 * size of the passing prefix, and optionally the position of the first
 * failing element as (*lf_out, *idx_out). When every element of the chosen
 * leaf passes, *idx_out == (*lf_out)->n.count and the first failing element
 * (if any) is the first element of the next leaf.
 *
 * Each inner level binary-searches the first failing separator in [1, count)
 * and descends into the child before it: children left of that child contain
 * only passing elements (each is bounded above by a passing separator), and
 * their csize[] prefix is accumulated into the count. */

/* Score partition: passing means (inclusive ? score <= v : score < v).
 * Score predicates ignore the member, so inner levels compare only the
 * contiguous sepscore cache and never dereference an element. */
static unsigned long zbtPartitionScore(zbtree *t, double v, int inclusive,
                                       zbtLeaf **lf_out, int *idx_out) {
    unsigned long cnt = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int lo = 1, hi = (int)in->n.count;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            int pass = inclusive ? (in->sepscore[mid] <= v)
                                 : (in->sepscore[mid] < v);
            if (pass) lo = mid + 1; else hi = mid;
        }
        for (int i = 0; i < lo - 1; i++) cnt += in->csize[i];
        n = in->child[lo - 1];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    int lo = 0, hi = (int)lf->n.count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        double s = lf->elems[mid]->score;
        int pass = inclusive ? (s <= v) : (s < v);
        if (pass) lo = mid + 1; else hi = mid;
    }
    cnt += lo;
    if (lf_out) *lf_out = lf;
    if (idx_out) *idx_out = lo;
    return cnt;
}

/* Generic-predicate partition, used for lex ranges (the predicate has to
 * examine the member, so each binary-search probe dereferences one
 * element). */
typedef int (*zbtBeforeFn)(const zbtElem *e, void *arg);

static unsigned long zbtPartitionFn(zbtree *t, zbtBeforeFn before, void *arg,
                                    zbtLeaf **lf_out, int *idx_out) {
    unsigned long cnt = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int lo = 1, hi = (int)in->n.count;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (before(in->sep[mid], arg)) lo = mid + 1; else hi = mid;
        }
        for (int i = 0; i < lo - 1; i++) cnt += in->csize[i];
        n = in->child[lo - 1];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    int lo = 0, hi = (int)lf->n.count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (before(lf->elems[mid], arg)) lo = mid + 1; else hi = mid;
    }
    cnt += lo;
    if (lf_out) *lf_out = lf;
    if (idx_out) *idx_out = lo;
    return cnt;
}

/* Predicates for lex ranges. */
static int beforeNotGteMin(const zbtElem *e, void *arg) {
    return !zslLexValueGteMin(zbtGetEle((zbtElem *)e), (zlexrangespec *)arg);
}
static int beforeLteMax(const zbtElem *e, void *arg) {
    return zslLexValueLteMax(zbtGetEle((zbtElem *)e), (zlexrangespec *)arg);
}

/* Position 'it' on the partition point reported by a partition descent (the
 * first failing element) and return it, or NULL when the whole tree passes.
 *
 * When every element of the reported leaf passes, the partition point is the
 * separator of the next subtree, which in leaf order is the first element of
 * the next leaf. */
static zbtElem *zbtElemAtPartition(zbtLeaf *lf, int idx, zbtIter *it) {
    if (idx < (int)lf->n.count) {
        it->leaf = (zbtNode *)lf;
        it->idx = idx;
        return lf->elems[idx];
    }
    zbtLeaf *nx = lf->next;
    if (!nx || nx->n.count == 0) return NULL;
    it->leaf = (zbtNode *)nx;
    it->idx = 0;
    return nx->elems[0];
}

/* First element of a score range, or NULL if the range holds nothing.
 *
 * A single partition descent both counts the elements before the range start
 * and lands on the first element that is not before it. That element is in
 * range unless it already runs past the range end, which is the only extra
 * check needed: the order is monotonic, so if the first candidate exceeds the
 * end then no element can be in range. '*out_rank' receives its 1-based rank
 * and 'it' is positioned on it, ready to be stepped with zbtIterNext(). */
zbtElem *zbtFirstInRange(zbtree *t, zrangespec *range, unsigned long *out_rank,
                         zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtIter local;
    if (!it) it = &local;
    zbtLeaf *lf;
    int idx;
    unsigned long before =
        zbtPartitionScore(t, range->min, range->minex, &lf, &idx);
    zbtElem *e = zbtElemAtPartition(lf, idx, it);
    if (!e) return NULL;
    if (!zslValueLteMax(e->score, range)) return NULL;
    if (out_rank) *out_rank = before + 1;
    return e;
}

/* Last element of a score range, or NULL if the range holds nothing. The
 * partition counts everything up to the range end, so the last in-range
 * candidate is the element right before the partition point. */
zbtElem *zbtLastInRange(zbtree *t, zrangespec *range, unsigned long *out_rank,
                        zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf;
    int idx;
    unsigned long upto =
        zbtPartitionScore(t, range->max, !range->maxex, &lf, &idx);
    if (upto == 0) return NULL;
    /* A non-empty passing prefix always leaves the partition point at least
     * one slot into its leaf: the descent only enters a subtree whose minimum
     * passes, so the leaf's first element passes too. */
    serverAssert(idx >= 1);
    zbtElem *e = lf->elems[idx - 1];
    if (!zslValueGteMin(e->score, range)) return NULL;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = idx - 1; }
    if (out_rank) *out_rank = upto;
    return e;
}

/* Lex-range counterparts of the two functions above. */
zbtElem *zbtFirstInLexRange(zbtree *t, zlexrangespec *range,
                            unsigned long *out_rank, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtIter local;
    if (!it) it = &local;
    zbtLeaf *lf;
    int idx;
    unsigned long before = zbtPartitionFn(t, beforeNotGteMin, range, &lf, &idx);
    zbtElem *e = zbtElemAtPartition(lf, idx, it);
    if (!e) return NULL;
    if (!zslLexValueLteMax(zbtGetEle(e), range)) return NULL;
    if (out_rank) *out_rank = before + 1;
    return e;
}

zbtElem *zbtLastInLexRange(zbtree *t, zlexrangespec *range,
                           unsigned long *out_rank, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf;
    int idx;
    unsigned long upto = zbtPartitionFn(t, beforeLteMax, range, &lf, &idx);
    if (upto == 0) return NULL;
    serverAssert(idx >= 1);
    zbtElem *e = lf->elems[idx - 1];
    if (!zslLexValueGteMin(zbtGetEle(e), range)) return NULL;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = idx - 1; }
    if (out_rank) *out_rank = upto;
    return e;
}

/* Mirrors the skiplist zslNthIn*Range semantics: n >= 0 counts forward from
 * the first in-range element, n < 0 counts back from the last. The edge of
 * the range is found with one seek and the offset is then resolved by rank,
 * so the opposite edge never has to be located. Checking the resolved element
 * against the far bound is equivalent to comparing its rank with the opposite
 * edge's rank, the order being monotonic. */
zbtElem *zbtNthInRange(zbtree *t, zrangespec *range, long n,
                       unsigned long *out_rank, zbtIter *it) {
    if (n >= 0) {
        unsigned long frank;
        zbtElem *e = zbtFirstInRange(t, range, &frank, it);
        if (!e) return NULL;
        if (n > 0) {
            frank += (unsigned long)n;
            e = zbtElemByRank(t, frank, it);
            if (!e || !zslValueLteMax(e->score, range)) return NULL;
        }
        if (out_rank) *out_rank = frank;
        return e;
    }

    unsigned long lrank;
    zbtElem *e = zbtLastInRange(t, range, &lrank, it);
    if (!e) return NULL;
    if (n < -1) {
        long target = (long)lrank + 1 + n;
        if (target < 1) return NULL;
        lrank = (unsigned long)target;
        e = zbtElemByRank(t, lrank, it);
        if (!e || !zslValueGteMin(e->score, range)) return NULL;
    }
    if (out_rank) *out_rank = lrank;
    return e;
}

zbtElem *zbtNthInLexRange(zbtree *t, zlexrangespec *range, long n,
                          unsigned long *out_rank, zbtIter *it) {
    if (n >= 0) {
        unsigned long frank;
        zbtElem *e = zbtFirstInLexRange(t, range, &frank, it);
        if (!e) return NULL;
        if (n > 0) {
            frank += (unsigned long)n;
            e = zbtElemByRank(t, frank, it);
            if (!e || !zslLexValueLteMax(zbtGetEle(e), range)) return NULL;
        }
        if (out_rank) *out_rank = frank;
        return e;
    }

    unsigned long lrank;
    zbtElem *e = zbtLastInLexRange(t, range, &lrank, it);
    if (!e) return NULL;
    if (n < -1) {
        long target = (long)lrank + 1 + n;
        if (target < 1) return NULL;
        lrank = (unsigned long)target;
        e = zbtElemByRank(t, lrank, it);
        if (!e || !zslLexValueGteMin(zbtGetEle(e), range)) return NULL;
    }
    if (out_rank) *out_rank = lrank;
    return e;
}

/* Number of elements inside a score range. Two partition descents are enough:
 * the range holds everything up to its end minus everything before its start.
 * Inverted or empty ranges make the difference non-positive and count 0. */
unsigned long zbtCountInRange(zbtree *t, zrangespec *range) {
    if (t->length == 0) return 0;
    unsigned long before =
        zbtPartitionScore(t, range->min, range->minex, NULL, NULL);
    unsigned long upto =
        zbtPartitionScore(t, range->max, !range->maxex, NULL, NULL);
    return upto > before ? upto - before : 0;
}

/* Number of elements inside a lex range. See zbtCountInRange(). */
unsigned long zbtCountInLexRange(zbtree *t, zlexrangespec *range) {
    if (t->length == 0) return 0;
    unsigned long before = zbtPartitionFn(t, beforeNotGteMin, range, NULL, NULL);
    unsigned long upto = zbtPartitionFn(t, beforeLteMax, range, NULL, NULL);
    return upto > before ? upto - before : 0;
}

/*-----------------------------------------------------------------------------
 * Range deletion (also removes the members from the ZSET dict)
 *----------------------------------------------------------------------------*/

/* Drop the empty non-root leaf 'lf' out of the tree: unlink it from the sibling
 * chain, remove it from its parent and free it. Fixing up the parent can
 * cascade into inner-node rebalancing or collapse the root, but it never frees
 * another leaf, so leaf pointers held by the caller stay valid. */
static void zbtUnlinkEmptyLeaf(zbtree *t, zbtLeaf *lf) {
    serverAssert(lf->n.count == 0 && lf->n.parent);
    zbtInner *p = (zbtInner *)lf->n.parent;
    int pos = zbtChildIdx(p, (zbtNode *)lf);

    if (lf->prev) lf->prev->next = lf->next;
    else t->head = (zbtNode *)lf->next;
    if (lf->next) lf->next->prev = lf->prev;
    else t->tail = (zbtNode *)lf->prev;

    zbtRemoveChild(p, pos);
    zbtFreeNodeShallow(t, (zbtNode *)lf);

    /* The removal shifted the slots after 'pos' down; hand the fixup a slot
     * that still exists. Non-root inner nodes keep at least ZBT_INNER_MIN
     * children and the root collapses at one, so a slot always remains. */
    serverAssert(p->n.count >= 1);
    zbtFixupInnerAfterShrink(t, p,
        pos < (int)p->n.count ? pos : (int)p->n.count - 1);
}

/* Delete every element whose 1-based rank falls in [first, last] (inclusive),
 * removing each member from the companion dict 'd' as well.
 *
 * The window is contiguous in leaf order, so a single descent locates its start
 * and the sibling chain leads to the rest: no element is ever located
 * individually, and no leaf is descended to twice. Leaves that the window
 * consumes entirely are unlinked outright rather than refilled from a sibling
 * just to keep them alive. Cost is O(K) element work plus one rebalance per
 * emptied leaf, against O(K log N) for element-at-a-time deletion. */
static unsigned long zbtDeleteRankRange(zbtree *t, unsigned long first,
                                        unsigned long last, dict *d) {
    if (last > t->length) last = t->length;
    if (first < 1 || first > last) return 0;

    zbtIter it;
    if (!zbtElemByRank(t, first, &it)) return 0;
    zbtLeaf *lf = (zbtLeaf *)it.leaf;
    int idx = it.idx;

    unsigned long remaining = last - first + 1;
    unsigned long removed = 0;

    /* Only the first and the last leaf the window touches can come out of it
     * partially filled, and rebalancing them is left until the walk is over: a
     * merge frees a leaf, which could be the one the walk is about to step to.
     * The tree is not observable from outside in between (deleting from the
     * dict never reaches back into the tree), so a transiently underfull
     * boundary leaf is safe. */
    zbtLeaf *bfirst = NULL, *blast = NULL;

    while (remaining) {
        int avail = (int)lf->n.count - idx;
        int take = ((unsigned long)avail < remaining) ? avail : (int)remaining;
        zbtLeaf *next = lf->next; /* read before the tree is restructured */

        for (int k = 0; k < take; k++) {
            zbtElem *el = lf->elems[idx + k];
            dictDelete(d, zbtGetEle(el));
            t->alloc_size -= zmalloc_usable_size(el);
            zbtFreeElem(el);
        }
        memmove(&lf->elems[idx], &lf->elems[idx + take],
                ((int)lf->n.count - idx - take) * sizeof(zbtElem *));
        lf->n.count -= (uint32_t)take;
        t->length -= (unsigned long)take;
        removed += (unsigned long)take;
        remaining -= (unsigned long)take;

        if (lf->n.count == 0 && lf->n.parent) {
            zbtUnlinkEmptyLeaf(t, lf);
        } else {
            /* An empty leaf with no parent is the root of an empty tree, the
             * same shape zbtCreate() starts from; leave it in place. */
            zbtUpdateToRoot(t, (zbtNode *)lf);
            if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN) {
                if (!bfirst) bfirst = lf;
                else blast = lf;
            }
        }

        if (!remaining) break;
        lf = next;
        idx = 0;
        serverAssert(lf != NULL);
    }

    /* Rebalance the boundary leaves, the later one first: a merge always frees
     * the right-hand leaf of the pair it joins, so settling the later boundary
     * can never free the earlier one. Both counts are re-tested because
     * settling one boundary can fill the other. */
    if (blast && blast->n.parent && blast->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, blast);
    if (bfirst && bfirst->n.parent && bfirst->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, bfirst);
    return removed;
}

unsigned long zbtDeleteRangeByScore(zbtree *t, zrangespec *range, dict *d) {
    if (t->length == 0) return 0;
    unsigned long before =
        zbtPartitionScore(t, range->min, range->minex, NULL, NULL);
    unsigned long upto =
        zbtPartitionScore(t, range->max, !range->maxex, NULL, NULL);
    if (before >= upto) return 0;
    return zbtDeleteRankRange(t, before + 1, upto, d);
}

unsigned long zbtDeleteRangeByLex(zbtree *t, zlexrangespec *range, dict *d) {
    if (t->length == 0) return 0;
    unsigned long before = zbtPartitionFn(t, beforeNotGteMin, range, NULL, NULL);
    unsigned long upto = zbtPartitionFn(t, beforeLteMax, range, NULL, NULL);
    if (before >= upto) return 0;
    return zbtDeleteRankRange(t, before + 1, upto, d);
}

/* Delete elements whose 1-based rank is in [start, end] (inclusive). */
unsigned long zbtDeleteRangeByRank(zbtree *t, unsigned long start,
                                   unsigned long end, dict *d) {
    if (t->length == 0 || start > end) return 0;
    return zbtDeleteRankRange(t, start, end, d);
}

/*-----------------------------------------------------------------------------
 * Active defragmentation support
 *----------------------------------------------------------------------------*/

/* Replace element 'olde' with the (content-identical) relocated 'newe' in its
 * leaf slot and fix any separator pointers that referenced it. */
void zbtReplaceElem(zbtree *t, zbtElem *olde, zbtElem *newe) {
    zbtLeaf *lf = zbtFindLeaf(t, newe->score, zbtGetEle(newe));
    int found;
    int idx = zbtLeafSearch(lf, newe->score, zbtGetEle(newe), &found);
    serverAssert(found && lf->elems[idx] == olde);
    lf->elems[idx] = newe;
    zbtUpdateToRoot(t, (zbtNode *)lf);
}

static zbtNode *zbtDefragNode(zbtNode *n, void *(*fn)(void *)) {
    zbtNode *nn = fn(n);
    if (nn) n = nn;
    if (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < in->n.count; i++) {
            zbtNode *c = zbtDefragNode(in->child[i], fn);
            in->child[i] = c;
            c->parent = n;
        }
    }
    return n;
}

static void zbtCollectLeaves(zbtree *t, zbtNode *n, zbtLeaf **prev) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        lf->prev = *prev;
        if (*prev) (*prev)->next = lf;
        else t->head = (zbtNode *)lf;
        *prev = lf;
    } else {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < in->n.count; i++)
            zbtCollectLeaves(t, in->child[i], prev);
    }
}

/* Relocate every tree node using the provided defrag allocator, then rebuild
 * the leaf sibling chain and head/tail pointers. Element objects are handled
 * separately by the caller (via the ZSET dict scan + zbtReplaceElem). */
void zbtDefragNodes(zbtree *t, void *(*fn)(void *)) {
    t->root = zbtDefragNode(t->root, fn);
    t->root->parent = NULL;
    zbtLeaf *prev = NULL;
    zbtCollectLeaves(t, t->root, &prev);
    if (prev) prev->next = NULL;
    t->tail = (zbtNode *)prev;
}

/* Relocate a single leaf (if the allocator decides to move it) and repair all
 * external references: the parent child slot (or the root), the sibling links
 * and the head/tail pointers. Returns the current (possibly new) leaf. */
static zbtLeaf *zbtDefragRelocLeaf(zbtree *t, zbtLeaf *lf, void *(*fn)(void *)) {
    zbtLeaf *nl = fn(lf);
    if (!nl) return lf;
    if (nl->n.parent) {
        zbtInner *p = (zbtInner *)nl->n.parent;
        p->child[zbtChildIdx(p, (zbtNode *)lf)] = (zbtNode *)nl;
    } else {
        t->root = (zbtNode *)nl;
    }
    if (nl->prev) nl->prev->next = nl; else t->head = (zbtNode *)nl;
    if (nl->next) nl->next->prev = nl; else t->tail = (zbtNode *)nl;
    return nl;
}

/* Relocate a single inner node (if moved) and repair the grandparent child
 * slot (or the root) and every child's parent back-pointer. Returns the
 * current (possibly new) inner node. */
static zbtInner *zbtDefragRelocInner(zbtree *t, zbtInner *in, void *(*fn)(void *)) {
    zbtInner *ni = fn(in);
    if (!ni) return in;
    if (ni->n.parent) {
        zbtInner *p = (zbtInner *)ni->n.parent;
        p->child[zbtChildIdx(p, (zbtNode *)in)] = (zbtNode *)ni;
    } else {
        t->root = (zbtNode *)ni;
    }
    for (uint32_t i = 0; i < ni->n.count; i++)
        ni->child[i]->parent = (zbtNode *)ni;
    return ni;
}

/* Incremental variant of zbtDefragNodes(): relocate up to 'budget' tree nodes,
 * walking the leaves left to right and relocating each inner node right after
 * its last child is processed (post-order). The resume position is stored in
 * the tree as the (score, member) key of the first element of the next leaf to
 * process, so it survives element inserts/deletes/rebalances happening between
 * calls. Returns 1 if more work remains (bookmark saved) or 0 when the whole
 * tree has been relocated. Unlike zbtDefragNodes(), the leaf sibling chain and
 * head/tail are kept consistent after every single relocation, so the tree is
 * safe to query/modify between steps. */
int zbtDefragNodesIncremental(zbtree *t, void *(*fn)(void *), unsigned int budget) {
    zbtLeaf *lf;
    if (t->defrag_resume)
        lf = zbtFindLeaf(t, t->defrag_resume_score, t->defrag_resume);
    else
        lf = (zbtLeaf *)t->head;

    unsigned int work = 0;
    while (lf) {
        zbtLeaf *next = lf->next; /* value stays valid across relocation */
        lf = zbtDefragRelocLeaf(t, lf, fn);
        work++;

        /* Post-order: relocate ancestors whose last child we just completed. */
        zbtNode *c = (zbtNode *)lf;
        while (c->parent) {
            zbtInner *p = (zbtInner *)c->parent;
            if (p->child[p->n.count - 1] != c) break;
            c = (zbtNode *)zbtDefragRelocInner(t, p, fn);
            work++;
        }

        if (!next) {
            /* Processed the last leaf; the cascade above relocated the root. */
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
        lf = next;
    }

    /* Empty tree (single empty root leaf) or nothing left. */
    if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
    return 0;
}

/*-----------------------------------------------------------------------------
 * Debugging / test verification
 *----------------------------------------------------------------------------*/

#ifdef REDIS_TEST
#include <assert.h>
#include "testhelp.h"

/* Subtree minimum obtained by descending to the leftmost leaf. zbtNodeMin()
 * trusts sep[0] instead, so the verifier needs this independent version to
 * actually check the separator invariant against the tree contents. */
static zbtElem *zbtNodeMinDescend(zbtNode *n) {
    while (!n->isleaf) n = ((zbtInner *)n)->child[0];
    return ((zbtLeaf *)n)->elems[0];
}

static unsigned long zbtVerifyNode(zbtree *t, zbtNode *n, int depth,
                                   int *leafdepth) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        if (n->parent) serverAssert(n->count >= ZBT_LEAF_MIN);
        serverAssert(n->count <= ZBT_LEAF_MAX);
        if (*leafdepth == -1) *leafdepth = depth;
        else serverAssert(*leafdepth == depth); /* all leaves same depth */
        for (uint32_t i = 1; i < n->count; i++) {
            zbtElem *a = lf->elems[i - 1], *b = lf->elems[i];
            serverAssert(zbtCompare(a->score, zbtGetEle(a), b) < 0);
        }
        return n->count;
    }
    zbtInner *in = (zbtInner *)n;
    if (n->parent) serverAssert(n->count >= ZBT_INNER_MIN);
    serverAssert(n->count >= 2 || !n->parent);
    serverAssert(n->count <= ZBT_INNER_MAX);
    unsigned long total = 0;
    for (uint32_t i = 0; i < n->count; i++) {
        serverAssert(in->child[i]->parent == n);
        serverAssert(zbtNodeMinDescend(in->child[i]) == in->sep[i]);
        serverAssert(in->sepscore[i] == in->sep[i]->score);
        unsigned long cs = zbtVerifyNode(t, in->child[i], depth + 1, leafdepth);
        serverAssert(cs == in->csize[i]);
        total += cs;
    }
    return total;
}

/* Panics if any structural invariant is violated. */
void zbtDebugVerify(zbtree *t) {
    int leafdepth = -1;
    unsigned long total = zbtVerifyNode(t, t->root, 0, &leafdepth);
    serverAssert(total == t->length);

    /* Verify the leaf chain matches in-order traversal and head/tail. */
    unsigned long chain = 0;
    zbtLeaf *lf = (zbtLeaf *)t->head;
    zbtLeaf *plf = NULL;
    zbtElem *prev_elem = NULL;
    while (lf) {
        serverAssert(lf->prev == plf);
        for (uint32_t i = 0; i < lf->n.count; i++) {
            if (prev_elem) {
                zbtElem *cur = lf->elems[i];
                serverAssert(zbtCompare(prev_elem->score, zbtGetEle(prev_elem), cur) < 0);
            }
            prev_elem = lf->elems[i];
            chain++;
        }
        plf = lf;
        lf = lf->next;
    }
    serverAssert(chain == t->length);
    serverAssert(t->tail == (zbtNode *)plf || (t->length == 0));
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

int zbtreeTest(int argc, char **argv, int flags) {
    UNUSED(argc);
    UNUSED(argv);
    UNUSED(flags);

    printf("Testing B+ tree operations with structure verification\n");

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

    /* --- Duplicate-score-heavy workload ---
     * With only 7 distinct scores every separator comparison degenerates to
     * the member tie-break, exercising the sdscmp arm of the binary searches
     * through splits, merges and borrows. */
    {
        const int M = 5000;
        zbtree *dt = zbtCreate();
        zbtElem **held = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "dup:%d", i * 7919 % M);
            sds s = sdsnew(buf);
            held[i] = zbtInsert(dt, (double)(i % 7), s);
            sdsfree(s);
            if (i % 97 == 0) zbtDebugVerify(dt);
        }
        zbtDebugVerify(dt);
        for (int i = 0; i < M; i++) {
            zbtElem *e = held[i];
            unsigned long rank = zbtGetRank(dt, e->score, zbtGetEle(e));
            assert(rank != 0);
            assert(zbtElemByRank(dt, rank, NULL) == e);
            assert(zbtRankByElem(dt, e) == rank);
        }
        /* Delete two thirds in scattered order, verifying as we go. */
        for (int i = 0; i < M; i++) {
            if (i % 3 == 0) continue;
            zbtDeleteElem(dt, held[i]);
            if (i % 101 == 0) zbtDebugVerify(dt);
        }
        zbtDebugVerify(dt);
        assert(dt->length == (unsigned long)(M + 2) / 3);
        zbtFree(dt);
        test_cond("Duplicate-score tie-break workload", 1);
    }

    /* --- Range counting against a brute-force scan ---
     * Four elements share every score, so range bounds land both on and
     * between distinct scores. The tree is bulk built, which packs leaves
     * exactly full, so bounds at multiples of ZBT_LEAF_MAX/4 fall on leaf
     * boundaries and exercise the exact-separator descent. */
    {
        const int M = 3000;
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "cnt:%06d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem((double)(i / 4), s);
            sdsfree(s);
        }
        zbtree *ct = zbtCreate();
        zbtBuildFromSorted(ct, arr, M);
        zfree(arr);
        zbtDebugVerify(ct);

        static const double bounds[] = {-100, 0, 1, 15, 16, 63, 64, 100,
                                        374, 749, 750, 10000};
        int nb = (int)(sizeof(bounds) / sizeof(bounds[0]));
        for (int a = 0; a < nb; a++) {
            for (int b = 0; b < nb; b++) {
                for (int ex = 0; ex < 4; ex++) {
                    zrangespec rs = {.min = bounds[a], .max = bounds[b],
                                     .minex = ex & 1, .maxex = (ex >> 1) & 1};
                    /* Brute-force reference over the whole leaf chain. */
                    unsigned long want = 0, wfrank = 0, wlrank = 0;
                    zbtElem *wfirst = NULL, *wsecond = NULL, *wlast = NULL;
                    unsigned long rank = 0;
                    zbtIter bit;
                    for (zbtElem *e = zbtFirst(ct, &bit); e;
                         e = zbtIterNext(&bit)) {
                        rank++;
                        if (zslValueGteMin(e->score, &rs) &&
                            zslValueLteMax(e->score, &rs)) {
                            if (!want) { wfirst = e; wfrank = rank; }
                            else if (want == 1) wsecond = e;
                            wlast = e; wlrank = rank;
                            want++;
                        }
                    }
                    serverAssert(zbtCountInRange(ct, &rs) == want);

                    /* First/last seek: element, rank and iterator position. */
                    unsigned long got_rank = 0;
                    zbtIter fit;
                    zbtElem *f = zbtFirstInRange(ct, &rs, &got_rank, &fit);
                    serverAssert(f == wfirst);
                    if (f) {
                        serverAssert(got_rank == wfrank);
                        serverAssert(zbtIterNext(&fit) == wsecond ||
                                     want == 1);
                    }
                    zbtIter lit;
                    zbtElem *l = zbtLastInRange(ct, &rs, &got_rank, &lit);
                    serverAssert(l == wlast);
                    if (l) {
                        serverAssert(got_rank == wlrank);
                        serverAssert(zbtIterPrev(&lit) == NULL || want >= 1);
                    }

                    /* Offset-based access must agree with the reference. */
                    static const long offs[] = {0, 1, 2, -1, -2, -3, 5000};
                    for (int oi = 0; oi < (int)(sizeof(offs)/sizeof(offs[0]));
                         oi++) {
                        long nth = offs[oi];
                        unsigned long target = 0;
                        if (nth >= 0) {
                            if ((unsigned long)nth < want)
                                target = wfrank + (unsigned long)nth;
                        } else {
                            long back = -nth - 1;
                            if ((unsigned long)back < want)
                                target = wlrank - (unsigned long)back;
                        }
                        zbtElem *want_e = target ?
                            zbtElemByRank(ct, target, NULL) : NULL;
                        got_rank = 0;
                        zbtElem *got_e =
                            zbtNthInRange(ct, &rs, nth, &got_rank, NULL);
                        serverAssert(got_e == want_e);
                        if (got_e) serverAssert(got_rank == target);
                    }
                }
            }
        }
        zbtFree(ct);

        /* Lex ranges: one shared score forces every comparison through the
         * member, which is what the lex predicates look at. */
        const int L = 2000;
        arr = zmalloc(sizeof(zbtElem *) * L);
        for (int i = 0; i < L; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "lex:%06d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem(1.0, s);
            sdsfree(s);
        }
        zbtree *lt = zbtCreate();
        zbtBuildFromSorted(lt, arr, L);
        zfree(arr);
        zbtDebugVerify(lt);

        static const int lexb[] = {-1, 0, 1, 63, 64, 999, 1999, 2000};
        int nl = (int)(sizeof(lexb) / sizeof(lexb[0]));
        for (int a = 0; a < nl; a++) {
            for (int b = 0; b < nl; b++) {
                for (int ex = 0; ex < 4; ex++) {
                    char lo[32], hi[32];
                    snprintf(lo, sizeof(lo), "lex:%06d", lexb[a]);
                    snprintf(hi, sizeof(hi), "lex:%06d", lexb[b]);
                    zlexrangespec ls = {.min = sdsnew(lo), .max = sdsnew(hi),
                                        .minex = ex & 1, .maxex = (ex >> 1) & 1};
                    unsigned long want = 0, wfrank = 0, wlrank = 0;
                    zbtElem *wfirst = NULL, *wsecond = NULL, *wlast = NULL;
                    unsigned long rank = 0;
                    zbtIter bit;
                    for (zbtElem *e = zbtFirst(lt, &bit); e;
                         e = zbtIterNext(&bit)) {
                        sds v = zbtGetEle(e);
                        rank++;
                        if (zslLexValueGteMin(v, &ls) &&
                            zslLexValueLteMax(v, &ls)) {
                            if (!want) { wfirst = e; wfrank = rank; }
                            else if (want == 1) wsecond = e;
                            wlast = e; wlrank = rank;
                            want++;
                        }
                    }
                    serverAssert(zbtCountInLexRange(lt, &ls) == want);

                    unsigned long got_rank = 0;
                    zbtIter fit;
                    zbtElem *f = zbtFirstInLexRange(lt, &ls, &got_rank, &fit);
                    serverAssert(f == wfirst);
                    if (f) {
                        serverAssert(got_rank == wfrank);
                        serverAssert(zbtIterNext(&fit) == wsecond ||
                                     want == 1);
                    }
                    zbtElem *l = zbtLastInLexRange(lt, &ls, &got_rank, NULL);
                    serverAssert(l == wlast);
                    if (l) serverAssert(got_rank == wlrank);

                    /* Offsets, including past both ends of the range. */
                    static const long loffs[] = {0, 1, -1, -2, 3000};
                    for (int oi = 0; oi < (int)(sizeof(loffs)/sizeof(loffs[0]));
                         oi++) {
                        long nth = loffs[oi];
                        unsigned long target = 0;
                        if (nth >= 0) {
                            if ((unsigned long)nth < want)
                                target = wfrank + (unsigned long)nth;
                        } else {
                            long back = -nth - 1;
                            if ((unsigned long)back < want)
                                target = wlrank - (unsigned long)back;
                        }
                        zbtElem *want_e = target ?
                            zbtElemByRank(lt, target, NULL) : NULL;
                        got_rank = 0;
                        zbtElem *got_e =
                            zbtNthInLexRange(lt, &ls, nth, &got_rank, NULL);
                        serverAssert(got_e == want_e);
                        if (got_e) serverAssert(got_rank == target);
                    }
                    sdsfree(ls.min);
                    sdsfree(ls.max);
                }
            }
        }
        zbtFree(lt);
        test_cond("Range count/seek/nth match brute force", 1);
    }

    /* --- Bottom-up bulk build and batched range deletion --- */
    static const int sizes[] = {1, ZBT_LEAF_MAX, ZBT_LEAF_MAX + 1,
                                ZBT_LEAF_MAX * ZBT_INNER_MAX + 3, 5000};
    for (int trial = 0; trial < (int)(sizeof(sizes) / sizeof(sizes[0]));
         trial++) {
        /* Cover boundary sizes around leaf/inner fan-out multiples. */
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

    /* --- Ascending insertion (tail-append fast path) ---
     * Strictly increasing scores take the append path for every insert;
     * interleaving descending and random keys forces it to decline and fall
     * back to a descent, including right at the tail leaf's boundary. */
    {
        const int M = 6000;
        zbtree *at = zbtCreate();
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "asc:%06d", i);
            sds s = sdsnew(buf);
            zbtElem *e = zbtInsert(at, (double)i, s);
            sdsfree(s);
            serverAssert(zbtLast(at, NULL) == e); /* landed at the very end */
            if (i % 251 == 0) zbtDebugVerify(at);
        }
        zbtDebugVerify(at);
        serverAssert(at->length == (unsigned long)M);
        /* Same scores, members ascending: ties must still append in order. */
        for (int i = 0; i < 500; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "tie:%06d", i);
            sds s = sdsnew(buf);
            zbtElem *e = zbtInsert(at, (double)M, s);
            sdsfree(s);
            serverAssert(zbtLast(at, NULL) == e);
        }
        /* Descending and interior keys must not be appended. */
        for (int i = 0; i < 1500; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "mix:%06d", i);
            sds s = sdsnew(buf);
            zbtInsert(at, (double)(-i), s);
            sdsfree(s);
            snprintf(buf, sizeof(buf), "ins:%06d", i);
            sds s2 = sdsnew(buf);
            zbtInsert(at, (double)(rand() % M), s2);
            sdsfree(s2);
            if (i % 173 == 0) zbtDebugVerify(at);
        }
        zbtDebugVerify(at);
        serverAssert(at->length == (unsigned long)M + 500 + 3000);
        zbtIter ait;
        unsigned long ac = 0;
        zbtElem *aprev = NULL;
        for (zbtElem *e = zbtFirst(at, &ait); e; e = zbtIterNext(&ait)) {
            if (aprev)
                serverAssert(zbtCompare(aprev->score, zbtGetEle(aprev), e) < 0);
            aprev = e;
            ac++;
        }
        serverAssert(ac == at->length);
        zbtFree(at);
        test_cond("Ascending insert appends at the tail", 1);
    }

    /* --- Score updates ---
     * Deltas are sized to hit each path of zbtUpdateScore(): tiny ones leave
     * the element where it is, mid-sized ones move it inside its leaf, and
     * large ones push it into a different subtree entirely. zbtDebugVerify()
     * re-derives every separator by descending to the leftmost leaf, so it
     * catches a stale sep[]/sepscore[] left behind by an in-place update. */
    {
        const int M = 6000;
        zbtree *ut = zbtCreate();
        zbtElem **held = zmalloc(sizeof(zbtElem *) * M);
        srand(4242);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "up:%06d", i);
            sds s = sdsnew(buf);
            /* Dense scores, so a small delta really can cross a neighbour. */
            held[i] = zbtInsert(ut, (double)(i / 2), s);
            sdsfree(s);
        }
        zbtDebugVerify(ut);

        static const double deltas[] = {0.25, -0.25, 1, -1, 3, -3, 40, -40,
                                        5000, -5000};
        int nd = (int)(sizeof(deltas) / sizeof(deltas[0]));
        for (int round = 0; round < 6; round++) {
            for (int i = 0; i < M; i++) {
                zbtElem *e = held[rand() % M];
                double want = e->score + deltas[rand() % nd];
                zbtUpdateScore(ut, e, want);
                serverAssert(e->score == want);
                /* The element must remain findable at its new score, which
                 * only holds if every separator on the path is accurate. */
                serverAssert(zbtGetRank(ut, want, zbtGetEle(e)) != 0);
                serverAssert(zbtRankByElem(ut, e) ==
                             zbtGetRank(ut, want, zbtGetEle(e)));
            }
            zbtDebugVerify(ut);
            serverAssert(ut->length == (unsigned long)M);
        }

        /* Every member is still reachable, and the order is intact. */
        for (int i = 0; i < M; i++)
            serverAssert(zbtGetRank(ut, held[i]->score,
                                    zbtGetEle(held[i])) != 0);
        zbtIter uit;
        unsigned long uc = 0;
        zbtElem *uprev = NULL;
        for (zbtElem *e = zbtFirst(ut, &uit); e; e = zbtIterNext(&uit)) {
            if (uprev)
                serverAssert(zbtCompare(uprev->score, zbtGetEle(uprev), e) < 0);
            uprev = e;
            uc++;
        }
        serverAssert(uc == (unsigned long)M);
        zfree(held);
        zbtFree(ut);
        test_cond("Score update in place and across leaves", 1);
    }

    /* --- Random-window range deletion ---
     * Windows of every shape (whole leaves, partial leading/trailing leaves,
     * spans crossing many levels) are removed from a multi-level tree until it
     * is empty, verifying the structure and the dict after each one. This is
     * what covers leaves that a single delete leaves short by more than one
     * element, and the leaves the window empties outright. */
    {
        const int M = 8000;
        srand(9876);
        for (int round = 0; round < 12; round++) {
            dict *d = dictCreate(&zsetDictType);
            zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
            for (int i = 0; i < M; i++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "rw:%08d", i);
                sds s = sdsnew(buf);
                arr[i] = zbtCreateElem((double)(i / 3), s);
                sdsfree(s);
            }
            zbtree *rt = zbtCreate();
            zbtBuildFromSorted(rt, arr, M);
            for (int i = 0; i < M; i++)
                serverAssert(dictAdd(d, arr[i], NULL) == DICT_OK);
            zfree(arr);
            zbtDebugVerify(rt);

            while (rt->length) {
                unsigned long len = rt->length;
                unsigned long lo = (unsigned long)(rand() % (int)len) + 1;
                unsigned long span;
                switch (round % 4) {
                case 0: span = 1; break;                       /* single */
                case 1: span = ZBT_LEAF_MAX; break;            /* ~one leaf */
                case 2: span = ZBT_LEAF_MAX * 3 + 7; break;    /* many leaves */
                default: span = (unsigned long)(rand() % 200) + 1; break;
                }
                unsigned long hi = lo + span - 1;
                if (hi > len) hi = len;

                /* Reference: the members that should survive, in order. */
                unsigned long want = hi - lo + 1;
                unsigned long got = zbtDeleteRangeByRank(rt, lo, hi, d);
                serverAssert(got == want);
                serverAssert(rt->length == len - want);
                serverAssert(dictSize(d) == rt->length);
                zbtDebugVerify(rt);

                /* Ranks stay dense and consistent with the leaf order. */
                zbtIter vit;
                unsigned long seen = 0;
                for (zbtElem *e = zbtFirst(rt, &vit); e; e = zbtIterNext(&vit)) {
                    seen++;
                    serverAssert(zbtRankByElem(rt, e) == seen);
                }
                serverAssert(seen == rt->length);
            }
            serverAssert(dictSize(d) == 0);
            dictRelease(d);
            zbtFree(rt);
        }
        test_cond("Random-window range delete", 1);
    }

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

    return 0;
}
#endif
