/* dasl.c - DASL (array-packed skip list) ported to C with double keys.
 *
 * Phase 0 isolated prototype. See dasl.h for the high-level description.
 *
 * This is a CORRECTED port of DASL. A line-for-line transcription of the
 * upstream C++ Insert_usplit()/Delete() turned out to be unusable as an
 * ordered index: validating it against an oracle (differential + structural
 * checks under ASan/UBSan) surfaced two correctness bugs that are latent in
 * the reference (its benchmark only times operations, never checks results):
 *
 *   1. The next[]-only "fast path" descent cannot reach an unpromoted front
 *      node - a new global-minimum whose leader is absent from the express
 *      lanes. The reference uses it for Contains/Scan AND inside Insert/Delete,
 *      so reads miss keys and mutations operate on the wrong node.
 *   2. Insert_usplit double-promotes: a node that was promoted, then split
 *      (shrinking below DASL_ARR_SIZE), then re-filled to DASL_ARR_SIZE gets
 *      promoted again, creating duplicate express-lane leaders.
 *
 * The structure here keeps DASL's shape (each node packs up to DASL_ARR_SIZE
 * elements; the bottom level holds every element; upper levels are express
 * lanes whose next[i] descend one level) but fixes both problems:
 *
 *   - All descents (read and write) forward-step along each level
 *     (`while (forward && forward->scores[0] <= score) advance`) before dropping
 *     a level, so front nodes are always reachable (fixes #1).
 *   - A node is promoted to the next level exactly once: the first time it
 *     fills to DASL_ARR_SIZE *and* is not already represented one level up
 *     (`dasl_represented`); it keeps that single express-lane entry until it
 *     empties (fixes #2). Splits are even (DASL_ARR_SIZE/2) and the new
 *     right-hand node is the one promoted.
 *
 * Two node types. Level-0 data lives in the slim `daslLeaf` (raw double score +
 * owned member sds, 16 bytes/slot); the express-lane index levels and their
 * heads are `daslNode` (the same raw double score + member sds, plus weights[]
 * and next[]). next[i] on a level-1 index node points at a daslLeaf; on level
 * >= 2 it points at a daslNode. Index routing and leaf lookup share one total
 * order: compare the raw double, then sdscmp the full member on a tie.
 *
 * This variant is validated against a sorted-array oracle across insert-only,
 * mixed insert/delete, and full-drain workloads (many seeds, ASan/UBSan).
 */

#include "fmacros.h"

/* server.h pulls in the zset range-spec types (zrangespec/zlexrangespec), the
 * dict API, and the structure-agnostic range predicates (zslValueGteMin/LteMax,
 * zslLexValueGteMin/LteMax) that the range queries below reuse. It must precede
 * dasl.h so __REDIS_H is defined when dasl.h declares its range API. */
#include "server.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "zmalloc.h"
#include "dasl.h"

#define DASL_SPLIT (DASL_ARR_SIZE / 2)

/*-----------------------------------------------------------------------------
 * Per-level order-statistics weight storage
 *
 * weights[] is a trailing flexible array on daslNode (see dasl.h). Because this
 * port promotes every split-off / append-spill node immediately and propagates
 * (never drops) the express-lane entry of a surviving leader, every non-leftmost
 * node is represented exactly one level up. A level-h slot therefore covers a
 * single child node, so its weight is bounded by DASL_ARR_SIZE^h: <= 64 at
 * level 1, <= 64^2 at level 2, and so on. The per-level width below tracks that
 * bound exactly, and dasl_wset asserts the value fits in debug builds, so any
 * future change that breaks the promotion invariant (letting un-promoted
 * siblings accumulate under one slot) trips loudly rather than truncating. The
 * differential test (dasl_composite_test.c) also asserts the DASL_ARR_SIZE^h cap
 * after every mutation.
 *----------------------------------------------------------------------------*/

/* Width (bytes) of one weight slot at index `level` (>= 1), sized to the
 * DASL_ARR_SIZE^level cap (DASL_ARR_SIZE == 64). */
static inline size_t dasl_weight_width(int level) {
    if (level <= 1) return sizeof(uint8_t);  /* 1: <= 64 */
    if (level <= 2) return sizeof(uint16_t); /* 2: <= 64^2 = 4096 */
    if (level <= 5) return sizeof(uint32_t); /* 4: <= 64^5 = 2^30 */
    return sizeof(uint64_t);                 /* 8: anything larger */
}

/* The per-level widths above assume the DASL_ARR_SIZE^level slot-weight cap fits
 * the chosen type. DASL_ARR_SIZE is overridable, so enforce it at compile time:
 * a too-large fanout would silently overflow uint8/uint16/uint32 weights. */
_Static_assert((unsigned long long)DASL_ARR_SIZE <= 255ULL,
               "dasl_weight_width: level-1 uint8 needs DASL_ARR_SIZE^1 <= 255");
_Static_assert((unsigned long long)DASL_ARR_SIZE * DASL_ARR_SIZE <= 65535ULL,
               "dasl_weight_width: level-2 uint16 needs DASL_ARR_SIZE^2 <= 65535");
_Static_assert((unsigned long long)DASL_ARR_SIZE * DASL_ARR_SIZE * DASL_ARR_SIZE *
                   DASL_ARR_SIZE * DASL_ARR_SIZE <= 4294967295ULL,
               "dasl_weight_width: levels 3-5 uint32 need DASL_ARR_SIZE^5 <= 2^32-1");

/* Total allocation size of an index node at `level`, including weights[]. */
static inline size_t daslNodeSize(int level) {
    return sizeof(daslNode) + (size_t)DASL_ARR_SIZE * dasl_weight_width(level);
}

/* Read weight slot `i` of an index node at `level`. */
static inline unsigned long dasl_wget(const daslNode *n, int level, int i) {
    switch (dasl_weight_width(level)) {
    case 1:  return ((const uint8_t *)n->weights)[i];
    case 2:  return ((const uint16_t *)n->weights)[i];
    case 4:  return ((const uint32_t *)n->weights)[i];
    default: return (unsigned long)((const uint64_t *)n->weights)[i];
    }
}

/* Write weight slot `i` of an index node at `level`. */
static inline void dasl_wset(daslNode *n, int level, int i, unsigned long w) {
    switch (dasl_weight_width(level)) {
    case 1:  assert(w <= UINT8_MAX);  ((uint8_t *)n->weights)[i] = (uint8_t)w; break;
    case 2:  assert(w <= UINT16_MAX); ((uint16_t *)n->weights)[i] = (uint16_t)w; break;
    case 4:  assert(w <= UINT32_MAX); ((uint32_t *)n->weights)[i] = (uint32_t)w; break;
    default: ((uint64_t *)n->weights)[i] = (uint64_t)w; break;
    }
}

/* Accumulate weights[lo, hi) of an index node at `level` into `dst`. Hoists the
 * width switch out of the per-slot loop, since rank/range scans are hot. A hi
 * of -1 (no qualifying slot) yields an empty sum. */
#define DASL_WSUM(dst, n, level, lo, hi) do {                              \
    int _lo = (lo), _hi = (hi);                                            \
    switch (dasl_weight_width(level)) {                                    \
    case 1: { const uint8_t *_w = (const uint8_t *)(n)->weights;           \
              for (int _i = _lo; _i < _hi; _i++) (dst) += _w[_i]; } break; \
    case 2: { const uint16_t *_w = (const uint16_t *)(n)->weights;         \
              for (int _i = _lo; _i < _hi; _i++) (dst) += _w[_i]; } break; \
    case 4: { const uint32_t *_w = (const uint32_t *)(n)->weights;         \
              for (int _i = _lo; _i < _hi; _i++) (dst) += _w[_i]; } break; \
    default:{ const uint64_t *_w = (const uint64_t *)(n)->weights;         \
              for (int _i = _lo; _i < _hi; _i++) (dst) += _w[_i]; } break; \
    }                                                                      \
} while (0)

/* Canonicalize -0.0 to +0.0 (zset treats signed zeros as equal). */
static inline double daslCanonScore(double s) { return s == 0 ? 0.0 : s; }

/* Composite comparison: raw double then sdscmp. Doubles compare in numeric
 * order, so this is the total order at every level - index nodes and leaves
 * alike store the score as a raw double and compare it the same way. Only a
 * score tie falls through to sdscmp on the full members. */
static inline int dasl_cmp(double sa, sds ma, double sb, sds mb) {
    if (sa < sb) return -1;
    if (sa > sb) return 1;
    return sdscmp(ma, mb);
}

/* Greatest index slot i in [0, n_key) with (scores[i], members[i]) <=
 * (score, mem), or -1 if none. Branchless binary search over the packed node
 * (~log2(DASL_ARR_SIZE) compares); see the original rationale - dasl_cmp's own
 * early-outs are well predicted, the loop control is not, so it is cmov-shaped. */
static int dasl_find_le(const daslNode *n, double score, sds mem) {
    int lo = 0, len = n->n_key;
    while (len > 0) {
        int half = len >> 1;
        int mid = lo + half;
        int le = dasl_cmp(n->scores[mid], n->members[mid], score, mem) <= 0;
        lo  = le ? mid + 1 : lo;
        len = le ? len - half - 1 : half;
    }
    return lo - 1;
}

/* Leaf flavour of dasl_find_le: greatest slot i with (scores[i], members[i]) <=
 * (score, mem), or -1 if none (including the empty leaf head, n_key == 0). */
static int dasl_leaf_find_le(const daslLeaf *n, double score, sds mem) {
    int lo = 0, len = n->n_key;
    while (len > 0) {
        int half = len >> 1;
        int mid = lo + half;
        int le = dasl_cmp(n->scores[mid], n->members[mid], score, mem) <= 0;
        lo  = le ? mid + 1 : lo;
        len = le ? len - half - 1 : half;
    }
    return lo - 1;
}

/* Does the node whose leader is (lscore, lmem) already own an express-lane
 * index entry one level up? A node graduates to the next level exactly once -
 * the first time it fills to DASL_ARR_SIZE - and keeps that entry until it
 * empties; after a split shrinks it below DASL_ARR_SIZE it keeps the entry, so
 * re-filling it must NOT promote it again (that is the duplicate-leader
 * corruption).
 *
 * `cover` is the covering node recorded by the descent at the level above (the
 * last node there with leader <= the inserted key, or that level's head). The
 * mirroring entry, if it exists, can only sit in `cover` itself or in cover's
 * first forward node. O(DASL_ARR_SIZE) local check. */
static int dasl_represented(const daslNode *cover, double lscore, sds lmem) {
    if (cover == NULL) return 0;
    int idx = dasl_find_le(cover, lscore, lmem);
    if (idx >= 0 && dasl_cmp(cover->scores[idx], cover->members[idx], lscore, lmem) == 0) return 1;
    if (cover->forward != NULL &&
        dasl_cmp(cover->forward->scores[0], cover->forward->members[0], lscore, lmem) == 0) return 1;
    return 0;
}

/* Sum of an index node's per-slot weights = number of level-0 elements under
 * it. */
static unsigned long dasl_nodeweight(const daslNode *n, int level) {
    unsigned long w = 0;
    DASL_WSUM(w, n, level, 0, n->n_key);
    return w;
}

/* Weight crossed when leaving index node `x` at level `h` (h >= 1): a real node
 * contributes the sum of its slot weights; a head contributes its stored prefix
 * weight. */
static unsigned long dasl_coverweight(const dasl *sl, const daslNode *x, int h) {
    if (x == sl->head[h]) return dasl_wget(x, h, 0);
    return dasl_nodeweight(x, h);
}

/* Weight of one index slot, computed from the immediate children's node weights
 * (the level below must already carry correct weights). Slot i of `parent`
 * covers the level-(child_level) nodes from next[i] up to (but excluding) the
 * child the next slot - or parent->forward's first slot - points at. The
 * children are leaves when child_level == 0, index nodes otherwise. */
static unsigned long dasl_index_slot_weight(const daslNode *parent, int i, int child_level) {
    void *stop = NULL;
    if (i + 1 < parent->n_key) stop = parent->next[i + 1];
    else if (parent->forward != NULL) stop = parent->forward->next[0];
    unsigned long w = 0;
    if (child_level == 0) {
        for (daslLeaf *m = parent->next[i]; m != NULL && (void *)m != stop; m = m->forward)
            w += (unsigned long)m->n_key;
    } else {
        for (daslNode *m = parent->next[i]; m != NULL && (void *)m != stop; m = m->forward)
            w += dasl_nodeweight(m, child_level);
    }
    return w;
}

/* Allocate an index node (level >= 1) holding the single element (score, mem)
 * with the borrowed member `mem`. weights[]/next[] are zeroed; weights are set
 * by daslRecomputePath after the mutation completes. */
static daslNode *daslNewNode(dasl *sl, int level, double score, sds mem) {
    size_t usable;
    daslNode *n = zmalloc_usable(daslNodeSize(level), &usable);
    n->forward = NULL;
    n->prev = NULL;
    n->n_key = 1;
    n->scores[0] = score;
    n->members[0] = mem;
    for (int i = 1; i < DASL_ARR_SIZE; i++) { n->scores[i] = 0; n->members[i] = NULL; }
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->next[i] = NULL;
    memset(n->weights, 0, (size_t)DASL_ARR_SIZE * dasl_weight_width(level));
    sl->alloc_size += usable;
    return n;
}

/* Allocate an empty index head (n_key == 0) at `level`. */
static daslNode *daslNewHead(dasl *sl, int level) {
    size_t usable;
    daslNode *n = zmalloc_usable(daslNodeSize(level), &usable);
    n->forward = NULL;
    n->prev = NULL;
    n->n_key = 0;
    for (int i = 0; i < DASL_ARR_SIZE; i++) { n->scores[i] = 0; n->members[i] = NULL; }
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->next[i] = NULL;
    memset(n->weights, 0, (size_t)DASL_ARR_SIZE * dasl_weight_width(level));
    sl->alloc_size += usable;
    return n;
}

/* Allocate an empty leaf (n_key == 0); used for the leaf head and as the blank
 * right sibling of a leaf split. */
static daslLeaf *daslNewLeafBlank(dasl *sl) {
    size_t usable;
    daslLeaf *n = zmalloc_usable(sizeof(*n), &usable);
    n->forward = NULL;
    n->prev = NULL;
    n->n_key = 0;
    for (int i = 0; i < DASL_ARR_SIZE; i++) { n->scores[i] = 0; n->members[i] = NULL; }
    sl->alloc_size += usable;
    return n;
}

/* Allocate a leaf holding the single element (score, mem) (mem is owned). */
static daslLeaf *daslNewLeaf(dasl *sl, double score, sds mem) {
    daslLeaf *n = daslNewLeafBlank(sl);
    n->scores[0] = score;
    n->members[0] = mem;
    n->n_key = 1;
    return n;
}

dasl *daslCreate(void) {
    size_t usable;
    dasl *sl = zmalloc_usable(sizeof(*sl), &usable);
    sl->max_height = 1;
    sl->length = 0;
    sl->alloc_size = usable;
    sl->tail = NULL;
    /* Index heads are allocated lazily per level; only the leaf head exists. */
    for (int i = 0; i < DASL_MAXHEIGHT; i++) sl->head[i] = NULL;
    sl->lhead = daslNewLeafBlank(sl);
    return sl;
}

void daslFree(dasl *sl) {
    /* Free the leaf chain (head + leaves), freeing each owned member. */
    daslLeaf *lf = sl->lhead;
    while (lf) {
        daslLeaf *next = lf->forward;
        for (int j = 0; j < lf->n_key; j++) sdsfree(lf->members[j]);
        zfree(lf);
        lf = next;
    }
    /* Free each index level's chain (head included); members there are
     * borrowed and must not be freed. */
    for (int i = 1; i < sl->max_height; i++) {
        daslNode *node = sl->head[i];
        while (node) {
            daslNode *next = node->forward;
            zfree(node);
            node = next;
        }
    }
    zfree(sl);
}

size_t daslAllocSize(const dasl *sl) { return sl->alloc_size; }

/* Hint the OS that the structure's memory is not needed before a fork (see
 * dismissMemory()). Walks the bottom-level chain, dismissing each leaf and its
 * owned member sds. Express-lane nodes/members are borrowed and skipped. */
void daslDismiss(dasl *sl) {
    daslLeaf *x = sl->lhead ? sl->lhead->forward : NULL;
    while (x != NULL) {
        daslLeaf *next = x->forward;
        for (int j = 0; j < x->n_key; j++) {
            sds m = x->members[j];
            if (m) dismissMemory(sdsAllocPtr(m), sdsAllocSize(m));
        }
        dismissMemory(x, 0);
        x = next;
    }
}

/* Grow the structure so that index level `h` (h >= 1) has an allocated head. */
static void daslEnsureHeight(dasl *sl, int h) {
    while (sl->max_height <= h) {
        sl->head[sl->max_height] = daslNewHead(sl, sl->max_height);
        sl->max_height++;
    }
}

/* Forward-aware descent: walk down from the top index level, at each level
 * stepping forward while the next node's leader is <= (score, mem), then
 * descending via the matching next[] pointer (or to the level-below head when
 * the target precedes every entry). Lands on the level-0 leaf that would
 * contain (score, mem). If `prev` is non-NULL it is filled with the covering
 * index node at each level [1, max_height-1]. Returns the level-0 leaf. */
static daslLeaf *daslDescend(const dasl *sl, double score, sds mem,
                             daslNode **prev) {
    int h = sl->max_height - 1;
    if (h == 0) {
        /* No index levels: scan the leaf chain from its head. */
        daslLeaf *lf = sl->lhead;
        while (lf->forward != NULL &&
               dasl_cmp(lf->forward->scores[0], lf->forward->members[0], score, mem) <= 0)
            lf = lf->forward;
        return lf;
    }
    daslNode *x = sl->head[h];
    while (1) {
        while (x->forward != NULL &&
               dasl_cmp(x->forward->scores[0], x->forward->members[0], score, mem) <= 0)
            x = x->forward;
        if (prev) prev[h] = x;
        int j = dasl_find_le(x, score, mem);
        if (h == 1) {
            /* Children are leaves. */
            daslLeaf *lf = (j < 0) ? sl->lhead : (daslLeaf *)x->next[j];
            while (lf->forward != NULL &&
                   dasl_cmp(lf->forward->scores[0], lf->forward->members[0], score, mem) <= 0)
                lf = lf->forward;
            return lf;
        }
        daslNode *nx = (j < 0) ? sl->head[h - 1] : (daslNode *)x->next[j];
        __builtin_prefetch(nx);
        x = nx;
        h--;
    }
}

/* Recompute the order-statistics weights affected by mutating (score, mem) by
 * `delta` (+1 for an insert, -1 for a delete). See the original commentary: the
 * common no-split insert/delete is O(height) (one region per level adjusted by
 * delta); structurally rearranged levels [1, top] are rebuilt from children.
 *
 * `added[h]`/`kept[h]` are split siblings / append-kept index nodes at level h.
 * `path_in` is the post-mutation covering path the caller already holds (valid
 * only for a plain in-node insert/delete); otherwise the path is rediscovered
 * here, which also yields the correct covers after a leader propagation. */
static void daslRecomputePath(dasl *sl, double score, sds mem,
                              daslNode **added, daslNode **kept, daslNode **path_in,
                              int top, int delta) {
    daslNode *path[DASL_MAXHEIGHT];
    if (path_in != NULL) {
        memcpy(path, path_in, sizeof(path));
    } else {
        for (int i = 0; i < DASL_MAXHEIGHT; i++) path[i] = NULL;
        daslDescend(sl, score, mem, path);
    }

    /* Level 0 carries no stored weights (every leaf slot is the implicit
     * constant 1), so nothing to recompute there. */

    /* Index levels bottom-up. */
    for (int h = 1; h < sl->max_height; h++) {
        if (h <= top) {
            /* Structurally modified: covering node on the path, any split
             * sibling, and any append-split "kept full" node rebuilt in full
             * from children. */
            daslNode *cand[3] = { path[h], added[h], kept ? kept[h] : NULL };
            for (int k = 0; k < 3; k++) {
                daslNode *x = cand[k];
                if (x == NULL || x == sl->head[h]) continue;
                for (int i = 0; i < x->n_key; i++)
                    dasl_wset(x, h, i, dasl_index_slot_weight(x, i, h - 1));
            }
        } else {
            /* Layout unchanged: the element sits under exactly one covering
             * region. A real covering node's slot is adjusted here; the
             * head-prefix case (path[h] == head[h]) is owned by the loop below. */
            daslNode *x = path[h];
            if (x != sl->head[h]) {
                int wi = dasl_find_le(x, score, mem);
                dasl_wset(x, h, wi, dasl_wget(x, h, wi) + delta);
            }
        }
    }

    /* Per-level head prefixes: level-0 elements before head[h]->forward. The
     * leaf head has no stored prefix (it is always 0). */
    for (int h = 1; h < sl->max_height; h++) {
        if (h <= top) {
            /* Built from the level below. */
            daslNode *f = sl->head[h]->forward;
            if (h - 1 == 0) {
                /* The level below is the leaf chain. */
                daslLeaf *stop = f ? (daslLeaf *)f->next[0] : NULL;
                unsigned long pref = 0;
                for (daslLeaf *m = sl->lhead->forward; m != NULL && m != stop; m = m->forward)
                    pref += (unsigned long)m->n_key;
                dasl_wset(sl->head[h], h, 0, pref);
            } else {
                daslNode *stop = f ? (daslNode *)f->next[0] : NULL;
                unsigned long pref = dasl_wget(sl->head[h - 1], h - 1, 0);
                for (daslNode *m = sl->head[h - 1]->forward; m != NULL && m != stop; m = m->forward)
                    pref += dasl_nodeweight(m, h - 1);
                dasl_wset(sl->head[h], h, 0, pref);
            }
        } else if (path[h] == sl->head[h]) {
            /* head[h]->forward is unchanged; the element falls in its prefix. */
            dasl_wset(sl->head[h], h, 0, dasl_wget(sl->head[h], h, 0) + delta);
        }
    }
}

/* A node's leader changed from (olds, old_mem) to (news, new_mem); rewrite the
 * mirroring index entry on each level above `level`, stopping once a level's
 * entry is not the leader (idx != 0) since higher levels then cannot mirror
 * it. */
static void daslPropagateLeader(dasl *sl, daslNode **prev, int level,
                                double olds, sds old_mem,
                                double news, sds new_mem) {
    for (int h = level + 1; h < sl->max_height; h++) {
        daslNode *p = prev[h];
        int idx = dasl_find_le(p, olds, old_mem);
        daslNode *target = p;
        if (!(idx >= 0 && dasl_cmp(p->scores[idx], p->members[idx], olds, old_mem) == 0)) {
            if (p->forward != NULL &&
                dasl_cmp(p->forward->scores[0], p->forward->members[0], olds, old_mem) == 0) {
                target = p->forward;
                idx = 0;
            } else {
                if (idx != 0) break; else continue;
            }
        }
        target->scores[idx] = news;
        target->members[idx] = new_mem;
        if (idx != 0) break;
    }
}

int daslContains(const dasl *sl, double score, sds ele) {
    double cs = daslCanonScore(score);
    daslLeaf *x = daslDescend(sl, cs, ele, NULL);
    int j = dasl_leaf_find_le(x, cs, ele);
    return j >= 0 && x->scores[j] == cs && sdscmp(x->members[j], ele) == 0;
}

/* Insert taking ownership of `ele` (no copy is made). daslInsert() wraps this
 * with an sdsdup so the public contract (caller keeps ownership) is preserved;
 * the score-reposition path uses it directly to keep the exact same buffer.
 *
 * The caller MUST guarantee the (score, member) is not already present. */
static daslCursor daslInsertOwned(dasl *sl, double score, sds ele) {
    double cscore = daslCanonScore(score);
    sds owned = ele;

    daslNode *prev[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) prev[i] = NULL;
    daslLeaf *cover_leaf = daslDescend(sl, cscore, owned, prev);

    /* Carried up the express lanes when a leaf/node graduates. */
    int level = 0;
    double up_score = cscore;
    sds up_mem = owned;
    void *down = NULL;       /* child the new index entry should point at */
    int to_index = 0;        /* leaf phase produced an index promotion */

    int leader_changed = 0;  /* an inserted element became a node leader: weights
                              * above the rebuilt levels need the full recompute */
    int can_reuse = 0;       /* plain in-leaf insert: reuse the descent covers */

    /* Level-0 leaf + slot that ends up holding the element (for the cursor). */
    daslLeaf *ins_node = NULL;
    int ins_slot = 0;

    daslNode *added[DASL_MAXHEIGHT];
    daslNode *kept[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) { added[i] = NULL; kept[i] = NULL; }

    /*-------------------------------------------------------------------------
     * Level-0 (leaf) phase.
     *-----------------------------------------------------------------------*/
    int at_head = (cover_leaf == sl->lhead);
    if (at_head && cover_leaf->forward == NULL) {
        /* Empty structure: create the first leaf. */
        daslLeaf *nn = daslNewLeaf(sl, cscore, owned);
        nn->prev = sl->lhead;
        sl->lhead->forward = nn;
        sl->tail = nn;
        ins_node = nn; ins_slot = 0;
    } else {
        daslLeaf *lt = at_head ? cover_leaf->forward : cover_leaf;
        int idx = dasl_leaf_find_le(lt, cscore, owned);
        if (idx == 0 && dasl_cmp(lt->scores[0], lt->members[0], cscore, owned) > 0) idx = -1;

        if (lt->n_key < DASL_ARR_SIZE) {
            /* Room: shift right of idx and insert at idx+1. */
            int pos = idx + 1;
            double old_ls = lt->scores[0];
            sds old_lm = lt->members[0];
            memmove(&lt->scores[pos + 1], &lt->scores[pos], (lt->n_key - pos) * sizeof(double));
            memmove(&lt->members[pos + 1], &lt->members[pos], (lt->n_key - pos) * sizeof(sds));
            lt->scores[pos] = cscore;
            lt->members[pos] = owned;
            lt->n_key++;
            ins_node = lt; ins_slot = pos;
            if (idx == -1) { /* leader changed */
                leader_changed = 1;
                daslPropagateLeader(sl, prev, 0, old_ls, old_lm, cscore, owned);
            }
            daslNode *cover = (1 < sl->max_height) ? prev[1] : NULL;
            if (lt->n_key == DASL_ARR_SIZE && !dasl_represented(cover, lt->scores[0], lt->members[0])) {
                up_score = lt->scores[0]; up_mem = lt->members[0]; down = lt;
                level = 1; to_index = 1; daslEnsureHeight(sl, 1);
            } else {
                can_reuse = (idx != -1);
            }
        } else if (idx == lt->n_key - 1) {
            /* Full leaf, key appends at the tail: keep lt full, spill the new
             * key into a fresh right leaf (append-only fill stays ~100%). */
            daslLeaf *add = daslNewLeaf(sl, cscore, owned);
            add->forward = lt->forward;
            add->prev = lt;
            if (add->forward != NULL) add->forward->prev = add;
            lt->forward = add;
            if (add->forward == NULL) sl->tail = add;
            ins_node = add; ins_slot = 0;
            up_score = add->scores[0];
            up_mem = add->members[0]; down = add;
            level = 1; to_index = 1; daslEnsureHeight(sl, 1);
        } else {
            /* Full leaf: even split. Upper half moves to a new right leaf. */
            daslLeaf *add = daslNewLeafBlank(sl);
            memcpy(add->scores, &lt->scores[DASL_SPLIT], DASL_SPLIT * sizeof(double));
            memcpy(add->members, &lt->members[DASL_SPLIT], DASL_SPLIT * sizeof(sds));
            add->n_key = DASL_SPLIT;
            add->forward = lt->forward;
            add->prev = lt;
            if (add->forward != NULL) add->forward->prev = add;
            lt->forward = add;
            if (add->forward == NULL) sl->tail = add;
            for (int i = DASL_SPLIT; i < DASL_ARR_SIZE; i++) { lt->scores[i] = 0; lt->members[i] = NULL; }
            lt->n_key = DASL_SPLIT;

            daslLeaf *into;
            int half_idx;
            if (idx < DASL_SPLIT) { into = lt; half_idx = idx; }
            else { into = add; half_idx = idx - DASL_SPLIT; }
            double old_ls = into->scores[0];
            sds old_lm = into->members[0];
            int pos = half_idx + 1;
            memmove(&into->scores[pos + 1], &into->scores[pos], (into->n_key - pos) * sizeof(double));
            memmove(&into->members[pos + 1], &into->members[pos], (into->n_key - pos) * sizeof(sds));
            into->scores[pos] = cscore;
            into->members[pos] = owned;
            into->n_key++;
            ins_node = into; ins_slot = pos;
            if (into == lt && half_idx == -1) {
                leader_changed = 1;
                daslPropagateLeader(sl, prev, 0, old_ls, old_lm, cscore, owned);
            }
            /* The split-off leaf must get an index entry one level up. */
            up_score = add->scores[0];
            up_mem = add->members[0]; down = add;
            level = 1; to_index = 1; daslEnsureHeight(sl, 1);
        }
    }

    /*-------------------------------------------------------------------------
     * Index (level >= 1) phase: insert (up_score, up_mem) pointing at `down`,
     * splitting and promoting as needed. Mirrors the leaf phase on the daslNode
     * arrays.
     *-----------------------------------------------------------------------*/
    while (to_index) {
        if (level >= sl->max_height || prev[level] == NULL) {
            daslEnsureHeight(sl, level);
            prev[level] = sl->head[level];
        }
        daslNode *p = prev[level];
        int at_head_n = (p == sl->head[level]);

        if (at_head_n && p->forward == NULL) {
            /* Empty level: create the first index node. */
            daslNode *nn = daslNewNode(sl, level, up_score, up_mem);
            nn->next[0] = down;
            nn->prev = p;
            p->forward = nn;
            added[level] = nn;
            down = nn;
            break;
        }

        daslNode *target = at_head_n ? p->forward : p;
        int idx = dasl_find_le(target, up_score, up_mem);
        if (idx == 0 && dasl_cmp(target->scores[0], target->members[0], up_score, up_mem) > 0) idx = -1;

        if (target->n_key < DASL_ARR_SIZE) {
            int pos = idx + 1;
            double old_leader = target->scores[0];
            sds old_leader_mem = target->members[0];
            memmove(&target->scores[pos + 1], &target->scores[pos], (target->n_key - pos) * sizeof(double));
            memmove(&target->members[pos + 1], &target->members[pos], (target->n_key - pos) * sizeof(sds));
            memmove(&target->next[pos + 1], &target->next[pos], (target->n_key - pos) * sizeof(void *));
            {
                size_t ws = dasl_weight_width(level);
                memmove(target->weights + (size_t)(pos + 1) * ws, target->weights + (size_t)pos * ws,
                        (size_t)(target->n_key - pos) * ws);
            }
            target->next[pos] = down;
            target->scores[pos] = up_score;
            target->members[pos] = up_mem;
            target->n_key++;
            if (idx == -1) {
                leader_changed = 1;
                daslPropagateLeader(sl, prev, level, old_leader, old_leader_mem, up_score, up_mem);
            }
            down = target;
            daslNode *cover = (level + 1 < sl->max_height) ? prev[level + 1] : NULL;
            if (target->n_key == DASL_ARR_SIZE &&
                !dasl_represented(cover, target->scores[0], target->members[0])) {
                up_score = target->scores[0];
                up_mem = target->members[0];
                level++;
                continue;
            }
            break;
        } else if (idx == target->n_key - 1) {
            /* Full node, key appends: keep target full, spill into a new node. */
            kept[level] = target;
            daslNode *add = daslNewNode(sl, level, up_score, up_mem);
            add->next[0] = down;
            add->forward = target->forward;
            add->prev = target;
            if (add->forward != NULL) add->forward->prev = add;
            target->forward = add;
            added[level] = add;
            down = add;
            up_score = add->scores[0];
            up_mem = add->members[0];
            level++;
            daslEnsureHeight(sl, level);
            continue;
        } else {
            /* Full: even split. Upper half moves to a new right node `add`. */
            daslNode *add = daslNewNode(sl, level, target->scores[DASL_SPLIT], target->members[DASL_SPLIT]);
            size_t ws = dasl_weight_width(level);
            memcpy(add->scores, &target->scores[DASL_SPLIT], DASL_SPLIT * sizeof(double));
            memcpy(add->members, &target->members[DASL_SPLIT], DASL_SPLIT * sizeof(sds));
            memcpy(add->next, &target->next[DASL_SPLIT], DASL_SPLIT * sizeof(void *));
            memcpy(add->weights, target->weights + (size_t)DASL_SPLIT * ws, (size_t)DASL_SPLIT * ws);
            add->n_key = DASL_SPLIT;
            add->forward = target->forward;
            add->prev = target;
            if (add->forward != NULL) add->forward->prev = add;
            target->forward = add;
            added[level] = add;
            for (int i = DASL_SPLIT; i < DASL_ARR_SIZE; i++) {
                target->scores[i] = 0;
                target->members[i] = NULL;
                target->next[i] = NULL;
            }
            memset(target->weights + (size_t)DASL_SPLIT * ws, 0, (size_t)DASL_SPLIT * ws);
            target->n_key = DASL_SPLIT;

            daslNode *into;
            int half_idx;
            if (idx < DASL_SPLIT) { into = target; half_idx = idx; }
            else { into = add; half_idx = idx - DASL_SPLIT; }
            double old_leader = into->scores[0];
            sds old_leader_mem = into->members[0];
            int pos = half_idx + 1;
            memmove(&into->scores[pos + 1], &into->scores[pos], (into->n_key - pos) * sizeof(double));
            memmove(&into->members[pos + 1], &into->members[pos], (into->n_key - pos) * sizeof(sds));
            memmove(&into->next[pos + 1], &into->next[pos], (into->n_key - pos) * sizeof(void *));
            memmove(into->weights + (size_t)(pos + 1) * ws, into->weights + (size_t)pos * ws,
                    (size_t)(into->n_key - pos) * ws);
            into->next[pos] = down;
            into->scores[pos] = up_score;
            into->members[pos] = up_mem;
            into->n_key++;
            if (into == target && half_idx == -1) {
                leader_changed = 1;
                daslPropagateLeader(sl, prev, level, old_leader, old_leader_mem, up_score, up_mem);
            }
            down = add;
            up_score = add->scores[0];
            up_mem = add->members[0];
            level++;
            daslEnsureHeight(sl, level);
            continue;
        }
    }

    sl->length++;
    sl->alloc_size += sdsAllocSize(owned);
    if (can_reuse) {
        daslRecomputePath(sl, cscore, owned, added, NULL, prev, 0, +1);
    } else {
        int ins_top = leader_changed ? sl->max_height : level;
        daslRecomputePath(sl, cscore, owned, added, kept, NULL, ins_top, +1);
    }

    daslCursor c = { ins_node, ins_slot };
    return c;
}

daslCursor daslInsert(dasl *sl, double score, sds ele) {
    return daslInsertOwned(sl, score, sdsdup(ele));
}

/* Remove the index entry for leader (delscore, delmem) at level `h`; if that
 * empties the index node, unlink and free it. The member here is borrowed (it
 * owns nothing), so only the pointer slot is cleared - never sdsfree'd. */
static void daslUnlinkIndexEntry(dasl *sl, daslNode **prev, int h, double delscore, sds delmem) {
    daslNode *p = prev[h];
    int idx = dasl_find_le(p, delscore, delmem);
    daslNode *target = p;
    if (!(idx >= 0 && dasl_cmp(p->scores[idx], p->members[idx], delscore, delmem) == 0)) {
        if (p->forward != NULL &&
            dasl_cmp(p->forward->scores[0], p->forward->members[0], delscore, delmem) == 0) {
            target = p->forward;
            idx = 0;
        } else {
            return; /* not present at this level */
        }
    }
    memmove(&target->scores[idx], &target->scores[idx + 1], (target->n_key - idx - 1) * sizeof(double));
    memmove(&target->members[idx], &target->members[idx + 1], (target->n_key - idx - 1) * sizeof(sds));
    memmove(&target->next[idx], &target->next[idx + 1], (target->n_key - idx - 1) * sizeof(void *));
    {
        size_t ws = dasl_weight_width(h);
        memmove(target->weights + (size_t)idx * ws, target->weights + (size_t)(idx + 1) * ws,
                (size_t)(target->n_key - idx - 1) * ws);
    }
    target->scores[target->n_key - 1] = 0;
    target->members[target->n_key - 1] = NULL;
    target->next[target->n_key - 1] = NULL;
    dasl_wset(target, h, target->n_key - 1, 0);
    target->n_key--;
    if (target->n_key == 0) {
        daslNode *pred = target->prev;
        pred->forward = target->forward;
        if (target->forward != NULL) target->forward->prev = pred;
        if (prev[h] == target) prev[h] = NULL;
        sl->alloc_size -= zmalloc_usable_size(target);
        zfree(target);
    }
}

/* Core delete. When `kept` is non-NULL the removed element's owned member sds
 * is returned through it (and NOT freed) so the caller can transfer ownership;
 * otherwise the member is freed. Returns 1 if an element was removed, else 0. */
static int daslDeleteEx(dasl *sl, double score, sds ele, sds *kept) {
    double cscore = daslCanonScore(score);

    daslNode *prev[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) prev[i] = NULL;
    daslLeaf *n0 = daslDescend(sl, cscore, ele, prev);

    int idx = dasl_leaf_find_le(n0, cscore, ele);
    if (!(idx >= 0 && n0->scores[idx] == cscore && sdscmp(n0->members[idx], ele) == 0))
        return 0; /* not found */

    sds dead_mem = n0->members[idx];

    /* (score, member) whose post-delete descent visits the affected ancestor
     * chain. */
    double rscore = cscore;
    sds rmem = ele;

    if (idx > 0) {
        /* Non-leader: shift it out at level 0; no upward fixup needed. */
        memmove(&n0->scores[idx], &n0->scores[idx + 1], (n0->n_key - idx - 1) * sizeof(double));
        memmove(&n0->members[idx], &n0->members[idx + 1], (n0->n_key - idx - 1) * sizeof(sds));
        n0->scores[n0->n_key - 1] = 0;
        n0->members[n0->n_key - 1] = NULL;
        n0->n_key--;
    } else if (n0->n_key > 1) {
        /* Leader delete, leaf survives: new leader is the old slot 1. */
        double news_s = n0->scores[1];
        sds news_m = n0->members[1];
        memmove(&n0->scores[0], &n0->scores[1], (n0->n_key - 1) * sizeof(double));
        memmove(&n0->members[0], &n0->members[1], (n0->n_key - 1) * sizeof(sds));
        n0->scores[n0->n_key - 1] = 0;
        n0->members[n0->n_key - 1] = NULL;
        n0->n_key--;
        daslPropagateLeader(sl, prev, 0, cscore, dead_mem, news_s, news_m);
        rscore = daslCanonScore(news_s); rmem = news_m;
    } else {
        /* Single-key leaf: unlink at level 0 via its back-pointer, then remove
         * its index entries on every level above. */
        daslLeaf *pred = n0->prev;
        pred->forward = n0->forward;
        if (n0->forward != NULL) n0->forward->prev = pred;
        else sl->tail = (pred == sl->lhead) ? NULL : pred; /* removed the last leaf */
        for (int hh = 1; hh < sl->max_height; hh++)
            daslUnlinkIndexEntry(sl, prev, hh, cscore, dead_mem);
        sl->alloc_size -= zmalloc_usable_size(n0);
        zfree(n0);
    }

    /* Shrink height while the top index level is empty. */
    while (sl->max_height > 1 && sl->head[sl->max_height - 1]->forward == NULL) {
        sl->alloc_size -= zmalloc_usable_size(sl->head[sl->max_height - 1]);
        zfree(sl->head[sl->max_height - 1]);
        sl->head[sl->max_height - 1] = NULL;
        sl->max_height--;
    }

    sl->length--;

    if (sl->length > 0) {
        daslNode *added[DASL_MAXHEIGHT];
        for (int i = 0; i < DASL_MAXHEIGHT; i++) added[i] = NULL;
        int del_top = (idx > 0) ? 0 : sl->max_height;
        daslRecomputePath(sl, rscore, rmem, added, NULL, (idx > 0) ? prev : NULL, del_top, -1);
    }

    sl->alloc_size -= sdsAllocSize(dead_mem);
    if (kept) *kept = dead_mem;
    else sdsfree(dead_mem);

    return 1;
}

int daslDelete(dasl *sl, double score, sds ele) {
    return daslDeleteEx(sl, score, ele, NULL);
}

/* Move the element (oldscore, member) to newscore, preserving the exact owned
 * member sds buffer. Returns a cursor to the new position, or a null cursor if
 * (oldscore, member) is not present. */
daslCursor daslReposition(dasl *sl, double oldscore, double newscore, sds member) {
    double cs = daslCanonScore(oldscore);
    daslLeaf *x = daslDescend(sl, cs, member, NULL);
    int j = dasl_leaf_find_le(x, cs, member);
    if (j < 0 || x->scores[j] != cs || sdscmp(x->members[j], member) != 0) {
        daslCursor n = { NULL, 0 };
        return n;
    }
    daslCursor c = { x, j };
    return daslUpdateScore(sl, &c, newscore);
}

sds daslScan(const dasl *sl, double score, sds ele, int scan_num) {
    double cs = daslCanonScore(score);
    daslLeaf *x = daslDescend(sl, cs, ele, NULL);
    int j = dasl_leaf_find_le(x, cs, ele);
    if (j < 0) {
        /* key precedes every element: start at the first real leaf. */
        x = sl->lhead->forward;
        j = 0;
        if (x == NULL) return NULL; /* empty structure */
    }

    sds last = x->members[j];
    int visited = 1;
    while (visited < scan_num) {
        if (j + 1 < x->n_key) {
            j++;
        } else if (x->forward != NULL) {
            x = x->forward;
            j = 0;
        } else {
            break;
        }
        last = x->members[j];
        visited++;
    }
    return last;
}

unsigned long daslGetRank(const dasl *sl, double score, sds ele) {
    double cs = daslCanonScore(score);

    unsigned long rank = 0;
    daslLeaf *lf;
    int h = sl->max_height - 1;
    if (h == 0) {
        lf = sl->lhead;
    } else {
        daslNode *x = sl->head[h];
        while (1) {
            while (x->forward != NULL &&
                   dasl_cmp(x->forward->scores[0], x->forward->members[0], cs, ele) <= 0) {
                rank += dasl_coverweight(sl, x, h);
                x = x->forward;
            }
            int j = dasl_find_le(x, cs, ele);
            DASL_WSUM(rank, x, h, 0, j);
            if (h - 1 == 0) {
                lf = (j < 0) ? sl->lhead : (daslLeaf *)x->next[j];
                break;
            }
            x = (j < 0) ? sl->head[h - 1] : (daslNode *)x->next[j];
            h--;
        }
    }
    /* Leaf level: each slot weighs 1, each leaf stepped over weighs n_key. */
    while (lf->forward != NULL &&
           dasl_cmp(lf->forward->scores[0], lf->forward->members[0], cs, ele) <= 0) {
        rank += (unsigned long)lf->n_key;
        lf = lf->forward;
    }
    int j = dasl_leaf_find_le(lf, cs, ele);
    if (j >= 0) {
        rank += (unsigned long)j;
        if (lf->scores[j] == cs && sdscmp(lf->members[j], ele) == 0)
            return rank + 1; /* 1-based */
    }
    return 0; /* not present */
}

daslCursor daslGetElementByRank(const dasl *sl, unsigned long rank) {
    daslCursor c = { NULL, 0 };
    if (rank == 0 || rank > sl->length) return c;

    unsigned long trav = 0; /* elements passed so far */
    daslLeaf *lf;
    int h = sl->max_height - 1;
    if (h == 0) {
        lf = sl->lhead;
    } else {
        daslNode *x = sl->head[h];
        while (1) {
            while (x->forward != NULL && trav + dasl_coverweight(sl, x, h) < rank) {
                trav += dasl_coverweight(sl, x, h);
                x = x->forward;
            }
            if (x == sl->head[h]) {
                /* Still in this level's head prefix: drop a level. */
                if (h - 1 == 0) { lf = sl->lhead; break; }
                x = sl->head[h - 1]; h--; continue;
            }
            int j = 0;
            while (j < x->n_key) {
                unsigned long wj = dasl_wget(x, h, j);
                if (trav + wj >= rank) break;
                trav += wj; j++;
            }
            if (h - 1 == 0) { lf = (daslLeaf *)x->next[j]; break; }
            x = (daslNode *)x->next[j]; h--;
        }
    }
    /* Leaf level: walk leaves, then slots (each weighs 1). */
    while (lf->forward != NULL && trav + (unsigned long)lf->n_key < rank) {
        trav += (unsigned long)lf->n_key;
        lf = lf->forward;
    }
    int j = 0;
    while (j < lf->n_key && trav + 1 < rank) { trav++; j++; }
    c.node = lf; c.slot = j;
    return c;
}

double daslCursorScore(const daslCursor *c) {
    return c->node->scores[c->slot];
}

sds daslCursorMember(const daslCursor *c) {
    return c->node->members[c->slot];
}

sds daslGetNodeElement(const daslCursor *c) {
    return c->node->members[c->slot];
}

/*-----------------------------------------------------------------------------
 * Cursor-based iteration over the in-order (level-0) element chain
 *----------------------------------------------------------------------------*/

daslCursor daslFirst(const dasl *sl) {
    daslCursor c = { NULL, 0 };
    daslLeaf *x = sl->lhead->forward;
    if (x != NULL) { c.node = x; c.slot = 0; }
    return c;
}

daslCursor daslLast(const dasl *sl) {
    daslCursor c = { NULL, 0 };
    if (sl->tail != NULL) { c.node = sl->tail; c.slot = sl->tail->n_key - 1; }
    return c;
}

daslCursor daslNext(daslCursor c) {
    if (c.node == NULL) return c;
    if (c.slot + 1 < c.node->n_key) { c.slot++; return c; }
    if (c.node->forward != NULL) { c.node = c.node->forward; c.slot = 0; }
    else { c.node = NULL; c.slot = 0; }
    return c;
}

daslCursor daslPrev(daslCursor c) {
    if (c.node == NULL) return c;
    if (c.slot > 0) { c.slot--; return c; }
    /* Roll back to the previous leaf's last slot. The first real leaf's prev is
     * the leaf head (n_key == 0), which means "off the front". */
    daslLeaf *p = c.node->prev;
    if (p == NULL || p->n_key == 0) { c.node = NULL; c.slot = 0; }
    else { c.node = p; c.slot = p->n_key - 1; }
    return c;
}

int daslDeleteCursor(dasl *sl, daslCursor *c) {
    if (c->node == NULL) return 0;
    double score = c->node->scores[c->slot];
    sds member = c->node->members[c->slot];
    return daslDelete(sl, score, member);
}

/* Change a cursor element's score in place when ordering is preserved, else
 * delete + re-insert; return a cursor to the element's new position. */
daslCursor daslUpdateScore(dasl *sl, daslCursor *c, double newscore) {
    daslLeaf *node = c->node;
    int slot = c->slot;
    sds member = node->members[slot];
    double ns = daslCanonScore(newscore);

    /* Fast path: a non-leader slot (slot != 0, so no express-lane leader to
     * rewrite) whose order relative to its neighbours is unchanged. */
    if (slot != 0) {
        daslCursor p = daslPrev(*c);
        daslCursor n = daslNext(*c);
        int okprev = (p.node == NULL) ||
            dasl_cmp(p.node->scores[p.slot], p.node->members[p.slot], ns, member) < 0;
        int oknext = (n.node == NULL) ||
            dasl_cmp(ns, member, n.node->scores[n.slot], n.node->members[n.slot]) < 0;
        if (okprev && oknext) {
            node->scores[slot] = ns;
            return *c;
        }
    }

    /* Slow path: reposition via delete + re-insert, preserving the exact member
     * sds buffer (daslDeleteEx hands it back instead of freeing it, and
     * daslInsertOwned re-stores it without copying). */
    double oldscore = node->scores[slot];
    sds skept = NULL;
    daslDeleteEx(sl, oldscore, member, &skept);
    return daslInsertOwned(sl, newscore, skept);
}

unsigned long daslGetRankByCursor(const dasl *sl, daslCursor c) {
    if (c.node == NULL) return 0;
    return daslGetRank(sl, c.node->scores[c.slot], c.node->members[c.slot]);
}

/*-----------------------------------------------------------------------------
 * Range queries by score
 *----------------------------------------------------------------------------*/

/* Whether any element falls inside the score range. Mirrors zslIsInRange(). */
static int daslIsInRange(const dasl *sl, zrangespec *range) {
    if (range->min > range->max ||
        (range->min == range->max && (range->minex || range->maxex)))
        return 0;
    daslCursor last = daslLast(sl);
    if (last.node == NULL || !zslValueGteMin(daslCursorScore(&last), range))
        return 0;
    daslCursor first = daslFirst(sl);
    if (!zslValueLteMax(daslCursorScore(&first), range))
        return 0;
    return 1;
}

/* Number of elements whose score is below `edge` (score < edge when orEqual==0,
 * score <= edge when orEqual==1). O(log N) via the daslGetRank descent shape. */
static unsigned long daslCountByScore(const dasl *sl, double edge, int orEqual) {
    unsigned long cnt = 0;
    daslLeaf *lf;
    int h = sl->max_height - 1;
    if (h == 0) {
        lf = sl->lhead;
    } else {
        daslNode *x = sl->head[h];
        while (1) {
            while (x->forward != NULL) {
                double leader = x->forward->scores[0];
                int q = orEqual ? (leader <= edge) : (leader < edge);
                if (!q) break;
                cnt += dasl_coverweight(sl, x, h);
                x = x->forward;
            }
            int j = -1;
            for (int i = 0; i < x->n_key; i++) {
                double s = x->scores[i];
                int q = orEqual ? (s <= edge) : (s < edge);
                if (q) j = i; else break;
            }
            DASL_WSUM(cnt, x, h, 0, j);
            if (h - 1 == 0) {
                lf = (j < 0) ? sl->lhead : (daslLeaf *)x->next[j];
                break;
            }
            x = (j < 0) ? sl->head[h - 1] : (daslNode *)x->next[j];
            h--;
        }
    }
    /* Leaf level: scores are raw doubles. */
    while (lf->forward != NULL) {
        double leader = lf->forward->scores[0];
        int q = orEqual ? (leader <= edge) : (leader < edge);
        if (!q) break;
        cnt += (unsigned long)lf->n_key;
        lf = lf->forward;
    }
    int j = -1;
    for (int i = 0; i < lf->n_key; i++) {
        int q = orEqual ? (lf->scores[i] <= edge) : (lf->scores[i] < edge);
        if (q) j = i; else break;
    }
    if (j >= 0) cnt += (unsigned long)(j + 1);
    return cnt;
}

daslCursor daslNthInRange(dasl *sl, zrangespec *range, long n, unsigned long *out_rank) {
    daslCursor c = { NULL, 0 };
    if (!daslIsInRange(sl, range)) return c;

    unsigned long lo = daslCountByScore(sl, range->min, range->minex ? 1 : 0);
    unsigned long hi = daslCountByScore(sl, range->max, range->maxex ? 0 : 1);
    if (hi <= lo) return c;

    unsigned long target;
    if (n >= 0) {
        target = lo + 1 + (unsigned long)n;
        if (target > hi) return c;
    } else {
        long t = (long)hi + 1 + n; /* n is negative; n == -1 -> hi */
        if (t < (long)(lo + 1)) return c;
        target = (unsigned long)t;
    }
    c = daslGetElementByRank(sl, target);
    if (c.node != NULL && out_rank) *out_rank = target;
    return c;
}

/*-----------------------------------------------------------------------------
 * Range queries by lexicographic member
 *----------------------------------------------------------------------------*/

/* sdscmp() that treats shared.minstring/maxstring as -inf/+inf, matching the
 * static sdscmplex() in t_zset.c (which is not exported). */
static int dasl_sdscmplex(sds a, sds b) {
    if (a == b) return 0;
    if (a == shared.minstring || b == shared.maxstring) return -1;
    if (a == shared.maxstring || b == shared.minstring) return 1;
    return sdscmp(a, b);
}

/* Whether any element falls inside the lex range. Mirrors zslIsInLexRange(). */
static int daslIsInLexRange(const dasl *sl, zlexrangespec *range) {
    int cmp = dasl_sdscmplex(range->min, range->max);
    if (cmp > 0 || (cmp == 0 && (range->minex || range->maxex)))
        return 0;
    daslCursor last = daslLast(sl);
    if (last.node == NULL || !zslLexValueGteMin(daslCursorMember(&last), range))
        return 0;
    daslCursor first = daslFirst(sl);
    if (!zslLexValueLteMax(daslCursorMember(&first), range))
        return 0;
    return 1;
}

/* True while a member belongs to the low "below-min" prefix (fails GteMin). */
static int dasl_lex_below_min(sds m, zlexrangespec *range) {
    return !zslLexValueGteMin(m, range);
}

/* Count elements whose member keeps `inprefix` true. Lex queries assume a
 * uniform score, so member order matches composite order and the qualifiers
 * form a prefix; accumulate their weights in O(log N), as daslCountByScore. */
static unsigned long daslCountLexPrefix(const dasl *sl, zlexrangespec *range,
                                        int (*inprefix)(sds, zlexrangespec *)) {
    unsigned long cnt = 0;
    daslLeaf *lf;
    int h = sl->max_height - 1;
    if (h == 0) {
        lf = sl->lhead;
    } else {
        daslNode *x = sl->head[h];
        while (1) {
            while (x->forward != NULL && inprefix(x->forward->members[0], range)) {
                cnt += dasl_coverweight(sl, x, h);
                x = x->forward;
            }
            int j = -1;
            for (int i = 0; i < x->n_key; i++) {
                if (inprefix(x->members[i], range)) j = i; else break;
            }
            DASL_WSUM(cnt, x, h, 0, j);
            if (h - 1 == 0) {
                lf = (j < 0) ? sl->lhead : (daslLeaf *)x->next[j];
                break;
            }
            x = (j < 0) ? sl->head[h - 1] : (daslNode *)x->next[j];
            h--;
        }
    }
    while (lf->forward != NULL && inprefix(lf->forward->members[0], range)) {
        cnt += (unsigned long)lf->n_key;
        lf = lf->forward;
    }
    int j = -1;
    for (int i = 0; i < lf->n_key; i++) {
        if (inprefix(lf->members[i], range)) j = i; else break;
    }
    if (j >= 0) cnt += (unsigned long)(j + 1);
    return cnt;
}

daslCursor daslNthInLexRange(dasl *sl, zlexrangespec *range, long n, unsigned long *out_rank) {
    daslCursor c = { NULL, 0 };
    if (!daslIsInLexRange(sl, range)) return c;

    unsigned long lo = daslCountLexPrefix(sl, range, dasl_lex_below_min);
    unsigned long hi = daslCountLexPrefix(sl, range, zslLexValueLteMax);
    if (hi <= lo) return c;

    unsigned long target;
    if (n >= 0) {
        target = lo + 1 + (unsigned long)n;
        if (target > hi) return c;
    } else {
        long t = (long)hi + 1 + n;
        if (t < (long)(lo + 1)) return c;
        target = (unsigned long)t;
    }
    c = daslGetElementByRank(sl, target);
    if (c.node != NULL && out_rank) *out_rank = target;
    return c;
}

/*-----------------------------------------------------------------------------
 * Range deletes
 *----------------------------------------------------------------------------*/

unsigned long daslDeleteRangeByScore(dasl *sl, zrangespec *range, dict *dict) {
    unsigned long removed = 0;
    daslCursor c;
    while ((c = daslNthInRange(sl, range, 0, NULL)).node != NULL) {
        double s = daslCursorScore(&c);
        sds m = daslCursorMember(&c);
        if (dict) dictDelete(dict, m);
        daslDelete(sl, s, m);
        removed++;
    }
    return removed;
}

unsigned long daslDeleteRangeByLex(dasl *sl, zlexrangespec *range, dict *dict) {
    unsigned long removed = 0;
    daslCursor c;
    while ((c = daslNthInLexRange(sl, range, 0, NULL)).node != NULL) {
        double s = daslCursorScore(&c);
        sds m = daslCursorMember(&c);
        if (dict) dictDelete(dict, m);
        daslDelete(sl, s, m);
        removed++;
    }
    return removed;
}

unsigned long daslDeleteRangeByRank(dasl *sl, unsigned int start, unsigned int end, dict *dict) {
    unsigned long removed = 0;
    while (start <= end) {
        daslCursor c = daslGetElementByRank(sl, start);
        if (c.node == NULL) break;
        double s = daslCursorScore(&c);
        sds m = daslCursorMember(&c);
        if (dict) dictDelete(dict, m);
        daslDelete(sl, s, m);
        removed++;
        end--;
    }
    return removed;
}

/*-----------------------------------------------------------------------------
 * Active defragmentation
 *----------------------------------------------------------------------------*/

/* Relocate every node of the structure using `defragfn` and repair all
 * structural links. Processed bottom-up so that when an index level's next[]
 * pointers are re-derived the level below is already at its final addresses.
 * Member sds pointers are left untouched. */
void daslDefragNodes(dasl *sl, void *(*defragfn)(void *)) {
    /* Level 0: relocate the leaf head and every leaf, repairing forward/prev
     * and the tail. */
    daslLeaf *nlh = defragfn(sl->lhead);
    if (nlh) sl->lhead = nlh;
    {
        daslLeaf *pl = sl->lhead;
        daslLeaf *cur = pl->forward;
        while (cur != NULL) {
            daslLeaf *ncur = defragfn(cur);
            if (ncur) cur = ncur;
            pl->forward = cur;
            cur->prev = pl;
            if (cur->forward == NULL) sl->tail = cur;
            pl = cur;
            cur = cur->forward;
        }
    }

    for (int h = 1; h < sl->max_height; h++) {
        daslNode *nh = defragfn(sl->head[h]);
        if (nh) sl->head[h] = nh;

        daslNode *pnode = sl->head[h];
        daslNode *cur = pnode->forward;
        while (cur != NULL) {
            daslNode *ncur = defragfn(cur);
            if (ncur) cur = ncur;
            pnode->forward = cur;
            cur->prev = pnode;
            pnode = cur;
            cur = cur->forward;
        }

        /* Re-derive next[] from the (already relocated) level below. Both chains
         * are ascending, so a single lockstep walk links them. The children are
         * leaves at level 1, index nodes above. */
        if (h == 1) {
            /* Children are leaves: compare the leaf leader (score + member)
             * against the index slot's score + member (same total order). */
            daslLeaf *child = sl->lhead->forward;
            daslNode *node = sl->head[h]->forward;
            while (node != NULL) {
                for (int i = 0; i < node->n_key; i++) {
                    while (child != NULL &&
                           dasl_cmp(child->scores[0], child->members[0],
                                         node->scores[i], node->members[i]) < 0)
                        child = child->forward;
                    node->next[i] = child;
                }
                node = node->forward;
            }
        } else {
            daslNode *child = sl->head[h - 1]->forward;
            daslNode *node = sl->head[h]->forward;
            while (node != NULL) {
                for (int i = 0; i < node->n_key; i++) {
                    while (child != NULL &&
                           dasl_cmp(child->scores[0], child->members[0],
                                    node->scores[i], node->members[i]) < 0)
                        child = child->forward;
                    node->next[i] = child;
                }
                node = node->forward;
            }
        }
    }
}

/* Repoint every reference to a member whose sds buffer moved from `oldmem` to
 * `newmem`: the level-0 owner slot, plus the express-lane borrowers that mirror
 * it when it is a node leader. */
void daslDefragMember(dasl *sl, double score, sds oldmem, sds newmem) {
    double cs = daslCanonScore(score);

    daslNode *prev[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) prev[i] = NULL;
    daslLeaf *n0 = daslDescend(sl, cs, oldmem, prev);

    int idx = dasl_leaf_find_le(n0, cs, oldmem);
    if (!(idx >= 0 && n0->scores[idx] == cs && sdscmp(n0->members[idx], oldmem) == 0))
        return; /* not present (should not happen for a live element) */
    n0->members[idx] = newmem;
    if (idx != 0) return; /* non-leader: no express-lane mirror to fix */

    for (int h = 1; h < sl->max_height; h++) {
        daslNode *p = prev[h];
        int j = dasl_find_le(p, cs, oldmem);
        daslNode *target = p;
        if (!(j >= 0 && dasl_cmp(p->scores[j], p->members[j], cs, oldmem) == 0)) {
            if (p->forward != NULL &&
                dasl_cmp(p->forward->scores[0], p->forward->members[0], cs, oldmem) == 0) {
                target = p->forward;
                j = 0;
            } else {
                break;
            }
        }
        target->members[j] = newmem;
        if (j != 0) break;
    }
}

void daslPrint(const dasl *sl) {
    printf("DASL height=%d length=%lu\n", sl->max_height, sl->length);
    for (int i = sl->max_height - 1; i >= 1; i--) {
        printf("L%d:", i);
        daslNode *x = sl->head[i]->forward;
        while (x != NULL) {
            printf(" [");
            for (int j = 0; j < x->n_key; j++)
                printf("%s%g:%s", j ? "," : "", x->scores[j],
                       x->members[j] ? x->members[j] : "(nil)");
            printf("]");
            x = x->forward;
        }
        printf("\n");
    }
    printf("L0:");
    daslLeaf *lf = sl->lhead->forward;
    while (lf != NULL) {
        printf(" [");
        for (int j = 0; j < lf->n_key; j++)
            printf("%s%g:%s", j ? "," : "", lf->scores[j],
                   lf->members[j] ? lf->members[j] : "(nil)");
        printf("]");
        lf = lf->forward;
    }
    printf("\n");
}
