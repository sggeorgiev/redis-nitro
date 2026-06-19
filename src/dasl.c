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
 * keys; the bottom level holds every key; upper levels are express lanes whose
 * next[i] descend one level) but fixes both problems:
 *
 *   - All descents (read and write) forward-step along each level
 *     (`while (forward && forward->keys[0] <= key) advance`) before dropping a
 *     level, so front nodes are always reachable (fixes #1).
 *   - A node is promoted to the next level exactly once: the first time it
 *     fills to DASL_ARR_SIZE *and* is not already represented one level up
 *     (`dasl_represented`); it keeps that single express-lane entry until it
 *     empties (fixes #2). Splits are even (DASL_ARR_SIZE/2) and the new
 *     right-hand node is the one promoted.
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

#include "zmalloc.h"
#include "endianconv.h"
#include "dasl.h"

#define DASL_SPLIT (DASL_ARR_SIZE / 2)

/* Empty array slots hold the maximum composite key (all 0xFF) and a NULL
 * member; intra-node search is bounded by n_key and never inspects empties, so
 * the sentinel is belt-and-suspenders for debugging. */
static const daslKey DASL_EMPTY_KEY = { { 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                          0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF } };

/* Pack (score, ele) into a fixed-width composite key: bytes [0,8) hold the
 * score in an order-preserving "sortable" encoding (sign-bit/all-bits flip then
 * big-endian, so memcmp matches numeric order) and bytes [8,16) hold the first
 * 8 bytes of the member, zero padded. memcmp of the 16 bytes therefore orders
 * by score first, then by member prefix; ties are resolved by the full sds. */
static void daslMakeKey(daslKey *k, double score, sds ele) {
    uint64_t bits;
    if (score == 0) score = 0; /* canonicalize -0.0 to +0.0 (zset treats them equal) */
    memcpy(&bits, &score, sizeof(bits));
    bits = (bits & (1ULL << 63)) ? ~bits : (bits ^ (1ULL << 63));
    bits = htonu64(bits); /* big-endian: byte 0 is most significant */
    memcpy(k->b, &bits, DASL_SCORE_SIZE);
    size_t n = sdslen(ele);
    if (n > DASL_MEMBER_PREFIX) n = DASL_MEMBER_PREFIX;
    memset(k->b + DASL_SCORE_SIZE, 0, DASL_MEMBER_PREFIX);
    memcpy(k->b + DASL_SCORE_SIZE, ele, n);
}

/* Compare composite key + full member of two entries. memcmp the 16-byte
 * composite first; on a tie compare the full members with sdscmp. Members are
 * present at every level (owned at level 0, borrowed above), so the tie-break
 * is always available. */
static inline int dasl_cmp(const daslKey *a, sds ma, const daslKey *b, sds mb) {
    int c = memcmp(a->b, b->b, DASL_CK_SIZE);
    if (c) return c < 0 ? -1 : 1;
    return sdscmp(ma, mb);
}

/* Greatest slot index i in [0, n_key) with (keys[i], members[i]) <=
 * (target, mem), or -1 if none (including an empty head whose n_key == 0). */
static int dasl_find_le(const daslNode *n, const daslKey *target, sds mem) {
    for (int i = 0; i < n->n_key; i++)
        if (dasl_cmp(&n->keys[i], n->members[i], target, mem) > 0) return i - 1;
    return n->n_key - 1;
}

/* Does the node whose leader key is `leader` already own an express-lane index
 * entry one level up? A node graduates to the next level exactly once - the
 * first time it fills to DASL_ARR_SIZE - and keeps that entry until it empties;
 * after a split shrinks it below DASL_ARR_SIZE it keeps the entry, so re-filling
 * it must NOT promote it again (that is the duplicate-leader corruption).
 *
 * `cover` is the covering node recorded by the descent at the level above (the
 * last node there with leader <= the inserted key, or that level's head). The
 * mirroring entry, if it exists, can only sit in `cover` itself - because the
 * node's leader is >= cover's leader and <= the inserted key, and index leaders
 * are unique per level - or, when `cover` is the level's (empty) head, in
 * cover's first forward node. So this is an O(DASL_ARR_SIZE) local check rather
 * than a scan of the whole express lane. */
static int dasl_represented(const daslNode *cover, const daslKey *leader, sds lmem) {
    if (cover == NULL) return 0;
    int idx = dasl_find_le(cover, leader, lmem);
    if (idx >= 0 && dasl_cmp(&cover->keys[idx], cover->members[idx], leader, lmem) == 0) return 1;
    if (cover->forward != NULL &&
        dasl_cmp(&cover->forward->keys[0], cover->forward->members[0], leader, lmem) == 0) return 1;
    return 0;
}

/* Sum of a node's per-slot weights = number of level-0 elements it covers. */
static unsigned long dasl_nodeweight(const daslNode *n) {
    unsigned long w = 0;
    for (int i = 0; i < n->n_key; i++) w += n->weights[i];
    return w;
}

/* Weight crossed when leaving node `x` at level `h`: a real node contributes
 * the sum of its slot weights; a head contributes its stored prefix weight. */
static unsigned long dasl_coverweight(const dasl *sl, const daslNode *x, int h) {
    if (x == sl->head[h]) return x->weights[0];
    return dasl_nodeweight(x);
}

/* Weight of one index slot, computed from the immediate children's node weights
 * (the level below must already carry correct weights). Slot i of `parent`
 * covers the level-(h-1) nodes from next[i] up to (but excluding) the child that
 * the next slot - or parent->forward's first slot - points at. O(fanout). */
static unsigned long dasl_index_slot_weight(const daslNode *parent, int i) {
    const daslNode *stop = NULL;
    if (i + 1 < parent->n_key) stop = parent->next[i + 1];
    else if (parent->forward != NULL) stop = parent->forward->next[0];
    unsigned long w = 0;
    for (const daslNode *m = parent->next[i]; m != NULL && m != stop; m = m->forward)
        w += dasl_nodeweight(m);
    return w;
}

/* Allocate a node holding the single key `key` with member `mem` (a real
 * data/index node). The caller owns `mem`: at level 0 it must be an owned sds
 * copy; at express-lane levels it is a borrowed pointer to the level-0 owner. */
static daslNode *daslNewNode(dasl *sl, const daslKey *key, sds mem) {
    size_t usable;
    daslNode *n = zmalloc_usable(sizeof(*n), &usable);
    n->forward = NULL;
    n->prev = NULL;
    n->n_key = 1;
    n->keys[0] = *key;
    n->members[0] = mem;
    for (int i = 1; i < DASL_ARR_SIZE; i++) { n->keys[i] = DASL_EMPTY_KEY; n->members[i] = NULL; }
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->next[i] = NULL;
    /* weights are set by daslRecomputePath after the mutation completes; start
     * from a clean slate so an unrecomputed read (there are none) is defined. */
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->weights[i] = 0;
    sl->alloc_size += usable;
    return n;
}

/* Allocate an empty per-level sentinel head (n_key == 0). */
static daslNode *daslNewHead(dasl *sl) {
    size_t usable;
    daslNode *n = zmalloc_usable(sizeof(*n), &usable);
    n->forward = NULL;
    n->prev = NULL;
    n->n_key = 0;
    for (int i = 0; i < DASL_ARR_SIZE; i++) { n->keys[i] = DASL_EMPTY_KEY; n->members[i] = NULL; }
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->next[i] = NULL;
    /* On a head only weights[0] is meaningful (the prefix weight); start at 0. */
    for (int i = 0; i < DASL_ARR_SIZE; i++) n->weights[i] = 0;
    sl->alloc_size += usable;
    return n;
}

dasl *daslCreate(void) {
    size_t usable;
    dasl *sl = zmalloc_usable(sizeof(*sl), &usable);
    sl->max_height = 1;
    sl->length = 0;
    sl->alloc_size = usable;
    sl->tail = NULL;
    /* Heads are allocated lazily per level; only level 0 exists initially. */
    for (int i = 0; i < DASL_MAXHEIGHT; i++) sl->head[i] = NULL;
    sl->head[0] = daslNewHead(sl);
    return sl;
}

void daslFree(dasl *sl) {
    /* Each node belongs to exactly one level's forward chain, so walking
     * every active level's chain (head included) frees the whole structure.
     * Only the level-0 chain owns its members; express-lane members are
     * borrowed pointers into level-0 nodes and must not be freed. */
    for (int i = 0; i < sl->max_height; i++) {
        daslNode *node = sl->head[i];
        while (node) {
            daslNode *next = node->forward;
            if (i == 0)
                for (int j = 0; j < node->n_key; j++) sdsfree(node->members[j]);
            zfree(node);
            node = next;
        }
    }
    zfree(sl);
}

size_t daslAllocSize(const dasl *sl) { return sl->alloc_size; }

/* Hint the OS that the structure's memory is not needed before a fork (see
 * dismissMemory()). Walks the bottom-level chain, dismissing each node and its
 * owned member sds. Express-lane nodes/members are borrowed and skipped. */
void daslDismiss(dasl *sl) {
    daslNode *x = sl->head[0] ? sl->head[0]->forward : NULL;
    while (x != NULL) {
        daslNode *next = x->forward;
        for (int j = 0; j < x->n_key; j++) {
            sds m = x->members[j];
            if (m) dismissMemory(sdsAllocPtr(m), sdsAllocSize(m));
        }
        dismissMemory(x, 0);
        x = next;
    }
}

/* Grow the structure so that level `h` has an allocated head. */
static void daslEnsureHeight(dasl *sl, int h) {
    while (sl->max_height <= h) {
        sl->head[sl->max_height] = daslNewHead(sl);
        sl->max_height++;
    }
}

/* Forward-aware descent: walk down from the top level, at each level stepping
 * forward while the next node's leader is <= key, then descending via the
 * matching next[] pointer (or to the level-below head when key precedes every
 * entry). Lands on the level-0 node that would contain `key`. If `prev` is
 * non-NULL it is filled with the covering node at each level. Returns the
 * level-0 node. */
static daslNode *daslDescend(const dasl *sl, const daslKey *key, sds mem, daslNode **prev) {
    int h = sl->max_height - 1;
    daslNode *x = sl->head[h];
    while (1) {
        while (x->forward != NULL &&
               dasl_cmp(&x->forward->keys[0], x->forward->members[0], key, mem) <= 0)
            x = x->forward;
        if (prev) prev[h] = x;
        if (h == 0) return x;
        int j = dasl_find_le(x, key, mem);
        x = (j < 0) ? sl->head[h - 1] : x->next[j];
        h--;
    }
}

/* Recompute the order-statistics weights affected by mutating (key, mem). The
 * post-mutation descent for (key, mem) visits exactly the covering node at each
 * level - the ancestor chain whose subtree counts changed - so recomputing
 * those bottom-up (plus any split siblings in `added`, indexed by the level the
 * sibling lives on) restores all weights in O(height * fanout) = O(log N).
 * `added[h]` may be NULL when no node was split at level h (e.g. all deletes).
 *
 * When `path_in` is non-NULL it is the post-mutation covering path the caller
 * already computed (the mutation's own descent, with the receiving node patched
 * in at each touched level), so the redundant internal descent is skipped. When
 * it is NULL the path is rediscovered here (used by the structurally-complex
 * single-key-node delete, where the caller's prev[] has freed/NULL entries). */
static void daslRecomputePath(dasl *sl, const daslKey *key, sds mem, daslNode **added,
                              daslNode **path_in) {
    daslNode *path[DASL_MAXHEIGHT];
    if (path_in != NULL) {
        memcpy(path, path_in, sizeof(path));
    } else {
        for (int i = 0; i < DASL_MAXHEIGHT; i++) path[i] = NULL;
        daslDescend(sl, key, mem, path);
    }

    /* Level 0: the covering node and any split sibling have all-1 slot weights. */
    daslNode *l0[2] = { path[0], added[0] };
    for (int k = 0; k < 2; k++)
        if (l0[k] != NULL && l0[k] != sl->head[0])
            for (int i = 0; i < l0[k]->n_key; i++) l0[k]->weights[i] = 1;

    /* Index levels bottom-up: covering node on the path + any split sibling. */
    for (int h = 1; h < sl->max_height; h++) {
        daslNode *cand[2] = { path[h], added[h] };
        for (int k = 0; k < 2; k++) {
            daslNode *x = cand[k];
            if (x == NULL || x == sl->head[h]) continue;
            for (int i = 0; i < x->n_key; i++) x->weights[i] = dasl_index_slot_weight(x, i);
        }
    }

    /* Per-level head prefixes: keys before head[h]->forward. Built from the
     * level below: the lower head's prefix plus the level-(h-1) nodes between
     * the two heads' forwards. */
    sl->head[0]->weights[0] = 0;
    for (int h = 1; h < sl->max_height; h++) {
        daslNode *f = sl->head[h]->forward;
        daslNode *stop = f ? f->next[0] : NULL;
        unsigned long pref = sl->head[h - 1]->weights[0];
        for (daslNode *m = sl->head[h - 1]->forward; m != NULL && m != stop; m = m->forward)
            pref += dasl_nodeweight(m);
        sl->head[h]->weights[0] = pref;
    }
}

/* A node's leader key changed from `olds` to `news`; rewrite the mirroring
 * index entry on each level above `level`, stopping once a level's entry is
 * not the leader (idx != 0) since higher levels then cannot mirror it. */
static void daslPropagateLeader(dasl *sl, daslNode **prev, int level,
                                const daslKey *olds, sds old_mem,
                                const daslKey *news, sds new_mem) {
    for (int h = level + 1; h < sl->max_height; h++) {
        daslNode *p = prev[h];
        int idx = dasl_find_le(p, olds, old_mem);
        daslNode *target = p;
        if (!(idx >= 0 && dasl_cmp(&p->keys[idx], p->members[idx], olds, old_mem) == 0)) {
            if (p->forward != NULL &&
                dasl_cmp(&p->forward->keys[0], p->forward->members[0], olds, old_mem) == 0) {
                target = p->forward;
                idx = 0;
            } else {
                if (idx != 0) break; else continue;
            }
        }
        target->keys[idx] = *news;
        target->members[idx] = new_mem;
        if (idx != 0) break;
    }
}

int daslContains(const dasl *sl, double score, sds ele) {
    daslKey key;
    daslMakeKey(&key, score, ele);
    daslNode *x = daslDescend(sl, &key, ele, NULL);
    int j = dasl_find_le(x, &key, ele);
    return j >= 0 && dasl_cmp(&x->keys[j], x->members[j], &key, ele) == 0;
}

/* Insert taking ownership of `ele` (no copy is made). On a duplicate the
 * passed sds is freed and a null cursor is returned. daslInsert() wraps this
 * with an sdsdup so the public contract (caller keeps ownership) is preserved;
 * the score-reposition path uses it directly to keep the exact same buffer. */
static daslCursor daslInsertOwned(dasl *sl, double score, sds ele) {
    daslKey key;
    daslMakeKey(&key, score, ele);

    daslNode *prev[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) prev[i] = sl->head[i];
    daslDescend(sl, &key, ele, prev);

    /* Reject duplicates (checked at level 0). */
    {
        daslNode *x = prev[0];
        int j = dasl_find_le(x, &key, ele);
        if (j >= 0 && dasl_cmp(&x->keys[j], x->members[j], &key, ele) == 0) {
            sdsfree(ele);
            daslCursor dup = { NULL, 0 };
            return dup;
        }
    }

    /* The member is now owned by the structure. It lives in exactly one
     * level-0 slot; express-lane entries only ever borrow a pointer to it. */
    sds owned = ele;

    int level = 0;
    daslKey up_key = key;   /* leader key to insert at the level above */
    sds up_mem = owned;     /* borrowed member that travels with up_key */
    daslNode *down = NULL;  /* node the new index entry should point at */

    /* Level-0 node + slot that ends up holding the inserted element. Recorded
     * by the level-0 branch below so we can return the cursor directly instead
     * of re-descending (the level-0 node is never touched again after level 0). */
    daslNode *ins_node = NULL;
    int ins_slot = 0;

    /* Split siblings / new index nodes created during this insert, indexed by
     * the level they live on. Fed to daslRecomputePath so their slot weights
     * get rebuilt even when they are off the post-insert descent path. */
    daslNode *added[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) added[i] = NULL;

    /* Post-mutation covering path for the weight recompute, seeded from the
     * descent's covering chain and patched with the receiving node at the one
     * level a simple insert touches. This is reused (skipping the recompute's
     * own descent) ONLY for a plain in-node insert: no leader change, no split,
     * no promotion. Those structural changes shift the post-mutation covering
     * node at higher levels away from the snapshotted prev[] (e.g. a leader
     * change propagated up moves the cover from a head onto its forward node),
     * so `can_reuse` stays 0 for them and recompute descends afresh. */
    daslNode *path[DASL_MAXHEIGHT];
    for (int i = 0; i < DASL_MAXHEIGHT; i++) path[i] = prev[i];
    int can_reuse = 0;

    while (1) {
        /* A brand-new top level reached via promotion/split has a freshly
         * allocated head, so the snapshotted prev[] is stale; anchor at head. */
        if (level >= sl->max_height || prev[level] == NULL) {
            daslEnsureHeight(sl, level);
            prev[level] = sl->head[level];
        }
        daslNode *p = prev[level];
        int at_head = (p == sl->head[level]);

        if (at_head && p->forward == NULL) {
            /* Case 1: empty level - create the first node. */
            if (level == 0) {
                daslNode *nn = daslNewNode(sl, &key, owned);
                nn->prev = p;
                p->forward = nn;
                sl->tail = nn;
                down = nn;
                ins_node = nn;
                ins_slot = 0;
                path[level] = nn;
            } else {
                daslNode *nn = daslNewNode(sl, &up_key, up_mem);
                nn->next[0] = down;
                nn->prev = p;
                p->forward = nn;
                added[level] = nn;
                down = nn;
                path[level] = nn;
            }
            break;
        }

        daslKey ins_key = (level == 0) ? key : up_key;
        sds ins_mem = (level == 0) ? owned : up_mem;
        daslNode *ins_down = (level == 0) ? NULL : down;
        daslNode *target = at_head ? p->forward : p;

        int idx = dasl_find_le(target, &ins_key, ins_mem);
        if (idx == 0 && dasl_cmp(&target->keys[0], target->members[0], &ins_key, ins_mem) > 0) idx = -1;

        if (target->n_key < DASL_ARR_SIZE) {
            /* Room: shift right of idx and insert at idx+1. */
            int pos = idx + 1;
            daslKey old_leader = target->keys[0];
            sds old_leader_mem = target->members[0];
            memmove(&target->keys[pos + 1], &target->keys[pos],
                    (target->n_key - pos) * sizeof(daslKey));
            memmove(&target->members[pos + 1], &target->members[pos],
                    (target->n_key - pos) * sizeof(sds));
            memmove(&target->next[pos + 1], &target->next[pos],
                    (target->n_key - pos) * sizeof(daslNode *));
            memmove(&target->weights[pos + 1], &target->weights[pos],
                    (target->n_key - pos) * sizeof(unsigned long));
            target->keys[pos] = ins_key;
            target->members[pos] = ins_mem;
            target->next[pos] = ins_down;
            target->n_key++;
            if (idx == -1) /* leader changed */
                daslPropagateLeader(sl, prev, level, &old_leader, old_leader_mem, &ins_key, ins_mem);
            down = target;
            path[level] = target;
            if (level == 0) { ins_node = target; ins_slot = pos; }
            /* Promote iff the node just became full and is not already
             * represented one level up (graduate exactly once). prev[level+1]
             * is the covering node from the descent, so this is an O(1) local
             * check (NULL when level+1 is a not-yet-existing top level). */
            daslNode *cover = (level + 1 < sl->max_height) ? prev[level + 1] : NULL;
            if (target->n_key == DASL_ARR_SIZE &&
                !dasl_represented(cover, &target->keys[0], target->members[0])) {
                up_key = target->keys[0];
                up_mem = target->members[0];
                level++;
                continue;
            }
            /* Reached here without promoting. Safe to reuse the path iff this
             * was a single-level (no lower split/promote drove us up) in-node
             * insert that did not change the node's leader. */
            can_reuse = (level == 0 && idx != -1);
            break;
        } else {
            /* Full: even split. Upper half moves to a new right node `add`. */
            daslNode *add = daslNewNode(sl, &target->keys[DASL_SPLIT], target->members[DASL_SPLIT]);
            memcpy(add->keys, &target->keys[DASL_SPLIT], DASL_SPLIT * sizeof(daslKey));
            memcpy(add->members, &target->members[DASL_SPLIT], DASL_SPLIT * sizeof(sds));
            memcpy(add->next, &target->next[DASL_SPLIT], DASL_SPLIT * sizeof(daslNode *));
            memcpy(add->weights, &target->weights[DASL_SPLIT], DASL_SPLIT * sizeof(unsigned long));
            add->n_key = DASL_SPLIT;
            add->forward = target->forward;
            add->prev = target;
            if (add->forward != NULL) add->forward->prev = add;
            target->forward = add;
            /* `add` is the new right-hand node at this level; daslRecomputePath
             * rebuilds its weights even if the descent path goes through the
             * left half instead. */
            added[level] = add;
            /* Maintain the level-0 tail pointer when the last node is split. */
            if (level == 0 && add->forward == NULL) sl->tail = add;
            for (int i = DASL_SPLIT; i < DASL_ARR_SIZE; i++) {
                target->keys[i] = DASL_EMPTY_KEY;
                target->members[i] = NULL;
                target->next[i] = NULL;
                target->weights[i] = 0;
            }
            target->n_key = DASL_SPLIT;

            /* Insert the new key into whichever half it belongs to. */
            daslNode *into;
            int half_idx;
            if (idx < DASL_SPLIT) { into = target; half_idx = idx; }
            else { into = add; half_idx = idx - DASL_SPLIT; }
            daslKey old_leader = into->keys[0];
            sds old_leader_mem = into->members[0];
            int pos = half_idx + 1;
            memmove(&into->keys[pos + 1], &into->keys[pos],
                    (into->n_key - pos) * sizeof(daslKey));
            memmove(&into->members[pos + 1], &into->members[pos],
                    (into->n_key - pos) * sizeof(sds));
            memmove(&into->next[pos + 1], &into->next[pos],
                    (into->n_key - pos) * sizeof(daslNode *));
            memmove(&into->weights[pos + 1], &into->weights[pos],
                    (into->n_key - pos) * sizeof(unsigned long));
            into->keys[pos] = ins_key;
            into->members[pos] = ins_mem;
            into->next[pos] = ins_down;
            into->n_key++;
            if (into == target && half_idx == -1)
                daslPropagateLeader(sl, prev, level, &old_leader, old_leader_mem, &ins_key, ins_mem);
            path[level] = into;
            if (level == 0) { ins_node = into; ins_slot = pos; }

            /* The split-off node `add` must get an index entry one level up. */
            down = add;
            up_key = add->keys[0];
            up_mem = add->members[0];
            level++;
            daslEnsureHeight(sl, level);
            continue;
        }
    }

    sl->length++;
    sl->alloc_size += sdsAllocSize(owned);
    daslRecomputePath(sl, &key, ele, added, can_reuse ? path : NULL);

    /* The level-0 branch recorded the node/slot that received the element, so
     * the cursor is returned directly (mirrors zslInsert returning the new
     * node) without a third descent. */
    daslCursor c = { ins_node, ins_slot };
    return c;
}

daslCursor daslInsert(dasl *sl, double score, sds ele) {
    return daslInsertOwned(sl, score, sdsdup(ele));
}

/* Remove the index entry for leader (delk, delmem) at level `h`; if that
 * empties the index node, unlink and free it. The member here is borrowed (it
 * owns nothing), so only the pointer slot is cleared - never sdsfree'd. */
static void daslUnlinkIndexEntry(dasl *sl, daslNode **prev, int h, const daslKey *delk, sds delmem) {
    daslNode *p = prev[h];
    int idx = dasl_find_le(p, delk, delmem);
    daslNode *target = p;
    /* The mirroring entry is in the covering node p, or - when the deleted key
     * is exactly p's forward leader - in p->forward. */
    if (!(idx >= 0 && dasl_cmp(&p->keys[idx], p->members[idx], delk, delmem) == 0)) {
        if (p->forward != NULL &&
            dasl_cmp(&p->forward->keys[0], p->forward->members[0], delk, delmem) == 0) {
            target = p->forward;
            idx = 0;
        } else {
            return; /* not present at this level */
        }
    }
    memmove(&target->keys[idx], &target->keys[idx + 1],
            (target->n_key - idx - 1) * sizeof(daslKey));
    memmove(&target->members[idx], &target->members[idx + 1],
            (target->n_key - idx - 1) * sizeof(sds));
    memmove(&target->next[idx], &target->next[idx + 1],
            (target->n_key - idx - 1) * sizeof(daslNode *));
    memmove(&target->weights[idx], &target->weights[idx + 1],
            (target->n_key - idx - 1) * sizeof(unsigned long));
    target->keys[target->n_key - 1] = DASL_EMPTY_KEY;
    target->members[target->n_key - 1] = NULL;
    target->next[target->n_key - 1] = NULL;
    target->weights[target->n_key - 1] = 0;
    target->n_key--;
    if (target->n_key == 0) {
        /* Unlink via the O(1) back-pointer (head for the first real node). */
        daslNode *pred = target->prev;
        pred->forward = target->forward;
        if (target->forward != NULL) target->forward->prev = pred;
        if (prev[h] == target) prev[h] = NULL;
        sl->alloc_size -= zmalloc_usable_size(target);
        zfree(target);
    }
}

/* Core delete. When `kept` is non-NULL the removed element's owned member sds
 * is returned through it (and NOT freed) so the caller can transfer ownership
 * (used by the score-reposition path to preserve a shared buffer); otherwise
 * the member is freed. Returns 1 if an element was removed, 0 if not found. */
static int daslDeleteEx(dasl *sl, double score, sds ele, sds *kept) {
    daslKey key;
    daslMakeKey(&key, score, ele);

    daslNode *prev[DASL_MAXHEIGHT];

    /* Forward-aware descent recording the covering node at each level.
     * Predecessors for unlinking come from each node's prev back-pointer, so
     * no predecessor needs to be tracked here. */
    daslDescend(sl, &key, ele, prev);

    daslNode *n0 = prev[0];
    int idx = dasl_find_le(n0, &key, ele);
    if (!(idx >= 0 && dasl_cmp(&n0->keys[idx], n0->members[idx], &key, ele) == 0))
        return 0; /* not found */

    /* The owned level-0 member for the deleted element. Freed only at the very
     * end: leader propagation and index unlink below still match against it. */
    sds dead_mem = n0->members[idx];

    /* Key whose post-delete descent visits the affected ancestor chain, used for
     * the weight recompute. For a leader delete the ancestor index entries are
     * rewritten to the new leader, so we recompute along the new leader's path,
     * not the (now absent) deleted key's. */
    daslKey rkey = key;
    sds rmem = ele;

    if (idx > 0) {
        /* Non-leader: shift it out at level 0; no upward fixup needed (only
         * leaders are mirrored on express lanes). */
        memmove(&n0->keys[idx], &n0->keys[idx + 1], (n0->n_key - idx - 1) * sizeof(daslKey));
        memmove(&n0->members[idx], &n0->members[idx + 1], (n0->n_key - idx - 1) * sizeof(sds));
        memmove(&n0->weights[idx], &n0->weights[idx + 1], (n0->n_key - idx - 1) * sizeof(unsigned long));
        n0->keys[n0->n_key - 1] = DASL_EMPTY_KEY;
        n0->members[n0->n_key - 1] = NULL;
        n0->weights[n0->n_key - 1] = 0;
        n0->n_key--;
    } else if (n0->n_key > 1) {
        /* Leader delete, node survives: new leader is the old slot 1. */
        daslKey news = n0->keys[1];
        sds news_mem = n0->members[1];
        memmove(&n0->keys[0], &n0->keys[1], (n0->n_key - 1) * sizeof(daslKey));
        memmove(&n0->members[0], &n0->members[1], (n0->n_key - 1) * sizeof(sds));
        memmove(&n0->weights[0], &n0->weights[1], (n0->n_key - 1) * sizeof(unsigned long));
        n0->keys[n0->n_key - 1] = DASL_EMPTY_KEY;
        n0->members[n0->n_key - 1] = NULL;
        n0->weights[n0->n_key - 1] = 0;
        n0->n_key--;
        daslPropagateLeader(sl, prev, 0, &key, dead_mem, &news, news_mem);
        rkey = news;
        rmem = news_mem;
    } else {
        /* Single-key node: unlink at level 0 via its back-pointer (O(1)), then
         * remove its index entries on every level above. */
        daslNode *pred = n0->prev;
        pred->forward = n0->forward;
        if (n0->forward != NULL) n0->forward->prev = pred;
        else sl->tail = (pred == sl->head[0]) ? NULL : pred; /* removed the last node */
        for (int hh = 1; hh < sl->max_height; hh++)
            daslUnlinkIndexEntry(sl, prev, hh, &key, dead_mem);
        sl->alloc_size -= zmalloc_usable_size(n0);
        zfree(n0);
        prev[0] = NULL;
    }

    /* Shrink height while the top level is empty. */
    while (sl->max_height > 1 && sl->head[sl->max_height - 1]->forward == NULL) {
        sl->alloc_size -= zmalloc_usable_size(sl->head[sl->max_height - 1]);
        zfree(sl->head[sl->max_height - 1]);
        sl->head[sl->max_height - 1] = NULL;
        sl->max_height--;
    }

    sl->length--;

    /* Rebuild the weights along the affected path (delete creates no new nodes,
     * so there are no split siblings to feed in). Skip when the list is empty. */
    if (sl->length > 0) {
        daslNode *added[DASL_MAXHEIGHT];
        for (int i = 0; i < DASL_MAXHEIGHT; i++) added[i] = NULL;
        /* Reuse the descent's prev[] as the recompute path only for a plain
         * non-leader delete (idx > 0): no leader changed and no node was freed,
         * so prev[] is still the valid post-delete covering chain. The leader
         * delete (propagates a new leader up, shifting higher covers) and the
         * single-key unlink (frees nodes, NULLs prev[]) descend afresh. */
        daslRecomputePath(sl, &rkey, rmem, added, (idx > 0) ? prev : NULL);
    }

    /* Every borrowed reference to the owned member has now been rewritten or
     * removed. Either hand it back to the caller (ownership transfer) or free
     * it now. */
    sl->alloc_size -= sdsAllocSize(dead_mem);
    if (kept) *kept = dead_mem;
    else sdsfree(dead_mem);

    return 1;
}

int daslDelete(dasl *sl, double score, sds ele) {
    return daslDeleteEx(sl, score, ele, NULL);
}

/* Move the element (oldscore, member) to newscore, preserving the exact owned
 * member sds buffer (so an external borrower of that pointer - the zset dict
 * key - stays valid). Returns a cursor to the element's new position, or a
 * null cursor if (oldscore, member) is not present. */
daslCursor daslReposition(dasl *sl, double oldscore, double newscore, sds member) {
    daslKey key;
    daslMakeKey(&key, oldscore, member);
    daslNode *x = daslDescend(sl, &key, member, NULL);
    int j = dasl_find_le(x, &key, member);
    if (j < 0 || dasl_cmp(&x->keys[j], x->members[j], &key, member) != 0) {
        daslCursor n = { NULL, 0 };
        return n;
    }
    daslCursor c = { x, j };
    return daslUpdateScore(sl, &c, newscore);
}

/* Recover the original double from a composite key's sortable score bytes
 * (exact inverse of the encoding in daslMakeKey). */
static double daslDecodeScore(const daslKey *k) {
    uint64_t bits;
    memcpy(&bits, k->b, DASL_SCORE_SIZE);
    bits = ntohu64(bits);
    bits = (bits & (1ULL << 63)) ? (bits ^ (1ULL << 63)) : ~bits;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return d;
}

sds daslScan(const dasl *sl, double score, sds ele, int scan_num) {
    daslKey key;
    daslMakeKey(&key, score, ele);
    daslNode *x = daslDescend(sl, &key, ele, NULL);
    int j = dasl_find_le(x, &key, ele);
    if (j < 0) {
        /* key precedes every element: start at the first real node. */
        x = sl->head[0]->forward;
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
    daslKey key;
    daslMakeKey(&key, score, ele);

    int h = sl->max_height - 1;
    daslNode *x = sl->head[h];
    unsigned long rank = 0;
    while (1) {
        /* Step forward while the next node's leader is still <= the target,
         * accumulating the weight crossed. */
        while (x->forward != NULL &&
               dasl_cmp(&x->forward->keys[0], x->forward->members[0], &key, ele) <= 0) {
            rank += dasl_coverweight(sl, x, h);
            x = x->forward;
        }
        int j = dasl_find_le(x, &key, ele);
        /* Add the weights of the slots strictly before the landing slot. */
        for (int t = 0; t < j; t++) rank += x->weights[t];
        if (h == 0) {
            if (j >= 0 && dasl_cmp(&x->keys[j], x->members[j], &key, ele) == 0)
                return rank + 1; /* 1-based */
            return 0;            /* not present */
        }
        x = (j < 0) ? sl->head[h - 1] : x->next[j];
        h--;
    }
}

daslCursor daslGetElementByRank(const dasl *sl, unsigned long rank) {
    daslCursor c = { NULL, 0 };
    if (rank == 0 || rank > sl->length) return c;

    int h = sl->max_height - 1;
    daslNode *x = sl->head[h];
    unsigned long trav = 0; /* elements passed so far (0-based frontier) */
    while (1) {
        /* Step forward while doing so keeps the cumulative count below rank. */
        while (x->forward != NULL && trav + dasl_coverweight(sl, x, h) < rank) {
            trav += dasl_coverweight(sl, x, h);
            x = x->forward;
        }
        if (x == sl->head[h]) {
            /* Still in this level's head prefix: drop a level (cannot happen at
             * level 0, whose head prefix weight is always 0). */
            x = sl->head[h - 1];
            h--;
            continue;
        }
        int j = 0;
        while (j < x->n_key && trav + x->weights[j] < rank) { trav += x->weights[j]; j++; }
        if (h == 0) { c.node = x; c.slot = j; return c; }
        x = x->next[j];
        h--;
    }
}

double daslCursorScore(const daslCursor *c) {
    return daslDecodeScore(&c->node->keys[c->slot]);
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
    daslNode *x = sl->head[0]->forward;
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
    /* Roll over to the next node's first slot, or off the end. */
    if (c.node->forward != NULL) { c.node = c.node->forward; c.slot = 0; }
    else { c.node = NULL; c.slot = 0; }
    return c;
}

daslCursor daslPrev(daslCursor c) {
    if (c.node == NULL) return c;
    if (c.slot > 0) { c.slot--; return c; }
    /* Roll back to the previous node's last slot. The first real node's prev is
     * the level-0 head (n_key == 0), which means "off the front". */
    daslNode *p = c.node->prev;
    if (p == NULL || p->n_key == 0) { c.node = NULL; c.slot = 0; }
    else { c.node = p; c.slot = p->n_key - 1; }
    return c;
}

int daslDeleteCursor(dasl *sl, daslCursor *c) {
    if (c->node == NULL) return 0;
    double score = daslDecodeScore(&c->node->keys[c->slot]);
    sds member = c->node->members[c->slot];
    return daslDelete(sl, score, member);
}

/* Change a cursor element's score in place when ordering is preserved, else
 * delete + re-insert; return a cursor to the element's new position. Mirrors
 * zslUpdateScore() (which keeps the same node pointer; here the cursor may move
 * because array-packed nodes relocate elements on reposition). */
daslCursor daslUpdateScore(dasl *sl, daslCursor *c, double newscore) {
    daslNode *node = c->node;
    int slot = c->slot;
    sds member = node->members[slot];

    /* Fast path: a non-leader slot (slot != 0, so no express-lane leader to
     * rewrite) whose order relative to its level-0 neighbours is unchanged. We
     * can rewrite the composite key in place; weights and links are untouched. */
    if (slot != 0) {
        daslKey nk;
        daslMakeKey(&nk, newscore, member);
        daslCursor p = daslPrev(*c);
        daslCursor n = daslNext(*c);
        int okprev = (p.node == NULL) ||
            dasl_cmp(&p.node->keys[p.slot], p.node->members[p.slot], &nk, member) < 0;
        int oknext = (n.node == NULL) ||
            dasl_cmp(&nk, member, &n.node->keys[n.slot], n.node->members[n.slot]) < 0;
        if (okprev && oknext) {
            node->keys[slot] = nk;
            return *c;
        }
    }

    /* Slow path: reposition via delete + re-insert, preserving the exact member
     * sds buffer (daslDeleteEx hands it back instead of freeing it, and
     * daslInsertOwned re-stores it without copying), so external borrowers of
     * the member pointer stay valid. */
    double oldscore = daslDecodeScore(&node->keys[slot]);
    sds kept = NULL;
    daslDeleteEx(sl, oldscore, member, &kept);
    return daslInsertOwned(sl, newscore, kept);
}

unsigned long daslGetRankByCursor(const dasl *sl, daslCursor c) {
    if (c.node == NULL) return 0;
    double score = daslDecodeScore(&c.node->keys[c.slot]);
    return daslGetRank(sl, score, c.node->members[c.slot]);
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
 * score <= edge when orEqual==1). Elements are score-major sorted, so the
 * qualifiers form a prefix; this accumulates their weights in O(log N) using
 * the same descent shape as daslGetRank(). */
static unsigned long daslCountByScore(const dasl *sl, double edge, int orEqual) {
    int h = sl->max_height - 1;
    daslNode *x = sl->head[h];
    unsigned long cnt = 0;
    while (1) {
        /* Step forward while the whole next node still qualifies. */
        while (x->forward != NULL) {
            double fs = daslDecodeScore(&x->forward->keys[0]);
            int q = orEqual ? (fs <= edge) : (fs < edge);
            if (!q) break;
            cnt += dasl_coverweight(sl, x, h);
            x = x->forward;
        }
        /* Last slot in x that still qualifies (slots are ascending by score). */
        int j = -1;
        for (int i = 0; i < x->n_key; i++) {
            double s = daslDecodeScore(&x->keys[i]);
            int q = orEqual ? (s <= edge) : (s < edge);
            if (q) j = i; else break;
        }
        for (int t = 0; t < j; t++) cnt += x->weights[t];
        if (h == 0) {
            if (j >= 0) cnt += x->weights[j]; /* level-0 slot weights are 1 */
            return cnt;
        }
        /* Descend through the last qualifying slot (its subtree holds the
         * boundary); its weight is accounted at the level below. */
        x = (j < 0) ? sl->head[h - 1] : x->next[j];
        h--;
    }
}

daslCursor daslNthInRange(dasl *sl, zrangespec *range, long n, unsigned long *out_rank) {
    daslCursor c = { NULL, 0 };
    if (!daslIsInRange(sl, range)) return c;

    /* In-range elements occupy 1-based ranks [lo+1, hi]: lo counts the low
     * prefix failing GteMin, hi counts elements passing LteMax. */
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
    int h = sl->max_height - 1;
    daslNode *x = sl->head[h];
    unsigned long cnt = 0;
    while (1) {
        while (x->forward != NULL && inprefix(x->forward->members[0], range)) {
            cnt += dasl_coverweight(sl, x, h);
            x = x->forward;
        }
        int j = -1;
        for (int i = 0; i < x->n_key; i++) {
            if (inprefix(x->members[i], range)) j = i; else break;
        }
        for (int t = 0; t < j; t++) cnt += x->weights[t];
        if (h == 0) {
            if (j >= 0) cnt += x->weights[j];
            return cnt;
        }
        x = (j < 0) ? sl->head[h - 1] : x->next[j];
        h--;
    }
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
    /* Each delete invalidates cursors, so re-anchor at the new first-in-range
     * element every iteration: O(removed * log N). */
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
    /* start/end are 1-based inclusive. After each delete ranks shift down by
     * one, so always remove the element now at rank `start` and shrink end. */
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
 * Member sds pointers are left untouched (the node struct is copied verbatim),
 * so daslDefragMember can subsequently move the strings. */
void daslDefragNodes(dasl *sl, void *(*defragfn)(void *)) {
    for (int h = 0; h < sl->max_height; h++) {
        /* Relocate this level's sentinel head. */
        daslNode *nh = defragfn(sl->head[h]);
        if (nh) sl->head[h] = nh;

        /* Relocate every node on the level-h forward chain, repairing the
         * same-level forward/prev links (and the level-0 tail) as we go. */
        daslNode *pnode = sl->head[h];
        daslNode *cur = pnode->forward;
        while (cur != NULL) {
            daslNode *ncur = defragfn(cur);
            if (ncur) cur = ncur;
            pnode->forward = cur;
            cur->prev = pnode;
            if (h == 0 && cur->forward == NULL) sl->tail = cur;
            pnode = cur;
            cur = cur->forward;
        }

        /* Re-derive the descent pointers next[] for this index level from the
         * (already relocated) level below. Each slot mirrors a level-(h-1)
         * node whose leader equals the slot's key; both chains are ascending,
         * so a single lockstep walk links them. Heads carry no usable next[]. */
        if (h >= 1) {
            daslNode *child = sl->head[h - 1]->forward;
            daslNode *node = sl->head[h]->forward;
            while (node != NULL) {
                for (int i = 0; i < node->n_key; i++) {
                    while (child != NULL &&
                           dasl_cmp(&child->keys[0], child->members[0],
                                    &node->keys[i], node->members[i]) < 0)
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
 * it when it is a node leader. (score, oldmem) is located via the normal
 * descent, so `oldmem` must still be valid when this is called. */
void daslDefragMember(dasl *sl, double score, sds oldmem, sds newmem) {
    daslKey key;
    daslMakeKey(&key, score, oldmem);

    daslNode *prev[DASL_MAXHEIGHT];
    daslDescend(sl, &key, oldmem, prev);

    daslNode *n0 = prev[0];
    int idx = dasl_find_le(n0, &key, oldmem);
    if (!(idx >= 0 && dasl_cmp(&n0->keys[idx], n0->members[idx], &key, oldmem) == 0))
        return; /* not present (should not happen for a live element) */
    n0->members[idx] = newmem;
    if (idx != 0) return; /* non-leader: no express-lane mirror to fix */

    /* Rewrite the mirroring borrowed pointer on each level above, stopping once
     * a level's entry is not the leader (same shape as daslPropagateLeader). */
    for (int h = 1; h < sl->max_height; h++) {
        daslNode *p = prev[h];
        int j = dasl_find_le(p, &key, oldmem);
        daslNode *target = p;
        if (!(j >= 0 && dasl_cmp(&p->keys[j], p->members[j], &key, oldmem) == 0)) {
            if (p->forward != NULL &&
                dasl_cmp(&p->forward->keys[0], p->forward->members[0], &key, oldmem) == 0) {
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
    for (int i = sl->max_height - 1; i >= 0; i--) {
        printf("L%d:", i);
        daslNode *x = sl->head[i]->forward;
        while (x != NULL) {
            printf(" [");
            for (int j = 0; j < x->n_key; j++)
                printf("%s%g:%s", j ? "," : "", daslDecodeScore(&x->keys[j]),
                       x->members[j] ? x->members[j] : "(nil)");
            printf("]");
            x = x->forward;
        }
        printf("\n");
    }
}
