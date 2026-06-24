/* dasl.h - DASL (Doubly-Array Skip List) ported to C.
 *
 * This is a faithful C port of the array-packed skip list from
 * https://github.com/ (DASL reference: src/skiplist.h). It is an
 * ISOLATED prototype (Phase 0): it is NOT wired into the Redis server
 * build yet and is exercised only by a standalone harness.
 *
 * Two node types. Level-0 data lives in `daslLeaf`, and the express-lane index
 * levels (>= 1) and their per-level heads are `daslNode`. Both store the zset
 * composite the same way: a raw double `scores[]` slot plus a parallel
 * `members[]` sds slot (16 bytes/slot). Routing and leaf lookup share one total
 * order - compare the raw double, then sdscmp the full member on a tie.
 *
 * Differences vs. the C++ reference:
 *   - The index key is the zset composite (double score, sds member), stored
 *     verbatim: index nodes carry the score as a raw double in `scores[]` and
 *     the full member sds in `members[]`, exactly like the leaves. Doubles
 *     compare in numeric order, so ordering is the raw double then sdscmp of the
 *     full member on a tie - the same total order at every level. (Earlier
 *     revisions packed the composite into a fixed-width sortable byte key for
 *     branch-free routing; that is gone now that index slots hold raw scores.)
 *   - Empty index slots hold score 0 plus a NULL member; intra-node search is
 *     bounded by n_key and never inspects empties.
 *   - C++ new/delete -> zmalloc/zfree, with allocation-size tracking.
 *   - Only the canonical operations are ported: Insert, Contains, Delete,
 *     Scan. Benchmark-only variants are dropped.
 *   - Two correctness bugs latent in the reference are fixed (see dasl.c):
 *     all descents are forward-aware (unpromoted front nodes stay reachable),
 *     and a node is promoted to the next level exactly once (no duplicate
 *     express-lane leaders). Splits are even (DASL_ARR_SIZE/2).
 *
 * The structural design is unchanged: each node packs up to DASL_ARR_SIZE
 * elements; the bottom level (leaf head forward chain) holds all elements, and
 * upper levels are express lanes whose `next[i]` pointers descend one level.
 *
 * Member ownership: at the bottom level (leaf chain) each slot's `members[i]`
 * is an owned sds copy of the inserted member; at express-lane levels
 * `members[i]` is a *borrowed* pointer to the leaf-level owner's sds, used only
 * for tie-break comparison. Borrowed pointers are never freed and always
 * move/propagate in lock-step with their score slot.
 */

#ifndef DASL_H
#define DASL_H

#include <stddef.h>

#include "sds.h"

#ifndef DASL_ARR_SIZE
#define DASL_ARR_SIZE 64   /* elements packed per node (must be a power of two); overridable for tests */
#endif
#define DASL_MAXHEIGHT 32  /* enough for 2^64 elements */

/* Index node: an express-lane node at level >= 1, and the per-level sentinel
 * head for those levels. Packs up to DASL_ARR_SIZE composite entries (raw
 * double score + member sds, exactly like a leaf), the order-statistics
 * weights[], and the descent pointers next[]. Level-0 data lives in the
 * separate, slimmer daslLeaf below; accordingly next[i] points at a daslLeaf
 * when this node is at level 1, and at a daslNode when it is at level >= 2
 * (hence the void* element type - the descender knows the level and casts). On
 * a head only weights[0] is used (the prefix weight: level-0 elements preceding
 * head->forward). */
typedef struct daslNode {
    double scores[DASL_ARR_SIZE]; /* scores[0] is the node's leader score; ascending */
    sds members[DASL_ARR_SIZE];  /* borrowed pointer to the level-0 owner sds */
    struct daslNode *forward;    /* next node at the same level */
    struct daslNode *prev;       /* previous node at the same level (head for the
                                  * first real node, NULL for a head); lets delete
                                  * find a predecessor in O(1) instead of scanning */
    int n_key;                   /* number of occupied slots */
    void *next[DASL_ARR_SIZE];   /* per-slot descent pointers: level-1 children are
                                  * daslLeaf*, level>=2 children are daslNode* */
    /* Order-statistics weight per slot, stored as a trailing flexible array so a
     * node only pays for the width its level needs (see dasl_weight_width in
     * dasl.c): a level-h slot covers one child node, so its weight is capped at
     * DASL_ARR_SIZE^h, and the width is uint8 at level 1, uint16 at level 2,
     * uint32 at levels 3-5, uint64 above. The raw bytes are never read directly -
     * all access goes through dasl_wget/dasl_wset (and DASL_WSUM), which cast to
     * the level's type. The buffer starts 8-byte aligned (the last fixed member
     * is a pointer array), so wider casts are aligned. On a head only weights[0]
     * is used (the level-0 prefix weight). */
    unsigned char weights[];
} daslNode;

/* Level-0 data node and the level-0 sentinel head. Holds the score as a raw
 * double plus the owned full member sds - the same composite layout as an index
 * node. A leaf carries neither weights[] (every leaf slot weighs the implicit
 * constant 1) nor next[] (descent pointers live only on index levels), so a
 * leaf slot costs 16 bytes (double + sds pointer) instead of the wider index
 * slot (which also carries a weight and a descent pointer). Leaves form a
 * doubly-linked chain (forward/prev) anchored at the structure's leaf head. */
typedef struct daslLeaf {
    double scores[DASL_ARR_SIZE]; /* scores[0] is the leader; ascending */
    sds members[DASL_ARR_SIZE];   /* full owned member per slot */
    struct daslLeaf *forward;     /* next leaf on the level-0 chain */
    struct daslLeaf *prev;        /* previous leaf (the leaf head for the first
                                   * real leaf, NULL for the head itself) */
    int n_key;                    /* number of occupied slots */
} daslLeaf;

typedef struct dasl {
    daslLeaf *lhead;                /* level-0 (data) sentinel head */
    daslNode *head[DASL_MAXHEIGHT]; /* index sentinel head per level; [0] unused */
    daslLeaf *tail;                 /* last node on the level-0 chain (NULL if empty) */
    int max_height;                 /* number of levels currently in use (>=1) */
    unsigned long length;           /* number of distinct elements */
    size_t alloc_size;              /* tracked heap usage, like zslAllocSize() */
} dasl;

/* Cursor referencing a single element (leaf + slot), as returned by
 * daslGetElementByRank. node==NULL means "no such element". Cursors only ever
 * reference level-0 data, so node is a daslLeaf*. */
typedef struct daslCursor {
    daslLeaf *node;
    int slot;
} daslCursor;

dasl *daslCreate(void);
void daslFree(dasl *sl);

/* Insert (score, ele). The member is copied (the caller keeps ownership of
 * `ele`). Returns a cursor to the inserted element. Mirrors zslInsert(), which
 * returns the inserted zskiplistNode*.
 *
 * The caller MUST guarantee (score, member) is not already present; the zset
 * dict (keyed by member) enforces uniqueness, so no DASL-side duplicate check
 * is performed. Inserting a duplicate corrupts the structure. */
daslCursor daslInsert(dasl *sl, double score, sds ele);

/* Returns 1 if (score, ele) is present, 0 otherwise. */
int daslContains(const dasl *sl, double score, sds ele);

/* Delete (score, ele). Returns 1 if removed, 0 if not found. Mirrors the
 * value-based deletion path zsetDel()/zslDelete() use internally. */
int daslDelete(dasl *sl, double score, sds ele);

/* Delete the element a cursor points at. Returns 1 if removed, 0 if the cursor
 * is null. Mirrors zslDelete(zsl, node) (deletion of an already-located node). */
int daslDeleteCursor(dasl *sl, daslCursor *c);

/* Forward scan starting at the largest key <= (score, ele), visiting up to
 * `scan_num` keys. Returns the member of the last key visited, or NULL if the
 * structure is empty (diagnostic/benchmark use). The returned sds is owned by
 * the structure; do not free it. */
sds daslScan(const dasl *sl, double score, sds ele, int scan_num);

/* Order statistics. Rank is 1-based, matching zslGetRank(). */

/* 1-based rank of (score, ele) in ascending order, or 0 if not present. */
unsigned long daslGetRank(const dasl *sl, double score, sds ele);

/* Element at 1-based rank `rank`. The returned cursor's node is NULL when
 * `rank` is 0 or greater than the number of elements. */
daslCursor daslGetElementByRank(const dasl *sl, unsigned long rank);

/* Decode the (score, member) referenced by a cursor returned above. The member
 * sds is owned by the structure; do not free it. */
double daslCursorScore(const daslCursor *c);
sds daslCursorMember(const daslCursor *c);

/* Alias of daslCursorMember(), named to mirror zslGetNodeElement() so call
 * sites translate as zslGetNodeElement(ln) -> daslGetNodeElement(&cursor). */
sds daslGetNodeElement(const daslCursor *c);

/* True when a cursor points at no element (past either end / empty list). */
static inline int daslCursorIsNull(const daslCursor *c) { return c->node == NULL; }

/* Cursor-based iteration over the in-order (level-0) element chain. These
 * replace the raw zskiplist walks `ln = ln->level[0].forward` (daslNext) and
 * `ln = ln->backward` (daslPrev). daslFirst/daslLast return the smallest/
 * largest element. All return a null cursor when there is no such element. */
daslCursor daslFirst(const dasl *sl);
daslCursor daslLast(const dasl *sl);
daslCursor daslNext(daslCursor c);
daslCursor daslPrev(daslCursor c);

/* Change the score of the element a cursor points at, repositioning it as
 * needed (the member is preserved). Returns a cursor to the element at its new
 * position. Mirrors zslUpdateScore(). */
daslCursor daslUpdateScore(dasl *sl, daslCursor *c, double newscore);

/* Move (oldscore, member) to newscore, preserving the exact owned member sds
 * buffer so an external borrower of that pointer (e.g. the zset dict key) stays
 * valid. Returns a cursor to the new position, or a null cursor if not present.
 * `member` must be the structure-owned sds for the element being moved. */
daslCursor daslReposition(dasl *sl, double oldscore, double newscore, sds member);

/* 1-based rank of the element a cursor points at, or 0 for a null cursor.
 * Mirrors zslGetRankByNode(). */
unsigned long daslGetRankByCursor(const dasl *sl, daslCursor c);

static inline unsigned long daslLength(const dasl *sl) { return sl->length; }
size_t daslAllocSize(const dasl *sl);

/* Hint the OS (madvise) that the structure's heap is not needed, used before
 * fork to reduce COW. Mirrors the per-node dismissal the skiplist did. */
void daslDismiss(dasl *sl);

/* Active-defrag support.
 *
 * daslDefragNodes relocates every daslNode (heads included) using the supplied
 * allocator `defragfn` (which must move the allocation and free the old one,
 * returning NULL when no move happened), fixing all head[]/forward/prev/next[]
 * and tail links. It does NOT touch the member sds buffers (their pointers are
 * copied verbatim), so external borrowers of those pointers stay valid; call it
 * before relocating the member strings.
 *
 * daslDefragMember repoints every reference to a member sds whose backing
 * buffer moved from `oldmem` to `newmem` (the level-0 owner slot plus any
 * express-lane borrowers). The (score, oldmem) pair must still be locatable,
 * so call it while `oldmem` is still alive and only free it afterwards. */
void daslDefragNodes(dasl *sl, void *(*defragfn)(void *));
void daslDefragMember(dasl *sl, double score, sds oldmem, sds newmem);

/* Range queries / range deletes. These reuse the zset range-spec types and the
 * structure-agnostic predicate/parse helpers (zslValueGteMin/LteMax,
 * zslLexValueGteMin/LteMax, zslParseRange/zslParseLexRange), so they are only
 * declared when server.h (which defines zrangespec/zlexrangespec/dict) is in
 * scope. The standalone harness, which does not include server.h, sees only
 * the core API above. */
#ifdef __REDIS_H

/* Nth element within a score/lex range. n is 0-based forward, negative for
 * reverse (-1 = last in range). Returns a null cursor when out of bounds. If
 * out_rank != NULL it receives the returned element's 1-based absolute rank.
 * Mirror zslNthInRange()/zslNthInLexRange(). */
daslCursor daslNthInRange(dasl *sl, zrangespec *range, long n, unsigned long *out_rank);
daslCursor daslNthInLexRange(dasl *sl, zlexrangespec *range, long n, unsigned long *out_rank);

/* Delete every element within the given range, calling dictDelete(dict, member)
 * for each removed element (dict may be NULL to skip). Return the count removed.
 * Mirror zslDeleteRangeByScore()/ByLex()/ByRank(); rank bounds are 1-based
 * inclusive. */
unsigned long daslDeleteRangeByScore(dasl *sl, zrangespec *range, dict *dict);
unsigned long daslDeleteRangeByLex(dasl *sl, zlexrangespec *range, dict *dict);
unsigned long daslDeleteRangeByRank(dasl *sl, unsigned int start, unsigned int end, dict *dict);

#endif /* __REDIS_H */

/* Debug: print every level. */
void daslPrint(const dasl *sl);

#endif /* DASL_H */
