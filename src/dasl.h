/* dasl.h - DASL (Doubly-Array Skip List) ported to C.
 *
 * This is a faithful C port of the array-packed skip list from
 * https://github.com/ (DASL reference: src/skiplist.h). It is an
 * ISOLATED prototype (Phase 0): it is NOT wired into the Redis server
 * build yet and is exercised only by a standalone harness.
 *
 * Differences vs. the C++ reference:
 *   - The key is the zset composite (double score, sds member). It is packed
 *     into a fixed-width 16-byte `daslKey`: bytes [0,8) hold the score in an
 *     order-preserving "sortable" encoding (so byte-wise memcmp matches numeric
 *     order) and bytes [8,16) hold the first 8 bytes of the member (zero
 *     padded). A parallel `members[]` array holds the full sds. Ordering is a
 *     single 16-byte memcmp; ties on the composite fall back to sdscmp of the
 *     full member. See dasl.c for the encoding/comparison details.
 *   - Empty array slots use an all-0xFF composite (the maximum key) plus a NULL
 *     member as a sentinel; intra-node search is bounded by n_key and never
 *     inspects empties, so the sentinel is belt-and-suspenders for debugging.
 *   - C++ new/delete -> zmalloc/zfree, with allocation-size tracking.
 *   - Only the canonical operations are ported: Insert, Contains, Delete,
 *     Scan. Benchmark-only variants are dropped.
 *   - Two correctness bugs latent in the reference are fixed (see dasl.c):
 *     all descents are forward-aware (unpromoted front nodes stay reachable),
 *     and a node is promoted to the next level exactly once (no duplicate
 *     express-lane leaders). Splits are even (DASL_ARR_SIZE/2).
 *
 * The structural design is unchanged: each node packs up to DASL_ARR_SIZE
 * keys; the bottom level (head[0] forward chain) holds all elements, and
 * upper levels are express lanes whose `next[i]` pointers descend one level.
 *
 * Member ownership: at the bottom level (head[0] chain) each slot's
 * `members[i]` is an owned sds copy of the inserted member; at express-lane
 * levels `members[i]` is a *borrowed* pointer to the bottom-level owner's sds,
 * used only for tie-break comparison. Borrowed pointers are never freed and
 * always move/propagate in lock-step with their composite key.
 */

#ifndef DASL_H
#define DASL_H

#include <stddef.h>

#include "sds.h"

#ifndef DASL_ARR_SIZE
#define DASL_ARR_SIZE 64   /* keys packed per node (must be a power of two); overridable for tests */
#endif
#define DASL_MAXHEIGHT 32  /* enough for 2^64 elements */

#define DASL_SCORE_SIZE 8                                   /* order-preserving score bytes */
#define DASL_MEMBER_PREFIX 8                                /* leading member bytes packed inline */
#define DASL_CK_SIZE (DASL_SCORE_SIZE + DASL_MEMBER_PREFIX) /* composite key width: 16 */

/* Fixed-width composite key: [0,8) sortable score, [8,16) member prefix. */
typedef struct daslKey {
    unsigned char b[DASL_CK_SIZE];
} daslKey;

/* Node layout note (memory): a level-0 data node (`is_leaf == 1`) holds every
 * element, so it dominates the structure's footprint, yet its `next[]` is always
 * NULL (descent pointers exist only on index levels) and every `weights[]` slot
 * is always 1. Those two arrays are therefore kept LAST and are NOT allocated on
 * leaves: leaves are sized `DASL_LEAF_SIZE` (the struct truncated before
 * `weights`), saving 2*DASL_ARR_SIZE pointers/longs (~38% of a node) per
 * element. Index nodes and the per-level heads are allocated full size. Code
 * must never read/write `weights[]`/`next[]` on a node with `is_leaf == 1`;
 * a leaf slot's order-statistics weight is the constant 1. */
typedef struct daslNode {
    daslKey keys[DASL_ARR_SIZE]; /* keys[0] is the node's leader key; ascending */
    sds members[DASL_ARR_SIZE];  /* full member per slot; owned at level 0, borrowed above */
    struct daslNode *forward;    /* next node at the same level */
    struct daslNode *prev;       /* previous node at the same level (head for the
                                  * first real node, NULL for a head); lets delete
                                  * find a predecessor in O(1) instead of scanning */
    int n_key;                   /* number of occupied key slots */
    int is_leaf;                 /* 1 for level-0 data nodes (no weights[]/next[]
                                  * allocated); 0 for index nodes and heads */
    /* The two arrays below are omitted on leaves (see DASL_LEAF_SIZE). */
    unsigned long weights[DASL_ARR_SIZE]; /* order-statistics weight per slot: at
                                  * index levels weights[i] counts the level-0
                                  * elements under next[i]'s subtree. On a head
                                  * node only weights[0] is used: the count of
                                  * level-0 keys preceding head->forward (the
                                  * prefix weight). (Leaf slots are implicitly 1.) */
    struct daslNode *next[DASL_ARR_SIZE]; /* per-key descent pointers (index levels) */
} daslNode;

/* Allocation size of a level-0 data leaf: the struct truncated just before the
 * index-only weights[]/next[] arrays. */
#define DASL_LEAF_SIZE offsetof(daslNode, weights)

typedef struct dasl {
    daslNode *head[DASL_MAXHEIGHT]; /* sentinel head node per level */
    daslNode *tail;                 /* last node on the level-0 chain (NULL if empty) */
    int max_height;                 /* number of levels currently in use (>=1) */
    unsigned long length;           /* number of distinct elements */
    size_t alloc_size;              /* tracked heap usage, like zslAllocSize() */
} dasl;

/* Cursor referencing a single element (node + slot), as returned by
 * daslGetElementByRank. node==NULL means "no such element". */
typedef struct daslCursor {
    daslNode *node;
    int slot;
} daslCursor;

dasl *daslCreate(void);
void daslFree(dasl *sl);

/* Insert (score, ele). The member is copied (the caller keeps ownership of
 * `ele`). No-op if an equal (score, member) is already present. Returns a
 * cursor to the inserted element, or a null cursor (node==NULL) on duplicate.
 * Mirrors zslInsert(), which returns the inserted zskiplistNode*. */
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
