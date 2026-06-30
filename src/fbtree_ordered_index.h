#ifndef FBTREE_ORDERED_INDEX_H
#define FBTREE_ORDERED_INDEX_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
/* Forward declare sds types to avoid C++ issues with sds.h macros.
 * C code will have sds.h included elsewhere; C++ benchmarks get the typedef here. */
typedef char *sds;
typedef const char *const_sds;

typedef struct fbtreeIndex fbtreeIndex;

/* Opaque iterator type that can be stack allocated */
typedef uint64_t fbtreeIterator[3];

/* Internal API for testing */
fbtreeIndex *fbtreeCreate(void);
sds fbtreeInsert(fbtreeIndex *fbt, sds string);
bool fbtreeDelete(fbtreeIndex *fbt, const_sds key);
/* Like fbtreeDelete, but matches the target leaf slot by byte-equality
 * (sdscmp == 0) rather than pointer identity. Returns true iff found+deleted. */
bool fbtreeDeleteByValue(fbtreeIndex *fbt, const_sds key);
sds fbtreePopMin(fbtreeIndex *fbt);
sds fbtreePopMax(fbtreeIndex *fbt);
void fbtreeEmpty(fbtreeIndex *fbt);
void fbtreeFree(fbtreeIndex *fbt);
unsigned long fbtreeLength(fbtreeIndex *fbt);
void fbtreeInitIterator(fbtreeIterator *iterator, fbtreeIndex *fbt);
void fbtreeResetIterator(fbtreeIterator *iterator);
fbtreeIndex *fbtreeIteratorGetIndex(fbtreeIterator *iterator);
bool fbtreeNext(fbtreeIterator *iterator, const_sds *pos);
bool fbtreePrev(fbtreeIterator *iterator, const_sds *pos);

void fbtreeSeekToRank(fbtreeIterator *iterator, unsigned long rank);
const_sds fbtreeGetAtRank(fbtreeIndex *fbt, unsigned long rank);
long fbtreeGetRankOfItem(fbtreeIndex *fbt, const_sds item);
/* Like fbtreeGetRankOfItem, but matches the leaf slot by byte-equality
 * (sdscmp == 0). Returns the 0-based rank, or -1 if absent. */
long fbtreeGetRankOfValue(fbtreeIndex *fbt, const_sds key);

/* Total heap bytes owned by the tree (fbt header + nodes + out-of-line long
 * prefixes + item sds). Returns the header size for an empty tree. */
size_t fbtreeMemUsage(fbtreeIndex *fbt);

/* Advise the kernel that the tree's item-sds pages won't be needed soon
 * (copy-on-write reduction after fork). Frees nothing. */
void fbtreeDismiss(fbtreeIndex *fbt);

/* Active-defrag relocation: move every node, long prefix, item sds, and the
 * header to fresh allocations via the provided callbacks, fixing all internal
 * pointers. 'defrag_alloc' relocates a plain allocation (returns NULL if not
 * moved); 'defrag_sds' relocates an sds. Returns the (possibly relocated)
 * index pointer. */
fbtreeIndex *fbtreeDefrag(fbtreeIndex *fbt, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds));

/* Incremental, latency-bounded active-defrag for large trees.
 *
 * Relocating a large tree in a single call (fbtreeDefrag) blocks the event
 * loop long enough to exceed the active-defrag latency budget, so the work is
 * split across many small steps driven by the defrag scheduler.
 *
 * Usage:
 *   ctx = fbtreeDefragStart(fbt, &fbt, alloc, sds_alloc);  // relocates header
 *   while (fbtreeDefragStep(fbt, ctx, &scanned)) { yield as needed }
 *   fbtreeDefragEnd(ctx);
 *
 * Each fbtreeDefragStep relocates at most one leaf (its item sds + struct,
 * fixing aliasing anchors and the leaf chain), then a final pass relocates the
 * (few) inner-node structs and long prefixes. Progress is tracked by a copied
 * key, not a live pointer, so the tree may be safely read/mutated by commands
 * between steps. Returns 1 while more work remains, 0 when done. */
typedef struct fbtreeDefragCtx fbtreeDefragCtx;
fbtreeDefragCtx *fbtreeDefragStart(fbtreeIndex *fbt, fbtreeIndex **fbt_out, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds));
int fbtreeDefragStep(fbtreeIndex *fbt, fbtreeDefragCtx *ctx, size_t *scanned);
void fbtreeDefragEnd(fbtreeDefragCtx *ctx);

/* Score seek - positions iterator at first element with score >= given score.
 * Always positions the iterator (even if no exact match). Use fbtreeNext to get elements.
 * If all elements have score < given score, iterator is positioned past end.
 * Returns the rank (0-indexed) of the position. If positioned past end,
 * returns the tree length (one past the last valid rank). Returns 0 for
 * an empty tree. */
long fbtreeSeekToScore(const char *score, fbtreeIterator *iterator);

/* Value seek - positions iterator at first element with value >= given value.
 * Uses full sds comparison (not just score prefix).
 * Always positions the iterator (even if no exact match). Use fbtreeNext to get elements.
 * If all elements have value < given value, iterator is positioned past end. */
void fbtreeSeekToValue(const_sds value, fbtreeIterator *iterator);

/* Optional callback invoked for each item being deleted, before sdsfree.
 * Pass NULL for callback/callback_ctx to skip. */

/* Range deletion */
unsigned long fbtreeDeleteRangeByRank(fbtreeIndex *fbt, unsigned long start_rank, unsigned long end_rank, void (*callback)(sds item, void *ctx), void *callback_ctx);
unsigned long fbtreeDeleteRangeByScore(fbtreeIndex *fbt, const char *min_score, const char *max_score, int min_ex, int max_ex, void (*callback)(sds item, void *ctx), void *callback_ctx);
unsigned long fbtreeDeleteRangeByValue(fbtreeIndex *fbt, const_sds min_val, const_sds max_val, int min_ex, int max_ex, void (*callback)(sds item, void *ctx), void *callback_ctx);

#endif /* FBTREE_ORDERED_INDEX_H */
