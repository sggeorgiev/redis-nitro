/* Zset adapter for fbtree - implements OrderedIndexOps using fbtree as backend.
 * Handles score normalization and [score][element] packing. */

#include <assert.h>
#include <string.h>
#include "endianconv.h"
#include "fbtree_ordered_index.h"
#include "ordered_index.h"
#include "sds.h"

/* For zset entries, the string contains: [8-byte normalized score][element bytes]
 * The normalized score is stored in big-endian for lexicographic ordering.
 * Score packing/unpacking is handled by zset_fbtree_adapter.c */

/* Verify iterator types are compatible */
static_assert(sizeof(OrderedIndexIterator) >= sizeof(fbtreeIterator), "Iterator size check");

#ifndef UNUSED
#define UNUSED(V) ((void)V)
#endif

#define SCORE_SIZE sizeof(double) /* Normalized score prefix size */
static_assert(SCORE_SIZE == 8, "Score size must be 8 bytes for fbtreeLookupByScore");

/* ========== Score Normalization ==========
 * Converts IEEE 754 double to a sortable 8-byte big-endian representation.
 * The transformation ensures lexicographic byte comparison matches numeric order.
 *
 * IEEE 754 double bit layout: [sign:1][exponent:11][mantissa:52]
 * - Positive numbers: flip sign bit (0->1) to sort after negatives
 * - Negative numbers: flip all bits to reverse their order
 */

static inline uint64_t scoreToSortable(double score) {
    uint64_t bits;
    memcpy(&bits, &score, sizeof(bits));
    /* Flip sign bit for positives, all bits for negatives */
    if (bits & (1ULL << 63)) {
        bits = ~bits;
    } else {
        bits ^= (1ULL << 63);
    }
    /* Convert to big-endian for lexicographic comparison */
    return htonu64(bits);
}

static inline double sortableToScore(uint64_t be) {
    /* Convert from big-endian */
    uint64_t bits = ntohu64(be);
    /* Reverse the transformation */
    if (bits & (1ULL << 63)) {
        bits ^= (1ULL << 63);
    } else {
        bits = ~bits;
    }
    double score;
    memcpy(&score, &bits, sizeof(score));
    return score;
}

/* Pack score and element into a single sds: [8-byte sortable score][element] */
static sds packScoreElement(double score, const_sds ele) {
    uint64_t sortable = scoreToSortable(score);
    size_t ele_len = sdslen(ele);
    sds packed = sdsnewlen(NULL, SCORE_SIZE + ele_len);
    memcpy(packed, &sortable, SCORE_SIZE);
    memcpy(packed + SCORE_SIZE, ele, ele_len);
    return packed;
}

/* Extract element from packed string (returns pointer into packed, not a copy) */
static inline const char *unpackElement(const_sds packed, size_t *len) {
    *len = sdslen(packed) - SCORE_SIZE;
    return packed + SCORE_SIZE;
}

/* Extract score from packed string */
static inline double unpackScore(const_sds packed) {
    uint64_t sortable;
    memcpy(&sortable, packed, SCORE_SIZE);
    return sortableToScore(sortable);
}

/* ========== OrderedIndexOps Implementation ========== */

static OrderedIndex *zsetFbtreeCreate(void) {
    return (OrderedIndex *)fbtreeCreate();
}

static void zsetFbtreeFree(OrderedIndex *idx) {
    fbtreeFree((fbtreeIndex *)idx);
}

static OrderedIndexItem *zsetFbtreeInsert(OrderedIndex *idx, double score, const_sds ele) {
    sds packed = packScoreElement(score, ele);
    return (OrderedIndexItem *)fbtreeInsert((fbtreeIndex *)idx, packed);
}

static void zsetFbtreeDelete(OrderedIndex *idx, OrderedIndexItem *pos) {
    fbtreeDelete((fbtreeIndex *)idx, (const_sds)pos);
}

static OrderedIndexItem *zsetFbtreeUpdateScore(OrderedIndex *idx, OrderedIndexItem *pos, double newscore) {
    /* Extract element, delete old, insert with new score */
    const_sds packed = (const_sds)pos;
    size_t ele_len;
    const char *ele = unpackElement(packed, &ele_len);

    /* Create temp sds for element (insert will take ownership of new packed string) */
    sds ele_copy = sdsnewlen(ele, ele_len);
    sds new_packed = packScoreElement(newscore, ele_copy);
    sdsfree(ele_copy);

    // TODO: optimize? what if score is the same? what if order doesn't change? What if it moves within the same leaf node?
    fbtreeDelete((fbtreeIndex *)idx, packed);
    return (OrderedIndexItem *)fbtreeInsert((fbtreeIndex *)idx, new_packed);
}

static OrderedIndexItem *zsetFbtreePopFirst(OrderedIndex *idx) {
    return (OrderedIndexItem *)fbtreePopMin((fbtreeIndex *)idx);
}

static OrderedIndexItem *zsetFbtreePopLast(OrderedIndex *idx) {
    return (OrderedIndexItem *)fbtreePopMax((fbtreeIndex *)idx);
}

static void zsetFbtreeFreeItem(OrderedIndexItem *item) {
    sdsfree((sds)item);
}

static unsigned long zsetFbtreeDeleteRangeByScore(OrderedIndex *idx, double min, double max, int min_ex, int max_ex) {
    uint64_t min_sortable = scoreToSortable(min);
    uint64_t max_sortable = scoreToSortable(max);
    return fbtreeDeleteRangeByScore((fbtreeIndex *)idx, (const char *)&min_sortable, (const char *)&max_sortable, min_ex, max_ex, NULL, NULL);
}

static unsigned long zsetFbtreeDeleteRangeByRank(OrderedIndex *idx, unsigned long start, unsigned long end) {
    return fbtreeDeleteRangeByRank((fbtreeIndex *)idx, start, end, NULL, NULL);
}

static int zsetFbtreeDeleteByValue(OrderedIndex *idx, double score, const_sds ele) {
    sds key = packScoreElement(score, ele);
    int r = fbtreeDeleteByValue((fbtreeIndex *)idx, key) ? 1 : 0;
    sdsfree(key);
    return r;
}

static unsigned long zsetFbtreeLength(OrderedIndex *idx) {
    return fbtreeLength((fbtreeIndex *)idx);
}

static OrderedIndexItem *zsetFbtreeGetByRank(OrderedIndex *idx, unsigned long rank) {
    return (OrderedIndexItem *)fbtreeGetAtRank((fbtreeIndex *)idx, rank);
}

static long zsetFbtreeGetRank(OrderedIndex *idx, const OrderedIndexItem *pos) {
    return fbtreeGetRankOfItem((fbtreeIndex *)idx, (const_sds)pos);
}

static long zsetFbtreeGetRankByValue(OrderedIndex *idx, double score, const_sds ele) {
    sds key = packScoreElement(score, ele);
    long rank = fbtreeGetRankOfValue((fbtreeIndex *)idx, key);
    sdsfree(key);
    return rank;
}

static void zsetFbtreeGetElementRaw(const OrderedIndexItem *pos, const char **ptr, size_t *len) {
    *ptr = unpackElement((const_sds)pos, len);
}

static double zsetFbtreeGetScore(const OrderedIndexItem *pos) {
    return unpackScore((const_sds)pos);
}

static void zsetFbtreeInitIterator(OrderedIndexIterator *iter, OrderedIndex *idx) {
    fbtreeInitIterator((fbtreeIterator *)iter, (fbtreeIndex *)idx);
}

static void zsetFbtreeResetIterator(OrderedIndexIterator *iter) {
    fbtreeResetIterator((fbtreeIterator *)iter);
}

static bool zsetFbtreeNext(OrderedIndexIterator *iter, OrderedIndexItem **pos) {
    return fbtreeNext((fbtreeIterator *)iter, (const_sds *)pos);
}

static bool zsetFbtreePrev(OrderedIndexIterator *iter, OrderedIndexItem **pos) {
    return fbtreePrev((fbtreeIterator *)iter, (const_sds *)pos);
}

static void zsetFbtreeSeekToRank(OrderedIndexIterator *iter, unsigned long rank) {
    fbtreeSeekToRank((fbtreeIterator *)iter, rank);
}

static void zsetFbtreeSeekToScoreRange(OrderedIndexIterator *iter, double min, double max, int min_ex, int max_ex, long offset) {
    fbtreeIterator *fbt_iter = (fbtreeIterator *)iter;
    fbtreeIndex *fbt = fbtreeIteratorGetIndex(fbt_iter);
    if (!fbt) return;

    /* Empty range check */
    if (min > max || (min == max && (min_ex || max_ex))) {
        fbtreeResetIterator(fbt_iter);
        return;
    }

    uint64_t sortable;
    if (offset >= 0) {
        /* Positive offset: seek from the start of the range */
        sortable = scoreToSortable(min);
        if (min_ex) {
            /* Exclusive min: increment to the next representable score so
             * fbtreeSeekToScore lands past all elements with score == min. */
            uint64_t native = ntohu64(sortable);
            native++;
            sortable = htonu64(native);
        }
    } else {
        /* Negative offset: seek from the end of the range.
         * Find the rank one past the last in-range element, then
         * target = one_past + offset (offset is negative, e.g. -1 → last). */
        sortable = scoreToSortable(max);
        if (!max_ex) {
            /* Inclusive max: increment to next score so seek lands past all
             * elements with score == max. */
            uint64_t native = ntohu64(sortable);
            native++;
            sortable = htonu64(native);
        }
    }
    long target = offset + fbtreeSeekToScore((const char *)&sortable, fbt_iter);

    /* Negative target means offset went past the start of the range.
     * We need to avoid underflow with fbtreeSeekToRank. */
    if (target < 0) {
        fbtreeResetIterator(fbt_iter);
    } else {
        fbtreeSeekToRank(fbt_iter, (unsigned long)target);
    }
}

static size_t zsetFbtreeMemUsage(OrderedIndex *idx) {
    return fbtreeMemUsage((fbtreeIndex *)idx);
}

static void zsetFbtreeDismiss(OrderedIndex *idx) {
    fbtreeDismiss((fbtreeIndex *)idx);
}

static OrderedIndex *zsetFbtreeDefrag(OrderedIndex *idx, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds)) {
    return (OrderedIndex *)fbtreeDefrag((fbtreeIndex *)idx, defrag_alloc, defrag_sds);
}

static void *zsetFbtreeDefragStart(OrderedIndex *idx, OrderedIndex **idx_out, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds)) {
    fbtreeIndex *newfbt = (fbtreeIndex *)idx;
    fbtreeDefragCtx *ctx = fbtreeDefragStart((fbtreeIndex *)idx, &newfbt, defrag_alloc, defrag_sds);
    *idx_out = (OrderedIndex *)newfbt;
    return ctx;
}

static int zsetFbtreeDefragStep(OrderedIndex *idx, void *ctx, size_t *scanned) {
    return fbtreeDefragStep((fbtreeIndex *)idx, (fbtreeDefragCtx *)ctx, scanned);
}

static void zsetFbtreeDefragEnd(void *ctx) {
    fbtreeDefragEnd((fbtreeDefragCtx *)ctx);
}

const OrderedIndexOps fbtreeOrderedIndexOps = {
    /* Lifecycle */
    .create = zsetFbtreeCreate,
    .free = zsetFbtreeFree,
    /* Modification */
    .insert = zsetFbtreeInsert,
    .deleteItem = zsetFbtreeDelete,
    .update_score = zsetFbtreeUpdateScore,
    .pop_first = zsetFbtreePopFirst,
    .pop_last = zsetFbtreePopLast,
    .free_item = zsetFbtreeFreeItem,
    .delete_range_by_score = zsetFbtreeDeleteRangeByScore,
    .delete_range_by_rank = zsetFbtreeDeleteRangeByRank,
    .delete_by_value = zsetFbtreeDeleteByValue,
    /* Query */
    .length = zsetFbtreeLength,
    .get_by_rank = zsetFbtreeGetByRank,
    .get_rank = zsetFbtreeGetRank,
    .get_rank_by_value = zsetFbtreeGetRankByValue,
    .get_element_raw = zsetFbtreeGetElementRaw,
    .get_score = zsetFbtreeGetScore,
    /* Iterator */
    .init_iterator = zsetFbtreeInitIterator,
    .reset_iterator = zsetFbtreeResetIterator,
    .next = zsetFbtreeNext,
    .prev = zsetFbtreePrev,
    .seek_to_rank = zsetFbtreeSeekToRank,
    .seek_to_score_range = zsetFbtreeSeekToScoreRange,
    /* Memory */
    .mem_usage = zsetFbtreeMemUsage,
    .dismiss = zsetFbtreeDismiss,
    .defrag = zsetFbtreeDefrag,
    .defrag_start = zsetFbtreeDefragStart,
    .defrag_step = zsetFbtreeDefragStep,
    .defrag_end = zsetFbtreeDefragEnd,
};

/* ========== Test Wrappers ========== */

uint64_t scoreToSortable_test(double score) {
    return scoreToSortable(score);
}

double sortableToScore_test(uint64_t be) {
    return sortableToScore(be);
}

sds packScoreElement_test(double score, const_sds ele) {
    return packScoreElement(score, ele);
}

const char *unpackElement_test(const_sds packed, size_t *len) {
    return unpackElement(packed, len);
}

double unpackScore_test(const_sds packed) {
    return unpackScore(packed);
}
