#ifndef ORDERED_INDEX_H
#define ORDERED_INDEX_H

#include <stdbool.h>
#include <stdint.h>
#include "sds.h"

/* Opaque types for ordered index, positions, and iterators */
typedef struct OrderedIndex OrderedIndex;
typedef struct OrderedIndexItem OrderedIndexItem;
typedef uint64_t OrderedIndexIterator[3];

/* Operations interface for ordered index implementations */
typedef struct OrderedIndexOps {
    /* Lifecycle */
    OrderedIndex *(*create)(void);
    void (*free)(OrderedIndex *idx);

    /* Modification */
    OrderedIndexItem *(*insert)(OrderedIndex *idx, double score, const_sds ele);
    void (*deleteItem)(OrderedIndex *idx, OrderedIndexItem *pos);
    OrderedIndexItem *(*update_score)(OrderedIndex *idx, OrderedIndexItem *pos, double newscore);
    OrderedIndexItem *(*pop_first)(OrderedIndex *idx);
    OrderedIndexItem *(*pop_last)(OrderedIndex *idx);
    void (*free_item)(OrderedIndexItem *item);
    unsigned long (*delete_range_by_score)(OrderedIndex *idx, double min, double max, int min_ex, int max_ex);
    unsigned long (*delete_range_by_rank)(OrderedIndex *idx, unsigned long start, unsigned long end); /* 0-based, inclusive */
    /* Delete the item with exactly this (score, ele); returns 1 if found+deleted, else 0. */
    int (*delete_by_value)(OrderedIndex *idx, double score, const_sds ele);

    /* Query */
    unsigned long (*length)(OrderedIndex *idx);
    OrderedIndexItem *(*get_by_rank)(OrderedIndex *idx, unsigned long rank); /* 0-based */
    long (*get_rank)(OrderedIndex *idx, const OrderedIndexItem *pos);        /* returns 0-based, -1 if not found */
    /* 0-based rank of (score, ele), or -1 if absent. */
    long (*get_rank_by_value)(OrderedIndex *idx, double score, const_sds ele);
    void (*get_element_raw)(const OrderedIndexItem *pos, const char **ptr, size_t *len);
    double (*get_score)(const OrderedIndexItem *pos);

    /* Iterator */
    void (*init_iterator)(OrderedIndexIterator *iter, OrderedIndex *idx);
    void (*reset_iterator)(OrderedIndexIterator *iter);
    bool (*next)(OrderedIndexIterator *iter, OrderedIndexItem **pos);
    bool (*prev)(OrderedIndexIterator *iter, OrderedIndexItem **pos);
    void (*seek_to_rank)(OrderedIndexIterator *iter, unsigned long rank); /* 0-based */
    void (*seek_to_score_range)(OrderedIndexIterator *iter, double min, double max, int min_ex, int max_ex, long offset);

    /* Memory */
    /* Total heap bytes owned by the ordered index (NOT including any external hashtable). */
    size_t (*mem_usage)(OrderedIndex *idx);
    /* Advise the kernel we won't need the backing pages soon (CoW reduction on fork). */
    void (*dismiss)(OrderedIndex *idx);

    /* Active-defrag relocation: relocate all internal allocations (nodes, item
     * sds, and the index header) using the provided callbacks, fixing internal
     * pointers. 'defrag_alloc' relocates a plain allocation (NULL if unmoved);
     * 'defrag_sds' relocates an sds. Returns the (possibly relocated) index. */
    OrderedIndex *(*defrag)(OrderedIndex *idx, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds));

    /* Incremental, latency-bounded active-defrag for large indexes.
     * 'defrag_start' relocates the index header, returns an opaque context
     * (or NULL if incremental defrag is not supported), and reports the
     * relocated index via *idx_out. 'defrag_step' performs one bounded chunk,
     * returning 1 while more work remains and 0 when done, and adds the number
     * of elements processed to *scanned. 'defrag_end' frees the context. */
    void *(*defrag_start)(OrderedIndex *idx, OrderedIndex **idx_out, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds));
    int (*defrag_step)(OrderedIndex *idx, void *ctx, size_t *scanned);
    void (*defrag_end)(void *ctx);
} OrderedIndexOps;

/* Inline wrappers for performance (compiler can inline these) */
static inline OrderedIndex *orderedIndexCreate(const OrderedIndexOps *ops) {
    return ops->create();
}

static inline void orderedIndexFree(const OrderedIndexOps *ops, OrderedIndex *idx) {
    ops->free(idx);
}

static inline OrderedIndexItem *orderedIndexInsert(const OrderedIndexOps *ops, OrderedIndex *idx, double score, const_sds ele) {
    return ops->insert(idx, score, ele);
}

static inline void orderedIndexDelete(const OrderedIndexOps *ops, OrderedIndex *idx, OrderedIndexItem *pos) {
    ops->deleteItem(idx, pos);
}

static inline OrderedIndexItem *orderedIndexUpdateScore(const OrderedIndexOps *ops, OrderedIndex *idx, OrderedIndexItem *pos, double newscore) {
    return ops->update_score(idx, pos, newscore);
}

static inline OrderedIndexItem *orderedIndexPopFirst(const OrderedIndexOps *ops, OrderedIndex *idx) {
    return ops->pop_first(idx);
}

static inline OrderedIndexItem *orderedIndexPopLast(const OrderedIndexOps *ops, OrderedIndex *idx) {
    return ops->pop_last(idx);
}

static inline void orderedIndexFreeItem(const OrderedIndexOps *ops, OrderedIndexItem *item) {
    ops->free_item(item);
}

static inline unsigned long orderedIndexDeleteRangeByScore(const OrderedIndexOps *ops, OrderedIndex *idx, double min, double max, int min_ex, int max_ex) {
    return ops->delete_range_by_score(idx, min, max, min_ex, max_ex);
}

static inline unsigned long orderedIndexDeleteRangeByRank(const OrderedIndexOps *ops, OrderedIndex *idx, unsigned long start, unsigned long end) {
    return ops->delete_range_by_rank(idx, start, end);
}

static inline int orderedIndexDeleteByValue(const OrderedIndexOps *ops, OrderedIndex *idx, double score, const_sds ele) {
    return ops->delete_by_value(idx, score, ele);
}

static inline unsigned long orderedIndexLength(const OrderedIndexOps *ops, OrderedIndex *idx) {
    return ops->length(idx);
}

static inline OrderedIndexItem *orderedIndexGetByRank(const OrderedIndexOps *ops, OrderedIndex *idx, unsigned long rank) {
    return ops->get_by_rank(idx, rank);
}

static inline long orderedIndexGetRank(const OrderedIndexOps *ops, OrderedIndex *idx, const OrderedIndexItem *pos) {
    return ops->get_rank(idx, pos);
}

static inline long orderedIndexGetRankByValue(const OrderedIndexOps *ops, OrderedIndex *idx, double score, const_sds ele) {
    return ops->get_rank_by_value(idx, score, ele);
}

static inline void orderedIndexGetElementRaw(const OrderedIndexOps *ops, const OrderedIndexItem *pos, const char **ptr, size_t *len) {
    ops->get_element_raw(pos, ptr, len);
}

static inline double orderedIndexGetScore(const OrderedIndexOps *ops, const OrderedIndexItem *pos) {
    return ops->get_score(pos);
}

static inline void orderedIndexInitIterator(const OrderedIndexOps *ops, OrderedIndexIterator *iter, OrderedIndex *idx) {
    ops->init_iterator(iter, idx);
}

static inline void orderedIndexResetIterator(const OrderedIndexOps *ops, OrderedIndexIterator *iter) {
    ops->reset_iterator(iter);
}

static inline bool orderedIndexNext(const OrderedIndexOps *ops, OrderedIndexIterator *iter, OrderedIndexItem **pos) {
    return ops->next(iter, pos);
}

static inline bool orderedIndexPrev(const OrderedIndexOps *ops, OrderedIndexIterator *iter, OrderedIndexItem **pos) {
    return ops->prev(iter, pos);
}

static inline void orderedIndexSeekToRank(const OrderedIndexOps *ops, OrderedIndexIterator *iter, unsigned long rank) {
    ops->seek_to_rank(iter, rank);
}

static inline void orderedIndexSeekToScoreRange(const OrderedIndexOps *ops, OrderedIndexIterator *iter, double min, double max, int min_ex, int max_ex, long offset) {
    ops->seek_to_score_range(iter, min, max, min_ex, max_ex, offset);
}

static inline size_t orderedIndexMemUsage(const OrderedIndexOps *ops, OrderedIndex *idx) {
    return ops->mem_usage(idx);
}

static inline void orderedIndexDismiss(const OrderedIndexOps *ops, OrderedIndex *idx) {
    ops->dismiss(idx);
}

static inline OrderedIndex *orderedIndexDefrag(const OrderedIndexOps *ops, OrderedIndex *idx, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds)) {
    return ops->defrag(idx, defrag_alloc, defrag_sds);
}

static inline void *orderedIndexDefragStart(const OrderedIndexOps *ops, OrderedIndex *idx, OrderedIndex **idx_out, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds)) {
    return ops->defrag_start(idx, idx_out, defrag_alloc, defrag_sds);
}

static inline int orderedIndexDefragStep(const OrderedIndexOps *ops, OrderedIndex *idx, void *ctx, size_t *scanned) {
    return ops->defrag_step(idx, ctx, scanned);
}

static inline void orderedIndexDefragEnd(const OrderedIndexOps *ops, void *ctx) {
    ops->defrag_end(ctx);
}

/* Available implementations */
extern const OrderedIndexOps skiplistOrderedIndexOps;
extern const OrderedIndexOps fbtreeOrderedIndexOps;

#endif /* ORDERED_INDEX_H */
