/* Feature B-Tree: a cache-optimized B+tree for sorted binary strings.
 * This is a general-purpose data structure - zset-specific logic is in zset_fbtree_adapter.c */

#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include "config.h"
#include "fbtree_ordered_index.h"
#include "fbtree_ordered_index_internal.h"
/* <assert.h> (included above for static_assert) defines assert(); redisassert.h
 * replaces it with the server's stack-trace-logging variant. Undef first to
 * avoid a macro-redefinition warning. */
#undef assert
#include "redisassert.h"
#include "zmalloc.h"
#include "sds.h"

#if HAVE_X86_SIMD
#include <immintrin.h>
#elif HAVE_ARM_NEON
#include <arm_neon.h>
#endif

typedef enum {
    ITER_AT_POSITION,  /* Positioned at a valid element */
    ITER_BEFORE_START, /* Positioned before first element (fbtreePrev fails, fbtreeNext works) */
    ITER_PAST_END,     /* Positioned past last element (fbtreeNext fails, fbtreePrev works) */
} IterState;

typedef struct {
    fbtreeIndex *fbt;
    leafNode *current_leaf;
    uint8_t current_index;
    uint8_t leaf_count;
    IterState state;
} iter;

static_assert(sizeof(fbtreeIterator) >= sizeof(iter), "Opaque iterator size check");

typedef struct {
    sds updated_anchor;  /* Pointer to updated anchor string if it's changed */
    node *new_node;      /* Pointer to new child node to insert (node split happened) */
    sds new_node_anchor; /* Pointer to new node's anchor string (node split happened) */
    sds inserted_item;   /* Pointer to the newly inserted item in the leaf */
} insertResult;

/* Hint for optimized tree traversal - allows skipping inner node searches */
typedef enum {
    HINT_NONE,     /* No hint - use normal search */
    HINT_LEFTMOST, /* Traverse to leftmost child at each level */
    HINT_RIGHTMOST /* Traverse to rightmost child at each level */
} TraversalHint;

typedef struct {
    sds updated_anchor;    /* Pointer to updated anchor string if it's changed */
    bool delete_executed;  /* True if key was found and deleted, False if not found no-op */
    bool node_underflowed; /* True if child dropped below MIN_FILL after delete */
    bool node_shrank;      /* True if a merge reduced this node's child count */
} deleteResult;

/* Conversion from user-facing opaque iterator type to internal struct */
static inline iter *iteratorFromOpaque(fbtreeIterator *iterator) {
    return (iter *)(void *)iterator;
}


static inline bool innerNodeHasLongPrefix(innerNode *inner) {
    return inner->prefix_len > EMBED_PREFIX_LEN;
}

static inline const char *innerNodeGetPrefix(innerNode *inner) {
    if (innerNodeHasLongPrefix(inner)) {
        char **ptr = (char **)&inner->embedded_prefix[LONG_PREFIX_PTR_OFFSET];
        return *ptr;
    }
    return inner->embedded_prefix;
}

static void innerNodeFreePrefix(innerNode *inner) {
    if (innerNodeHasLongPrefix(inner)) {
        char **ptr = (char **)&inner->embedded_prefix[LONG_PREFIX_PTR_OFFSET];
        zfree(*ptr);
    }
}

static void innerNodeSetPrefix(innerNode *inner, const char *data, size_t len) {
    innerNodeFreePrefix(inner);
    inner->prefix_len = len;
    if (len > EMBED_PREFIX_LEN) {
        char **ptr = (char **)&inner->embedded_prefix[LONG_PREFIX_PTR_OFFSET];
        *ptr = zmalloc(len);
        memcpy(*ptr, data, len);
    } else if (len > 0) {
        memcpy(inner->embedded_prefix, data, len);
    }
}

static innerNode *innerNodeCreate(void) {
    /* zcalloc zeroes all fields — is_leaf defaults to false (0). */
    return zcalloc(sizeof(innerNode));
}

static leafNode *leafNodeCreate(void) {
    leafNode *node = zcalloc(sizeof(*node));
    node->header.is_leaf = true;
    return node;
}

static leafNode *leafNodeCreateWithItem(sds item) {
    leafNode *leaf = leafNodeCreate();
    leaf->values[0] = item;
    leaf->header.num_items = 1;
    return leaf;
}

fbtreeIndex *fbtreeCreate(void) {
    fbtreeIndex *fbt = zmalloc(sizeof(*fbt));
    fbt->root = NULL;
    fbt->leftmost_leaf = NULL;
    fbt->rightmost_leaf = NULL;
    return fbt;
}

/* Callback invoked for each item being deleted, before sdsfree.
 * Allows callers to perform side effects (e.g., hashtable removal). */
typedef void (*fbtreeItemCallback)(sds item, void *ctx);

static void freeNodeRecursive(node *n, fbtreeItemCallback callback, void *ctx) {
    if (!n) return;

    if (n->is_leaf) {
        leafNode *leaf = (leafNode *)n;
        for (int i = 0; i < leaf->header.num_items; i++) {
            if (callback) callback(leaf->values[i], ctx);
            sdsfree(leaf->values[i]);
        }
        zfree(leaf);
    } else {
        innerNode *inner = (innerNode *)n;
        innerNodeFreePrefix(inner);
        for (int i = 0; i < inner->header.num_items; i++) {
            if (inner->children[i] != NULL) {
                freeNodeRecursive(inner->children[i], callback, ctx);
            }
        }
        zfree(inner);
    }
}

/* Free all nodes and reset to empty. If callback is non-NULL, it is invoked
 * for each item before freeing. */
static void fbtreeDeleteAll(fbtreeIndex *fbt, fbtreeItemCallback callback, void *callback_ctx) {
    freeNodeRecursive(fbt->root, callback, callback_ctx);
    fbt->root = NULL;
    fbt->leftmost_leaf = NULL;
    fbt->rightmost_leaf = NULL;
}

void fbtreeEmpty(fbtreeIndex *fbt) {
    fbtreeDeleteAll(fbt, NULL, NULL);
}

void fbtreeFree(fbtreeIndex *fbt) {
    fbtreeEmpty(fbt);
    zfree(fbt);
}

static void recomputeFeatures(innerNode *inner) {
    for (int i = 0; i < inner->header.num_items; i++) {
        assert(sdslen(inner->anchors[i]) >= inner->prefix_len);
        for (int j = 0; j < FEATURE_SIZE; j++)
            inner->features[j][i] = getFeatureByte(inner->anchors[i], inner->prefix_len, j);
    }
}

/* Copy count children (anchors, child pointers, child_sizes, child_num_items,
 * features) from src starting at src_idx to dst starting at dst_idx.
 * Regions must not overlap — use innerNodeMoveChildren for overlapping shifts. */
static inline void innerNodeCopyChildren(innerNode *dst, int dst_idx, innerNode *src, int src_idx, int count) {
    for (int j = 0; j < FEATURE_SIZE; j++)
        memcpy(&dst->features[j][dst_idx], &src->features[j][src_idx], count);
    memcpy(&dst->anchors[dst_idx], &src->anchors[src_idx], count * sizeof(dst->anchors[0]));
    memcpy(&dst->children[dst_idx], &src->children[src_idx], count * sizeof(dst->children[0]));
    memcpy(&dst->child_sizes[dst_idx], &src->child_sizes[src_idx], count * sizeof(dst->child_sizes[0]));
    memcpy(&dst->child_num_items[dst_idx], &src->child_num_items[src_idx], count * sizeof(dst->child_num_items[0]));
}

/* Move count children within the same node from src_idx to dst_idx.
 * Handles overlapping regions (for insert/remove shifts). */
static inline void innerNodeMoveChildren(innerNode *node, int dst_idx, int src_idx, int count) {
    for (int j = 0; j < FEATURE_SIZE; j++)
        memmove(&node->features[j][dst_idx], &node->features[j][src_idx], count);
    memmove(&node->anchors[dst_idx], &node->anchors[src_idx], count * sizeof(node->anchors[0]));
    memmove(&node->children[dst_idx], &node->children[src_idx], count * sizeof(node->children[0]));
    memmove(&node->child_sizes[dst_idx], &node->child_sizes[src_idx], count * sizeof(node->child_sizes[0]));
    memmove(&node->child_num_items[dst_idx], &node->child_num_items[src_idx], count * sizeof(node->child_num_items[0]));
}

static bool updateCommonPrefix(innerNode *inner) {
    if (inner->header.num_items < 2) return false;

    /* Compute the common prefix of this node's first and last anchors.
     * Anchors are high keys of children, so this is the common prefix among
     * anchor values — NOT the common prefix of all keys in the subtree.
     *
     * This means prefix_len can exceed a child's prefix_len. The leftmost
     * child's key range extends below its high key (anchor), so its own
     * anchors may diverge earlier than the parent's anchors do. Example:
     * parent anchors "60_100" and "60_200" share prefix "60_" (len=3), but
     * child[0] contains keys "59_999" through "60_100", so child[0]'s
     * anchors only share prefix "" (len=0).
     *
     * This is handled correctly by findChildIndex: when a lookup key is
     * less than the prefix, it returns child index 0 (leftmost).  */
    const_sds first_anchor = inner->anchors[0];
    const_sds last_anchor = inner->anchors[inner->header.num_items - 1];
    size_t first_len = sdslen(first_anchor);
    size_t last_len = sdslen(last_anchor);
    size_t max_len = first_len < last_len ? first_len : last_len;
    size_t len = 0;
    while (len < max_len && first_anchor[len] == last_anchor[len]) len++;

    bool prefix_changed = (len != inner->prefix_len) ||
                          (len > 0 && memcmp(innerNodeGetPrefix(inner), first_anchor, len) != 0);
    if (prefix_changed) {
        innerNodeSetPrefix(inner, first_anchor, len);
        recomputeFeatures(inner);
        return true;
    }
    return false;
}

static size_t getSubtreeSize(node *n) {
    if (n->is_leaf) {
        return n->num_items;
    } else {
        innerNode *inner = (innerNode *)n;
        size_t total = 0;
        for (int i = 0; i < inner->header.num_items; i++) {
            total += inner->child_sizes[i];
        }
        return total;
    }
}

/* Insert a child into an inner node in sorted order. Returns true if parent's anchor/feature needs to be updated */
static bool innerNodeInsert(innerNode *parent, const node *child, sds child_anchor, size_t insert_index) {
    assert(parent->header.num_items < NODE_SIZE);

    /* shift higher elements to make space */
    size_t num_to_move = parent->header.num_items - insert_index;
    if (num_to_move > 0) {
        innerNodeMoveChildren(parent, insert_index + 1, insert_index, num_to_move);
    }

    /* insert child - features stored pre-biased for SIMD comparison */
    for (int j = 0; j < FEATURE_SIZE; j++)
        parent->features[j][insert_index] = getFeatureByte(child_anchor, parent->prefix_len, j);
    parent->anchors[insert_index] = child_anchor;
    parent->children[insert_index] = (node *)child;
    parent->child_sizes[insert_index] = getSubtreeSize((node *)child);
    parent->child_num_items[insert_index] = ((node *)child)->num_items;
    parent->header.num_items++;

    /* update prefix - might need to initialize, or common length could become shorter */
    if (insert_index == 0 || insert_index + 1 == parent->header.num_items) {
        updateCommonPrefix(parent);
    }

    bool anchor_changed = (num_to_move == 0);
    return anchor_changed;
}

/* Remove child at given index from inner node. Caller must free the child. */
static void innerNodeRemoveChild(innerNode *parent, int index) {
    assert(index < parent->header.num_items);
    int num_to_move = parent->header.num_items - index - 1;
    if (num_to_move > 0) {
        innerNodeMoveChildren(parent, index, index + 1, num_to_move);
    }
    parent->header.num_items--;
    if (index == 0 || index == parent->header.num_items) {
        updateCommonPrefix(parent);
    }
}

/* Remove children at indices [start_idx, end_idx] inclusive from inner node.
 * Caller must free the removed children. Does NOT update prefix/features
 * since the caller may be doing further modifications. */
static void innerNodeRemoveChildrenRange(innerNode *parent, int start_idx, int end_idx) {
    assert(start_idx >= 0 && end_idx < parent->header.num_items && start_idx <= end_idx);
    int remove_count = end_idx - start_idx + 1;
    int num_to_move = parent->header.num_items - end_idx - 1;
    if (num_to_move > 0) {
        innerNodeMoveChildren(parent, start_idx, end_idx + 1, num_to_move);
    }
    parent->header.num_items -= remove_count;
}

/* Refresh a child's cached metadata in the parent: child_sizes, anchor,
 * features, child_num_items. Call after the child's contents changed (e.g.,
 * after truncation or merge). Does NOT update common prefix. */
static void innerNodeRefreshChildMeta(innerNode *parent, int index) {
    parent->child_sizes[index] = getSubtreeSize(parent->children[index]);
    parent->child_num_items[index] = parent->children[index]->num_items;
    sds anchor = nodeHighKey(parent->children[index]);
    parent->anchors[index] = anchor;
    for (int j = 0; j < FEATURE_SIZE; j++)
        parent->features[j][index] = getFeatureByte(anchor, parent->prefix_len, j);
}

/* TODO: Support asymmetric inner node splits for append/prepend (hint-gated),
 * matching what leafNodeSplit does. Sequential loads would achieve ~100% inner
 * node load factor instead of ~50%. Requires passing TraversalHint through. */
static innerNode *innerNodeSplit(innerNode *left_node) {
    assert(left_node->header.num_items == NODE_SIZE);
    innerNode *right_node = innerNodeCreate();

    /* move higher half of elements to right node */
    const size_t num_left_keys = NODE_SIZE / 2;
    const size_t num_right_keys = NODE_SIZE - num_left_keys;

    innerNodeCopyChildren(right_node, 0, left_node, num_left_keys, num_right_keys);

    right_node->header.num_items = num_right_keys;
    left_node->header.num_items = num_left_keys;

    /* each covers a smaller range of the dataset, so prefix could be longer now */
    /* Copy prefix from left to right before updateCommonPrefix potentially changes it */
    innerNodeSetPrefix(right_node, innerNodeGetPrefix(left_node), left_node->prefix_len); // TODO: only copy size of prefix? Optimize to avoid copy maybe?
    updateCommonPrefix(right_node);
    updateCommonPrefix(left_node);
    return right_node;
}

static int leafNodeBinarySearch(leafNode *leaf, const_sds string) {
    int left = 0, right = leaf->header.num_items;
    while (left < right) {
        int mid = (left + right) / 2;
        if (sdscmp(leaf->values[mid], string) < 0) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

static insertResult leafNodeInsert(leafNode *leaf, sds string) {
    assert(leaf->header.num_items < NODE_SIZE);

    int insert_index = leafNodeBinarySearch(leaf, string);
    int count = leaf->header.num_items;

    insertResult result = {
        .inserted_item = string,
        .updated_anchor = (insert_index == count) ? string : NULL};

    /* Shift elements right to make space */
    memmove(&leaf->values[insert_index + 1], &leaf->values[insert_index],
            (count - insert_index) * sizeof(sds));

    /* Insert at position - take ownership */
    leaf->values[insert_index] = string;
    leaf->header.num_items++;

    return result;
}

/* Helper to link a new right leaf after an existing left leaf */
static void linkLeafRight(leafNode *left, leafNode *right) {
    right->prev = left;
    right->next = left->next;
    left->next = right;
    if (right->next) right->next->prev = right;
}

/* Unlink a leaf from the doubly-linked list */
static void unlinkLeaf(leafNode *leaf) {
    if (leaf->prev) leaf->prev->next = leaf->next;
    if (leaf->next) leaf->next->prev = leaf->prev;
}

static insertResult leafNodeSplit(leafNode *left_leaf, sds string, TraversalHint hint) {
    assert(left_leaf->header.num_items == NODE_SIZE);

    /* Binary search to find insertion point - reuse for pattern detection */
    int insert_index = leafNodeBinarySearch(left_leaf, string);

    if (insert_index == NODE_SIZE && hint == HINT_RIGHTMOST) {
        /* Append at tree edge: create new node with just the new item, left unchanged */
        leafNode *right_leaf = leafNodeCreateWithItem(string);
        linkLeafRight(left_leaf, right_leaf);
        return (insertResult){
            .new_node = (node *)right_leaf,
            .new_node_anchor = string,
            .inserted_item = right_leaf->values[0]};
    }

    if (insert_index == 0 && hint == HINT_LEFTMOST) {
        /* Prepend at tree edge: move all items to new right node, left gets just new item */
        leafNode *right_leaf = leafNodeCreate();
        memcpy(right_leaf->values, left_leaf->values, NODE_SIZE * sizeof(sds));
        right_leaf->header.num_items = NODE_SIZE;
        left_leaf->values[0] = string;
        left_leaf->header.num_items = 1;
        linkLeafRight(left_leaf, right_leaf);
        return (insertResult){
            .updated_anchor = string,
            .new_node = (node *)right_leaf,
            .new_node_anchor = leafNodeHighKey(right_leaf),
            .inserted_item = left_leaf->values[0]};
    }

    /* Middle insert: standard 50/50 split */
    leafNode *right_leaf = leafNodeCreate();
    size_t num_left = NODE_SIZE / 2;
    size_t num_right = NODE_SIZE - num_left;

    memcpy(right_leaf->values, &left_leaf->values[num_left], num_right * sizeof(sds));
    left_leaf->header.num_items = num_left;
    right_leaf->header.num_items = num_right;
    linkLeafRight(left_leaf, right_leaf);

    /* Insert into appropriate leaf */
    insertResult leaf_result = (insert_index <= (int)num_left)
                                   ? leafNodeInsert(left_leaf, string)
                                   : leafNodeInsert(right_leaf, string);

    return (insertResult){
        .updated_anchor = leafNodeHighKey(left_leaf),
        .new_node = (node *)right_leaf,
        .new_node_anchor = leafNodeHighKey(right_leaf),
        .inserted_item = leaf_result.inserted_item};
}

/* Scalar implementation - always available for testing */
static void featureSearchSIMD_scalar(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                     int num_keys,
                                     const unsigned char target[FEATURE_SIZE],
                                     int *out_left,
                                     int *out_right) {
    int left = 0, right = num_keys;

    for (int i = 0; i < num_keys; i++) {
        int cmp = 0;
        for (int row = 0; row < FEATURE_SIZE && cmp == 0; row++) {
            signed char target_biased = (signed char)(target[row] ^ FEATURE_BIAS);
            cmp = target_biased - (signed char)features[row][i];
        }
        if (cmp <= 0) {
            left = i;
            break;
        }
        left = i + 1;
    }

    right = left;
    for (int i = left; i < num_keys; i++) {
        int cmp = 0;
        for (int row = 0; row < FEATURE_SIZE && cmp == 0; row++) {
            signed char target_biased = (signed char)(target[row] ^ FEATURE_BIAS);
            cmp = target_biased - (signed char)features[row][i];
        }
        if (cmp < 0) break;
        right = i + 1;
    }

    *out_left = left;
    *out_right = right;
}

/* SIMD feature search: finds range [out_left, out_right) of keys matching target.
 * Features are stored pre-biased (XOR'd with 0x80), so we only bias the target.
 * Bitmasks track candidates (ge_mask: target >= key, le_mask: target <= key). */
#if HAVE_X86_SIMD

ATTRIBUTE_TARGET_AVX2
static void featureSearchSIMD_avx2(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                   int num_keys,
                                   const unsigned char target[FEATURE_SIZE],
                                   int *out_left,
                                   int *out_right) {
    uint64_t valid_mask = (num_keys >= 64) ? ~0ULL : (1ULL << num_keys) - 1;
    uint64_t ge_mask = valid_mask;
    uint64_t le_mask = valid_mask;

    for (int row = 0; row < FEATURE_SIZE && (ge_mask || le_mask); row++) {
        /* Bias target to match pre-biased features */
        __m256i target_biased = _mm256_set1_epi8((char)(target[row] ^ FEATURE_BIAS));
        uint64_t gt_this = 0, lt_this = 0;

        /* Process 64 feature bytes in 2x32-byte chunks */
        for (int chunk = 0; chunk < 2; chunk++) {
            __m256i feat = _mm256_loadu_si256((const __m256i *)&features[row][chunk * 32]);
            __m256i gt = _mm256_cmpgt_epi8(target_biased, feat);
            __m256i lt = _mm256_cmpgt_epi8(feat, target_biased);
            gt_this |= (uint64_t)(uint32_t)_mm256_movemask_epi8(gt) << (chunk * 32);
            lt_this |= (uint64_t)(uint32_t)_mm256_movemask_epi8(lt) << (chunk * 32);
        }

        /* Narrow candidate set: eliminate keys where comparison is decided */
        uint64_t undecided = ge_mask & le_mask;
        ge_mask &= ~(lt_this & undecided);
        le_mask &= ~(gt_this & undecided);
    }

    *out_left = le_mask ? __builtin_ctzll(le_mask) : num_keys;
    *out_right = (le_mask & ~ge_mask) ? __builtin_ctzll(le_mask & ~ge_mask) : num_keys;
}

ATTRIBUTE_TARGET_SSE2
static void featureSearchSIMD_sse2(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                   int num_keys,
                                   const unsigned char target[FEATURE_SIZE],
                                   int *out_left,
                                   int *out_right) {
    uint64_t valid_mask = (num_keys >= 64) ? ~0ULL : (1ULL << num_keys) - 1;
    uint64_t ge_mask = valid_mask;
    uint64_t le_mask = valid_mask;

    for (int row = 0; row < FEATURE_SIZE && (ge_mask || le_mask); row++) {
        /* Bias target to match pre-biased features */
        __m128i target_biased = _mm_set1_epi8((char)(target[row] ^ FEATURE_BIAS));
        uint64_t gt_this = 0, lt_this = 0;

        /* Process 64 feature bytes in 4x16-byte chunks */
        for (int chunk = 0; chunk < 4; chunk++) {
            __m128i feat = _mm_loadu_si128((const __m128i *)&features[row][chunk * 16]);
            __m128i gt = _mm_cmpgt_epi8(target_biased, feat);
            __m128i lt = _mm_cmplt_epi8(target_biased, feat);
            gt_this |= (uint64_t)(uint16_t)_mm_movemask_epi8(gt) << (chunk * 16);
            lt_this |= (uint64_t)(uint16_t)_mm_movemask_epi8(lt) << (chunk * 16);
        }

        uint64_t undecided = ge_mask & le_mask;
        ge_mask &= ~(lt_this & undecided);
        le_mask &= ~(gt_this & undecided);
    }

    *out_left = le_mask ? __builtin_ctzll(le_mask) : num_keys;
    *out_right = (le_mask & ~ge_mask) ? __builtin_ctzll(le_mask & ~ge_mask) : num_keys;
}

static void featureSearchSIMD(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                              int num_keys,
                              const unsigned char target[FEATURE_SIZE],
                              int *out_left,
                              int *out_right) {
    if (__builtin_cpu_supports("avx2")) {
        featureSearchSIMD_avx2(features, num_keys, target, out_left, out_right);
    } else {
        featureSearchSIMD_sse2(features, num_keys, target, out_left, out_right);
    }
}

#elif HAVE_ARM_NEON

/* Convert 16-byte NEON comparison result to 16-bit mask (one bit per byte). */
static uint16_t neon_movemask_16(uint8x16_t v) {
    /* Position bits: byte i contributes to bit i of result */
    static const uint8x16_t shift_amt = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    uint8x16_t masked = vshrq_n_u8(v, 7); /* isolate high bit: 0x01 or 0x00 */
    uint8x16_t shifted = vshlq_u8(masked, vreinterpretq_s8_u8(shift_amt));
    /* Sum each half to get one byte with 8 bits packed */
    uint8_t lo = vaddv_u8(vget_low_u8(shifted));
    uint8_t hi = vaddv_u8(vget_high_u8(shifted));
    return (uint16_t)lo | ((uint16_t)hi << 8);
}

/* Convert 4x16-byte NEON vectors to 64-bit mask (one bit per byte). */
static uint64_t neon_movemask_64(uint8x16_t v0, uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
    return (uint64_t)neon_movemask_16(v0) |
           ((uint64_t)neon_movemask_16(v1) << 16) |
           ((uint64_t)neon_movemask_16(v2) << 32) |
           ((uint64_t)neon_movemask_16(v3) << 48);
}

static void featureSearchSIMD(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                              int num_keys,
                              const unsigned char target[FEATURE_SIZE],
                              int *out_left,
                              int *out_right) {
    /* Index vector for validity mask generation */
    static const uint8x16_t neon_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    /* ge[chunk] tracks positions where target >= feature (not yet proven less)
     * le[chunk] tracks positions where target <= feature (not yet proven greater)
     * Initialize both with validity mask (0xFF = valid, 0x00 = invalid) */
    uint8x16_t ge[4], le[4];
    int full_chunks = num_keys / 16;
    for (int chunk = 0; chunk < full_chunks; chunk++)
        ge[chunk] = le[chunk] = vdupq_n_u8(0xFF);
    for (int chunk = full_chunks; chunk < 4; chunk++) {
        int remaining = num_keys - chunk * 16;
        if (remaining <= 0)
            ge[chunk] = le[chunk] = vdupq_n_u8(0x00);
        else
            ge[chunk] = le[chunk] = vcltq_u8(neon_indices, vdupq_n_u8((uint8_t)remaining));
    }

    for (int row = 0; row < FEATURE_SIZE; row++) {
        int8x16_t target_biased = vdupq_n_s8((int8_t)(target[row] ^ FEATURE_BIAS));

        for (int chunk = 0; chunk < 4; chunk++) {
            int8x16_t feat = vld1q_s8((const int8_t *)&features[row][chunk * 16]);
            uint8x16_t gt = vcgtq_s8(target_biased, feat); /* target > feature */
            uint8x16_t lt = vcltq_s8(target_biased, feat); /* target < feature */

            /* undecided = positions where both ge and le are still set */
            uint8x16_t undecided = vandq_u8(ge[chunk], le[chunk]);

            /* ge &= ~(lt & undecided): if target < feature and was undecided, target is not >= */
            ge[chunk] = vbicq_u8(ge[chunk], vandq_u8(lt, undecided));

            /* le &= ~(gt & undecided): if target > feature and was undecided, target is not <= */
            le[chunk] = vbicq_u8(le[chunk], vandq_u8(gt, undecided));
        }
    }

    /* Extract final bitmasks */
    uint64_t le_mask = neon_movemask_64(le[0], le[1], le[2], le[3]);
    uint64_t ge_mask = neon_movemask_64(ge[0], ge[1], ge[2], ge[3]);

    *out_left = le_mask ? __builtin_ctzll(le_mask) : num_keys;
    *out_right = (le_mask & ~ge_mask) ? __builtin_ctzll(le_mask & ~ge_mask) : num_keys;
}

#else

static void featureSearchSIMD(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                              int num_keys,
                              const unsigned char target[FEATURE_SIZE],
                              int *out_left,
                              int *out_right) {
    featureSearchSIMD_scalar(features, num_keys, target, out_left, out_right);
}
#endif /* HAVE_X86_SIMD */

/* Test wrappers to expose static functions for unit testing */
void featureSearchSIMD_test_wrapper(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                    int num_keys,
                                    const unsigned char target[FEATURE_SIZE],
                                    int *out_left,
                                    int *out_right) {
    featureSearchSIMD(features, num_keys, target, out_left, out_right);
}

#if HAVE_X86_SIMD
void featureSearchSIMD_avx2_test_wrapper(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                         int num_keys,
                                         const unsigned char target[FEATURE_SIZE],
                                         int *out_left,
                                         int *out_right) {
    featureSearchSIMD_avx2(features, num_keys, target, out_left, out_right);
}

void featureSearchSIMD_sse2_test_wrapper(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                         int num_keys,
                                         const unsigned char target[FEATURE_SIZE],
                                         int *out_left,
                                         int *out_right) {
    featureSearchSIMD_sse2(features, num_keys, target, out_left, out_right);
}
#endif

void featureSearchSIMD_scalar_test_wrapper(char features[FEATURE_SIZE][FEATURE_ROW_SIZE],
                                           int num_keys,
                                           const unsigned char target[FEATURE_SIZE],
                                           int *out_left,
                                           int *out_right) {
    featureSearchSIMD_scalar(features, num_keys, target, out_left, out_right);
}

/* Find child index for insertion using feature vectors and anchors.
 * Two-pass approach:
 * 1. SIMD pass: narrow candidates using feature comparison
 * 2. Anchor pass: binary search on anchors within narrowed range (only if needed)
 *
 * Compares the full prefix at each node. If the key falls outside the prefix
 * range (less than all anchors or greater than all anchors), short-circuits
 * to child 0 or num_items without proceeding to features/anchors. */
static int findChildIndex(innerNode *inner, const_sds string) {
    if (inner->prefix_len > 0) {
        const char *prefix = innerNodeGetPrefix(inner);
        size_t slen = sdslen(string);
        size_t cmp_len = inner->prefix_len < slen ? inner->prefix_len : slen;
        int cmp = memcmp(string, prefix, cmp_len);
        if (cmp == 0 && slen < inner->prefix_len) cmp = -1; /* string shorter than prefix */
        if (cmp < 0) return 0;
        if (cmp > 0) return inner->header.num_items;
    }

    /* Extract target feature bytes */
    unsigned char target[FEATURE_SIZE];
    size_t slen = sdslen(string);
    for (int j = 0; j < FEATURE_SIZE; j++) {
        size_t idx = inner->prefix_len + j;
        target[j] = (idx < slen) ? (unsigned char)string[idx] : 0;
    }

    /* Pass 1: SIMD feature search to narrow range */
    int left, right;
    featureSearchSIMD(inner->features, inner->header.num_items, target, &left, &right);

    /* Pass 2: Binary search on anchors within narrowed range (handles collisions) */
    while (left < right) {
        int mid = (left + right) / 2;
        int cmp = sdscmp(string, inner->anchors[mid]);
        if (cmp <= 0) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return left;
}

static insertResult innerNodeHandleChildSplit(innerNode *parent, node *new_child, sds new_child_anchor, size_t new_child_idx) {
    if (parent->header.num_items == NODE_SIZE) {
        /* We're full - need to split */
        innerNode *new_right_parent = innerNodeSplit(parent);

        innerNode *insert_node = parent;
        bool insert_in_right_parent = new_child_idx > parent->header.num_items;
        if (insert_in_right_parent) {
            insert_node = new_right_parent;
            new_child_idx -= parent->header.num_items;
        }

        innerNodeInsert(insert_node, new_child, new_child_anchor, new_child_idx);
        insertResult result = {
            .updated_anchor = parent->anchors[parent->header.num_items - 1],
            .new_node = (node *)new_right_parent,
            .new_node_anchor = new_right_parent->anchors[new_right_parent->header.num_items - 1]};
        return result;
    } else {
        /* We're not full - just insert new child */
        bool anchor_changed = innerNodeInsert(parent, new_child, new_child_anchor, new_child_idx);
        insertResult result = {
            .updated_anchor = anchor_changed ? new_child_anchor : NULL,
        };
        return result;
    }
}

static insertResult subtreeInsert(node *n, sds string, TraversalHint hint) {
    assert(n);
    if (n->is_leaf) {
        leafNode *leaf = (leafNode *)n;
        if (leaf->header.num_items == NODE_SIZE) {
            return leafNodeSplit(leaf, string, hint);
        } else {
            return leafNodeInsert(leaf, string);
        }
    } else {
        /* inner node - find correct child for insert */
        innerNode *parent = (innerNode *)n;
        assert(parent->header.num_items > 0);

        /* Use hint to skip search when possible */
        int child_idx;
        if (hint == HINT_RIGHTMOST) {
            child_idx = parent->header.num_items - 1;
        } else if (hint == HINT_LEFTMOST) {
            child_idx = 0;
        } else {
            child_idx = findChildIndex(parent, string);
            if (child_idx == parent->header.num_items) child_idx--;
        }

        insertResult child_insert_result = subtreeInsert(parent->children[child_idx], string, hint);

        if (child_insert_result.updated_anchor) {
            parent->anchors[child_idx] = child_insert_result.updated_anchor;
            /* Anchor changed - prefix may need to shrink if this is first or last child */
            if (child_idx == 0 || child_idx == parent->header.num_items - 1) {
                updateCommonPrefix(parent);
            }
            for (int j = 0; j < FEATURE_SIZE; j++)
                parent->features[j][child_idx] = getFeatureByte(child_insert_result.updated_anchor, parent->prefix_len, j);
        }

        if (child_insert_result.new_node) {
            /* Our child split - recalculate original child's size since it lost elements */
            parent->child_sizes[child_idx] = getSubtreeSize(parent->children[child_idx]);
            parent->child_num_items[child_idx] = parent->children[child_idx]->num_items;
            /* Our child split, and we need to insert the new child just after the existing one */
            insertResult result = innerNodeHandleChildSplit(parent, child_insert_result.new_node, child_insert_result.new_node_anchor, child_idx + 1);
            result.inserted_item = child_insert_result.inserted_item;
            return result;
        } else {
            /* No split - just increment size for the inserted element */
            parent->child_sizes[child_idx]++;
            parent->child_num_items[child_idx] = parent->children[child_idx]->num_items;
            /* subtree root did not split, so no new child node to deal with */
            bool parent_anchor_changed = (child_idx == parent->header.num_items - 1);
            insertResult result = {
                .updated_anchor = parent_anchor_changed ? parent->anchors[child_idx] : NULL,
                .inserted_item = child_insert_result.inserted_item,
            };
            return result;
        }
    }
}

sds fbtreeInsert(fbtreeIndex *fbt, sds string) {
    if (fbt->root == NULL) {
        leafNode *leaf = leafNodeCreateWithItem(string);
        fbt->root = (node *)leaf;
        fbt->leftmost_leaf = leaf;
        fbt->rightmost_leaf = leaf;
        return leaf->values[0];
    }

    /* Detect append/prepend patterns for optimized insert path */
    TraversalHint hint = HINT_NONE;
    leafNode *rightmost = fbt->rightmost_leaf;
    leafNode *leftmost = fbt->leftmost_leaf;

    if (rightmost && sdscmp(string, leafNodeHighKey(rightmost)) > 0) {
        hint = HINT_RIGHTMOST;
    } else if (leftmost && sdscmp(string, leafNodeLowKey(leftmost)) < 0) {
        hint = HINT_LEFTMOST;
    }

    /* Insert with hint - skips inner node searches for append/prepend */
    insertResult result = subtreeInsert(fbt->root, string, hint);

    if (result.new_node) {
        innerNode *new_root = innerNodeCreate();
        sds left_anchor = result.updated_anchor;
        if (!left_anchor) {
            left_anchor = nodeHighKey(fbt->root);
        }
        innerNodeInsert(new_root, fbt->root, left_anchor, 0);
        innerNodeInsert(new_root, result.new_node, result.new_node_anchor, 1);
        fbt->root = (node *)new_root;
    }

    /* Update leaf caches using linked list - O(1) */
    if (leftmost && leftmost->prev) {
        fbt->leftmost_leaf = leftmost->prev;
    }
    if (rightmost && rightmost->next) {
        fbt->rightmost_leaf = rightmost->next;
    }
    return result.inserted_item;
}

/* Remove item from leaf without freeing it. Returns the removed item. */
static sds leafNodeRemoveAt(leafNode *leaf, int delete_index) {
    assert(delete_index >= 0 && delete_index < leaf->header.num_items);
    sds item = leaf->values[delete_index];
    leaf->header.num_items--;
    memmove(&leaf->values[delete_index], &leaf->values[delete_index + 1],
            (leaf->header.num_items - delete_index) * sizeof(sds));
    return item;
}

/* Remove and free values at indices [start_idx, end_idx] inclusive from leaf.
 * Returns the number of items removed. */
static int leafNodeRemoveRange(leafNode *leaf, int start_idx, int end_idx) {
    assert(start_idx >= 0 && end_idx < leaf->header.num_items && start_idx <= end_idx);
    int remove_count = end_idx - start_idx + 1;

    for (int i = start_idx; i <= end_idx; i++) {
        sdsfree(leaf->values[i]);
    }

    int num_to_move = leaf->header.num_items - end_idx - 1;
    if (num_to_move > 0) {
        memmove(&leaf->values[start_idx], &leaf->values[end_idx + 1], num_to_move * sizeof(sds));
    }
    leaf->header.num_items -= remove_count;
    return remove_count;
}

/* Merge right leaf into left leaf by appending right's items after left's.
 * Preserves sorted order since left < right in the tree. Right becomes empty.
 * Caller must ensure combined count fits: left->num_items + right->num_items <= NODE_SIZE. */
static void leafNodeMerge(leafNode *left, leafNode *right) {
    assert(left->header.num_items + right->header.num_items <= NODE_SIZE);
    memcpy(&left->values[left->header.num_items], right->values, right->header.num_items * sizeof(sds));
    left->header.num_items += right->header.num_items;
    right->header.num_items = 0;
}

/* Merge right inner node into left inner node by appending right's children
 * after left's. Preserves tree order since left < right. Right becomes empty.
 * Caller must ensure combined count fits: left->num_items + right->num_items <= NODE_SIZE. */
static void innerNodeMerge(innerNode *left, innerNode *right) {
    assert(left->header.num_items + right->header.num_items <= NODE_SIZE);
    bool same_prefix_len = (left->prefix_len == right->prefix_len);
    innerNodeCopyChildren(left, left->header.num_items, right, 0, right->header.num_items);
    left->header.num_items += right->header.num_items;
    right->header.num_items = 0;
    /* Features depend on prefix_len (the offset into anchor strings), not the
     * prefix bytes themselves. If both nodes had the same prefix_len, the copied
     * features are already correct and we only need updateCommonPrefix to check
     * whether the merged range shortened the prefix (which recomputes features
     * internally if so). If prefix_len differed, the copied features used the
     * wrong offset, so we must recompute if updateCommonPrefix didn't already. */
    if (!updateCommonPrefix(left) && !same_prefix_len) {
        recomputeFeatures(left);
    }
}

/* Attempt to merge the child at child_idx with a sibling in the parent.
 * Called when the child has underflowed (num_items < MIN_FILL).
 * Always merges right into left. Returns the index of the surviving (merged)
 * child, or -1 if no merge was possible. */
static void mergeNodeChildren(fbtreeIndex *fbt, innerNode *inner);
static int tryMergeChild(fbtreeIndex *fbt, innerNode *parent, int child_idx) {
    int child_count = parent->child_num_items[child_idx];
    if (child_count >= MIN_FILL) return -1;

    /* Pick the best sibling: prefer fewer items, check left then right */
    int best_sib = -1;
    if (child_idx > 0 && parent->child_num_items[child_idx - 1] + child_count <= NODE_SIZE) {
        best_sib = child_idx - 1;
    }
    if (child_idx < parent->header.num_items - 1 &&
        parent->child_num_items[child_idx + 1] + child_count <= NODE_SIZE) {
        if (best_sib < 0 || parent->child_num_items[child_idx + 1] < parent->child_num_items[best_sib]) {
            best_sib = child_idx + 1;
        }
    }
    if (best_sib < 0) return -1;

    /* Always merge right into left */
    int left_idx = (best_sib < child_idx) ? best_sib : child_idx;
    int right_idx = (best_sib < child_idx) ? child_idx : best_sib;

    node *left_child = parent->children[left_idx];
    node *right_child = parent->children[right_idx];

    if (left_child->is_leaf) {
        leafNode *left_leaf = (leafNode *)left_child;
        leafNode *right_leaf = (leafNode *)right_child;
        leafNodeMerge(left_leaf, right_leaf);

        /* Update leaf linked list */
        left_leaf->next = right_leaf->next;
        if (right_leaf->next) right_leaf->next->prev = left_leaf;

        /* Update rightmost cache if the freed (right) node was the rightmost leaf.
         * The left leaf can never be the leftmost_leaf being freed here since we
         * always free the right node. */
        if (fbt->rightmost_leaf == right_leaf) fbt->rightmost_leaf = left_leaf;

        zfree(right_leaf);
    } else {
        innerNode *left_inner = (innerNode *)left_child;
        innerNode *right_inner = (innerNode *)right_child;
        int boundary = left_inner->header.num_items;
        innerNodeMerge(left_inner, right_inner);

        innerNodeFreePrefix(right_inner);
        zfree(right_inner);

        /* Children at the merge boundary are newly-adjacent siblings that
         * were previously in separate parents. Check if they can merge.
         * After a merge, the surviving child may still be underflowed and
         * mergeable with its new neighbor, so loop until stable.
         * Finally, check all children — the merge may have brought together
         * children from both sides that have underflowed neighbors. */
        if (boundary > 0 && boundary < left_inner->header.num_items) {
            left_inner->child_num_items[boundary - 1] = left_inner->children[boundary - 1]->num_items;
            left_inner->child_num_items[boundary] = left_inner->children[boundary]->num_items;
            int ci = (left_inner->child_num_items[boundary] < MIN_FILL) ? boundary : (left_inner->child_num_items[boundary - 1] < MIN_FILL) ? boundary - 1
                                                                                                                                            : -1;
            while (ci >= 0 && ci < left_inner->header.num_items) {
                int merged_idx = tryMergeChild(fbt, left_inner, ci);
                if (merged_idx < 0) break;
                ci = merged_idx;
                left_inner->child_num_items[ci] = left_inner->children[ci]->num_items;
            }
        }
        /* Scan all children for underflow — the boundary merge above handles
         * the merge seam, but children from both sides may have pre-existing
         * underflows that are now mergeable with their new neighbors.
         * child_num_items is inline in the node, so this is a cache-hot scan. */
        mergeNodeChildren(fbt, left_inner);
    }

    /* Parent fixup: copy right's anchor/features to left's slot */
    parent->anchors[left_idx] = parent->anchors[right_idx];
    for (int j = 0; j < FEATURE_SIZE; j++)
        parent->features[j][left_idx] = parent->features[j][right_idx];
    parent->child_sizes[left_idx] += parent->child_sizes[right_idx];
    parent->child_num_items[left_idx] = parent->children[left_idx]->num_items;

    /* Remove the right child entry from parent */
    innerNodeRemoveChild(parent, right_idx);

    /* We changed anchors[left_idx] before removing right_idx.
     * innerNodeRemoveChild handles prefix updates when the removed index is
     * first or last, but if left_idx == 0 and right_idx > 0, the first anchor
     * changed without innerNodeRemoveChild knowing. Update prefix in that case.
     * Note: we must NOT let the prefix grow longer than children can support,
     * so only shorten or keep the same. */
    if (left_idx == 0 && right_idx > 0 && right_idx < parent->header.num_items) {
        /* Only update if the first anchor changed - prefix might need shortening */
        updateCommonPrefix(parent);
    }

    return left_idx;
}

static deleteResult leafNodeDelete(fbtreeIndex *fbt, leafNode *leaf, const_sds item, bool by_value) {
    assert(leaf->header.num_items > 0);

    /* Find the slot to delete. Pointer identity for the original-item delete
     * path; byte-equality (via binary search on the sorted leaf) for the
     * value-based delete path. */
    int delete_index = -1;
    if (by_value) {
        int idx = leafNodeBinarySearch(leaf, item);
        if (idx < leaf->header.num_items && sdscmp(leaf->values[idx], item) == 0) {
            delete_index = idx;
        }
    } else {
        for (int i = 0; i < leaf->header.num_items; i++) {
            if (leaf->values[i] == item) {
                delete_index = i;
                break;
            }
        }
    }
    if (delete_index < 0) return (deleteResult){0};

    /* Update leaf caches before delete if this leaf will become empty.
     * Done here while leaf is hot in cache to avoid extra fetches. */
    if (leaf->header.num_items == 1) {
        if (leaf == fbt->leftmost_leaf) fbt->leftmost_leaf = leaf->next;
        if (leaf == fbt->rightmost_leaf) fbt->rightmost_leaf = leaf->prev;
    }

    sdsfree(leafNodeRemoveAt(leaf, delete_index));

    deleteResult result = {
        .delete_executed = true,
        .updated_anchor = (delete_index == leaf->header.num_items) ? leafNodeHighKey(leaf) : NULL,
        .node_underflowed = leaf->header.num_items < MIN_FILL};
    return result;
}

static deleteResult subtreeDeleteItem(fbtreeIndex *fbt, node *n, const_sds item, bool by_value) {
    if (n->is_leaf)
        return leafNodeDelete(fbt, (leafNode *)n, item, by_value);

    innerNode *inner = (innerNode *)n;
    int index = findChildIndex(inner, item);
    if (index == inner->header.num_items) return (deleteResult){0};

    deleteResult child_result = subtreeDeleteItem(fbt, inner->children[index], item, by_value);
    if (!child_result.delete_executed) return child_result;

    /* Update child size after delete */
    inner->child_sizes[index]--;

    if (child_result.updated_anchor) {
        inner->anchors[index] = child_result.updated_anchor;
        /* If the first or last anchor changed, the common prefix may need shortening */
        if (index == 0 || index == inner->header.num_items - 1) {
            updateCommonPrefix(inner);
        }
        for (int j = 0; j < FEATURE_SIZE; j++)
            inner->features[j][index] = getFeatureByte(child_result.updated_anchor, inner->prefix_len, j);
    }

    /* Remove empty child */
    if (inner->child_sizes[index] == 0) {
        node *empty_child = inner->children[index];
        if (empty_child->is_leaf) unlinkLeaf((leafNode *)empty_child);
        zfree(empty_child);
        innerNodeRemoveChild(inner, index);
        /* Anchor update: if we removed last child, new last child's anchor bubbles up */
        sds new_anchor = (inner->header.num_items > 0 && index == inner->header.num_items)
                             ? inner->anchors[inner->header.num_items - 1]
                             : NULL;
        return (deleteResult){
            .updated_anchor = new_anchor,
            .delete_executed = true,
            .node_underflowed = inner->header.num_items < MIN_FILL,
            .node_shrank = true};
    }

    inner->child_num_items[index] = inner->children[index]->num_items;

    /* Attempt merge if child underflowed */
    bool merged = false;
    if (child_result.node_underflowed && inner->header.num_items > 1) {
        int merged_idx = tryMergeChild(fbt, inner, index);
        if (merged_idx >= 0) merged = true;
    }

    /* If the child shrank (from a merge deeper in the tree), its sibling
     * may be underflowed and now able to merge with the smaller child.
     * Check immediate neighbors of the child. */
    if (child_result.node_shrank && !child_result.node_underflowed && inner->header.num_items > 1) {
        inner->child_num_items[index] = inner->children[index]->num_items;
        if (index > 0) {
            inner->child_num_items[index - 1] = inner->children[index - 1]->num_items;
            if (inner->child_num_items[index - 1] < MIN_FILL) {
                if (tryMergeChild(fbt, inner, index - 1) >= 0) merged = true;
            }
        }
        if (index < inner->header.num_items - 1) {
            inner->child_num_items[index + 1] = inner->children[index + 1]->num_items;
            if (inner->child_num_items[index + 1] < MIN_FILL) {
                if (tryMergeChild(fbt, inner, index + 1) >= 0) merged = true;
            }
        }
    }

    /* Recompute updated_anchor: after a merge, the last child may have changed */
    sds updated_anchor = NULL;
    if (child_result.updated_anchor || index >= inner->header.num_items) {
        updated_anchor = inner->anchors[inner->header.num_items - 1];
    }

    deleteResult result = {
        .updated_anchor = updated_anchor,
        .delete_executed = true,
        .node_underflowed = inner->header.num_items < MIN_FILL,
        .node_shrank = merged};
    return result;
}

/* Helper to handle root cleanup after delete */
static void fbtreePostDeleteCleanup(fbtreeIndex *fbt) {
    if (getSubtreeSize(fbt->root) == 0) {
        zfree(fbt->root);
        fbt->root = NULL;
        fbt->leftmost_leaf = NULL;
        fbt->rightmost_leaf = NULL;
    } else if (!fbt->root->is_leaf && fbt->root->num_items == 1) {
        innerNode *old_root = (innerNode *)fbt->root;
        fbt->root = old_root->children[0];
        innerNodeFreePrefix(old_root);
        zfree(old_root);
    }
}

bool fbtreeDelete(fbtreeIndex *fbt, const_sds item) {
    if (fbt->root == NULL) return false;

    deleteResult result = subtreeDeleteItem(fbt, fbt->root, item, false);
    if (!result.delete_executed) return false;

    fbtreePostDeleteCleanup(fbt);
    return true;
}

/* Like fbtreeDelete, but matches the leaf slot by byte-equality (sdscmp == 0)
 * instead of pointer identity. Navigation through inner nodes is by value, so
 * this finds and deletes the (unique) item whose stored sds equals key.
 * Returns true iff an item was found and deleted. */
bool fbtreeDeleteByValue(fbtreeIndex *fbt, const_sds key) {
    if (fbt->root == NULL) return false;

    deleteResult result = subtreeDeleteItem(fbt, fbt->root, key, true);
    if (!result.delete_executed) return false;

    fbtreePostDeleteCleanup(fbt);
    return true;
}

static deleteResult subtreePop(fbtreeIndex *fbt, node *n, TraversalHint hint, bool free_item) {
    assert(hint != HINT_NONE);
    if (n->is_leaf) {
        leafNode *leaf = (leafNode *)n;
        int idx = (hint == HINT_LEFTMOST) ? 0 : leaf->header.num_items - 1;
        sds item = leafNodeRemoveAt(leaf, idx);
        if (free_item) sdsfree(item);
        return (deleteResult){
            .delete_executed = true,
            .updated_anchor = (hint == HINT_RIGHTMOST) ? leafNodeHighKey(leaf) : NULL,
            .node_underflowed = leaf->header.num_items < MIN_FILL};
    }

    innerNode *inner = (innerNode *)n;
    int index = (hint == HINT_LEFTMOST) ? 0 : inner->header.num_items - 1;

    deleteResult child_result = subtreePop(fbt, inner->children[index], hint, free_item);
    if (!child_result.delete_executed) return child_result;

    inner->child_sizes[index]--;

    if (child_result.updated_anchor) {
        inner->anchors[index] = child_result.updated_anchor;
        /* If the first or last anchor changed, the common prefix may need shortening */
        if (index == 0 || index == inner->header.num_items - 1) {
            updateCommonPrefix(inner);
        }
        for (int j = 0; j < FEATURE_SIZE; j++)
            inner->features[j][index] = getFeatureByte(child_result.updated_anchor, inner->prefix_len, j);
    }

    /* Remove empty child */
    if (inner->child_sizes[index] == 0) {
        node *empty_child = inner->children[index];
        if (empty_child->is_leaf) unlinkLeaf((leafNode *)empty_child);
        zfree(empty_child);
        innerNodeRemoveChild(inner, index);
        sds new_anchor = (inner->header.num_items > 0 && index == inner->header.num_items)
                             ? inner->anchors[inner->header.num_items - 1]
                             : NULL;
        return (deleteResult){
            .updated_anchor = new_anchor,
            .delete_executed = true,
            .node_underflowed = inner->header.num_items < MIN_FILL};
    }

    inner->child_num_items[index] = inner->children[index]->num_items;

    /* Attempt merge if child underflowed */
    if (child_result.node_underflowed && inner->header.num_items > 1) {
        tryMergeChild(fbt, inner, index);
    }

    /* Recompute updated_anchor: after a merge, the last child may have changed */
    sds updated_anchor = NULL;
    if (child_result.updated_anchor || index >= inner->header.num_items) {
        updated_anchor = inner->anchors[inner->header.num_items - 1];
    }

    deleteResult result = {
        .updated_anchor = updated_anchor,
        .delete_executed = true,
        .node_underflowed = inner->header.num_items < MIN_FILL};
    return result;
}

/* Pop and return the minimum element. Returns NULL if tree is empty.
 * Caller is responsible for freeing the returned sds. */
sds fbtreePopMin(fbtreeIndex *fbt) {
    if (!fbt->root || !fbt->leftmost_leaf) return NULL;

    leafNode *leaf = fbt->leftmost_leaf;
    sds item = leaf->values[0];

    /* Update cache before delete */
    if (leaf->header.num_items == 1)
        fbt->leftmost_leaf = leaf->next;

    subtreePop(fbt, fbt->root, HINT_LEFTMOST, false);
    fbtreePostDeleteCleanup(fbt);
    return item;
}

/* Pop and return the maximum element. Returns NULL if tree is empty.
 * Caller is responsible for freeing the returned sds. */
sds fbtreePopMax(fbtreeIndex *fbt) {
    if (!fbt->root || !fbt->rightmost_leaf) return NULL;

    leafNode *leaf = fbt->rightmost_leaf;
    sds item = leaf->values[leaf->header.num_items - 1];

    /* Update cache before delete */
    if (leaf->header.num_items == 1)
        fbt->rightmost_leaf = leaf->prev;

    subtreePop(fbt, fbt->root, HINT_RIGHTMOST, false);
    fbtreePostDeleteCleanup(fbt);
    return item;
}

/* Get element at given rank (0-indexed). Returns NULL if rank >= length */
const_sds fbtreeGetAtRank(fbtreeIndex *fbt, unsigned long rank) {
    if (!fbt->root) return NULL;

    node *current = fbt->root;
    unsigned long remaining = rank;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int i = 0;
        while (i < inner->header.num_items && remaining >= inner->child_sizes[i]) {
            remaining -= inner->child_sizes[i];
            i++;
        }
        if (i >= inner->header.num_items) return NULL;
        current = inner->children[i];
    }

    leafNode *leaf = (leafNode *)current;
    if (remaining >= leaf->header.num_items) return NULL;
    return leaf->values[remaining];
}

/* Get rank of an item given a direct pointer to it (from hashtable lookup).
 * The item pointer must be a valid pointer into a leaf node's values array. */
long fbtreeGetRankOfItem(fbtreeIndex *fbt, const_sds item) {
    if (!fbt->root || !item) return -1;

    long rank = 0;
    node *current = fbt->root;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int child_idx = findChildIndex(inner, item);
        if (child_idx >= inner->header.num_items) return -1;

        for (int i = 0; i < child_idx; i++) {
            rank += inner->child_sizes[i];
        }
        current = inner->children[child_idx];
    }

    leafNode *leaf = (leafNode *)current;

    /* Find position by pointer comparison (item is known to be in this leaf) */
    for (int i = 0; i < leaf->header.num_items; i++) {
        if (leaf->values[i] == item) {
            return rank + i;
        }
    }
    return -1; /* Not found */
}

/* Like fbtreeGetRankOfItem, but matches the leaf slot by byte-equality
 * (sdscmp == 0) instead of pointer identity. Returns the 0-based rank, or -1
 * if no item equal to key is present. */
long fbtreeGetRankOfValue(fbtreeIndex *fbt, const_sds key) {
    if (!fbt->root || !key) return -1;

    long rank = 0;
    node *current = fbt->root;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int child_idx = findChildIndex(inner, key);
        if (child_idx >= inner->header.num_items) return -1;

        for (int i = 0; i < child_idx; i++) {
            rank += inner->child_sizes[i];
        }
        current = inner->children[child_idx];
    }

    leafNode *leaf = (leafNode *)current;

    /* Binary-search the sorted leaf, then verify byte-equality. */
    int idx = leafNodeBinarySearch(leaf, key);
    if (idx < leaf->header.num_items && sdscmp(leaf->values[idx], key) == 0) {
        return rank + idx;
    }
    return -1; /* Not found */
}

unsigned long fbtreeLength(fbtreeIndex *fbt) {
    return fbt->root ? getSubtreeSize(fbt->root) : 0;
}

/* DFS accumulation of heap bytes owned by a subtree: the node allocation
 * itself, any out-of-line long prefix on inner nodes, and the full allocation
 * (header + payload) of every item sds stored in the leaves. Anchors alias
 * item sds already counted in the leaves, so they are not added here. */
static size_t subtreeMemUsage(node *n) {
    size_t total = zmalloc_size(n);
    if (n->is_leaf) {
        leafNode *leaf = (leafNode *)n;
        for (int i = 0; i < leaf->header.num_items; i++) {
            total += sdsAllocSize(leaf->values[i]);
        }
    } else {
        innerNode *inner = (innerNode *)n;
        if (innerNodeHasLongPrefix(inner)) {
            total += zmalloc_size((void *)innerNodeGetPrefix(inner));
        }
        for (int i = 0; i < inner->header.num_items; i++) {
            total += subtreeMemUsage(inner->children[i]);
        }
    }
    return total;
}

/* Total heap bytes owned by the tree: the fbt header, every node, every
 * out-of-line long prefix, and every item sds. Returns just the header size
 * for an empty tree. */
size_t fbtreeMemUsage(fbtreeIndex *fbt) {
    size_t total = zmalloc_size(fbt);
    if (fbt->root) total += subtreeMemUsage(fbt->root);
    return total;
}

/* Advise the kernel that the backing pages for stored item sds won't be
 * needed soon, to reduce copy-on-write after fork. Walks the leaf chain and
 * dismisses each item's allocation. Node structs are below page size, so
 * dismissing them is a no-op; we skip them. Nothing is freed. */
void fbtreeDismiss(fbtreeIndex *fbt) {
    for (leafNode *leaf = fbt->leftmost_leaf; leaf != NULL; leaf = leaf->next) {
        for (int i = 0; i < leaf->header.num_items; i++) {
            sds item = leaf->values[i];
            zmadvise_dontneed(sdsAllocPtr(item));
        }
    }
}

/* Active-defrag relocation of a subtree. Reallocates the node itself, any
 * out-of-line long prefix, and (for leaves) every item sds, fixing all
 * internal pointers. The doubly linked leaf chain is rebuilt in traversal
 * order via *prev_leaf / *first_leaf. *high_key_out receives the (possibly
 * relocated) rightmost item sds of this subtree, which ancestor anchors alias.
 * Returns the (possibly relocated) node pointer. */
static node *fbtreeDefragNode(node *n,
                              void *(*defrag_alloc)(void *),
                              sds (*defrag_sds)(sds),
                              sds *high_key_out,
                              leafNode **prev_leaf,
                              leafNode **first_leaf) {
    if (n->is_leaf) {
        leafNode *leaf = (leafNode *)n;
        /* Relocate item sds first so the leaf and any ancestor anchors can be
         * updated to the new pointers. */
        for (int i = 0; i < leaf->header.num_items; i++) {
            sds ns = defrag_sds(leaf->values[i]);
            if (ns) leaf->values[i] = ns;
        }
        leafNode *newleaf;
        if ((newleaf = defrag_alloc(leaf))) leaf = newleaf;
        /* Rebuild the doubly linked leaf chain in left-to-right order. */
        leaf->prev = *prev_leaf;
        leaf->next = NULL;
        if (*prev_leaf)
            (*prev_leaf)->next = leaf;
        else
            *first_leaf = leaf;
        *prev_leaf = leaf;
        *high_key_out = leafNodeHighKey(leaf);
        return (node *)leaf;
    }

    innerNode *inner = (innerNode *)n;
    /* Relocate an out-of-line long prefix buffer, if any. */
    if (innerNodeHasLongPrefix(inner)) {
        char **ptr = (char **)&inner->embedded_prefix[LONG_PREFIX_PTR_OFFSET];
        char *newp = defrag_alloc(*ptr);
        if (newp) *ptr = newp;
    }
    for (int i = 0; i < inner->header.num_items; i++) {
        sds child_high = NULL;
        inner->children[i] =
            fbtreeDefragNode(inner->children[i], defrag_alloc, defrag_sds, &child_high, prev_leaf, first_leaf);
        /* The anchor aliases the rightmost item sds of the child subtree, which
         * may have moved. The bytes are unchanged, so the cached feature rows
         * stay valid; only the alias pointer needs updating. */
        inner->anchors[i] = child_high;
    }
    innerNode *newinner;
    if ((newinner = defrag_alloc(inner))) inner = newinner;
    *high_key_out = inner->anchors[inner->header.num_items - 1];
    return (node *)inner;
}

/* Relocate the whole tree (nodes, long prefixes, item sds, and the header) to
 * fresh allocations using the caller's defrag callbacks. Returns the (possibly
 * relocated) index pointer. */
fbtreeIndex *fbtreeDefrag(fbtreeIndex *fbt, void *(*defrag_alloc)(void *), sds (*defrag_sds)(sds)) {
    if (fbt->root) {
        leafNode *prev_leaf = NULL, *first_leaf = NULL;
        sds high_key = NULL;
        fbt->root = fbtreeDefragNode(fbt->root, defrag_alloc, defrag_sds, &high_key, &prev_leaf, &first_leaf);
        fbt->leftmost_leaf = first_leaf;
        fbt->rightmost_leaf = prev_leaf;
    }
    fbtreeIndex *newfbt;
    if ((newfbt = defrag_alloc(fbt))) fbt = newfbt;
    return fbt;
}

void fbtreeResetIterator(fbtreeIterator *iterator) {
    iter *it = iteratorFromOpaque(iterator);
    it->fbt = NULL;
    it->current_leaf = NULL;
    it->current_index = 0;
    it->leaf_count = 0;
    it->state = ITER_AT_POSITION;
}

void fbtreeInitIterator(fbtreeIterator *iterator, fbtreeIndex *fbt) {
    iter *it = iteratorFromOpaque(iterator);
    it->fbt = fbt->root ? fbt : NULL;
    it->current_leaf = NULL;
    it->current_index = 0;
    it->leaf_count = 0;
    it->state = ITER_AT_POSITION;
}

fbtreeIndex *fbtreeIteratorGetIndex(fbtreeIterator *iterator) {
    return iteratorFromOpaque(iterator)->fbt;
}

bool fbtreeNext(fbtreeIterator *iterator, const_sds *pos) {
    iter *it = iteratorFromOpaque(iterator);
    if (!it->fbt) return false;
    if (it->state == ITER_PAST_END) return false;

    if (!it->current_leaf) {
        /* First call - use cached leftmost leaf for O(1) start */
        it->current_leaf = it->fbt->leftmost_leaf;
        it->current_index = 0;
        it->leaf_count = it->current_leaf ? it->current_leaf->header.num_items : 0;
        it->state = ITER_AT_POSITION;
    }

    while (it->current_leaf) {
        if (it->current_index < it->leaf_count) {
            *pos = it->current_leaf->values[it->current_index++];
            return true;
        }
        it->current_leaf = it->current_leaf->next;
        it->current_index = 0;
        it->leaf_count = it->current_leaf ? it->current_leaf->header.num_items : 0;
    }
    it->state = ITER_PAST_END;
    return false;
}

bool fbtreePrev(fbtreeIterator *iterator, const_sds *pos) {
    iter *it = iteratorFromOpaque(iterator);
    if (!it->fbt) return false;
    if (it->state == ITER_BEFORE_START) return false;

    if (!it->current_leaf) {
        /* First call - use cached rightmost leaf for O(1) start */
        it->current_leaf = it->fbt->rightmost_leaf;
        it->leaf_count = it->current_leaf ? it->current_leaf->header.num_items : 0;
        it->current_index = it->leaf_count;
        it->state = ITER_AT_POSITION;
    }

    while (it->current_leaf) {
        if (it->current_index > 0) {
            *pos = it->current_leaf->values[--it->current_index];
            return true;
        }
        it->current_leaf = it->current_leaf->prev;
        it->leaf_count = it->current_leaf ? it->current_leaf->header.num_items : 0;
        it->current_index = it->leaf_count;
    }
    it->state = ITER_BEFORE_START;
    return false;
}

void fbtreeSeekToRank(fbtreeIterator *iterator, unsigned long rank) {
    iter *it = iteratorFromOpaque(iterator);
    if (!it->fbt || !it->fbt->root) {
        it->current_leaf = NULL;
        return;
    }

    node *current = it->fbt->root;
    unsigned long remaining = rank;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int i = 0;
        while (i < inner->header.num_items && remaining >= inner->child_sizes[i]) {
            remaining -= inner->child_sizes[i];
            i++;
        }
        if (i >= inner->header.num_items) {
            /* Rank beyond tree — position past end */
            it->current_leaf = NULL;
            it->state = ITER_PAST_END;
            return;
        }
        current = inner->children[i];
    }

    leafNode *leaf = (leafNode *)current;
    if (remaining >= leaf->header.num_items) {
        /* Rank beyond tree — position past end */
        it->current_leaf = NULL;
        it->state = ITER_PAST_END;
        return;
    }
    it->current_leaf = leaf;
    it->leaf_count = leaf->header.num_items;
    it->current_index = (uint8_t)remaining;
    it->state = (rank == 0) ? ITER_BEFORE_START : ITER_AT_POSITION;
}

#define SCORE_SIZE sizeof(double) /* Normalized score prefix size */

/* Find child index for score lookup (8-byte prefix).
 * Compares the full prefix at each node (up to SCORE_SIZE bytes). */
static int findChildIndexByScore(innerNode *inner, const char *score) {
    /* Compare prefix bytes (up to 8) */
    if (inner->prefix_len > 0) {
        size_t cmp_len = inner->prefix_len < SCORE_SIZE ? inner->prefix_len : SCORE_SIZE;
        const char *prefix = innerNodeGetPrefix(inner);
        int cmp = memcmp(score, prefix, cmp_len);
        if (cmp < 0) return 0;
        if (cmp > 0) return inner->header.num_items;
    }

    /* If node prefix covers entire score, first child has first match */
    if (inner->prefix_len >= SCORE_SIZE) return 0;

    /* Extract feature bytes - no bounds check, score is always 8 bytes */
    unsigned char target[FEATURE_SIZE];
    for (int j = 0; j < FEATURE_SIZE; j++)
        target[j] = (unsigned char)score[inner->prefix_len + j];

    int left, right;
    featureSearchSIMD(inner->features, inner->header.num_items, target, &left, &right);

    /* Binary search with fixed 8-byte comparison */
    while (left < right) {
        int mid = (left + right) / 2;
        if (memcmp(inner->anchors[mid], score, SCORE_SIZE) < 0)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

/* Binary search in leaf for first element with score >= given score */
static int leafNodeBinarySearchByScore(leafNode *leaf, const char *score) {
    int left = 0, right = leaf->header.num_items;
    while (left < right) {
        int mid = (left + right) / 2;
        if (memcmp(leaf->values[mid], score, SCORE_SIZE) < 0)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

/* Seek to first element with score >= given score. Always positions iterator.
 * If score > all elements, positions past end (fbtreeNext fails, fbtreePrev works).
 * If score < all elements, positions at start (fbtreePrev fails, fbtreeNext works).
 * Returns the rank (0-indexed) of the position. If positioned past end, returns
 * the tree length (one past the last valid rank). Returns 0 for an empty tree. */
long fbtreeSeekToScore(const char *score, fbtreeIterator *iterator) {
    iter *it = iteratorFromOpaque(iterator);
    fbtreeIndex *fbt = it->fbt;
    it->current_leaf = NULL;
    it->current_index = 0;
    it->leaf_count = 0;
    it->state = ITER_AT_POSITION;

    if (!fbt || !fbt->root) {
        it->fbt = NULL;
        return 0;
    }

    node *current = fbt->root;
    long rank = 0;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int child_idx = findChildIndexByScore(inner, score);
        if (child_idx >= inner->header.num_items)
            child_idx = inner->header.num_items - 1;
        for (int i = 0; i < child_idx; i++)
            rank += inner->child_sizes[i];
        current = inner->children[child_idx];
    }

    leafNode *leaf = (leafNode *)current;
    int pos = leafNodeBinarySearchByScore(leaf, score);
    rank += pos;

    it->fbt = fbt;
    it->current_leaf = leaf;
    it->leaf_count = leaf->header.num_items;
    it->current_index = pos;

    if (pos >= leaf->header.num_items) {
        /* Score beyond tree max */
        it->current_leaf = NULL;
        it->leaf_count = 0;
        it->current_index = 0;
        it->state = ITER_PAST_END;
    } else if (pos == 0 && leaf == fbt->leftmost_leaf) {
        /* At first element of tree - nothing before this */
        it->state = ITER_BEFORE_START;
    }
    return rank;
}

/* Seek to first element with value >= given value using full sds comparison.
 * If value > all elements, positions past end (fbtreeNext fails, fbtreePrev works).
 * If value < all elements, positions at start (fbtreePrev fails, fbtreeNext works). */
void fbtreeSeekToValue(const_sds value, fbtreeIterator *iterator) {
    iter *it = iteratorFromOpaque(iterator);
    fbtreeIndex *fbt = it->fbt;
    it->current_leaf = NULL;
    it->current_index = 0;
    it->leaf_count = 0;
    it->state = ITER_AT_POSITION;

    if (!fbt || !fbt->root) {
        it->fbt = NULL;
        return;
    }

    node *current = fbt->root;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        int child_idx = findChildIndex(inner, value);
        if (child_idx >= inner->header.num_items) child_idx = inner->header.num_items - 1;
        current = inner->children[child_idx];
    }

    leafNode *leaf = (leafNode *)current;
    int pos = leafNodeBinarySearch(leaf, value);

    it->fbt = fbt;
    it->current_leaf = leaf;
    it->leaf_count = leaf->header.num_items;
    it->current_index = pos;

    if (pos >= leaf->header.num_items) {
        /* Value beyond tree max */
        it->current_leaf = NULL;
        it->leaf_count = 0;
        it->current_index = 0;
        it->state = ITER_PAST_END;
    } else if (pos == 0 && leaf == fbt->leftmost_leaf) {
        /* At first element of tree - nothing before this */
        it->state = ITER_BEFORE_START;
    }
}

/* ========== Range Deletion ========== */

/* Boundary paths captured by path builders for use by deleteRangeCore and
 * deleteRangeSameLeaf. Stack-allocated; all arrays bounded by MAX_TREE_DEPTH. */
typedef struct {
    /* Shared path from root to split point */
    node *shared_path[MAX_TREE_DEPTH];
    int shared_left_idx[MAX_TREE_DEPTH];  /* left boundary child index at each level */
    int shared_right_idx[MAX_TREE_DEPTH]; /* right boundary child index at each level */
    int shared_depth;                     /* number of levels in shared path */

    /* Left sub-path from split point's left child down to start leaf */
    node *left_sub_path[MAX_TREE_DEPTH];
    int left_sub_idx[MAX_TREE_DEPTH]; /* child index at each level */
    int left_sub_depth;

    /* Right sub-path from split point's right child down to end leaf */
    node *right_sub_path[MAX_TREE_DEPTH];
    int right_sub_idx[MAX_TREE_DEPTH]; /* child index at each level */
    int right_sub_depth;

    /* Leaf-level boundary indices */
    leafNode *start_leaf;
    int start_idx; /* first index to delete in start_leaf */
    leafNode *end_leaf;
    int end_idx; /* last index to delete in end_leaf */

} BoundaryPaths;

/* Descend from an inner node to a leaf, recording the path. At each inner
 * node level, findChild is called to determine which child to descend into.
 * Returns the leaf node reached.
 *
 * When called with a compile-time-constant function pointer (e.g.,
 * findChildByScoreWrapper), the compiler can inline both this helper and the
 * callback at -O2, producing specialized code with no indirect calls. */
typedef int (*findChildFn)(innerNode *inner, const void *key);

static leafNode *descendSubPath(node *start,
                                findChildFn findChild,
                                const void *key,
                                node *path[],
                                int path_idx[],
                                int *path_depth) {
    node *current = start;
    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        path[*path_depth] = current;
        int ci = findChild(inner, key);
        if (ci >= inner->header.num_items) ci = inner->header.num_items - 1;
        path_idx[*path_depth] = ci;
        (*path_depth)++;
        current = inner->children[ci];
    }
    return (leafNode *)current;
}

/* Wrappers to match the findChildFn signature */
static int findChildByScoreWrapper(innerNode *inner, const void *key) {
    return findChildIndexByScore(inner, (const char *)key);
}

static int findChildByValueWrapper(innerNode *inner, const void *key) {
    return findChildIndex(inner, (const_sds)key);
}

/* Comparison function for leaf-level index resolution.
 * Returns <0, 0, or >0 like memcmp/sdscmp. */
typedef int (*leafCmpFn)(const_sds element, const void *key);

static int leafCmpByScore(const_sds element, const void *key) {
    return memcmp(element, key, SCORE_SIZE);
}

static int leafCmpByValue(const_sds element, const void *key) {
    return sdscmp(element, (const_sds)key);
}

/* Resolve the start index (first element to delete) in a leaf.
 * Binary search for first element >= key, then advance past equals
 * if the bound is exclusive. */
static int resolveStartIdx(const leafNode *leaf, const void *key, int exclusive, leafCmpFn cmp) {
    int lo = 0, hi = leaf->header.num_items;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cmp(leaf->values[mid], key) < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (exclusive) {
        while (lo < leaf->header.num_items && cmp(leaf->values[lo], key) == 0) lo++;
    }
    return lo;
}

/* Resolve the end index (last element to delete) in a leaf.
 * Binary search for first element >= key, then:
 * - inclusive: advance past equals, subtract 1 (last element <= key)
 * - exclusive: subtract 1 (last element < key) */
static int resolveEndIdx(const leafNode *leaf, const void *key, int exclusive, leafCmpFn cmp) {
    int lo = 0, hi = leaf->header.num_items;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cmp(leaf->values[mid], key) < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (exclusive) {
        return lo - 1;
    }
    while (lo < leaf->header.num_items && cmp(leaf->values[lo], key) == 0) lo++;
    return lo - 1;
}

/* Check all children of an inner node for underflow and merge any that
 * are below MIN_FILL with a sibling. */
static void mergeNodeChildren(fbtreeIndex *fbt, innerNode *inner) {
    if (inner->header.num_items <= 1) return;
    for (int i = inner->header.num_items - 1; i >= 0; i--) {
        if (inner->child_num_items[i] < MIN_FILL) {
            tryMergeChild(fbt, inner, i);
        }
    }
}

/* Recursively merge underflowed children top-down through a subtree.
 * max_depth: maximum number of levels to recurse.
 * Exercised by the white-box unit tests (which #include this .c directly);
 * marked unused so the standalone server build stays warning-clean. */
static __attribute__((unused)) void mergeSubtreeUnderflowed(fbtreeIndex *fbt, node *n, int max_depth) {
    if (!n || n->is_leaf || max_depth <= 0) return;
    innerNode *inner = (innerNode *)n;
    if (inner->header.num_items <= 1) return;

    for (int i = inner->header.num_items - 1; i >= 0; i--) {
        if (inner->child_num_items[i] < MIN_FILL) {
            tryMergeChild(fbt, inner, i);
        }
    }

    for (int i = 0; i < inner->header.num_items; i++) {
        mergeSubtreeUnderflowed(fbt, inner->children[i], max_depth - 1);
        /* Recursion may have merged children within children[i], changing
         * its num_items. Update the cache so the re-check pass below sees
         * the correct value. */
        inner->child_num_items[i] = inner->children[i]->num_items;
    }

    /* Re-check after recursion — child-level merges may have caused new
     * underflows visible from this level. */
    for (int i = inner->header.num_items - 1; i >= 0; i--) {
        if (inner->header.num_items > 1 && inner->child_num_items[i] < MIN_FILL) {
            tryMergeChild(fbt, inner, i);
        }
    }
}

/* Merge underflowed boundary nodes along a sub-path (leg) bottom-up.
 * At each level, check the boundary child for underflow. If a merge happens,
 * re-check at the same level (the merged node might be mergeable with its
 * new sibling). Update path pointers as merges happen.
 *
 * Also check the boundary child's immediate siblings: the range delete may
 * have shrunk the boundary child (without underflowing it), making a
 * previously-unmergeable underflowed sibling now fit. */
static void mergeBoundaryPath(fbtreeIndex *fbt, node *path[], int path_idx[], int path_depth) {
    for (int d = path_depth - 1; d >= 0; d--) {
        innerNode *inner = (innerNode *)path[d];
        if (inner->header.num_items <= 1) continue;
        int ci = path_idx[d];
        if (ci >= inner->header.num_items) continue;
        inner->child_num_items[ci] = inner->children[ci]->num_items;
        while (inner->header.num_items > 1 && inner->child_num_items[ci] < MIN_FILL) {
            int merged_idx = tryMergeChild(fbt, inner, ci);
            if (merged_idx < 0) break;
            ci = merged_idx;
            path_idx[d] = ci;
            inner->child_num_items[ci] = inner->children[ci]->num_items;
        }
        /* The boundary child may have shrunk without underflowing. Check
         * its immediate siblings — they may be underflowed and now able
         * to merge with the smaller boundary child. */
        if (ci > 0 && inner->header.num_items > 1) {
            inner->child_num_items[ci - 1] = inner->children[ci - 1]->num_items;
            if (inner->child_num_items[ci - 1] < MIN_FILL) {
                tryMergeChild(fbt, inner, ci - 1);
            }
        }
        if (ci < inner->header.num_items - 1 && inner->header.num_items > 1) {
            inner->child_num_items[ci + 1] = inner->children[ci + 1]->num_items;
            if (inner->child_num_items[ci + 1] < MIN_FILL) {
                tryMergeChild(fbt, inner, ci + 1);
            }
        }
    }
}

/* Merge underflowed nodes after range deletion. Three phases:
 *
 * 1. Legs: Each sub-path had children removed from one side. The boundary
 *    child at each level may be underflowed and needs merging with a sibling.
 *    Also checks the boundary child's immediate siblings — the boundary
 *    child may have shrunk (without underflowing), making a previously-
 *    unmergeable underflowed sibling now fit.
 *    Handled by mergeBoundaryPath (O(sub-path depth) per leg).
 *
 * 2. Crotch: The split node has two boundary children (top of each leg).
 *    mergeNodeChildren checks all children for underflow in one pass.
 *
 * 3. Common path: Above the split node, each level has a single boundary
 *    child — same as a leg. Handled by mergeBoundaryPath on the shared path
 *    (excluding the split node, which was already handled in Phase 2). */
static void mergeAfterRangeDelete(fbtreeIndex *fbt, BoundaryPaths *bp) {
    if (!fbt->root || fbt->root->is_leaf) return;

    /* Phase 1: Merge within each leg (sub-path) bottom-up. */
    mergeBoundaryPath(fbt, bp->left_sub_path, bp->left_sub_idx, bp->left_sub_depth);
    mergeBoundaryPath(fbt, bp->right_sub_path, bp->right_sub_idx, bp->right_sub_depth);

    /* Leg merges may have changed the split node's boundary children's
     * num_items. Update the cache so Phase 2 sees correct values. */
    if (bp->shared_depth > 0) {
        innerNode *split = (innerNode *)bp->shared_path[bp->shared_depth - 1];
        if (!split->header.is_leaf) {
            int li = bp->shared_left_idx[bp->shared_depth - 1];
            int ri = bp->shared_right_idx[bp->shared_depth - 1];
            if (li < split->header.num_items)
                split->child_num_items[li] = split->children[li]->num_items;
            if (ri < split->header.num_items && ri != li)
                split->child_num_items[ri] = split->children[ri]->num_items;
        }
    }

    /* Phase 2: Merge at the crotch (split node). It has two boundary
     * children (top of each leg), so mergeNodeChildren handles both. */
    if (bp->shared_depth > 0) {
        innerNode *split = (innerNode *)bp->shared_path[bp->shared_depth - 1];
        if (!split->header.is_leaf) {
            mergeNodeChildren(fbt, split);
        }
        /* Update the split node's parent's cache — mergeNodeChildren may
         * have changed the split node's num_items. */
        if (bp->shared_depth >= 2) {
            innerNode *split_parent = (innerNode *)bp->shared_path[bp->shared_depth - 2];
            int ci = bp->shared_left_idx[bp->shared_depth - 2];
            if (ci < split_parent->header.num_items)
                split_parent->child_num_items[ci] = split_parent->children[ci]->num_items;
        }
    }

    /* Phase 3: Merge along the common path above the split node.
     * Each level has a single boundary child, same as a leg. */
    mergeBoundaryPath(fbt, bp->shared_path, bp->shared_left_idx, bp->shared_depth - 1);
}

/* Delete a range within a single leaf. The shared_path records the path from
 * root to the leaf's parent for inner node fixup after removal.
 * Returns the number of elements deleted. */
static unsigned long deleteRangeSameLeaf(fbtreeIndex *fbt,
                                         BoundaryPaths *bp,
                                         fbtreeItemCallback callback,
                                         void *callback_ctx) {
    /* Empty range check for score/value paths where resolved indices
     * may indicate no matching elements. Rank paths never hit this. */
    if (bp->start_idx > bp->end_idx || bp->start_idx >= bp->start_leaf->header.num_items || bp->end_idx < 0) return 0;

    leafNode *leaf = bp->start_leaf;
    int start_idx = bp->start_idx;
    int end_idx = bp->end_idx;
    unsigned long deleted = (unsigned long)(end_idx - start_idx + 1);

    /* Update leftmost/rightmost caches if this leaf will become empty */
    if (start_idx == 0 && end_idx == leaf->header.num_items - 1) {
        if (leaf == fbt->leftmost_leaf) fbt->leftmost_leaf = leaf->next;
        if (leaf == fbt->rightmost_leaf) fbt->rightmost_leaf = leaf->prev;
    }

    /* Invoke callback for items being removed */
    if (callback) {
        for (int i = start_idx; i <= end_idx; i++) {
            callback(leaf->values[i], callback_ctx);
        }
    }

    leafNodeRemoveRange(leaf, start_idx, end_idx);

    /* Walk back up shared path: fix sizes and anchors */
    for (int d = bp->shared_depth - 1; d >= 0; d--) {
        innerNode *inner = (innerNode *)bp->shared_path[d];
        int ci = bp->shared_left_idx[d];
        inner->child_sizes[ci] -= deleted;

        if (inner->child_sizes[ci] == 0) {
            node *empty = inner->children[ci];
            if (empty->is_leaf) unlinkLeaf((leafNode *)empty);
            zfree(empty);
            innerNodeRemoveChild(inner, ci);
        } else {
            innerNodeRefreshChildMeta(inner, ci);
            if (ci == 0 || ci == inner->header.num_items - 1) updateCommonPrefix(inner);
        }
    }

    /* Merge pass */
    mergeAfterRangeDelete(fbt, bp);

    fbtreePostDeleteCleanup(fbt);
    return deleted;
}

/* Core range deletion engine for the diverged-path case (boundaries in
 * different leaves). Splices the leaf chain, trims boundary leaves, fixes
 * up inner nodes bottom-to-top, and runs a merge pass.
 * If callback is non-NULL, it is invoked for each deleted item before sdsfree.
 * Returns the number of elements deleted. */
static unsigned long deleteRangeCore(fbtreeIndex *fbt, BoundaryPaths *bp, fbtreeItemCallback callback, void *callback_ctx) {
    /* Empty range check: score/value path builders can produce boundary
     * indices that fall outside the leaf (no matching elements in that leaf).
     * Rank paths never hit this since ranks are pre-validated. */
    if (bp->start_idx >= bp->start_leaf->header.num_items && bp->end_idx < 0) return 0;
    if (bp->start_idx >= bp->start_leaf->header.num_items && bp->start_leaf->next == bp->end_leaf && bp->end_idx < 0)
        return 0;
    if (bp->end_idx < 0 && bp->start_leaf == bp->end_leaf) return 0;

    /* --- Phase 1: Boundaries diverged. Process the split. --- */
    int split_depth = bp->shared_depth - 1;
    innerNode *split_node = (innerNode *)bp->shared_path[split_depth];
    int li = bp->shared_left_idx[split_depth];
    int ri = bp->shared_right_idx[split_depth];

    leafNode *start_leaf = bp->start_leaf;
    int start_idx = bp->start_idx;
    leafNode *end_leaf = bp->end_leaf;
    int end_idx = bp->end_idx;

    /* --- Phase 2: Splice leaf chain and trim boundary leaves --- */

    /* For score/value-based paths, the boundary indices might be out of range:
     * - start_idx >= start_leaf->num_items: no matching elements in start leaf
     * - end_idx < 0: no matching elements in end leaf
     * In these cases, the boundary leaf is not trimmed at all. */
    bool start_leaf_untouched = (start_idx >= start_leaf->header.num_items);
    bool end_leaf_untouched = (end_idx < 0);

    /* Determine the surviving boundary leaves after trimming */
    bool start_leaf_dies = !start_leaf_untouched && (start_idx == 0);
    bool end_leaf_dies = !end_leaf_untouched && (end_idx == end_leaf->header.num_items - 1);

    /* The last surviving leaf on the left side */
    leafNode *left_survivor = start_leaf_dies ? start_leaf->prev : start_leaf;
    /* The first surviving leaf on the right side */
    leafNode *right_survivor = end_leaf_dies ? end_leaf->next : end_leaf;

    /* Update leftmost/rightmost caches */
    if (fbt->leftmost_leaf == start_leaf && start_leaf_dies) {
        fbt->leftmost_leaf = right_survivor;
    }
    if (fbt->rightmost_leaf == end_leaf && end_leaf_dies) {
        fbt->rightmost_leaf = left_survivor;
    }

    /* Splice the linked list: connect left_survivor <-> right_survivor.
     * All leaves between them become unreachable from the list. */
    if (left_survivor)
        left_survivor->next = right_survivor;
    if (right_survivor)
        right_survivor->prev = left_survivor;

    /* Count deleted elements during splice */
    unsigned long deleted = 0;

    /* Trim boundary leaves (free deleted sds values but keep the leaf node alive).
     * Even for leaves that will be fully freed via freeNodeRecursive, we must
     * zero their num_items so getSubtreeSize returns correct counts during fixup.
     * Skip trimming entirely for untouched boundary leaves (score/value edge case). */
    if (start_leaf_untouched) {
        /* No elements to delete in start leaf — leave it completely untouched */
    } else if (!start_leaf_dies) {
        for (int i = start_idx; i < start_leaf->header.num_items; i++) {
            if (callback) callback(start_leaf->values[i], callback_ctx);
            sdsfree(start_leaf->values[i]);
            deleted++;
        }
        start_leaf->header.num_items = start_idx;
    } else {
        for (int i = 0; i < start_leaf->header.num_items; i++) {
            if (callback) callback(start_leaf->values[i], callback_ctx);
            sdsfree(start_leaf->values[i]);
            deleted++;
        }
        start_leaf->header.num_items = 0;
    }
    if (end_leaf_untouched) {
        /* No elements to delete in end leaf — leave it completely untouched */
    } else if (!end_leaf_dies) {
        for (int i = 0; i <= end_idx; i++) {
            if (callback) callback(end_leaf->values[i], callback_ctx);
            sdsfree(end_leaf->values[i]);
            deleted++;
        }
        int remaining = end_leaf->header.num_items - end_idx - 1;
        memmove(&end_leaf->values[0], &end_leaf->values[end_idx + 1], remaining * sizeof(sds));
        end_leaf->header.num_items = remaining;
    } else {
        for (int i = 0; i < end_leaf->header.num_items; i++) {
            if (callback) callback(end_leaf->values[i], callback_ctx);
            sdsfree(end_leaf->values[i]);
            deleted++;
        }
        end_leaf->header.num_items = 0;
    }

    /* --- Phase 3: Fix up inner nodes bottom-to-top --- */

    /* Fix left subtree: remove children to the right of the left boundary child.
     * These subtrees are fully within the deleted range. Their leaves are already
     * disconnected from the linked list, so freeNodeRecursive is safe. */
    for (int d = bp->left_sub_depth - 1; d >= 0; d--) {
        innerNode *inner = (innerNode *)bp->left_sub_path[d];
        int ci = bp->left_sub_idx[d];

        for (int i = ci + 1; i < inner->header.num_items; i++) {
            deleted += getSubtreeSize(inner->children[i]);
            freeNodeRecursive(inner->children[i], callback, callback_ctx);
        }
        inner->header.num_items = ci + 1;

        /* Update or remove the boundary child */
        if (getSubtreeSize(inner->children[ci]) == 0) {
            zfree(inner->children[ci]);
            inner->header.num_items = ci;
        } else {
            innerNodeRefreshChildMeta(inner, ci);
        }
        updateCommonPrefix(inner);
    }

    /* Fix right subtree: remove children to the left of the right boundary child */
    for (int d = bp->right_sub_depth - 1; d >= 0; d--) {
        innerNode *inner = (innerNode *)bp->right_sub_path[d];
        int ci = bp->right_sub_idx[d];

        for (int i = 0; i < ci; i++) {
            deleted += getSubtreeSize(inner->children[i]);
            freeNodeRecursive(inner->children[i], callback, callback_ctx);
        }
        if (ci > 0) {
            innerNodeRemoveChildrenRange(inner, 0, ci - 1);
        }

        /* Boundary child is now at index 0 (if we removed left children).
         * Update the recorded index so the merge pass can use it. */
        bp->right_sub_idx[d] = 0;
        int new_ci = 0;
        if (getSubtreeSize(inner->children[new_ci]) == 0) {
            zfree(inner->children[new_ci]);
            innerNodeRemoveChildrenRange(inner, 0, 0);
        } else {
            innerNodeRefreshChildMeta(inner, new_ci);
        }
        updateCommonPrefix(inner);
    }

    /* Fix the split node: remove fully-deleted middle children, update boundary children */
    {
        /* Free middle children subtrees (fully inside the range) */
        for (int i = li + 1; i < ri; i++) {
            deleted += getSubtreeSize(split_node->children[i]);
            freeNodeRecursive(split_node->children[i], callback, callback_ctx);
        }

        /* Check if boundary subtrees are now empty */
        size_t left_size = getSubtreeSize(split_node->children[li]);
        size_t right_size = getSubtreeSize(split_node->children[ri]);

        int remove_start = li + 1;
        int remove_end = ri - 1;

        if (left_size == 0) {
            freeNodeRecursive(split_node->children[li], callback, callback_ctx);
            remove_start = li;
        }
        if (right_size == 0) {
            freeNodeRecursive(split_node->children[ri], callback, callback_ctx);
            remove_end = ri;
        }

        if (remove_start <= remove_end) {
            innerNodeRemoveChildrenRange(split_node, remove_start, remove_end);
        }

        /* Refresh metadata for surviving boundary children. After removal,
         * they occupy consecutive indices starting at li. */
        ri = li + (left_size > 0) + (right_size > 0) - 1;
        for (int i = li; i <= ri; i++) {
            innerNodeRefreshChildMeta(split_node, i);
        }
        updateCommonPrefix(split_node);
    }

    /* Fix ancestors above the split node */
    for (int d = split_depth - 1; d >= 0; d--) {
        innerNode *inner = (innerNode *)bp->shared_path[d];
        int ci = bp->shared_left_idx[d];

        if (getSubtreeSize(inner->children[ci]) == 0) {
            zfree(inner->children[ci]);
            innerNodeRemoveChild(inner, ci);
        } else {
            innerNodeRefreshChildMeta(inner, ci);
            if (ci == 0 || ci == inner->header.num_items - 1) updateCommonPrefix(inner);
        }
    }

    /* --- Phase 4: Post-fixup merge pass for underflowed boundary nodes --- */

    /* Merge pass */
    mergeAfterRangeDelete(fbt, bp);

    fbtreePostDeleteCleanup(fbt);

    return deleted;
}

/* Delete elements at ranks [start_rank, end_rank] (0-indexed, inclusive).
 * If callback is non-NULL, it is invoked for each deleted item before sdsfree.
 * Returns the number of elements deleted. */
unsigned long fbtreeDeleteRangeByRank(fbtreeIndex *fbt,
                                      unsigned long start_rank,
                                      unsigned long end_rank,
                                      void (*callback)(sds item, void *ctx),
                                      void *callback_ctx) {
    if (!fbt->root) return 0;

    unsigned long length = fbtreeLength(fbt);
    if (start_rank >= length) return 0;
    if (end_rank >= length) end_rank = length - 1;
    if (start_rank > end_rank) return 0;
    if (start_rank == 0 && end_rank == length - 1) {
        fbtreeDeleteAll(fbt, callback, callback_ctx);
        return length;
    }

    /* Build boundary paths by descending the tree using child_sizes accumulation */
    BoundaryPaths bp;
    memset(&bp, 0, sizeof(bp));

    node *current = fbt->root;
    unsigned long left_remaining = start_rank;
    unsigned long right_remaining = end_rank;
    int depth = 0;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        bp.shared_path[depth] = current;

        int li = 0;
        unsigned long left_acc = 0;
        while (li < inner->header.num_items && left_acc + inner->child_sizes[li] <= left_remaining) {
            left_acc += inner->child_sizes[li];
            li++;
        }
        bp.shared_left_idx[depth] = li;
        left_remaining -= left_acc;

        int ri = 0;
        unsigned long right_acc = 0;
        while (ri < inner->header.num_items && right_acc + inner->child_sizes[ri] <= right_remaining) {
            right_acc += inner->child_sizes[ri];
            ri++;
        }
        bp.shared_right_idx[depth] = ri;
        right_remaining -= right_acc;

        depth++;

        if (li == ri) {
            current = inner->children[li];
        } else {
            bp.shared_depth = depth;

            /* Descend left sub-path */
            node *left_current = inner->children[li];
            while (!left_current->is_leaf) {
                innerNode *left_inner = (innerNode *)left_current;
                bp.left_sub_path[bp.left_sub_depth] = left_current;
                int ci = 0;
                unsigned long acc = 0;
                while (ci < left_inner->header.num_items && acc + left_inner->child_sizes[ci] <= left_remaining) {
                    acc += left_inner->child_sizes[ci];
                    ci++;
                }
                bp.left_sub_idx[bp.left_sub_depth] = ci;
                left_remaining -= acc;
                bp.left_sub_depth++;
                left_current = left_inner->children[ci];
            }
            bp.start_leaf = (leafNode *)left_current;
            bp.start_idx = (int)left_remaining;

            /* Descend right sub-path */
            node *right_current = inner->children[ri];
            while (!right_current->is_leaf) {
                innerNode *right_inner = (innerNode *)right_current;
                bp.right_sub_path[bp.right_sub_depth] = right_current;
                int ci = 0;
                unsigned long acc = 0;
                while (ci < right_inner->header.num_items && acc + right_inner->child_sizes[ci] <= right_remaining) {
                    acc += right_inner->child_sizes[ci];
                    ci++;
                }
                bp.right_sub_idx[bp.right_sub_depth] = ci;
                right_remaining -= acc;
                bp.right_sub_depth++;
                right_current = right_inner->children[ci];
            }
            bp.end_leaf = (leafNode *)right_current;
            bp.end_idx = (int)right_remaining;

            return deleteRangeCore(fbt, &bp, callback, callback_ctx);
        }
    }

    /* Both boundaries in the same leaf */
    bp.shared_depth = depth;
    bp.start_leaf = (leafNode *)current;
    bp.end_leaf = (leafNode *)current;
    bp.start_idx = (int)left_remaining;
    bp.end_idx = (int)right_remaining;
    return deleteRangeSameLeaf(fbt, &bp, callback, callback_ctx);
}

/* Delete elements with score prefix in [min_score, max_score].
 * min_ex/max_ex: if true, the corresponding bound is exclusive.
 * Score is an 8-byte big-endian normalized prefix (as stored in the tree).
 * If callback is non-NULL, it is invoked for each deleted item before sdsfree.
 * Returns the number of elements deleted. */
unsigned long fbtreeDeleteRangeByScore(fbtreeIndex *fbt,
                                       const char *min_score,
                                       const char *max_score,
                                       int min_ex,
                                       int max_ex,
                                       void (*callback)(sds item, void *ctx),
                                       void *callback_ctx) {
    if (!fbt->root) return 0;

    /* Delete-all short-circuit */
    sds first = leafNodeLowKey(fbt->leftmost_leaf);
    sds last = leafNodeHighKey(fbt->rightmost_leaf);
    int min_covers = min_ex ? memcmp(min_score, first, SCORE_SIZE) < 0 : memcmp(min_score, first, SCORE_SIZE) <= 0;
    int max_covers = max_ex ? memcmp(max_score, last, SCORE_SIZE) > 0 : memcmp(max_score, last, SCORE_SIZE) >= 0;
    if (min_covers && max_covers) {
        unsigned long count = fbtreeLength(fbt);
        fbtreeDeleteAll(fbt, callback, callback_ctx);
        return count;
    }

    /* Quick check: empty range */
    int range_cmp = memcmp(min_score, max_score, SCORE_SIZE);
    if (range_cmp > 0 || (range_cmp == 0 && (min_ex || max_ex))) return 0;

    /* Build boundary paths by descending the tree using score prefix comparison */
    BoundaryPaths bp;
    memset(&bp, 0, sizeof(bp));

    node *current = fbt->root;
    int depth = 0;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        bp.shared_path[depth] = current;

        int li = findChildIndexByScore(inner, min_score);
        if (li >= inner->header.num_items) li = inner->header.num_items - 1;
        int ri = findChildIndexByScore(inner, max_score);
        if (ri >= inner->header.num_items) ri = inner->header.num_items - 1;

        bp.shared_left_idx[depth] = li;
        bp.shared_right_idx[depth] = ri;
        depth++;

        if (li == ri) {
            current = inner->children[li];
        } else {
            bp.shared_depth = depth;

            /* Descend left sub-path */
            bp.start_leaf = descendSubPath(
                inner->children[li], findChildByScoreWrapper, min_score,
                bp.left_sub_path, bp.left_sub_idx, &bp.left_sub_depth);
            bp.start_idx = resolveStartIdx(bp.start_leaf, min_score, min_ex, leafCmpByScore);

            /* Descend right sub-path */
            bp.end_leaf = descendSubPath(
                inner->children[ri], findChildByScoreWrapper, max_score,
                bp.right_sub_path, bp.right_sub_idx, &bp.right_sub_depth);
            bp.end_idx = resolveEndIdx(bp.end_leaf, max_score, max_ex, leafCmpByScore);

            return deleteRangeCore(fbt, &bp, callback, callback_ctx);
        }
    }

    /* Both boundaries in the same leaf */
    bp.shared_depth = depth;
    bp.start_leaf = (leafNode *)current;
    bp.end_leaf = (leafNode *)current;
    bp.start_idx = resolveStartIdx(bp.start_leaf, min_score, min_ex, leafCmpByScore);
    bp.end_idx = resolveEndIdx(bp.end_leaf, max_score, max_ex, leafCmpByScore);
    return deleteRangeSameLeaf(fbt, &bp, callback, callback_ctx);
}

/* Delete elements with value in [min_val, max_val] using full sds comparison.
 * min_ex/max_ex: if true, the corresponding bound is exclusive.
 * If callback is non-NULL, it is invoked for each deleted item before sdsfree.
 * Returns the number of elements deleted. */
unsigned long fbtreeDeleteRangeByValue(fbtreeIndex *fbt,
                                       const_sds min_val,
                                       const_sds max_val,
                                       int min_ex,
                                       int max_ex,
                                       void (*callback)(sds item, void *ctx),
                                       void *callback_ctx) {
    if (!fbt->root) return 0;

    /* Delete-all short-circuit */
    sds first = leafNodeLowKey(fbt->leftmost_leaf);
    sds last = leafNodeHighKey(fbt->rightmost_leaf);
    int min_covers = min_ex ? sdscmp(min_val, first) < 0 : sdscmp(min_val, first) <= 0;
    int max_covers = max_ex ? sdscmp(max_val, last) > 0 : sdscmp(max_val, last) >= 0;
    if (min_covers && max_covers) {
        unsigned long count = fbtreeLength(fbt);
        fbtreeDeleteAll(fbt, callback, callback_ctx);
        return count;
    }

    /* Quick check: empty range */
    int range_cmp = sdscmp(min_val, max_val);
    if (range_cmp > 0 || (range_cmp == 0 && (min_ex || max_ex))) return 0;

    /* Build boundary paths by descending the tree using full sds comparison */
    BoundaryPaths bp;
    memset(&bp, 0, sizeof(bp));

    node *current = fbt->root;
    int depth = 0;

    while (!current->is_leaf) {
        innerNode *inner = (innerNode *)current;
        bp.shared_path[depth] = current;

        int li = findChildIndex(inner, min_val);
        if (li >= inner->header.num_items) li = inner->header.num_items - 1;
        int ri = findChildIndex(inner, max_val);
        if (ri >= inner->header.num_items) ri = inner->header.num_items - 1;

        bp.shared_left_idx[depth] = li;
        bp.shared_right_idx[depth] = ri;
        depth++;

        if (li == ri) {
            current = inner->children[li];
        } else {
            bp.shared_depth = depth;

            /* Descend left sub-path */
            bp.start_leaf = descendSubPath(
                inner->children[li], findChildByValueWrapper, min_val,
                bp.left_sub_path, bp.left_sub_idx, &bp.left_sub_depth);
            bp.start_idx = resolveStartIdx(bp.start_leaf, min_val, min_ex, leafCmpByValue);

            /* Descend right sub-path */
            bp.end_leaf = descendSubPath(
                inner->children[ri], findChildByValueWrapper, max_val,
                bp.right_sub_path, bp.right_sub_idx, &bp.right_sub_depth);
            bp.end_idx = resolveEndIdx(bp.end_leaf, max_val, max_ex, leafCmpByValue);

            return deleteRangeCore(fbt, &bp, callback, callback_ctx);
        }
    }

    /* Both boundaries in the same leaf */
    bp.shared_depth = depth;
    bp.start_leaf = (leafNode *)current;
    bp.end_leaf = (leafNode *)current;
    bp.start_idx = resolveStartIdx(bp.start_leaf, min_val, min_ex, leafCmpByValue);
    bp.end_idx = resolveEndIdx(bp.end_leaf, max_val, max_ex, leafCmpByValue);
    return deleteRangeSameLeaf(fbt, &bp, callback, callback_ctx);
}

/* ========== Debug Functions ========== */

typedef struct {
    bool valid;
    size_t size;
    leafNode *leftmost_leaf;
    leafNode *rightmost_leaf;
} validateResult;

static void printBinaryString(const_sds s) {
    size_t len = sdslen(s);
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        if (c >= 32 && c < 127) {
            printf("%c", c);
        } else {
            printf("\033[2m%02x\033[0m", (unsigned char)c);
        }
    }
}

static void printIndent(int depth) {
    for (int j = 0; j < depth; j++) printf("│  ");
}

static validateResult validateNode(node *n, int depth, bool verbose);

static validateResult validateLeaf(leafNode *leaf, int depth, bool verbose) {
    size_t count = leaf->header.num_items;
    bool valid = (count <= NODE_SIZE);

    if (verbose) {
        printf(" Leaf (%zu items)\n", count);
        if (!valid) printf(" \033[31m[num_items %zu exceeds NODE_SIZE %d]\033[0m", count, NODE_SIZE);
        printf("\n");

        for (uint32_t i = 0; i < count; i++) {
            if (i % 8 == 0) {
                printIndent(depth);
                printf("├─");
            }
            printBinaryString(leaf->values[i]);
            printf(" ");
            if (i % 8 == 7) printf("\n");
        }
        if (count > 0 && count % 8 != 0) printf("\n");
    }
    return (validateResult){.valid = valid, .size = count, .leftmost_leaf = leaf, .rightmost_leaf = leaf};
}

static validateResult validateInner(innerNode *inner, int depth, bool verbose) {
    bool valid = true;
    size_t total_size = 0;
    leafNode *leftmost = NULL;
    leafNode *rightmost = NULL;

    if (verbose) {
        printf(" Inner (prefix=%zu, keys=%d)\n", inner->prefix_len, inner->header.num_items);
    }

    for (int i = 0; i < inner->header.num_items; i++) {
        const_sds anchor = inner->anchors[i];
        node *child = inner->children[i];

        /* Validate anchor starts with prefix */
        const char *prefix = innerNodeGetPrefix(inner);
        bool prefix_ok = sdslen(anchor) >= inner->prefix_len &&
                         memcmp(anchor, prefix, inner->prefix_len) == 0;

        /* Validate anchor matches child's high key */
        const_sds expected = nodeHighKey(child);
        bool anchor_ok = (expected == anchor);

        /* Validate feature matches anchor */
        bool feature_ok = true;
        for (int j = 0; j < FEATURE_SIZE && feature_ok; j++)
            feature_ok = (inner->features[j][i] == getFeatureByte(anchor, inner->prefix_len, j));

        /* Recursively validate child and get its size */
        if (verbose) {
            printIndent(depth);
            printf("\u251c\u2500[%02d] size=%zu anchor=", i, inner->child_sizes[i]);
            printBinaryString(anchor);
        }

        validateResult child_result = validateNode(child, depth + 1, verbose);

        /* Track leftmost/rightmost leaves */
        if (i == 0) leftmost = child_result.leftmost_leaf;
        rightmost = child_result.rightmost_leaf;

        /* Validate stored size matches actual size */
        bool size_ok = (inner->child_sizes[i] == child_result.size);

        /* Validate child_num_items matches child's actual num_items */
        bool num_items_ok = (inner->child_num_items[i] == inner->children[i]->num_items);

        valid = valid && prefix_ok && anchor_ok && feature_ok && size_ok && num_items_ok && child_result.valid;
        total_size += child_result.size;

        if (verbose && (!prefix_ok || !anchor_ok || !feature_ok || !size_ok || !num_items_ok)) {
            printIndent(depth);
            printf("   \033[31m");
            if (!prefix_ok) printf("prefix ");
            if (!anchor_ok) printf("anchor ");
            if (!feature_ok) printf("feature ");
            if (!size_ok) printf("size(%zu!=%zu) ", inner->child_sizes[i], child_result.size);
            if (!num_items_ok) printf("num_items(%u!=%u) ", inner->child_num_items[i], inner->children[i]->num_items);
            printf("FAIL\033[0m\n");
        }
    }
    return (validateResult){.valid = valid, .size = total_size, .leftmost_leaf = leftmost, .rightmost_leaf = rightmost};
}

static validateResult validateNode(node *n, int depth, bool verbose) {
    if (!n) return (validateResult){.valid = true, .size = 0};

    if (n->is_leaf) {
        return validateLeaf((leafNode *)n, depth, verbose);
    } else {
        return validateInner((innerNode *)n, depth, verbose);
    }
}

bool fbtreeDebugValidate(fbtreeIndex *fbt, bool verbose) {
    unsigned long length = fbt->root ? getSubtreeSize(fbt->root) : 0;
    if (verbose) printf("FBTree (length=%lu)\n", length);
    if (!fbt->root) {
        /* Empty tree: caches must be NULL */
        if (fbt->leftmost_leaf || fbt->rightmost_leaf) {
            if (verbose) printf("\033[31mERROR: empty tree has non-NULL leaf cache\033[0m\n");
            return false;
        }
        return true;
    }

    validateResult result = validateNode(fbt->root, 0, verbose);

    /* Also verify total size matches computed length */
    bool length_ok = (result.size == length);
    if (!length_ok && verbose) {
        printf("\033[31mERROR: tree size %zu != computed length %lu\033[0m\n", result.size, length);
    }

    /* Verify leaf caches point to actual leftmost/rightmost leaves */
    leafNode *actual_leftmost = result.leftmost_leaf;
    leafNode *actual_rightmost = result.rightmost_leaf;
    bool caches_ok = (fbt->leftmost_leaf == actual_leftmost && fbt->rightmost_leaf == actual_rightmost);
    if (!caches_ok && verbose) {
        printf("\033[31mERROR: leaf cache mismatch (leftmost: %p vs %p, rightmost: %p vs %p)\033[0m\n",
               (void *)fbt->leftmost_leaf, (void *)actual_leftmost,
               (void *)fbt->rightmost_leaf, (void *)actual_rightmost);
    }

    return result.valid && length_ok && caches_ok;
}

/* Validate merge enforcement: no non-root node has num_items < MIN_FILL
 * unless all siblings have num_items + node.num_items > NODE_SIZE.
 * Returns -1 if the property holds, or the depth (0-indexed from root)
 * of the first violation found. */
static int validateMergeEnforcementInner(innerNode *inner, int depth) {
    /* Check each child: if it has num_items < MIN_FILL, verify no sibling could absorb it */
    for (int i = 0; i < inner->header.num_items; i++) {
        int child_items = inner->children[i]->num_items;
        if (child_items < MIN_FILL) {
            /* Check left sibling */
            if (i > 0 && inner->children[i - 1]->num_items + child_items <= NODE_SIZE) {
                return depth;
            }
            /* Check right sibling */
            if (i < inner->header.num_items - 1 && inner->children[i + 1]->num_items + child_items <= NODE_SIZE) {
                return depth;
            }
        }

        /* Recurse into inner children */
        if (!inner->children[i]->is_leaf) {
            int result = validateMergeEnforcementInner((innerNode *)inner->children[i], depth + 1);
            if (result >= 0) return result;
        }
    }
    return -1;
}

/* ===== Incremental, latency-bounded active-defrag ===== */

struct fbtreeDefragCtx {
    void *(*defrag_alloc)(void *);
    sds (*defrag_sds)(sds);
    int phase;        /* 0 = relocate leaves, 1 = relocate inner nodes, 2 = done */
    bool have_resume; /* whether resume_key has been set */
    sds resume_key;   /* byte-copy of the last relocated leaf's high key */
};

fbtreeDefragCtx *fbtreeDefragStart(fbtreeIndex *fbt,
                                   fbtreeIndex **fbt_out,
                                   void *(*defrag_alloc)(void *),
                                   sds (*defrag_sds)(sds)) {
    fbtreeIndex *newfbt;
    if ((newfbt = defrag_alloc(fbt))) fbt = newfbt;
    fbtreeDefragCtx *ctx = zcalloc(sizeof(*ctx));
    ctx->defrag_alloc = defrag_alloc;
    ctx->defrag_sds = defrag_sds;
    ctx->phase = (fbt->root == NULL) ? 2 : 0;
    *fbt_out = fbt;
    return ctx;
}

/* Relocate inner-node structs and out-of-line long prefixes in a subtree,
 * post-order. Leaves were already relocated in phase 0 (their structs are
 * final and inner->children already point at them), so leaf children are left
 * untouched here. Few inner nodes exist, so this completes quickly. */
static node *fbtreeDefragInnerSubtree(node *n, void *(*defrag_alloc)(void *)) {
    if (n->is_leaf) return n;
    innerNode *inner = (innerNode *)n;
    if (innerNodeHasLongPrefix(inner)) {
        char **ptr = (char **)&inner->embedded_prefix[LONG_PREFIX_PTR_OFFSET];
        char *newp = defrag_alloc(*ptr);
        if (newp) *ptr = newp;
    }
    for (int i = 0; i < inner->header.num_items; i++)
        inner->children[i] = fbtreeDefragInnerSubtree(inner->children[i], defrag_alloc);
    innerNode *newinner;
    if ((newinner = defrag_alloc(inner))) inner = newinner;
    return (node *)inner;
}

int fbtreeDefragStep(fbtreeIndex *fbt, fbtreeDefragCtx *ctx, size_t *scanned) {
    if (ctx->phase == 2) return 0;

    if (ctx->phase == 1) {
        if (fbt->root) fbt->root = fbtreeDefragInnerSubtree(fbt->root, ctx->defrag_alloc);
        ctx->phase = 2;
        if (scanned) (*scanned)++;
        return 0;
    }

    /* phase 0: relocate the next leaf in key order. */
    leafNode *target;
    if (!ctx->have_resume) {
        target = fbt->leftmost_leaf;
    } else {
        /* Re-descend by the saved key (not a stored pointer) so concurrent
         * mutations between steps cannot leave us with a dangling leaf. */
        node *cur = fbt->root;
        while (cur && !cur->is_leaf) {
            innerNode *inner = (innerNode *)cur;
            int ci = findChildIndex(inner, ctx->resume_key);
            if (ci >= inner->header.num_items) ci = inner->header.num_items - 1;
            cur = inner->children[ci];
        }
        leafNode *resume_leaf = (leafNode *)cur;
        target = resume_leaf ? resume_leaf->next : NULL;
    }

    if (!target) {
        ctx->phase = 1; /* leaves done; relocate inner nodes next */
        return 1;
    }

    /* Record the root-to-leaf path so we can fix parent pointers/anchors. */
    node *path[MAX_TREE_DEPTH];
    int path_idx[MAX_TREE_DEPTH];
    int path_depth = 0;
    leafNode *leaf = descendSubPath(fbt->root, findChildByValueWrapper, target->values[0], path, path_idx, &path_depth);

    /* 1) Relocate each item sds in place. */
    for (int i = 0; i < leaf->header.num_items; i++) {
        sds ns = ctx->defrag_sds(leaf->values[i]);
        if (ns) leaf->values[i] = ns;
    }

    /* 2) Fix anchors that alias this leaf's (possibly relocated) high key. The
     * immediate parent's anchor always aliases it; ancestors alias it only
     * while this leaf is the rightmost child along the path. Do this before
     * returning so no anchor dangles between steps. */
    if (leaf->header.num_items > 0) {
        sds new_high = leaf->values[leaf->header.num_items - 1];
        for (int d = path_depth - 1; d >= 0; d--) {
            innerNode *in = (innerNode *)path[d];
            int ci = path_idx[d];
            in->anchors[ci] = new_high;
            if (ci != in->header.num_items - 1) break;
        }
    }

    /* 3) Relocate the leaf struct, fixing the leaf chain and the parent's
     * child pointer (or the root for a single-leaf tree). */
    leafNode *newleaf;
    if ((newleaf = ctx->defrag_alloc(leaf))) {
        if (newleaf->prev)
            newleaf->prev->next = newleaf;
        else
            fbt->leftmost_leaf = newleaf;
        if (newleaf->next)
            newleaf->next->prev = newleaf;
        else
            fbt->rightmost_leaf = newleaf;
        if (path_depth > 0) {
            innerNode *parent = (innerNode *)path[path_depth - 1];
            parent->children[path_idx[path_depth - 1]] = (node *)newleaf;
        } else {
            fbt->root = (node *)newleaf;
        }
        leaf = newleaf;
    }

    /* 4) Remember where to resume via a byte-copy of the new high key. */
    if (ctx->resume_key) sdsfree(ctx->resume_key);
    ctx->resume_key = (leaf->header.num_items > 0) ? sdsdup(leaf->values[leaf->header.num_items - 1]) : sdsempty();
    ctx->have_resume = true;

    if (scanned) (*scanned) += leaf->header.num_items;
    return 1;
}

void fbtreeDefragEnd(fbtreeDefragCtx *ctx) {
    if (!ctx) return;
    if (ctx->resume_key) sdsfree(ctx->resume_key);
    zfree(ctx);
}

bool fbtreeDebugValidateMergeEnforcement(fbtreeIndex *fbt) {
    if (!fbt->root || fbt->root->is_leaf) return true;
    return validateMergeEnforcementInner((innerNode *)fbt->root, 0) < 0;
}

/* Like fbtreeDebugValidateMergeEnforcement but returns the depth of the
 * first violation, or -1 if no violations. */
int fbtreeDebugValidateMergeEnforcementDepth(fbtreeIndex *fbt) {
    if (!fbt->root || fbt->root->is_leaf) return -1;
    return validateMergeEnforcementInner((innerNode *)fbt->root, 0);
}
