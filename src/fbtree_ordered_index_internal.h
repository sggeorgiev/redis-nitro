/* Internal header for fbtree_ordered_index — struct layouts and constants.
 * Include this ONLY in fbtree_ordered_index.c and its unit tests.
 * Production code outside the fbtree module should use the opaque public
 * API in fbtree_ordered_index.h. */

#ifndef FBTREE_ORDERED_INDEX_INTERNAL_H
#define FBTREE_ORDERED_INDEX_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fbtree_ordered_index.h"
#include "sds.h"

/* Architecture-specific constants for node sizing.
 * 64-bit: optimized for 8-byte pointers, targets 1792-byte innerNode (jemalloc size class)
 * 32-bit: uses same logical fanout, smaller nodes due to 4-byte pointers */
#define NODE_SIZE 61

#if SIZE_MAX == UINT64_MAX   /* 64-bit */
#define EMBED_PREFIX_LEN 254 /* Tuned so innerNode fits in 2048-byte jemalloc size class */
#elif SIZE_MAX == UINT32_MAX /* 32-bit */
#define EMBED_PREFIX_LEN 30  /* Tuned to fit innerNode exactly in 1024-byte jemalloc size class */
#endif

#define MIN_FILL (NODE_SIZE / 4) /* Minimum items before node underflows */
#define FEATURE_SIZE 4
#define FEATURE_ROW_SIZE 64 /* size of cache line */
/* SIMD comparison instructions (e.g., _mm256_cmpgt_epi8) operate on signed
 * bytes, but our feature bytes are unsigned [0,255]. XOR with 0x80 maps
 * unsigned order to signed order: 0→-128, 127→-1, 128→0, 255→127.
 * Features are stored pre-biased; search targets are biased at lookup time. */
#define FEATURE_BIAS 0x80
#define MAX_TREE_DEPTH 16   /* NODE_SIZE=61, so depth 6 handles >61^6 = ~51 billion elements */

/* Common header for all node types */
typedef struct {
    bool is_leaf;
    uint8_t num_items;
} node;

typedef struct {
    node header;
    char embedded_prefix[EMBED_PREFIX_LEN]; /* Short prefix inline; long prefix stores char* at aligned offset */
    size_t prefix_len;                      /* Common prefix length of this node's anchor values (NOT all subtree keys).
                                             * Features are the 4 bytes at offset prefix_len in each anchor, so the
                                             * prefix skips identical leading bytes to keep features discriminating.
                                             * Note: this can exceed a child's prefix_len — see updateCommonPrefix. */
    char features[FEATURE_SIZE][FEATURE_ROW_SIZE];
    sds anchors[NODE_SIZE]; /* High keys of children (rightmost value in each child's subtree) */
    node *children[NODE_SIZE];
    size_t child_sizes[NODE_SIZE];      /* subtree element counts for rank queries */
    uint8_t child_num_items[NODE_SIZE]; /* direct item count of each child */
} innerNode;

typedef struct leafNode {
    node header;
    struct leafNode *prev;
    struct leafNode *next;
    sds values[NODE_SIZE];
} leafNode;

struct fbtreeIndex {
    node *root;
    leafNode *leftmost_leaf;  /* Cache for fast-path prepend */
    leafNode *rightmost_leaf; /* Cache for fast-path append */
};

/* Architecture-specific size assertions */
#if SIZE_MAX == UINT64_MAX /* 64-bit */
static_assert(sizeof(innerNode) == 2048, "64-bit innerNode should fit in 2048-byte jemalloc size class");
static_assert(sizeof(leafNode) == 512, "64-bit leafNode should fit perfectly in jemalloc size class");
#elif SIZE_MAX == UINT32_MAX /* 32-bit */
static_assert(sizeof(innerNode) == 1024, "32-bit innerNode should fit exactly in 1024-byte jemalloc size class");
static_assert(sizeof(leafNode) == 256, "32-bit leafNode should fit perfectly in jemalloc size class");
#endif
static_assert(NODE_SIZE <= FEATURE_ROW_SIZE, "NODE_SIZE must fit in feature row");

/* Get low_key (minimum) from leaf node - leaves are always kept sorted */
static inline sds leafNodeLowKey(leafNode *leaf) {
    return (leaf->header.num_items == 0) ? NULL : leaf->values[0];
}

/* Get high_key pointer from leaf node */
static inline sds leafNodeHighKey(leafNode *leaf) {
    return (leaf->header.num_items == 0) ? NULL : leaf->values[leaf->header.num_items - 1];
}

/* Get high_key (maximum anchor) from any node type */
static inline sds nodeHighKey(node *n) {
    if (n->is_leaf) {
        return leafNodeHighKey((leafNode *)n);
    }
    innerNode *inner = (innerNode *)n;
    return (inner->header.num_items == 0) ? NULL : inner->anchors[inner->header.num_items - 1];
}

/* Get feature byte j from string s, biased for SIMD signed comparison.
 * Returns 0 ^ FEATURE_BIAS if the string is shorter than prefix_len + j. */
static inline char getFeatureByte(const_sds s, size_t prefix_len, int j) {
    size_t idx = prefix_len + j;
    unsigned char raw = (idx < sdslen(s)) ? (unsigned char)s[idx] : 0;
    return (char)(raw ^ FEATURE_BIAS);
}

/* Long prefix: when prefix_len > EMBED_PREFIX_LEN, pointer stored at aligned offset within embedded_prefix. */
#define LONG_PREFIX_PTR_OFFSET \
    ((sizeof(void *) - (offsetof(innerNode, embedded_prefix) % sizeof(void *))) % sizeof(void *))
static_assert(EMBED_PREFIX_LEN >= LONG_PREFIX_PTR_OFFSET + sizeof(char *), "embedded_prefix must fit aligned pointer");

/* Debug functions — test-only, not part of the public API. */
bool fbtreeDebugValidate(fbtreeIndex *fbt, bool verbose);
bool fbtreeDebugValidateMergeEnforcement(fbtreeIndex *fbt);
int fbtreeDebugValidateMergeEnforcementDepth(fbtreeIndex *fbt);

#endif /* FBTREE_ORDERED_INDEX_INTERNAL_H */
