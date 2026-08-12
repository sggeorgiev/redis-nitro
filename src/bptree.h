/* Bptree -- A prefix-compressed B+tree implementation.
 *
 * Copyright (c) 2011-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 *
 * PREFIX B+TREE
 * -------------
 * An in-memory ordered map keyed by variable-length, binary-safe byte strings
 * (memcmp / lexicographic order), mapping each unique key to an opaque
 * `void *` value. The public API deliberately mirrors rax.h so it can be used
 * as an ordered dictionary with range scans.
 *
 * The tree is a classic B+tree (all values live in the leaves; inner nodes
 * hold only routing separators) with two forms of prefix compression:
 *
 *   1. Per-node head compression. Every node factors out the longest common
 *      prefix shared by all of its keys and stores it once; each cell keeps
 *      only the remaining suffix.
 *
 *   2. Shortest-separator truncation. When a leaf splits, the separator copied
 *      up to the parent is the right half's first key truncated to the shortest
 *      byte string that still routes correctly. Inner nodes therefore store
 *      short routing strings rather than full keys, keeping fanout high.
 *
 * Trees are always ranked: inner nodes maintain per-child subtree element
 * counts so bptRankOf() and bptSelect() run in O(log N).
 *
 * NODE LAYOUT (slotted page)
 * --------------------------
 * Every node is a single allocation that is right-sized to its contents and
 * resized on each mutation, exactly like a rax node (the only other allocations
 * are out-of-line "overflow" blobs for very long keys). `pageSize` is NOT the
 * allocation size: it is the *maximum* node size and the split threshold, so a
 * node holds between one entry and `pageSize` bytes and typically far less.
 * Leaves and inner nodes share the same slotted-page layout:
 *
 *   [ header ][ prefix ][ packed cells --> ] -- free -- [ <-- dir[0..n-1] ]
 *                                                         ^ end = cap
 *
 * The node's common-prefix bytes sit immediately after the header; cells are
 * packed forward from the (pointer-aligned) end of the prefix, growing upward.
 * The directory lives at the very end of the allocation and grows backward, so
 * it -- and only it -- moves when the node is resized, while the cell offsets it
 * stores stay put (that is what makes resizing cheap). It is kept sorted by key,
 * so cells can be compacted without disturbing its meaning. A cell is:
 *
 *   [ u32 suffixLen ][ u8 flags ][ inline suffix bytes ][ blob* ? ][ payload ]
 *
 * where payload is the `void *` value in a leaf and the child `bptNode *` in an
 * inner node. Keys whose suffix would exceed an inline threshold set the
 * overflow flag: only a prefix is stored inline plus a pointer to a heap blob
 * holding the tail, so there is no maximum key length.
 *
 * KEY DIRECTORY
 * -------------
 * A directory entry is fixed width -- the head of the key's suffix (zero padded)
 * plus the u16 offset of its cell -- which makes the region a dense array that
 * a binary search can walk without leaving one or two cache lines. Comparing
 * the padded heads reproduces the verdict the full keys would give unless they
 * are equal, so a probe reads the variable-length cell (a random offset inside
 * the cell region, almost always a separate line) only for keys that agree on
 * their whole head. The heads are a pure cache of the cell bytes: compaction
 * and resizing move offsets around but never touch them, and anything that
 * changes a key -- including a change of the node prefix, which shifts where
 * every suffix begins -- rebuilds the node and its directory from scratch.
 * Inner-node directory entries additionally store a per-child subtree element
 * count (ranked maintenance).
 *
 * RESIZING
 * --------
 * A node grows to the exact bytes its content needs (grabbing the whole
 * allocator size class so subsequent inserts stay in place until the next class
 * boundary) and shrinks back on every removal, skipping the realloc whenever the
 * allocator's usable size already covers the target. A resize can move the node,
 * so each mutation republishes the possibly-moved node into its parent slot (or
 * the root), the leaf chain and the cached tail. Right-sizing also cancels most
 * of the memory cost of the merge-only deletion policy below: an underfull node
 * now only occupies the bytes of the content it still holds.
 *
 * LEAF CHAINING
 * -------------
 * Leaves are doubly linked (next/prev), so both forward and backward range
 * scans advance in O(1) per step without a parent stack.
 *
 * DELETION
 * --------
 * Merge only: a node that drops below the minimum fill is fused with an
 * adjacent sibling when the two fit in a single page (cascading upward, and
 * collapsing the root when it is left with a single child). Nothing is
 * redistributed between siblings, so when they do not fit the node just stays
 * underfull until a later merge or insert refills it. The tree therefore
 * guarantees no minimum node occupancy, only correct ordering and uniform leaf
 * depth.
 */

#ifndef BPTREE_H
#define BPTREE_H

#include <stdint.h>
#include <stddef.h>

/* Default / bounds for the page size -- the maximum node size and split
 * threshold, NOT the allocation size (nodes are right-sized to their content).
 * pageSize is clamped into this range and rounded up to a power of two by
 * bptNewRanked().
 *
 * The default trades tree height for node size, which pays off only because the
 * key directory keeps a search inside a couple of cache lines however many keys
 * a node holds: a wide node lowers the height (fewer dependent cache misses per
 * lookup), scans cross fewer pages, and the per-node header is amortized over
 * more keys. Small trees are unaffected -- a node is only ever as large as its
 * contents. */
#define BPT_PAGE_MIN 256
#define BPT_PAGE_MAX 65536
#define BPT_PAGE_DEFAULT 4096

/* A single node of the tree. The struct only describes the fixed header; the
 * common prefix, the packed cell heap and (at the end) the key directory follow in
 * the same right-sized allocation of `cap` bytes. Fields are documented in
 * bptree.c. */
typedef struct bptNode {
    uint32_t isleaf;      /* 1 if leaf, 0 if inner node. */
    uint32_t nkeys;       /* Number of keys/separators in this node. */
    uint32_t heapEnd;     /* First free byte past the last cell; cells live in
                             [dataStart, heapEnd) and grow upward. */
    uint32_t deadBytes;   /* Reclaimable bytes in the cell heap. */
    uint32_t cap;         /* Current allocation size in bytes (a power-of-two
                             page, or less once the node is right-sized). The
                             directory ends at this offset. */
    uint32_t prefixLen;   /* Length of the node common prefix (stored right
                             after the header). */
    uint32_t leftCount;   /* Inner only: number of elements in the subtree
                             rooted at `leftmost` (child 0). Unused on leaves. */
    struct bptNode *next; /* Leaf only: next leaf in key order (else NULL). */
    struct bptNode *prev; /* Leaf only: previous leaf in key order (else NULL). */
    struct bptNode *leftmost; /* Inner only: child for keys < separator 0. */
} bptNode;

typedef struct bptree {
    bptNode *root;      /* Root node (a leaf when the tree has height 1). */
    bptNode *tail;      /* Rightmost leaf (holds the maximum key). Kept up to
                         * date across split/merge/defrag so bptSeek("$") and
                         * bptAppend() are O(1) in the common case. */
    uint64_t numele;    /* Number of elements (keys) stored. */
    uint64_t numnodes;  /* Number of nodes (pages) currently allocated. */
    uint64_t version;   /* Bumped on every structural mutation; used by safe
                         * iterators to detect that they must re-seek. */
    size_t *alloc_size; /* If non-NULL, the tree accounts its used memory
                         * (pages, overflow blobs and the header) at this
                         * location. Mirrors rax's alloc_size hook. */
    uint32_t pageSize;  /* Maximum node size / split threshold in bytes (power
                         * of two); nodes are right-sized below this. */
    uint32_t maxInline; /* Max inline suffix bytes before a cell goes overflow. */
    uint32_t height;    /* Number of levels (1 == root is a leaf). */
} bptree;

/* Iterator over a bptree. See bptStart()/bptSeek()/bptNext()/bptPrev().
 *
 * A successful bptSeek() materializes the current key/value into the iterator,
 * and the first bptNext()/bptPrev() returns that element without touching tree
 * pages. Without BPT_ITER_SAFE, mutating the tree while iterating is forbidden.
 * With it, iteration re-seeks by key after any mutation (slower); the flag is
 * preserved across bptSeek() calls. */
#define BPT_ITER_STATIC_LEN 128
#define BPT_ITER_JUST_SEEKED (1<<0) /* Return the current element on the first
                                       step, then clear the flag. */
#define BPT_ITER_EOF (1<<1)         /* No more elements in the chosen direction. */
#define BPT_ITER_SAFE (1<<2)        /* Tolerate mutations between steps by
                                       re-seeking (slower). */
typedef struct bptIterator {
    int flags;
    bptree *bt;
    unsigned char *key;   /* Current key (materialized full key). */
    size_t key_len;       /* Current key length. */
    size_t key_max;       /* Capacity of the key buffer. */
    unsigned char key_static_string[BPT_ITER_STATIC_LEN];
    void *data;           /* Value associated with the current key. */
    bptNode *node;        /* Current leaf. */
    int idx;              /* Slot index within the current leaf. */
    uint64_t version;     /* bt->version observed at last positioning. */
} bptIterator;

/* Optional relocation callback used by bptDefrag(): given a pointer previously
 * returned by the tree's allocator, it either returns a new pointer (the block
 * was moved and its bytes copied, the old block freed) or NULL (left in place).
 * This mirrors Redis's activeDefragAlloc() contract. */
typedef void *(*bptReallocFn)(void *ptr);

/* Exported API. */
/* Create a ranked tree with the given page size. If alloc_size is non-NULL the
 * tree keeps *alloc_size updated with the bytes it holds (pages + overflow
 * blobs + the header), like raxNewEx()'s alloc_size argument. Inner nodes
 * maintain per-child subtree element counts so bptRankOf() and bptSelect()
 * run in O(log N). */
bptree *bptNewRanked(uint32_t pageSize, size_t *alloc_size);
void bptFree(bptree *bt);
uint64_t bptSize(bptree *bt);
uint64_t bptNumNodes(bptree *bt);
void bptDefrag(bptree **btref, bptReallocFn fn);
/* Relocate one leaf and the inner-node path leading to it. `cursor` is the
 * rank at which the next step starts; it is reset to zero when done. Returns 1
 * while more leaves remain, 0 when the pass is complete. */
int bptDefragStep(bptree **btref, uint64_t *cursor, bptReallocFn fn);

/* Keys are limited to 4GB (lengths are stored as uint32_t). */
int bptInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old);
/* Append-optimized insert for monotonically increasing keys (stream IDs,
 * PELs). When the key is strictly greater than the current maximum it is
 * placed at the end of the tail leaf in O(1), or starts a fresh rightmost
 * leaf when the tail is full ("append split", leaving the old tail ~100%
 * packed). Any other key falls back to a normal ordered insert, so the
 * semantics are identical to a non-overwrite insert for every input: returns 1
 * when inserted, 0 when the key already existed (*old is set and the stored
 * value is NOT overwritten). */
int bptAppend(bptree *bt, unsigned char *s, size_t len, void *data, void **old);
int bptRemove(bptree *bt, unsigned char *s, size_t len, void **old);

/* Rank queries.
 *
 * bptRankOf(): writes into *rank the number of keys strictly less than (s,len)
 * -- i.e. the 0-based position the key has (or would have) in key order -- and
 * returns 1 if the key is actually present, 0 otherwise. O(log N).
 *
 * bptSelect(): positions the iterator on the key at 0-based `index` (as if by a
 * seek), so the first bptNext()/bptPrev() returns it. Returns 1 on success, or
 * 0 (leaving the iterator at EOF) when index >= the number of elements. */
int bptRankOf(bptree *bt, unsigned char *s, size_t len, uint64_t *rank);
int bptSelect(bptIterator *it, uint64_t index);

void bptStart(bptIterator *it, bptree *bt);
int bptSeek(bptIterator *it, const char *op, unsigned char *ele, size_t len);
int bptNext(bptIterator *it);
int bptPrev(bptIterator *it);
int bptEOF(bptIterator *it);
void bptStop(bptIterator *it);

/* Invoke `cb` once per backing allocation of the tree (every node page and
 * every overflow blob), passing the block pointer, its size in bytes and the
 * caller's `ctx`. Used by dismissObject() to madvise(MADV_DONTNEED) cold pages
 * of a large sorted set. */
void bptForEachPage(bptree *bt,
                    void (*cb)(void *page, size_t size, void *ctx),
                    void *ctx);

#ifdef REDIS_TEST
int bptTest(int argc, char *argv[], int flags);
#endif

#endif /* BPTREE_H */
