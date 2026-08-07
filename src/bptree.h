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
 * NODE LAYOUT (slotted page)
 * --------------------------
 * Every node is a single fixed-size allocation of `pageSize` bytes (the only
 * exception being out-of-line "overflow" blobs for very long keys). Leaves and
 * inner nodes share the same slotted-page layout:
 *
 *   [ header ][ slot[0..n-1] (u16 offsets) ] --- free --- [ packed cells + prefix ]
 *
 * The slot array grows forward from just after the header and is kept sorted by
 * key; the cells (and the node's common-prefix bytes) are packed backward from
 * the end of the page. Each slot is a page-relative byte offset to a cell, so
 * cells can be compacted without disturbing slot meaning. A cell is:
 *
 *   [ u32 suffixLen ][ u8 flags ][ inline suffix bytes ][ blob* ? ][ payload ]
 *
 * where payload is the `void *` value in a leaf and the child `bptNode *` in an
 * inner node. Keys whose suffix would exceed an inline threshold set the
 * overflow flag: only a prefix is stored inline plus a pointer to a heap blob
 * holding the tail, so there is no maximum key length.
 *
 * LEAF CHAINING
 * -------------
 * Leaves are doubly linked (next/prev), so both forward and backward range
 * scans advance in O(1) per step without a parent stack.
 */

#ifndef BPTREE_H
#define BPTREE_H

#include <stdint.h>
#include <stddef.h>

/* Default / bounds for the page size. pageSize is clamped into this range and
 * rounded up to a power of two by bptNewEx(). */
#define BPT_PAGE_MIN 256
#define BPT_PAGE_MAX 65536
#define BPT_PAGE_DEFAULT 512

/* A single node (page) of the tree. The struct only describes the fixed
 * header; the slot array and the packed cell heap follow in the same
 * allocation of `pageSize` bytes. Fields are documented in bptree.c. */
typedef struct bptNode {
    uint32_t isleaf;      /* 1 if leaf, 0 if inner node. */
    uint32_t nkeys;       /* Number of slots (keys/separators) in this node. */
    uint32_t heapStart;   /* Lowest used byte offset; cells live in
                             [heapStart, pageSize). */
    uint32_t deadBytes;   /* Reclaimable bytes in the cell heap. */
    uint32_t prefixOff;   /* Offset of the node common-prefix bytes. */
    uint32_t prefixLen;   /* Length of the node common prefix. */
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
    uint32_t pageSize;  /* Node size in bytes (power of two). */
    uint32_t maxInline; /* Max inline suffix bytes before a cell goes overflow. */
    uint32_t height;    /* Number of levels (1 == root is a leaf). */
} bptree;

/* Iterator over a bptree. See bptStart()/bptSeek()/bptNext()/bptPrev().
 *
 * A successful bptSeek() materializes the current key/value into the iterator
 * (so bptCompare() is valid immediately after a seek, as with rax), and the
 * first bptNext()/bptPrev() returns that element without touching tree pages.
 * Without BPT_ITER_SAFE, mutating the tree while iterating is forbidden. With
 * it, iteration re-seeks by key after any mutation (slower); the flag is
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
bptree *bptNew(void);
/* Create a tree with the given page size. If alloc_size is non-NULL the tree
 * keeps *alloc_size updated with the bytes it holds (pages + overflow blobs +
 * the header), like raxNewEx()'s alloc_size argument. */
bptree *bptNewEx(uint32_t pageSize, size_t *alloc_size);
void bptFree(bptree *bt);
void bptFreeWithCallback(bptree *bt, void (*free_cb)(void *));
void bptFreeWithCbAndContext(bptree *bt,
                             void (*free_cb)(void *item, void *ctx),
                             void *ctx);
uint64_t bptSize(bptree *bt);
uint64_t bptNumNodes(bptree *bt);
void bptDefrag(bptree **btref, bptReallocFn fn);

/* Keys are limited to 4GB (lengths are stored as uint32_t). */
int bptInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old);
int bptTryInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old);
/* Append-optimized insert for monotonically increasing keys (stream IDs,
 * PELs). When the key is strictly greater than the current maximum it is
 * placed at the end of the tail leaf in O(1), or starts a fresh rightmost
 * leaf when the tail is full ("append split", leaving the old tail ~100%
 * packed). Any other key falls back to a normal ordered insert, so the
 * semantics are identical to bptTryInsert() for every input: returns 1 when
 * inserted, 0 when the key already existed (*old is set and the stored value
 * is NOT overwritten). */
int bptAppend(bptree *bt, unsigned char *s, size_t len, void *data, void **old);
int bptRemove(bptree *bt, unsigned char *s, size_t len, void **old);
int bptFind(bptree *bt, unsigned char *s, size_t len, void **value);

/* Single-walk upsert, mirroring rax's raxFindLink()/raxInsertAt(): locate a key
 * once, then commit an insert at the located position without re-walking in the
 * common case. The link captures where the key is (or would go). It is a
 * lightweight position, NOT a stable handle: any intervening mutation of the
 * SAME tree (insert, remove, or bptDefrag) invalidates it, so a bptFindLink()
 * must be followed immediately by at most one bptInsertAt() with no other
 * mutation of that tree in between. Mutating a different tree is fine. */
typedef struct bptNodeLink {
    bptNode *leaf;    /* Leaf holding the key, or where it would be inserted. */
    uint32_t pos;     /* Exact slot when found, else lower-bound insert index. */
    int found;        /* 1 if the key is present. */
    uint64_t version; /* bt->version observed at find time (staleness guard). */
} bptNodeLink;

/* Returns 1 and (if value != NULL) writes the current value when the key is
 * present; returns 0 otherwise. In both cases *link is filled so a subsequent
 * bptInsertAt() can commit without another full descent. */
int bptFindLink(bptree *bt, unsigned char *s, size_t len, void **value, bptNodeLink *link);
/* Commit against a link from an immediately preceding bptFindLink() on the same
 * (bt,s,len). If the key existed, overwrites its value (returning *old) and
 * returns 0; otherwise inserts and returns 1. */
int bptInsertAt(bptree *bt, unsigned char *s, size_t len, void *data, void **old, bptNodeLink *link);

void bptStart(bptIterator *it, bptree *bt);
int bptSeek(bptIterator *it, const char *op, unsigned char *ele, size_t len);
int bptNext(bptIterator *it);
int bptPrev(bptIterator *it);
int bptCompare(bptIterator *it, const char *op, unsigned char *key, size_t key_len);
int bptEOF(bptIterator *it);
void bptIteratorSetData(bptIterator *it, void *data);
void bptStop(bptIterator *it);
void bptShow(bptree *bt);

#ifdef REDIS_TEST
int bptTest(int argc, char *argv[], int flags);
#endif

#endif /* BPTREE_H */
