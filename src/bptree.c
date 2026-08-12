/* Bptree -- A prefix-compressed B+tree implementation.
 *
 * Copyright (c) 2011-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * See bptree.h for the high-level design (slotted pages, per-node head
 * compression, shortest-separator truncation, overflow blobs).
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdint.h>
#include "bptree.h"
#include "redisassert.h"

#ifndef BPT_MALLOC_INCLUDE
#define BPT_MALLOC_INCLUDE "bptree_malloc.h"
#endif
#include BPT_MALLOC_INCLUDE

/* ------------------------------------------------------------------------- */
/* On-page constants and small helpers                                       */
/* ------------------------------------------------------------------------- */

#define BPT_HDR_SIZE (sizeof(bptNode))

/* Bytes of each key's suffix replicated inline in its directory entry. A wider
 * head settles more comparisons without reading a cell, but every entry is paid
 * for by every key, and past two bytes the lost fanout costs more than the
 * saved cell reads for the key shapes measured (stream IDs, short strings).
 * Must be even, so the entry carries no padding. */
#ifndef BPT_DIR_HEAD
#define BPT_DIR_HEAD 2
#endif

/* Cell layout: [u32 suffixLen][u8 flags][inline suffix][blob* if overflow][payload].
 * The 3 non-inline, non-payload bytes of a cell header are 4 (len) + 1 (flags). */
#define BPT_CELL_HDR 5
#define BPT_PTR_SIZE ((uint32_t)sizeof(void *))
#define BPT_CFLAG_OVERFLOW 0x01

/* Smallest allocation a node ever takes: an empty node whose cell region starts
 * at the pointer-aligned header end, with no cells or directory yet. */
#define BPT_MIN_CAP ((((uint32_t)sizeof(bptNode)) + (BPT_PTR_SIZE - 1)) & \
                     ~((uint32_t)BPT_PTR_SIZE - 1))

/* Deletion threshold: a non-root node underflows when its live byte usage drops
 * below this fraction (in percent) of the page. An underflowing node is merged
 * with a sibling when the two fit in one page, and otherwise simply left
 * underfull -- see bptRemove(). */
#define BPT_MIN_FILL_PCT 40

/* Maximum tree height we support in the traversal stack. With a minimum fanout
 * of 2 this bounds the number of storable keys well beyond any practical use. */
#define BPT_MAX_HEIGHT 128

/* Overflow blob holding the tail of a long key's suffix (the bytes past the
 * inline portion). Owned by exactly one cell. */
typedef struct bptBlob {
    uint32_t len;            /* Number of tail bytes. */
    unsigned char data[];
} bptBlob;

/* An extracted (materialized) entry: a full key plus its payload. Used by the
 * structural operations (split/merge/prefix-rewrite) that rebuild nodes
 * from scratch. `key` is heap-owned by the extracting caller. */
typedef struct bptEnt {
    unsigned char *key;
    uint32_t keylen;
    void *payload;           /* value (leaf) or child bptNode* (inner). */
} bptEnt;

/* One directory entry per key, in key order: the head of the key's suffix
 * (zero padded) followed by the offset of its cell. Searches binary-search the
 * directory and compare `head` only, so a probe stays inside the dense
 * directory instead of jumping to a cell; the cell is read only when the heads
 * tie (see bptNodeSearch). */
typedef struct bptDirEnt {
    unsigned char head[BPT_DIR_HEAD];
    uint16_t off;            /* node-relative byte offset of the cell. */
} bptDirEnt;

#define BPT_DIR_SIZE ((uint32_t)sizeof(bptDirEnt))

/* Extra bytes appended to each directory entry of an inner node: the number of
 * elements in the subtree rooted at the entry's right child. Leaves do not
 * carry it, so their entries stay BPT_DIR_SIZE. */
#define BPT_DIR_CNT ((uint32_t)sizeof(uint32_t))
#define BPT_DIR_SIZE_RANKED (BPT_DIR_SIZE + BPT_DIR_CNT)

extern char __bptDirEntUnpadded[sizeof(bptDirEnt) == BPT_DIR_HEAD + 2 ? 1 : -1];

/* Directory entry stride for `n`: inner nodes store a per-child count after
 * {head,off}; leaves use the bare entry. */
static inline uint32_t bptStride(const bptree *bt, const bptNode *n) {
    (void)bt;
    return !n->isleaf ? BPT_DIR_SIZE_RANKED : BPT_DIR_SIZE;
}
/* Same, but for a node type known before the node exists (build/fit sizing). */
static inline uint32_t bptStrideFor(const bptree *bt, int isleaf) {
    (void)bt;
    return !isleaf ? BPT_DIR_SIZE_RANKED : BPT_DIR_SIZE;
}

/* The directory occupies the last nkeys*stride bytes of the allocation and
 * grows backward from `cap`. It therefore moves when the node is resized, while
 * the cell offsets it stores do not -- that is the point of the forward layout.
 * Entries have a variable stride, so they are addressed one at a time rather
 * than via array indexing. */
static inline bptDirEnt *bptDeAt(const bptree *bt, bptNode *n, uint32_t i) {
    uint32_t st = bptStride(bt, n);
    return (bptDirEnt *)((unsigned char *)n + n->cap - n->nkeys * st + i * st);
}
/* Directory-entry address when the entry count is not yet reflected in nkeys
 * (used while (re)building a node). */
static inline bptDirEnt *bptDeAtN(const bptree *bt, bptNode *n, uint32_t count,
                                  uint32_t i) {
    uint32_t st = bptStride(bt, n);
    return (bptDirEnt *)((unsigned char *)n + n->cap - count * st + i * st);
}
/* Offset of the cell holding key `i`. */
static inline uint32_t bptCellOff(const bptree *bt, bptNode *n, uint32_t i) {
    return bptDeAt(bt, n, i)->off;
}
/* Per-child subtree element count (inner nodes). Child 0 is the leftmost
 * child (count kept in the header); child k+1 is entry k's right child
 * (count kept just after that entry). */
static inline uint32_t bptChildCnt(const bptree *bt, bptNode *n, uint32_t cidx) {
    if (cidx == 0) return n->leftCount;
    uint32_t v;
    memcpy(&v, (unsigned char *)bptDeAt(bt, n, cidx - 1) + BPT_DIR_SIZE, BPT_DIR_CNT);
    return v;
}
static inline void bptSetChildCnt(const bptree *bt, bptNode *n, uint32_t cidx,
                                  uint32_t v) {
    if (cidx == 0) { n->leftCount = v; return; }
    memcpy((unsigned char *)bptDeAt(bt, n, cidx - 1) + BPT_DIR_SIZE, &v, BPT_DIR_CNT);
}
/* Total number of elements in the subtree rooted at `n`: nkeys for a leaf,
 * else the sum of the children's cached counts. O(fanout). */
static inline uint32_t bptSubtreeCount(const bptree *bt, bptNode *n) {
    if (n->isleaf) return n->nkeys;
    uint32_t c = n->leftCount;
    for (uint32_t i = 0; i < n->nkeys; i++) c += bptChildCnt(bt, n, i + 1);
    return c;
}
/* Copy the first BPT_DIR_HEAD suffix bytes into a directory head, zero padding
 * a shorter suffix. Comparing two heads with memcmp() yields either the same
 * verdict the full keys would give, or equality (meaning "undecided"): a
 * padding zero can only stand where one suffix ended, i.e. where it is a
 * prefix of the other and therefore the smaller of the two. */
static inline void bptDirHead(unsigned char *head, const unsigned char *s,
                              uint32_t slen) {
    uint32_t k = slen < BPT_DIR_HEAD ? slen : BPT_DIR_HEAD;
    memcpy(head, s, k);
    if (k < BPT_DIR_HEAD) memset(head + k, 0, BPT_DIR_HEAD - k);
}
static inline unsigned char *bptRaw(bptNode *n, uint32_t off) {
    return (unsigned char *)n + off;
}
/* The node common prefix sits immediately after the header. */
static inline unsigned char *bptPrefix(bptNode *n) {
    return (unsigned char *)n + BPT_HDR_SIZE;
}
/* Offset of the first cell: past the header and prefix, rounded up to a pointer
 * boundary so the payload/blob pointers at cell ends stay aligned. */
static inline uint32_t bptDataStart(bptNode *n) {
    uint32_t off = (uint32_t)BPT_HDR_SIZE + n->prefixLen;
    return (off + (BPT_PTR_SIZE - 1)) & ~((uint32_t)BPT_PTR_SIZE - 1);
}

static inline uint32_t bptCellSuffixLen(bptNode *n, uint32_t off) {
    uint32_t v;
    memcpy(&v, bptRaw(n, off), 4);
    return v;
}
static inline int bptCellOverflow(bptNode *n, uint32_t off) {
    return bptRaw(n, off)[4] & BPT_CFLAG_OVERFLOW;
}
/* Number of suffix bytes stored inline in the cell. */
static inline uint32_t bptCellInlineLen(bptree *bt, bptNode *n, uint32_t off) {
    if (bptCellOverflow(n, off)) return bt->maxInline;
    return bptCellSuffixLen(n, off);
}
static inline unsigned char *bptCellInline(bptNode *n, uint32_t off) {
    return bptRaw(n, off) + BPT_CELL_HDR;
}

/* Total on-page byte size of a cell given its suffix length and overflow-ness.
 * The trailing payload pointer (a value or a child bptNode*) occupies the last
 * BPT_PTR_SIZE bytes; for overflow cells the blob pointer occupies the
 * BPT_PTR_SIZE bytes just before it. The size is rounded up to a multiple of
 * BPT_PTR_SIZE so that, since every cell starts at a pointer-aligned page
 * offset (see bptBuildNode/bptCompact), both pointers are stored aligned. That
 * keeps them visible to a conservative pointer scanner (LeakSanitizer, which by
 * default only inspects aligned words) and mirrors rax's padded node layout. */
static inline uint32_t bptCellSize(bptree *bt, uint32_t suffixLen, int overflow) {
    uint32_t ilen = overflow ? bt->maxInline : suffixLen;
    uint32_t raw = BPT_CELL_HDR + ilen + (overflow ? BPT_PTR_SIZE : 0) + BPT_PTR_SIZE;
    return (raw + (BPT_PTR_SIZE - 1)) & ~(BPT_PTR_SIZE - 1);
}

/* Overflow blob pointer: the BPT_PTR_SIZE bytes immediately before the payload. */
static inline bptBlob *bptCellBlob(bptree *bt, bptNode *n, uint32_t off) {
    uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off), 1);
    bptBlob *b;
    memcpy(&b, bptRaw(n, off) + csize - 2 * BPT_PTR_SIZE, BPT_PTR_SIZE);
    return b;
}
static inline void bptCellSetBlob(bptree *bt, bptNode *n, uint32_t off, bptBlob *b) {
    uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off), 1);
    memcpy(bptRaw(n, off) + csize - 2 * BPT_PTR_SIZE, &b, BPT_PTR_SIZE);
}
/* Payload pointer: the last BPT_PTR_SIZE bytes of the cell (pointer aligned). */
static inline uint32_t bptCellPayloadOff(bptree *bt, bptNode *n, uint32_t off) {
    uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off), bptCellOverflow(n, off));
    return off + csize - BPT_PTR_SIZE;
}
static inline void *bptCellPayload(bptree *bt, bptNode *n, uint32_t off) {
    void *p;
    memcpy(&p, bptRaw(n, bptCellPayloadOff(bt, n, off)), BPT_PTR_SIZE);
    return p;
}
static inline void bptCellSetPayload(bptree *bt, bptNode *n, uint32_t off, void *p) {
    memcpy(bptRaw(n, bptCellPayloadOff(bt, n, off)), &p, BPT_PTR_SIZE);
}

/* Free bytes available for a new (directory entry + cell) without compaction:
 * the gap between the top of the cell region and the bottom of the directory. */
static inline uint32_t bptFreeBytes(bptree *bt, bptNode *n) {
    return (n->cap - n->nkeys * bptStride(bt, n)) - n->heapEnd;
}
/* Live bytes a node occupies (header + prefix + live cells + directory),
 * independent of the current allocation size. This is also exactly the size a
 * compacted node needs, so it doubles as the target for reallocation. */
static inline uint32_t bptUsedBytes(bptree *bt, bptNode *n) {
    return n->heapEnd - n->deadBytes + n->nkeys * bptStride(bt, n);
}

/* Longest common prefix length of two byte strings. */
static uint32_t bptLcp(const unsigned char *a, uint32_t alen,
                       const unsigned char *b, uint32_t blen) {
    uint32_t n = alen < blen ? alen : blen, i = 0;
    while (i < n && a[i] == b[i]) i++;
    return i;
}

/* memcmp with lexicographic tie-break by length. */
static int bptKeyCmp(const unsigned char *a, uint32_t alen,
                     const unsigned char *b, uint32_t blen) {
    uint32_t n = alen < blen ? alen : blen;
    int r = n ? memcmp(a, b, n) : 0;
    if (r) return r;
    if (alen < blen) return -1;
    if (alen > blen) return 1;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Memory accounting                                                         */
/* ------------------------------------------------------------------------- */

/* Allocate `size` bytes, adding the usable size to the tree's memory counter
 * (if any). Used for every long-lived tree allocation: pages, overflow blobs
 * and the tree header itself. Ephemeral scratch (extracted entry arrays,
 * compaction buffers, materialized keys) is NOT accounted. */
static inline void *bptMallocAccounted(bptree *bt, size_t size) {
    size_t usable;
    void *p = bpt_malloc_usable(size, &usable);
    if (p && bt->alloc_size) *bt->alloc_size += usable;
    return p;
}

/* Free a pointer previously obtained via bptMallocAccounted(), subtracting its usable
 * size from the tree's memory counter (if any). */
static inline void bptFreeAccounted(bptree *bt, void *p) {
    if (!p) return;
    if (bt->alloc_size) *bt->alloc_size -= bpt_malloc_usable_size(p);
    bpt_free(p);
}

/* ------------------------------------------------------------------------- */
/* Node allocation                                                           */
/* ------------------------------------------------------------------------- */

static bptNode *bptNewNode(bptree *bt, int isleaf) {
    size_t usable;
    bptNode *n = bpt_malloc_usable(BPT_MIN_CAP, &usable);
    if (!n) return NULL;
    if (bt->alloc_size) *bt->alloc_size += usable;
    bt->numnodes++;
    n->isleaf = isleaf ? 1 : 0;
    n->nkeys = 0;
    uint32_t cap = (uint32_t)(usable & ~((size_t)BPT_PTR_SIZE - 1));
    if (cap > bt->pageSize) cap = bt->pageSize;
    n->cap = cap;
    n->prefixLen = 0;
    n->leftCount = 0;
    n->heapEnd = bptDataStart(n);
    n->deadBytes = 0;
    n->next = NULL;
    n->prev = NULL;
    n->leftmost = NULL;
    return n;
}

/* Free the overflow blobs referenced by a node's cells (not the node itself). */
static void bptFreeNodeBlobs(bptree *bt, bptNode *n) {
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = bptCellOff(bt, n, i);
        if (bptCellOverflow(n, off)) bptFreeAccounted(bt, bptCellBlob(bt, n, off));
    }
}

/* Free a whole node (blobs + page). Values/children are the caller's concern. */
static void bptFreeNode(bptree *bt, bptNode *n) {
    bptFreeNodeBlobs(bt, n);
    bt->numnodes--;
    bptFreeAccounted(bt, n);
}

/* Child pointer for child index cidx (0 == leftmost, i+1 == cell i's child). */
static inline bptNode *bptChild(bptree *bt, bptNode *n, uint32_t cidx) {
    if (cidx == 0) return n->leftmost;
    return (bptNode *)bptCellPayload(bt, n, bptCellOff(bt, n, cidx - 1));
}

static inline void bptSetChild(bptree *bt, bptNode *n, uint32_t cidx,
                               bptNode *child) {
    if (cidx == 0) n->leftmost = child;
    else bptCellSetPayload(bt, n, bptCellOff(bt, n, cidx - 1), child);
}

/* ------------------------------------------------------------------------- */
/* Key materialization and comparison inside a node                          */
/* ------------------------------------------------------------------------- */

/* Reconstruct the full key of entry i into a freshly malloc'd buffer. */
static unsigned char *bptFullKey(bptree *bt, bptNode *n, uint32_t i, uint32_t *outlen) {
    uint32_t off = bptCellOff(bt, n, i);
    uint32_t slen = bptCellSuffixLen(n, off);
    uint32_t total = n->prefixLen + slen;
    unsigned char *buf = bpt_malloc(total ? total : 1);
    if (n->prefixLen) memcpy(buf, bptPrefix(n), n->prefixLen);
    uint32_t ilen = bptCellInlineLen(bt, n, off);
    memcpy(buf + n->prefixLen, bptCellInline(n, off), ilen);
    if (bptCellOverflow(n, off)) {
        bptBlob *b = bptCellBlob(bt, n, off);
        memcpy(buf + n->prefixLen + ilen, b->data, b->len);
    }
    *outlen = total;
    return buf;
}

/* Compare a search suffix (key bytes after the node prefix) against the full
 * suffix stored in cell at offset `off`. */
static int bptCmpSuffix(bptree *bt, bptNode *n, uint32_t off,
                        const unsigned char *s, uint32_t slen) {
    uint32_t clen = bptCellSuffixLen(n, off);
    uint32_t ilen = bptCellInlineLen(bt, n, off);
    uint32_t cmpn = slen < clen ? slen : clen;
    uint32_t within = cmpn < ilen ? cmpn : ilen;
    if (within) {
        int r = memcmp(s, bptCellInline(n, off), within);
        if (r) return r < 0 ? -1 : 1;
    }
    if (cmpn > within) {
        /* Overflow guaranteed (only overflow cells have ilen < clen). */
        bptBlob *b = bptCellBlob(bt, n, off);
        int r = memcmp(s + within, b->data, cmpn - within);
        if (r) return r < 0 ? -1 : 1;
    }
    if (slen < clen) return -1;
    if (slen > clen) return 1;
    return 0;
}

/* Result describing how a search key relates to a node's common prefix. */
typedef enum {
    BPT_PFX_BELOW,   /* key sorts before every key in the node. */
    BPT_PFX_ABOVE,   /* key sorts after every key in the node. */
    BPT_PFX_MATCH    /* key shares the full prefix; compare on suffix. */
} bptPfxRel;

static bptPfxRel bptPrefixRel(bptNode *n, const unsigned char *key, uint32_t len) {
    if (n->prefixLen == 0) return BPT_PFX_MATCH;
    uint32_t cmpn = len < n->prefixLen ? len : n->prefixLen;
    int r = cmpn ? memcmp(key, bptPrefix(n), cmpn) : 0;
    if (r < 0) return BPT_PFX_BELOW;
    if (r > 0) return BPT_PFX_ABOVE;
    /* Equal over the compared span. */
    if (len < n->prefixLen) return BPT_PFX_BELOW; /* key is a proper prefix. */
    return BPT_PFX_MATCH;
}

/* Lower-bound search within a node: returns the index of the first entry whose
 * key is >= the search key, and sets *found when an exact match exists. The
 * returned index is in [0, nkeys].
 *
 * Every probe reads only the directory, which is dense and contiguous: the
 * replicated suffix heads settle all but the ties, so a full cell -- sitting at
 * an unpredictable offset in the cell region, and usually on its own cache
 * line -- is fetched only for keys that agree on their first BPT_DIR_HEAD
 * suffix bytes. */
static uint32_t bptNodeSearch(bptree *bt, bptNode *n,
                              const unsigned char *key, uint32_t len, int *found) {
    *found = 0;
    bptPfxRel rel = bptPrefixRel(n, key, len);
    if (rel == BPT_PFX_BELOW) return 0;
    if (rel == BPT_PFX_ABOVE) return n->nkeys;

    const unsigned char *s = key + n->prefixLen;
    uint32_t slen = len - n->prefixLen;
    unsigned char head[BPT_DIR_HEAD];
    bptDirHead(head, s, slen);

    uint32_t st = bptStride(bt, n);
    unsigned char *base = (unsigned char *)n + n->cap - n->nkeys * st;
    uint32_t lo = 0, hi = n->nkeys;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        bptDirEnt *e = (bptDirEnt *)(base + mid * st);
        int c = memcmp(head, e->head, BPT_DIR_HEAD); /* sign(search - stored) */
        if (c == 0) c = bptCmpSuffix(bt, n, e->off, s, slen);
        if (c == 0) { *found = 1; return mid; }
        if (c > 0) lo = mid + 1;  /* stored < search */
        else hi = mid;            /* stored > search */
    }
    return lo;
}

/* ------------------------------------------------------------------------- */
/* Compaction and low-level cell writing                                     */
/* ------------------------------------------------------------------------- */

/* Write a cell for suffix s/slen with the given payload at offset `off`.
 * Allocates an overflow blob if needed. Returns the cell size. */
static uint32_t bptWriteCell(bptree *bt, bptNode *n, uint32_t off,
                             const unsigned char *s, uint32_t slen, void *payload) {
    int overflow = slen > bt->maxInline;
    uint32_t csize = bptCellSize(bt, slen, overflow);
    unsigned char *c = bptRaw(n, off);
    memcpy(c, &slen, 4);
    c[4] = overflow ? BPT_CFLAG_OVERFLOW : 0;
    uint32_t ilen = overflow ? bt->maxInline : slen;
    if (ilen) memcpy(c + BPT_CELL_HDR, s, ilen);
    if (overflow) {
        uint32_t tail = slen - bt->maxInline;
        bptBlob *b = bptMallocAccounted(bt, sizeof(bptBlob) + tail);
        b->len = tail;
        memcpy(b->data, s + bt->maxInline, tail);
        memcpy(c + csize - 2 * BPT_PTR_SIZE, &b, BPT_PTR_SIZE);
    }
    memcpy(c + csize - BPT_PTR_SIZE, &payload, BPT_PTR_SIZE);
    return csize;
}

/* Reclaim dead bytes by repacking all live cells against the start of the cell
 * region. The prefix (just after the header) never moves. Blob pointers stay
 * valid (only the cell bytes move). Cells are read via the directory, so
 * physical gaps left by removals are skipped; a scratch buffer avoids
 * clobbering a not-yet-moved cell when the physical order diverges from key
 * order. Only the cell offsets are rewritten -- the replicated heads describe
 * keys, which compaction does not change. */
static void bptCompact(bptree *bt, bptNode *n) {
    if (n->deadBytes == 0) return;
    uint32_t dataStart = bptDataStart(n);
    unsigned char *tmp = bpt_malloc(n->heapEnd);
    uint32_t off = dataStart;
    for (uint32_t i = 0; i < n->nkeys; i++) {
        bptDirEnt *e = bptDeAt(bt, n, i);
        uint32_t src = e->off;
        uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, src),
                                     bptCellOverflow(n, src));
        memcpy(tmp + off, bptRaw(n, src), csize);
        e->off = (uint16_t)off;
        off += csize;
    }
    memcpy(bptRaw(n, dataStart), tmp + dataStart, off - dataStart);
    n->heapEnd = off;
    n->deadBytes = 0;
    bpt_free(tmp);
}

/* ------------------------------------------------------------------------- */
/* Node right-sizing (rax-style exact-size reallocation)                     */
/* ------------------------------------------------------------------------- */

static inline uint32_t bptRoundCap(bptree *bt, uint32_t v) {
    v = (v + (BPT_PTR_SIZE - 1)) & ~((uint32_t)BPT_PTR_SIZE - 1);
    if (v < BPT_MIN_CAP) v = BPT_MIN_CAP;
    if (v > bt->pageSize) v = bt->pageSize;
    return v;
}

/* Directory base for `count` entries at capacity `cap`. */
static inline unsigned char *bptDirBase(const bptree *bt, bptNode *n,
                                        uint32_t cap, uint32_t count) {
    return (unsigned char *)n + cap - count * bptStride(bt, n);
}

/* Set n->cap to `newcap`, relocating the directory (nkeys entries) from the
 * old capacity anchor to the new one. The node block must already be large
 * enough for the larger of the two capacities. */
static inline void bptMoveDir(bptree *bt, bptNode *n, uint32_t newcap) {
    if (newcap == n->cap) return;
    unsigned char *dst = bptDirBase(bt, n, newcap, n->nkeys);
    unsigned char *src = bptDirBase(bt, n, n->cap, n->nkeys);
    if (n->nkeys) memmove(dst, src, n->nkeys * bptStride(bt, n));
    n->cap = newcap;
}

/* Reallocate node `n`'s page to hold `want` capacity bytes, updating the
 * tree's memory counter. Returns the (possibly moved) node with cap unset by
 * this helper -- the caller decides the final cap and directory placement. */
static bptNode *bptPageRealloc(bptree *bt, bptNode *n, uint32_t want) {
    size_t nu, ou;
    bptNode *nn = bpt_realloc_usable(n, want, &nu, &ou);
    if (bt->alloc_size) *bt->alloc_size += (size_t)nu - ou;
    return nn;
}

/* Reallocate `n` for a full rebuild that is about to overwrite all of its
 * content: the old cells/directory need not be preserved, so this simply resizes
 * the block to `need` bytes and grabs the whole allocator size class as the new
 * capacity. Returns the (possibly moved) node; the caller must republish it. */
static bptNode *bptReallocRaw(bptree *bt, bptNode *n, uint32_t need) {
    uint32_t want = bptRoundCap(bt, need);
    size_t usable = bpt_malloc_usable_size(n);
    if (want != n->cap || usable < want) {
        n = bptPageRealloc(bt, n, want);
        usable = bpt_malloc_usable_size(n);
    }
    uint32_t cap = (uint32_t)(usable & ~((size_t)BPT_PTR_SIZE - 1));
    if (cap > bt->pageSize) cap = bt->pageSize;
    n->cap = cap; /* content is about to be rewritten; no directory to move. */
    return n;
}

/* Grow `n` (preserving its live content) so it can hold `need` content bytes,
 * grabbing the whole allocator size class so subsequent inserts stay in place
 * until the next class boundary (rax skips the realloc the same way). Returns
 * the (possibly moved) node; the caller must republish it if it moved. */
static bptNode *bptReserve(bptree *bt, bptNode *n, uint32_t need) {
    if (need <= n->cap) return n;
    uint32_t want = bptRoundCap(bt, need);
    if (bpt_malloc_usable_size(n) < want) n = bptPageRealloc(bt, n, want);
    uint32_t cap = (uint32_t)(bpt_malloc_usable_size(n) & ~((size_t)BPT_PTR_SIZE - 1));
    if (cap > bt->pageSize) cap = bt->pageSize;
    bptMoveDir(bt, n, cap); /* grow: directory base moves up, memmove within block. */
    return n;
}

/* Shrink `n` back toward the bytes its live content needs, compacting first so
 * no live cell sits past the trimmed capacity. Returns the (possibly moved)
 * node; caller republishes.
 *
 * Reclaiming costs a compaction plus a realloc, i.e. a pass over the whole
 * node, so a removal only pays for it once a worthwhile slice of the page comes
 * back (or the node empties out). Trimming to the exact size on every removal,
 * the way rax does, is cheap for rax's tiny nodes but would recompact a whole
 * page here for the sake of one cell. The node keeps the slack until then, and
 * the next insert reuses it in place. */
static bptNode *bptTrim(bptree *bt, bptNode *n) {
    uint32_t want = bptRoundCap(bt, bptUsedBytes(bt, n));
    if (want >= n->cap) return n;
    if (n->nkeys && want > n->cap - n->cap / 4) return n;
    bptCompact(bt, n);
    bptMoveDir(bt, n, want); /* shrink: directory base moves down, within block. */
    n = bptPageRealloc(bt, n, want);
    return n;
}

/* Point a parent's child slot (or the tree root) at `child`. cidx is the child
 * index used during descent: 0 is the leftmost child, i+1 is separator i's
 * right child. */
static inline void bptLinkChild(bptree *bt, bptNode *child,
                                bptNode *parent, uint32_t cidx) {
    if (parent == NULL) bt->root = child;
    else if (cidx == 0) parent->leftmost = child;
    else bptCellSetPayload(bt, parent, bptCellOff(bt, parent, cidx - 1), child);
}

/* Repair a leaf's chain neighbours after it moved. The unique leaf with no
 * successor is the rightmost one, so it is also where the cached tail is fixed. */
static inline void bptFixLeafLinks(bptree *bt, bptNode *n) {
    if (n->prev) n->prev->next = n;
    if (n->next) n->next->prev = n;
    else bt->tail = n;
}

/* Republish a node that may have moved: patch its parent pointer and, for a
 * leaf, its chain neighbours and the cached tail. */
static void bptPublish(bptree *bt, bptNode *n, bptNode *parent, uint32_t cidx) {
    bptLinkChild(bt, n, parent, cidx);
    if (n->isleaf) bptFixLeafLinks(bt, n);
}

/* ------------------------------------------------------------------------- */
/* Building a node from a sorted list of entries                             */
/* ------------------------------------------------------------------------- */

/* Bytes required to store entries[start..start+count) in a single node,
 * given they'd share the common prefix `cp`. Includes header and directory. */
static uint32_t bptEntriesBytes(bptree *bt, bptEnt *ents, uint32_t start,
                                uint32_t count, uint32_t cp, int isleaf) {
    /* When a prefix is present the first cell offset is rounded down to a
     * pointer boundary, wasting up to BPT_PTR_SIZE-1 bytes; reserve that so a
     * "fits" verdict never overflows the page after alignment. */
    uint32_t total = BPT_HDR_SIZE + cp + count * bptStrideFor(bt, isleaf) +
                     (cp ? (BPT_PTR_SIZE - 1) : 0);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slen = ents[start + i].keylen - cp;
        total += bptCellSize(bt, slen, slen > bt->maxInline);
    }
    return total;
}

/* Common prefix length across sorted entries[start..start+count). Capped at
 * maxInline so the node prefix (stored inline in the page) is always bounded
 * and the remainder of each key lives in the suffix/overflow. Capping only
 * shortens a valid common prefix, so correctness is preserved; without it a
 * single-entry node would try to store the whole key as its prefix. */
static uint32_t bptEntriesCp(bptree *bt, bptEnt *ents, uint32_t start, uint32_t count) {
    if (count == 0) return 0;
    bptEnt *first = &ents[start], *last = &ents[start + count - 1];
    uint32_t cp = bptLcp(first->key, first->keylen, last->key, last->keylen);
    if (cp > bt->maxInline) cp = bt->maxInline;
    return cp;
}

static int bptEntriesFit(bptree *bt, bptEnt *ents, uint32_t start,
                         uint32_t count, int isleaf) {
    uint32_t cp = bptEntriesCp(bt, ents, start, count);
    return bptEntriesBytes(bt, ents, start, count, cp, isleaf) <= bt->pageSize;
}

/* Exact bytes a node built from entries[start..start+count) with stored prefix
 * `cp` occupies: aligned cell region start + packed cells + directory. */
static uint32_t bptBuiltSize(bptree *bt, bptEnt *ents, uint32_t start,
                             uint32_t count, uint32_t cp, int isleaf) {
    uint32_t off = ((uint32_t)BPT_HDR_SIZE + cp + (BPT_PTR_SIZE - 1)) &
                   ~((uint32_t)BPT_PTR_SIZE - 1);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slen = ents[start + i].keylen - cp;
        off += bptCellSize(bt, slen, slen > bt->maxInline);
    }
    return off + count * bptStrideFor(bt, isleaf);
}

/* (Re)build node `n` from entries[start..start+count) with an explicit
 * stored-prefix length `cp` (any value <= the entries' true common prefix is
 * valid; a shorter one just stores longer suffixes). The page is first resized
 * to exactly what the content needs, so `n` may move -- the returned pointer is
 * authoritative and the caller must republish it. next/prev/leftmost are
 * preserved across the move; any existing blobs must have been freed by the
 * caller. Must fit within the tree page size. */
static bptNode *bptBuildNodeCp(bptree *bt, bptNode *n, int isleaf,
                               bptEnt *ents, uint32_t start, uint32_t count,
                               uint32_t cp) {
    n = bptReallocRaw(bt, n, bptBuiltSize(bt, ents, start, count, cp, isleaf));
    n->isleaf = isleaf ? 1 : 0;
    n->nkeys = 0;
    n->deadBytes = 0;
    n->prefixLen = cp;
    if (cp) memcpy(bptRaw(n, BPT_HDR_SIZE), ents[start].key, cp);

    uint32_t off = bptDataStart(n);
    for (uint32_t i = 0; i < count; i++) {
        bptEnt *e = &ents[start + i];
        uint32_t slen = e->keylen - cp;
        uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
        assert(off + csize <= n->cap - count * bptStride(bt, n));
        bptWriteCell(bt, n, off, e->key + cp, slen, e->payload);
        bptDirEnt *de = bptDeAtN(bt, n, count, i);
        bptDirHead(de->head, e->key + cp, slen);
        de->off = (uint16_t)off;
        if (!isleaf) {
            /* Entry i's right child is ents[start+i].payload; record its
             * subtree element count from the child's own (maintained) counts. */
            uint32_t cc = bptSubtreeCount(bt, (bptNode *)e->payload);
            memcpy((unsigned char *)de + BPT_DIR_SIZE, &cc, BPT_DIR_CNT);
        }
        off += csize;
    }
    n->heapEnd = off;
    n->nkeys = count;
    return n;
}

static bptNode *bptBuildNode(bptree *bt, bptNode *n, int isleaf,
                             bptEnt *ents, uint32_t start, uint32_t count) {
    return bptBuildNodeCp(bt, n, isleaf, ents, start, count,
                          bptEntriesCp(bt, ents, start, count));
}

/* Extract all entries of a node into a fresh array of full keys. Each key is
 * heap-owned; free with bptFreeEnts(). Payloads are copied verbatim. */
static bptEnt *bptExtract(bptree *bt, bptNode *n, uint32_t *outcount) {
    uint32_t cnt = n->nkeys;
    bptEnt *ents = bpt_malloc(sizeof(bptEnt) * (cnt ? cnt : 1));
    for (uint32_t i = 0; i < cnt; i++) {
        ents[i].key = bptFullKey(bt, n, i, &ents[i].keylen);
        ents[i].payload = bptCellPayload(bt, n, bptCellOff(bt, n, i));
    }
    *outcount = cnt;
    return ents;
}

static void bptFreeEnts(bptEnt *ents, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) bpt_free(ents[i].key);
    bpt_free(ents);
}

/* Whether the split halves around `mid` both fit in a page. For a leaf the
 * halves are [0,mid) and [mid,count); for an inner node the middle entry moves
 * up, so the halves are [0,mid) and [mid+1,count). */
static int bptSplitHalvesFit(bptree *bt, bptEnt *ents, uint32_t count,
                             uint32_t mid, int isleaf) {
    if (!bptEntriesFit(bt, ents, 0, mid, isleaf)) return 0;
    if (isleaf) return bptEntriesFit(bt, ents, mid, count - mid, isleaf);
    return bptEntriesFit(bt, ents, mid + 1, count - mid - 1, isleaf);
}

/* Choose a split index in [1, count-1] that balances bytes between the two
 * halves, adjusting so that both halves are guaranteed to fit in a page. */
static uint32_t bptChooseSplit(bptree *bt, bptEnt *ents, uint32_t count, int isleaf) {
    uint32_t cp = bptEntriesCp(bt, ents, 0, count);
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slen = ents[i].keylen - cp;
        total += bptCellSize(bt, slen, slen > bt->maxInline);
    }
    uint32_t acc = 0, mid = 1;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slen = ents[i].keylen - cp;
        acc += bptCellSize(bt, slen, slen > bt->maxInline);
        if (acc * 2 >= total) { mid = i + 1; break; }
    }
    if (mid < 1) mid = 1;
    if (mid > count - 1) mid = count - 1;

    /* Nudge toward a fitting partition. One is guaranteed to exist: cells are
     * capped at ~pageSize/8 (maxInline), and even when a new key destroys a
     * half's common prefix — inflating every sibling suffix by up to the lost
     * prefix length — such a key is always first or last among the keys that
     * shared the prefix, so the boundary split isolating it leaves the other
     * half with its original (fitting) layout. If the guard is ever exhausted
     * anyway, bptBuildNode's heap assert fires rather than corrupting a page. */
    for (uint32_t guard = 0; guard <= count; guard++) {
        if (bptSplitHalvesFit(bt, ents, count, mid, isleaf)) break;
        if (!bptEntriesFit(bt, ents, 0, mid, isleaf) && mid > 1) { mid--; continue; }
        if (mid < count - 1) { mid++; continue; }
        break;
    }
    return mid;
}

/* Shortest separator: right's first key truncated to the first differing byte
 * (inclusive). Returns a malloc'd buffer. */
static unsigned char *bptShortestSep(const unsigned char *lk, uint32_t lklen,
                                     const unsigned char *rk, uint32_t rklen,
                                     uint32_t *outlen) {
    uint32_t i = bptLcp(lk, lklen, rk, rklen);
    uint32_t seplen = i + 1; /* rk > lk guarantees i < rklen. */
    if (seplen > rklen) seplen = rklen;
    unsigned char *sep = bpt_malloc(seplen ? seplen : 1);
    memcpy(sep, rk, seplen);
    *outlen = seplen;
    return sep;
}

/* ------------------------------------------------------------------------- */
/* Constructors / destructor                                                 */
/* ------------------------------------------------------------------------- */

static uint32_t bptRoundPow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

static bptree *bptNewEx(uint32_t pageSize, size_t *alloc_size) {
    if (pageSize < BPT_PAGE_MIN) pageSize = BPT_PAGE_MIN;
    if (pageSize > BPT_PAGE_MAX) pageSize = BPT_PAGE_MAX;
    pageSize = bptRoundPow2(pageSize);

    size_t hdr_usable;
    bptree *bt = bpt_malloc_usable(sizeof(*bt), &hdr_usable);
    if (!bt) return NULL;
    bt->numele = 0;
    bt->numnodes = 0;
    bt->version = 0;
    bt->alloc_size = alloc_size;
    if (bt->alloc_size) *bt->alloc_size += hdr_usable;
    bt->pageSize = pageSize;
    /* Inline threshold: keep overflow cells comfortably small so several fit in
     * a page. This bounds the maximum cell size well below the page, which in
     * turn guarantees that any node split leaves both halves within a page. */
    uint32_t usable = pageSize - (uint32_t)BPT_HDR_SIZE;
    bt->maxInline = usable / 8;
    if (bt->maxInline < 8) bt->maxInline = 8;
    bt->height = 1;
    bt->root = bptNewNode(bt, 1);
    if (!bt->root) {
        if (bt->alloc_size) *bt->alloc_size -= hdr_usable;
        bpt_free(bt);
        return NULL;
    }
    bt->tail = bt->root;
    return bt;
}

bptree *bptNewRanked(uint32_t pageSize, size_t *alloc_size) {
    return bptNewEx(pageSize, alloc_size);
}

/* Recursive free of the tree structure (leaf values are not owned). */
static void bptFreeRec(bptree *bt, bptNode *n) {
    if (!n->isleaf) {
        bptFreeRec(bt, n->leftmost);
        for (uint32_t i = 0; i < n->nkeys; i++)
            bptFreeRec(bt, (bptNode *)bptCellPayload(bt, n, bptCellOff(bt, n, i)));
    }
    bptFreeNode(bt, n);
}

void bptFree(bptree *bt) {
    if (!bt) return;
    bptFreeRec(bt, bt->root);
    bptFreeAccounted(bt, bt);
}

uint64_t bptSize(bptree *bt) {
    return bt->numele;
}

uint64_t bptNumNodes(bptree *bt) {
    return bt->numnodes;
}

/* ------------------------------------------------------------------------- */
/* Find                                                                      */
/* ------------------------------------------------------------------------- */

static int bptFind(bptree *bt, unsigned char *s, size_t len, void **value) {
    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, len, &found);
        uint32_t cidx = found ? idx + 1 : idx;
        n = bptChild(bt, n, cidx);
    }
    int found;
    uint32_t idx = bptNodeSearch(bt, n, s, len, &found);
    if (!found) return 0;
    if (value) *value = bptCellPayload(bt, n, bptCellOff(bt, n, idx));
    return 1;
}

/* ------------------------------------------------------------------------- */
/* In-place cell insertion / removal helpers                                 */
/* ------------------------------------------------------------------------- */

/* Insert a new (suffix,payload) cell at sorted position `pos`, assuming the key
 * shares the node's full prefix and there is room (after compaction). */
static void bptInsertCellAt(bptree *bt, bptNode *n, uint32_t pos,
                            const unsigned char *suffix, uint32_t slen, void *payload) {
    uint32_t st = bptStride(bt, n); /* leaf-only caller: st == BPT_DIR_SIZE. */
    uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
    if (bptFreeBytes(bt, n) < csize + st) bptCompact(bt, n);
    assert(bptFreeBytes(bt, n) >= csize + st);
    uint32_t off = n->heapEnd;
    bptWriteCell(bt, n, off, suffix, slen, payload);
    n->heapEnd += csize;
    /* The directory grows backward, so its base drops by one entry: only the
     * [0,pos) head shifts down; entries at/after pos keep their address. */
    unsigned char *old = (unsigned char *)n + n->cap - n->nkeys * st;
    unsigned char *dir = old - st;
    memmove(dir, old, pos * st);
    bptDirEnt *de = (bptDirEnt *)(dir + pos * st);
    bptDirHead(de->head, suffix, slen);
    de->off = (uint16_t)off;
    n->nkeys++;
}

/* Remove the cell at directory position `pos` in place (frees its blob). Works
 * for leaves and inner nodes: the wider inner entries carry their
 * per-child counts, which shift along with the entries. */
static void bptRemoveCellAt(bptree *bt, bptNode *n, uint32_t pos) {
    uint32_t st = bptStride(bt, n);
    unsigned char *dir = (unsigned char *)n + n->cap - n->nkeys * st;
    bptDirEnt *de = (bptDirEnt *)(dir + pos * st);
    uint32_t off = de->off;
    uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off),
                                 bptCellOverflow(n, off));
    if (bptCellOverflow(n, off)) bptFreeAccounted(bt, bptCellBlob(bt, n, off));
    n->deadBytes += csize;
    /* The directory grows backward, so its base rises by one entry: only the
     * [0,pos) head shifts up; entries after pos keep their address. */
    memmove(dir + st, dir, pos * st);
    n->nkeys--;
}

/* ------------------------------------------------------------------------- */
/* Insert                                                                    */
/* ------------------------------------------------------------------------- */

/* Split state propagated up the tree after a node overflows. */
typedef struct bptSplit {
    unsigned char *sep;  /* separator key pushed to the parent (malloc'd). */
    uint32_t seplen;
    bptNode *right;      /* new right sibling. */
} bptSplit;

/* Splice `right` immediately after `left` in the leaf chain and republish both
 * ends (also updates the cached tail when `right` becomes the rightmost). */
static void bptSpliceLeafRight(bptree *bt, bptNode *left, bptNode *right) {
    right->next = left->next;
    right->prev = left;
    left->next = right;
    bptFixLeafLinks(bt, left);
    bptFixLeafLinks(bt, right);
}

/* Splice `left` immediately before `n` in the leaf chain (prepend). */
static void bptSpliceLeafLeft(bptree *bt, bptNode *left, bptNode *n) {
    left->prev = n->prev;
    left->next = n;
    n->prev = left;
    bptFixLeafLinks(bt, left);
    bptFixLeafLinks(bt, n);
}

/* Hedge a singleton leaf's stored prefix to the LCP with its neighbour key,
 * capped at maxInline. Storing the whole key would make the next edge insert
 * miss BPT_PFX_MATCH and degenerate into one leaf per key (see bptAppend). */
static uint32_t bptHedgeCp(bptree *bt, const unsigned char *a, uint32_t alen,
                           const unsigned char *b, uint32_t blen) {
    uint32_t cp = bptLcp(a, alen, b, blen);
    if (cp > bt->maxInline) cp = bt->maxInline;
    return cp;
}

/* Rebuild a leaf, or split it, from a sorted entry list. `inspos` is the index
 * of the newly inserted entry in `ents` (the only entry that was not already
 * in `n`). On split, *sp is populated (caller must free sp->sep); otherwise
 * sp->right is NULL. Because a rebuild right-sizes the page, `n` may move: the
 * returned pointer is authoritative and the caller must republish it into its
 * parent. The leaf chain (and cached tail) are fixed up here.
 *
 * Overflow split policy is insert-point aware so sequential appends/prepends
 * keep the packed leaf ~full instead of converging to ~55% via a middle cut:
 *   - inspos == count-1 (append): leave `n` untouched; new right sibling holds
 *     only the new key.
 *   - inspos == 0 (prepend): leave `n` (old keys) as the right sibling; a new
 *     left leaf holds only the new key (returned as `n` so the parent slot
 *     stays correct).
 *   - otherwise: byte-balanced middle split (existing logic). */
static bptNode *bptLeafRebuildOrSplit(bptree *bt, bptNode *n,
                                      bptEnt *ents, uint32_t count,
                                      uint32_t inspos, bptSplit *sp) {
    sp->right = NULL;
    if (bptEntriesFit(bt, ents, 0, count, 1)) {
        bptFreeNodeBlobs(bt, n);
        n = bptBuildNode(bt, n, 1, ents, 0, count);
        bptFixLeafLinks(bt, n);
        return n;
    }
    assert(count >= 2 && inspos < count);

    /* Append split: new key is last. Keep the existing leaf packed. */
    if (inspos == count - 1) {
        bptNode *right = bptNewNode(bt, 1);
        sp->sep = bptShortestSep(ents[count - 2].key, ents[count - 2].keylen,
                                 ents[count - 1].key, ents[count - 1].keylen,
                                 &sp->seplen);
        uint32_t cp = bptHedgeCp(bt, ents[count - 2].key, ents[count - 2].keylen,
                                 ents[count - 1].key, ents[count - 1].keylen);
        right = bptBuildNodeCp(bt, right, 1, ents, count - 1, 1, cp);
        sp->right = right;
        bptSpliceLeafRight(bt, n, right);
        return n; /* unchanged -- still holds ents[0..count-2]. */
    }

    /* Prepend split: new key is first. Old leaf stays packed as the right
     * sibling; parent child slot is re-pointed at the new left leaf. */
    if (inspos == 0) {
        bptNode *left = bptNewNode(bt, 1);
        sp->sep = bptShortestSep(ents[0].key, ents[0].keylen,
                                 ents[1].key, ents[1].keylen, &sp->seplen);
        uint32_t cp = bptHedgeCp(bt, ents[0].key, ents[0].keylen,
                                 ents[1].key, ents[1].keylen);
        left = bptBuildNodeCp(bt, left, 1, ents, 0, 1, cp);
        bptSpliceLeafLeft(bt, left, n); /* n keeps old keys; becomes right. */
        sp->right = n;
        return left;
    }

    /* Middle split: byte-balanced cut. */
    uint32_t mid = bptChooseSplit(bt, ents, count, 1);
    bptNode *right = bptNewNode(bt, 1);
    /* Separator BEFORE we rebuild (keys still available in ents). */
    sp->sep = bptShortestSep(ents[mid - 1].key, ents[mid - 1].keylen,
                             ents[mid].key, ents[mid].keylen, &sp->seplen);

    bptFreeNodeBlobs(bt, n);
    n = bptBuildNode(bt, n, 1, ents, 0, mid);
    right = bptBuildNode(bt, right, 1, ents, mid, count - mid);
    sp->right = right;
    bptSpliceLeafRight(bt, n, right);
    return n;
}

/* Rebuild an inner node, or split it, from a leftmost child plus a sorted entry
 * list (entry.payload == child to the right of entry.key). Like the leaf
 * variant, `n` may move; the returned pointer is authoritative. */
static bptNode *bptInnerRebuildOrSplit(bptree *bt, bptNode *n, bptNode *leftmost,
                                       bptEnt *ents, uint32_t count, bptSplit *sp) {
    sp->right = NULL;
    if (bptEntriesFit(bt, ents, 0, count, 0)) {
        bptFreeNodeBlobs(bt, n);
        n = bptBuildNode(bt, n, 0, ents, 0, count);
        n->leftmost = leftmost;
        n->leftCount = bptSubtreeCount(bt, leftmost);
        return n;
    }
    uint32_t mid = bptChooseSplit(bt, ents, count, 0);
    /* Inner split moves the middle entry up: it is removed from both children. */
    sp->sep = bpt_malloc(ents[mid].keylen ? ents[mid].keylen : 1);
    memcpy(sp->sep, ents[mid].key, ents[mid].keylen);
    sp->seplen = ents[mid].keylen;
    bptNode *right = bptNewNode(bt, 0);
    bptNode *rleft = (bptNode *)ents[mid].payload;

    bptFreeNodeBlobs(bt, n);
    n = bptBuildNode(bt, n, 0, ents, 0, mid);
    n->leftmost = leftmost;
    n->leftCount = bptSubtreeCount(bt, leftmost);
    right = bptBuildNode(bt, right, 0, ents, mid + 1, count - mid - 1);
    right->leftmost = rleft;
    right->leftCount = bptSubtreeCount(bt, rleft);
    sp->right = right;
    return n;
}

/* Insert (sep,child) into inner node `n` at child position `pos`; rebuild or
 * split as needed. Returns the (possibly moved) `n`. */
static bptNode *bptInnerInsert(bptree *bt, bptNode *n, uint32_t pos,
                               unsigned char *sep, uint32_t seplen,
                               bptNode *child, bptSplit *sp) {
    uint32_t cnt;
    bptEnt *ents = bptExtract(bt, n, &cnt);
    bptEnt *merged = bpt_malloc(sizeof(bptEnt) * (cnt + 1));
    memcpy(merged, ents, sizeof(bptEnt) * pos);
    merged[pos].key = sep;
    merged[pos].keylen = seplen;
    merged[pos].payload = child;
    memcpy(merged + pos + 1, ents + pos, sizeof(bptEnt) * (cnt - pos));

    bptNode *leftmost = n->leftmost;
    n = bptInnerRebuildOrSplit(bt, n, leftmost, merged, cnt + 1, sp);

    bpt_free(ents); /* keys are freed below via merged (same pointers). */
    for (uint32_t i = 0; i < cnt + 1; i++)
        if (i != pos) bpt_free(merged[i].key);
    bpt_free(merged);
    return n;
}

/* Propagate a split from level `level` (the parent that must absorb the split)
 * up toward the root, creating a new root if necessary. Each inner insert may
 * move its node, so the node is republished into its own parent before moving
 * on. Allocation failures abort inside the allocator (zmalloc), as everywhere
 * else in this file: the tree offers no graceful OOM recovery once a structural
 * mutation started. */
static void bptPropagate(bptree *bt, bptNode **path, uint32_t *cidx,
                         int level, bptSplit sp) {
    while (level >= 0 && sp.right) {
        bptNode *parent = path[level];
        bptSplit up;
        parent = bptInnerInsert(bt, parent, cidx[level], sp.sep, sp.seplen,
                                sp.right, &up);
        bpt_free(sp.sep);
        bptNode *gp = level > 0 ? path[level - 1] : NULL;
        uint32_t gpc = level > 0 ? cidx[level - 1] : 0;
        bptLinkChild(bt, parent, gp, gpc);
        sp = up;
        level--;
    }
    if (sp.right) {
        bptNode *newroot = bptNewNode(bt, 0);
        newroot->leftmost = bt->root;
        bptSplit dummy;
        newroot = bptInnerInsert(bt, newroot, 0, sp.sep, sp.seplen, sp.right,
                                 &dummy);
        assert(dummy.right == NULL);
        bpt_free(sp.sep);
        bt->root = newroot;
        bt->height++;
    }
}

/* Ranked-count maintenance ------------------------------------------------- */

/* Cheap path: no split/merge changed the shape, so every inner node on the
 * descent stack still routes the mutated key through the same child. Nudge that
 * child's stored subtree count by `delta` (+1 on insert, -1 on remove). */
static void bptRankAdjustPath(bptree *bt, bptNode **path, uint32_t *cidx,
                              int depth, int delta) {
    for (int L = 0; L < depth; L++) {
        uint32_t c = bptChildCnt(bt, path[L], cidx[L]);
        bptSetChildCnt(bt, path[L], cidx[L], (uint32_t)((int64_t)c + delta));
    }
}

/* Expensive path: a split/merge (or root change) reshaped the tree, so
 * re-descend by key and recompute every on-path child count bottom-up. Each
 * parent count is read as the child's freshly-correct subtree total, so a
 * single pass fixes the whole path regardless of what the structural op did. */
static void bptRankRecompute(bptree *bt, const unsigned char *s, uint32_t len) {
    bptNode *path[BPT_MAX_HEIGHT];
    uint32_t cidx[BPT_MAX_HEIGHT];
    int depth = 0;
    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, len, &found);
        uint32_t c = found ? idx + 1 : idx;
        path[depth] = n;
        cidx[depth] = c;
        depth++;
        n = bptChild(bt, n, c);
    }
    for (int L = depth - 1; L >= 0; L--) {
        bptNode *child = bptChild(bt, path[L], cidx[L]);
        bptSetChildCnt(bt, path[L], cidx[L], bptSubtreeCount(bt, child));
    }
}

/* Cheap append path: an in-place tail insert grows the subtree count of the
 * last child at every level (the rightmost spine) by `delta`. */
static void bptRankAdjustRightmost(bptree *bt, int delta) {
    bptNode *n = bt->root;
    while (!n->isleaf) {
        uint32_t c = bptChildCnt(bt, n, n->nkeys);
        bptSetChildCnt(bt, n, n->nkeys, (uint32_t)((int64_t)c + delta));
        n = bptChild(bt, n, n->nkeys);
    }
}

static int bptInsertGeneric(bptree *bt, unsigned char *s, size_t len,
                            void *data, void **old, int overwrite) {
    bptNode *path[BPT_MAX_HEIGHT];
    uint32_t cidx[BPT_MAX_HEIGHT];
    int depth = 0;

    assert(len == (uint32_t)len); /* Keys are limited to 4GB (u32 lengths). */

    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, len, &found);
        uint32_t c = found ? idx + 1 : idx;
        assert(depth < BPT_MAX_HEIGHT);
        path[depth] = n;
        cidx[depth] = c;
        depth++;
        n = bptChild(bt, n, c);
    }

    int found;
    uint32_t pos = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
    if (found) {
        if (old) *old = bptCellPayload(bt, n, bptCellOff(bt, n, pos));
        if (overwrite) bptCellSetPayload(bt, n, bptCellOff(bt, n, pos), data);
        errno = 0;
        return 0; /* key already existed. */
    }
    if (old) *old = NULL;

    bptSplit sp;
    sp.right = NULL;
    bptNode *parent = depth ? path[depth - 1] : NULL;
    uint32_t pcidx = depth ? cidx[depth - 1] : 0;

    /* Fast path: key shares the full node prefix and the cell still fits within
     * the page size. Grow the leaf to make room (grabbing the allocator size
     * class, so this reallocs only at class boundaries), then insert in place. */
    bptPfxRel rel = bptPrefixRel(n, s, (uint32_t)len);
    uint32_t slen = (uint32_t)len - n->prefixLen;
    uint32_t csize = (rel == BPT_PFX_MATCH) ?
                     bptCellSize(bt, slen, slen > bt->maxInline) : 0;
    if (rel == BPT_PFX_MATCH &&
        bptUsedBytes(bt, n) + csize + BPT_DIR_SIZE <= bt->pageSize) {
        bptNode *nn = bptReserve(bt, n, bptUsedBytes(bt, n) + csize + BPT_DIR_SIZE);
        if (nn != n) { bptPublish(bt, nn, parent, pcidx); n = nn; }
        bptInsertCellAt(bt, n, pos, s + n->prefixLen, slen, data);
    } else {
        /* Slow path: rebuild (possibly changing the prefix) or split. */
        uint32_t cnt;
        bptEnt *ents = bptExtract(bt, n, &cnt);
        bptEnt *merged = bpt_malloc(sizeof(bptEnt) * (cnt + 1));
        memcpy(merged, ents, sizeof(bptEnt) * pos);
        merged[pos].key = s;
        merged[pos].keylen = (uint32_t)len;
        merged[pos].payload = data;
        memcpy(merged + pos + 1, ents + pos, sizeof(bptEnt) * (cnt - pos));
        n = bptLeafRebuildOrSplit(bt, n, merged, cnt + 1, pos, &sp);
        bptLinkChild(bt, n, parent, pcidx);
        bpt_free(ents);
        for (uint32_t i = 0; i < cnt + 1; i++)
            if (i != pos) bpt_free(merged[i].key);
        bpt_free(merged);
    }

    if (sp.right) {
        bptPropagate(bt, path, cidx, depth - 1, sp);
        bptRankRecompute(bt, s, (uint32_t)len); /* a split reshaped the path. */
    } else {
        bptRankAdjustPath(bt, path, cidx, depth, +1);
    }

    bt->numele++;
    bt->version++;
    errno = 0;
    return 1;
}

int bptInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old) {
    return bptInsertGeneric(bt, s, len, data, old, 1);
}

static int bptTryInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old) {
    return bptInsertGeneric(bt, s, len, data, old, 0);
}

/* See bptree.h. Fast paths, tried in order:
 *   1. Tail leaf shares the key's prefix and has room: O(1) in-place cell
 *      insert at the end of the tail leaf -- no descent, no rebuild.
 *   2. Tail leaf is full or the key breaks its stored prefix: "append split".
 *      The new key starts a fresh rightmost leaf and the old tail is left
 *      untouched (so sequential loads keep leaves ~100% packed instead of the
 *      ~55% a middle split leaves behind, and no entry extraction/rebuild is
 *      paid). Only the separator insert walks the tree, taking the rightmost
 *      child at every level with no key comparisons.
 * Keys that do not extend the maximum fall back to bptTryInsert(). */
int bptAppend(bptree *bt, unsigned char *s, size_t len, void *data, void **old) {
    assert(len == (uint32_t)len); /* Keys are limited to 4GB (u32 lengths). */
    bptNode *t = bt->tail;
    /* No maximum key to compare against; treat as a plain insert. */
    if (t->nkeys == 0) return bptTryInsert(bt, s, len, data, old);

    /* Compare against the current maximum key (last cell of the tail leaf).
     * cmp = sign(new key - max key). */
    bptPfxRel rel = bptPrefixRel(t, s, (uint32_t)len);
    int cmp;
    if (rel == BPT_PFX_BELOW) cmp = -1;
    else if (rel == BPT_PFX_ABOVE) cmp = 1;
    else cmp = bptCmpSuffix(bt, t, bptCellOff(bt, t, t->nkeys - 1),
                            s + t->prefixLen, (uint32_t)len - t->prefixLen);

    if (cmp == 0) { /* The key IS the current maximum: exists, don't touch. */
        if (old) *old = bptCellPayload(bt, t, bptCellOff(bt, t, t->nkeys - 1));
        errno = 0;
        return 0;
    }
    if (cmp < 0) /* Not an append: genuine ordered insert (or existing key). */
        return bptTryInsert(bt, s, len, data, old);
    if (old) *old = NULL;

    /* 1. In-place tail insert. */
    if (rel == BPT_PFX_MATCH) {
        uint32_t slen = (uint32_t)len - t->prefixLen;
        uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
        if (bptUsedBytes(bt, t) + csize + BPT_DIR_SIZE <= bt->pageSize) {
            uint32_t need = bptUsedBytes(bt, t) + csize + BPT_DIR_SIZE;
            if (need > t->cap) {
                /* Growth crosses a size class: walk the rightmost spine (last
                 * child at every level, no comparisons) to find the tail's
                 * parent so the possibly-moved leaf can be republished. */
                bptNode *parent = NULL;
                uint32_t pc = 0;
                bptNode *n = bt->root;
                while (!n->isleaf) {
                    parent = n;
                    pc = n->nkeys;
                    n = bptChild(bt, n, n->nkeys);
                }
                bptNode *nt = bptReserve(bt, t, need);
                if (nt != t) { bptPublish(bt, nt, parent, pc); t = nt; }
            }
            bptInsertCellAt(bt, t, t->nkeys, s + t->prefixLen, slen, data);
            bptRankAdjustRightmost(bt, +1);
            bt->numele++;
            bt->version++;
            errno = 0;
            return 1;
        }
    }

    /* 2. Append split. Collect the rightmost spine for the separator
     * propagation (no comparisons: always the last child). */
    bptNode *path[BPT_MAX_HEIGHT];
    uint32_t cidx[BPT_MAX_HEIGHT];
    int depth = 0;
    bptNode *n = bt->root;
    while (!n->isleaf) {
        assert(depth < BPT_MAX_HEIGHT);
        path[depth] = n;
        cidx[depth] = n->nkeys;
        depth++;
        n = bptChild(bt, n, n->nkeys);
    }
    assert(n == t);

    /* The separator (and the new leaf's hedged prefix) both come from the
     * previous maximum key. */
    uint32_t lastlen;
    unsigned char *lastkey = bptFullKey(bt, t, t->nkeys - 1, &lastlen);

    /* Build the new leaf with a hedged prefix: the bytes the new key shares
     * with its predecessor are very likely shared by future appends too.
     * Storing the whole key as the prefix instead would make the next append
     * miss BPT_PFX_MATCH almost every time and degenerate into one leaf per
     * key. */
    bptNode *right = bptNewNode(bt, 1);
    uint32_t cp = bptLcp(lastkey, lastlen, s, (uint32_t)len);
    if (cp > bt->maxInline) cp = bt->maxInline;
    bptEnt e = { s, (uint32_t)len, data };
    right = bptBuildNodeCp(bt, right, 1, &e, 0, 1, cp);

    assert(t->next == NULL); /* t is the tail. */
    right->prev = t;
    right->next = NULL;
    t->next = right;
    bt->tail = right;

    bptSplit sp;
    sp.sep = bptShortestSep(lastkey, lastlen, s, (uint32_t)len, &sp.seplen);
    sp.right = right;
    bpt_free(lastkey);
    bptPropagate(bt, path, cidx, depth - 1, sp);
    bptRankRecompute(bt, s, (uint32_t)len); /* an append split reshaped the spine. */

    bt->numele++;
    bt->version++;
    errno = 0;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Remove (merge / root collapse)                                            */
/* ------------------------------------------------------------------------- */

static int bptUnderflow(bptree *bt, bptNode *n) {
    if (n->nkeys == 0) return 1;
    return bptUsedBytes(bt, n) < (bt->pageSize * BPT_MIN_FILL_PCT) / 100;
}

/* Bytes a merge of L and R needs at minimum: one header, both live cell regions
 * and both directories. Fusing two nodes can only shorten the prefix they
 * factor out, which lengthens the suffixes they store, so this bounds the
 * merged node from below -- except for a node whose stored prefix was left
 * deliberately short (see bptAppend), where the bound may be pessimistic and a
 * viable merge goes unattempted; the merge-only policy already permits leaving
 * a node underfull. Worth checking because the alternative way to find out that
 * two nodes do not fit is to materialize every key in both, and an underfull
 * node that no sibling can absorb would pay that on every later removal. */
static inline uint32_t bptMergeFloor(bptree *bt, bptNode *L, bptNode *R) {
    uint32_t cellsL = L->heapEnd - bptDataStart(L) - L->deadBytes;
    uint32_t cellsR = R->heapEnd - bptDataStart(R) - R->deadBytes;
    return (uint32_t)BPT_HDR_SIZE + cellsL + cellsR +
           (L->nkeys + R->nkeys) * bptStride(bt, L);
}

/* Concatenate leaf entries of L and R (adjacent leaves, all L < all R). */
static bptEnt *bptCombineLeaf(bptree *bt, bptNode *L, bptNode *R, uint32_t *outn) {
    uint32_t ln, rn;
    bptEnt *le = bptExtract(bt, L, &ln);
    bptEnt *re = bptExtract(bt, R, &rn);
    bptEnt *all = bpt_malloc(sizeof(bptEnt) * (ln + rn ? ln + rn : 1));
    memcpy(all, le, sizeof(bptEnt) * ln);
    memcpy(all + ln, re, sizeof(bptEnt) * rn);
    bpt_free(le);
    bpt_free(re);
    *outn = ln + rn;
    return all;
}

/* Concatenate inner entries of L and R with the parent separator pulled in
 * between them: [L.entries] ++ [(parentSep, R.leftmost)] ++ [R.entries]. */
static bptEnt *bptCombineInner(bptree *bt, bptNode *L, bptNode *R,
                               const unsigned char *psep, uint32_t pseplen,
                               uint32_t *outn) {
    uint32_t ln, rn;
    bptEnt *le = bptExtract(bt, L, &ln);
    bptEnt *re = bptExtract(bt, R, &rn);
    bptEnt *all = bpt_malloc(sizeof(bptEnt) * (ln + rn + 1));
    memcpy(all, le, sizeof(bptEnt) * ln);
    all[ln].key = bpt_malloc(pseplen ? pseplen : 1);
    memcpy(all[ln].key, psep, pseplen);
    all[ln].keylen = pseplen;
    all[ln].payload = R->leftmost;
    memcpy(all + ln + 1, re, sizeof(bptEnt) * rn);
    bpt_free(le);
    bpt_free(re);
    *outn = ln + rn + 1;
    return all;
}

int bptRemove(bptree *bt, unsigned char *s, size_t len, void **old) {
    bptNode *path[BPT_MAX_HEIGHT];
    uint32_t cidx[BPT_MAX_HEIGHT];
    int depth = 0;

    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, len, &found);
        uint32_t c = found ? idx + 1 : idx;
        assert(depth < BPT_MAX_HEIGHT);
        path[depth] = n;
        cidx[depth] = c;
        depth++;
        n = bptChild(bt, n, c);
    }

    int found;
    uint32_t pos = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
    if (!found) {
        if (old) *old = NULL;
        return 0;
    }
    if (old) *old = bptCellPayload(bt, n, bptCellOff(bt, n, pos));
    bptRemoveCellAt(bt, n, pos);
    bt->numele--;
    bt->version++;
    int structural = 0; /* set when a merge/collapse reshapes the path. */

    /* Merge upward. `cur` is the (possibly) underflowing node: it is fused with
     * an adjacent sibling whenever both halves fit in one page, which shrinks
     * the level above and may cascade up to a root collapse. When they do not
     * fit, `cur` is simply left underfull: nothing is redistributed between
     * siblings, so a node can hold fewer bytes than BPT_MIN_FILL_PCT and an
     * inner node can even end up with a single child. Every read path routes
     * through such nodes normally, and the space is reclaimed by a later merge
     * (or by inserts refilling the node). */
    bptNode *cur = n;
    int plevel = depth - 1;

    while (1) {
        bptNode *parent = (cur == bt->root) ? NULL : path[plevel];
        uint32_t pcidx = (cur == bt->root) ? 0 : cidx[plevel];

        if (cur == bt->root) {
            if (!cur->isleaf && cur->nkeys == 0) {
                bptNode *only = cur->leftmost;
                bptFreeNode(bt, cur);
                bt->root = only;
                bt->height--;
                structural = 1;
            } else if (cur->isleaf && cur->nkeys == 0) {
                cur->prefixLen = 0;
                cur->heapEnd = bptDataStart(cur);
                cur->deadBytes = 0;
            } else {
                bptNode *nc = bptTrim(bt, cur);
                if (nc != cur) bptPublish(bt, nc, NULL, 0);
            }
            break;
        }

        /* Not underflowing (or no sibling to merge with): give back the bytes
         * the removal freed and stop. */
        if (!bptUnderflow(bt, cur) || parent->nkeys == 0) {
            bptNode *nc = bptTrim(bt, cur);
            if (nc != cur) bptPublish(bt, nc, parent, pcidx);
            break;
        }

        bptNode *L, *R;
        uint32_t sepIdx;
        if (pcidx > 0) { L = bptChild(bt, parent, pcidx - 1); R = cur; sepIdx = pcidx - 1; }
        else { L = cur; R = bptChild(bt, parent, pcidx + 1); sepIdx = pcidx; }

        if (bptMergeFloor(bt, L, R) > bt->pageSize) {
            bptNode *nc = bptTrim(bt, cur); /* Leave `cur` underfull, trimmed. */
            if (nc != cur) bptPublish(bt, nc, parent, pcidx);
            break;
        }

        /* Parent separator between L and R. */
        uint32_t psepLen;
        unsigned char *psep = bptFullKey(bt, parent, sepIdx, &psepLen);

        /* Materializing both sides leaves them untouched, so the !fits path
         * below can bail out with the tree unchanged. */
        uint32_t total;
        bptEnt *all;
        if (cur->isleaf)
            all = bptCombineLeaf(bt, L, R, &total);
        else
            all = bptCombineInner(bt, L, R, psep, psepLen, &total);

        if (!bptEntriesFit(bt, all, 0, total, cur->isleaf)) {
            bptFreeEnts(all, total);
            bpt_free(psep);
            bptNode *nc = bptTrim(bt, cur); /* Leave `cur` underfull, trimmed. */
            if (nc != cur) bptPublish(bt, nc, parent, pcidx);
            break;
        }

        /* Everything goes into L (which is right-sized and may move); R is
         * freed and its parent separator (holding R's child pointer) dropped. */
        if (cur->isleaf) {
            bptNode *rnext = R->next;
            bptFreeNodeBlobs(bt, L);
            L = bptBuildNode(bt, L, 1, all, 0, total);
            L->next = rnext;
            bptFixLeafLinks(bt, L); /* fixes neighbours and the cached tail. */
        } else {
            bptNode *lm = L->leftmost;
            bptFreeNodeBlobs(bt, L);
            L = bptBuildNode(bt, L, 0, all, 0, total);
            L->leftmost = lm;
            L->leftCount = bptSubtreeCount(bt, lm);
        }
        bptLinkChild(bt, L, parent, sepIdx);
        bptFreeNode(bt, R);
        bptFreeEnts(all, total);
        bpt_free(psep);
        bptRemoveCellAt(bt, parent, sepIdx);
        structural = 1;
        cur = parent;
        plevel--;
    }
    if (structural) bptRankRecompute(bt, s, (uint32_t)len);
    else bptRankAdjustPath(bt, path, cidx, depth, -1);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Iterator                                                                  */
/* ------------------------------------------------------------------------- */

void bptStart(bptIterator *it, bptree *bt) {
    it->flags = BPT_ITER_EOF;
    it->bt = bt;
    it->key = it->key_static_string;
    it->key_len = 0;
    it->key_max = BPT_ITER_STATIC_LEN;
    it->data = NULL;
    it->node = NULL;
    it->idx = 0;
    it->version = bt->version;
}

void bptStop(bptIterator *it) {
    if (it->key != it->key_static_string) bpt_free(it->key);
    it->key = it->key_static_string;
    it->key_max = BPT_ITER_STATIC_LEN;
}

int bptEOF(bptIterator *it) {
    return (it->flags & BPT_ITER_EOF) != 0;
}

/* Materialize the current (node,idx) element into the iterator buffers. */
static void bptIterMaterialize(bptIterator *it) {
    bptree *bt = it->bt;
    bptNode *n = it->node;
    uint32_t i = (uint32_t)it->idx;
    uint32_t off = bptCellOff(bt, n, i);
    uint32_t slen = bptCellSuffixLen(n, off);
    uint32_t total = n->prefixLen + slen;
    if (total > it->key_max) {
        /* Grow geometrically: scans over keys of increasing length would
         * otherwise realloc on every step. The buffer is fully rewritten
         * below, so the old content need not be preserved. */
        size_t nm = it->key_max;
        while (nm < total) nm *= 2;
        unsigned char *nk = (it->key == it->key_static_string) ?
            bpt_malloc(nm) : bpt_realloc(it->key, nm);
        it->key = nk;
        it->key_max = nm;
    }
    if (n->prefixLen) memcpy(it->key, bptPrefix(n), n->prefixLen);
    uint32_t ilen = bptCellInlineLen(bt, n, off);
    memcpy(it->key + n->prefixLen, bptCellInline(n, off), ilen);
    if (bptCellOverflow(n, off)) {
        bptBlob *b = bptCellBlob(bt, n, off);
        memcpy(it->key + n->prefixLen + ilen, b->data, b->len);
    }
    it->key_len = total;
    it->data = bptCellPayload(bt, n, off);
}

/* Descend to the leaf that would contain `key` and return the lower-bound
 * index (first entry >= key) plus the found flag. */
static bptNode *bptDescendLB(bptree *bt, const unsigned char *key, uint32_t len,
                             uint32_t *idx, int *found) {
    bptNode *n = bt->root;
    while (!n->isleaf) {
        int f;
        uint32_t i = bptNodeSearch(bt, n, key, len, &f);
        n = bptChild(bt, n, f ? i + 1 : i);
    }
    *idx = bptNodeSearch(bt, n, key, len, found);
    return n;
}

static bptNode *bptFirstLeaf(bptree *bt) {
    bptNode *n = bt->root;
    while (!n->isleaf) n = n->leftmost;
    return n;
}

/* Position forward at (leaf,pos), crossing empty/exhausted leaves. */
static int bptNormFwd(bptIterator *it, bptNode *leaf, int64_t pos) {
    while (leaf && pos >= (int64_t)leaf->nkeys) { leaf = leaf->next; pos = 0; }
    if (!leaf) { it->flags |= BPT_ITER_EOF; return 0; }
    it->node = leaf;
    it->idx = (int)pos;
    return 1;
}
static int bptNormBwd(bptIterator *it, bptNode *leaf, int64_t pos) {
    while (leaf && pos < 0) { leaf = leaf->prev; if (leaf) pos = (int64_t)leaf->nkeys - 1; }
    if (!leaf || pos < 0) { it->flags |= BPT_ITER_EOF; return 0; }
    it->node = leaf;
    it->idx = (int)pos;
    return 1;
}

/* See bptree.h. Sums the subtree counts of every child to the left of the
 * descent at each inner level, then adds the leaf-local index. */
int bptRankOf(bptree *bt, unsigned char *s, size_t len, uint64_t *rank) {
    uint64_t r = 0;
    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
        uint32_t cidx = found ? idx + 1 : idx;
        for (uint32_t k = 0; k < cidx; k++) r += bptChildCnt(bt, n, k);
        n = bptChild(bt, n, cidx);
    }
    int found;
    uint32_t pos = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
    r += pos;
    if (rank) *rank = r;
    return found;
}

/* See bptree.h. Descends by subtree count to the leaf holding element `index`,
 * then positions the iterator there (mirroring bptSeek's post-seek state). */
int bptSelect(bptIterator *it, uint64_t index) {
    bptree *bt = it->bt;
    int safe = it->flags & BPT_ITER_SAFE;
    if (index >= bt->numele) { it->flags = safe | BPT_ITER_EOF; return 0; }
    it->flags = safe | BPT_ITER_JUST_SEEKED;
    it->version = bt->version;

    uint64_t rem = index;
    bptNode *n = bt->root;
    while (!n->isleaf) {
        uint32_t cidx = 0;
        for (;;) {
            uint32_t cc = bptChildCnt(bt, n, cidx);
            if (rem < cc) break;
            rem -= cc;
            cidx++;
            assert(cidx <= n->nkeys);
        }
        n = bptChild(bt, n, cidx);
    }
    it->node = n;
    it->idx = (int)rem;
    bptIterMaterialize(it);
    return 1;
}

int bptSeek(bptIterator *it, const char *op, unsigned char *ele, size_t len) {
    bptree *bt = it->bt;
    int safe = it->flags & BPT_ITER_SAFE; /* preserved across seeks. */
    it->flags = safe | BPT_ITER_JUST_SEEKED;
    it->version = bt->version;

    int eq = 0, lt = 0, gt = 0;
    if (op[0] == '>') { gt = 1; if (op[1] == '=') eq = 1; }
    else if (op[0] == '<') { lt = 1; if (op[1] == '=') eq = 1; }
    else if (op[0] == '=') { eq = 1; }
    else if (op[0] == '^') { /* first */ }
    else if (op[0] == '$') { /* last */ }
    else { it->flags = safe | BPT_ITER_EOF; return 0; }

    if (op[0] == '^') {
        if (bptNormFwd(it, bptFirstLeaf(bt), 0)) bptIterMaterialize(it);
        return 1;
    }
    if (op[0] == '$') {
        bptNode *leaf = bt->tail; /* O(1): the rightmost leaf is cached. */
        if (bptNormBwd(it, leaf, (int64_t)leaf->nkeys - 1)) bptIterMaterialize(it);
        return 1;
    }

    uint32_t idx;
    int found;
    bptNode *leaf = bptDescendLB(bt, ele, (uint32_t)len, &idx, &found);

    int ok;
    if (eq && !lt && !gt) {          /* == */
        if (!found) { it->flags = safe | BPT_ITER_EOF; return 1; }
        it->node = leaf; it->idx = (int)idx;
        ok = 1;
    } else if (gt) {                 /* >= or > */
        int64_t pos = (int64_t)idx;
        if (!eq && found) pos = (int64_t)idx + 1; /* '>' skips the match. */
        ok = bptNormFwd(it, leaf, pos);
    } else {                         /* <= or < */
        int64_t pos;
        if (eq && found) pos = (int64_t)idx;      /* '<=' includes the match. */
        else pos = (int64_t)idx - 1;              /* '<' or not found. */
        ok = bptNormBwd(it, leaf, pos);
    }
    /* Materialize eagerly: the iterator owns a private copy of the current
     * key/value, so the first bptNext()/bptPrev() never touches tree pages
     * (which a safe-iterator caller may have mutated in the meantime). */
    if (ok) bptIterMaterialize(it);
    return 1;
}

/* Re-establish (node,idx) after a mutation, for safe iteration. Returns the
 * next element strictly in the given direction relative to the last key. */
static int bptReseek(bptIterator *it, int forward) {
    bptree *bt = it->bt;
    uint32_t idx;
    int found;
    bptNode *leaf = bptDescendLB(bt, it->key, (uint32_t)it->key_len, &idx, &found);
    if (forward) {
        int64_t pos = found ? (int64_t)idx + 1 : (int64_t)idx;
        return bptNormFwd(it, leaf, pos);
    } else {
        int64_t pos = (int64_t)idx - 1;
        return bptNormBwd(it, leaf, pos);
    }
}

int bptNext(bptIterator *it) {
    if (it->flags & BPT_ITER_EOF) return 0;
    if (it->flags & BPT_ITER_JUST_SEEKED) {
        /* Key/value were materialized by bptSeek(); return them without
         * touching tree pages. it->version deliberately stays at its seek-time
         * value: if the tree changed in between, the next step must re-seek. */
        it->flags &= ~BPT_ITER_JUST_SEEKED;
        return 1;
    }
    if ((it->flags & BPT_ITER_SAFE) && it->version != it->bt->version) {
        if (!bptReseek(it, 1)) return 0;
        bptIterMaterialize(it);
        it->version = it->bt->version;
        return 1;
    }
    if (!bptNormFwd(it, it->node, (int64_t)it->idx + 1)) return 0;
    bptIterMaterialize(it);
    return 1;
}

int bptPrev(bptIterator *it) {
    if (it->flags & BPT_ITER_EOF) return 0;
    if (it->flags & BPT_ITER_JUST_SEEKED) {
        /* See bptNext(): seek already materialized, version stays stale. */
        it->flags &= ~BPT_ITER_JUST_SEEKED;
        return 1;
    }
    if ((it->flags & BPT_ITER_SAFE) && it->version != it->bt->version) {
        if (!bptReseek(it, 0)) return 0;
        bptIterMaterialize(it);
        it->version = it->bt->version;
        return 1;
    }
    if (!bptNormBwd(it, it->node, (int64_t)it->idx - 1)) return 0;
    bptIterMaterialize(it);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Active defragmentation                                                    */
/* ------------------------------------------------------------------------- */

/* Relocate one node page and its overflow blobs via `fn`. */
static bptNode *bptDefragPage(bptree *bt, bptNode *n, bptReallocFn fn) {
    bptNode *moved = fn(n);
    if (moved) n = moved;

    /* Repair overflow blob pointers stored inside this (relocated) page. */
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = bptCellOff(bt, n, i);
        if (!bptCellOverflow(n, off)) continue;
        bptBlob *b = bptCellBlob(bt, n, off);
        bptBlob *nb = fn(b);
        if (nb) bptCellSetBlob(bt, n, off, nb);
    }
    return n;
}

/* Relocate a whole subtree, fixing parent child pointers and rebuilding the
 * leaf chain from left to right. */
static bptNode *bptDefragNode(bptree *bt, bptNode *n, bptReallocFn fn,
                              bptNode **prevLeaf) {
    n = bptDefragPage(bt, n, fn);
    if (n->isleaf) {
        n->prev = *prevLeaf;
        if (*prevLeaf) (*prevLeaf)->next = n;
        *prevLeaf = n;
        return n;
    }

    n->leftmost = bptDefragNode(bt, n->leftmost, fn, prevLeaf);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = bptCellOff(bt, n, i);
        bptNode *child = (bptNode *)bptCellPayload(bt, n, off);
        bptNode *nc = bptDefragNode(bt, child, fn, prevLeaf);
        if (nc != child) bptCellSetPayload(bt, n, off, nc);
    }
    return n;
}

/* Relocate the whole tree (header + every node + every overflow blob) using
 * the caller-provided relocation function (typically activeDefragAlloc). On
 * return *btref points at the tree's current address; all internal pointers
 * (root, parent/child links, leaf chain, blob pointers) have been fixed up.
 * Values stored against keys are NOT touched. */
void bptDefrag(bptree **btref, bptReallocFn fn) {
    bptree *bt = *btref;
    bptree *moved = fn(bt);
    if (moved) { bt = moved; *btref = bt; }

    bptNode *prevLeaf = NULL;
    bt->root = bptDefragNode(bt, bt->root, fn, &prevLeaf);
    prevLeaf->next = NULL; /* last leaf terminates the chain (a tree always
                            * has at least its root leaf). */
    bt->tail = prevLeaf;
}

/* Relocate one leaf and its root-to-leaf path. Repeating the path work is a
 * deliberate tradeoff: it keeps the cursor mutation-tolerant and bounds each
 * call to one leaf page without storing pointers that commands could invalidate
 * between active-defrag cycles. */
int bptDefragStep(bptree **btref, uint64_t *cursor, bptReallocFn fn) {
    bptree *bt = *btref;

    if (*cursor == 0) {
        bptree *moved = fn(bt);
        if (moved) {
            bt = moved;
            *btref = bt;
        }
    }
    if (*cursor >= bt->numele) {
        *cursor = 0;
        return 0;
    }

    uint64_t rank = *cursor;
    uint64_t rem = rank;
    bptNode *parent = NULL;
    uint32_t parent_cidx = 0;
    bptNode *n = bt->root;

    for (;;) {
        bptNode *old = n;
        n = bptDefragPage(bt, n, fn);
        if (parent) bptSetChild(bt, parent, parent_cidx, n);
        else bt->root = n;

        if (n->isleaf) {
            if (n != old) {
                if (n->prev) n->prev->next = n;
                if (n->next) n->next->prev = n;
                if (bt->tail == old) bt->tail = n;
            }
            break;
        }

        uint32_t cidx = 0;
        for (;;) {
            uint32_t count = bptChildCnt(bt, n, cidx);
            if (rem < count) break;
            rem -= count;
            cidx++;
            assert(cidx <= n->nkeys);
        }
        parent = n;
        parent_cidx = cidx;
        n = bptChild(bt, n, cidx);
    }

    assert(rem < n->nkeys);
    *cursor = rank + (n->nkeys - rem);
    if (*cursor >= bt->numele) {
        *cursor = 0;
        return 0;
    }
    return 1;
}

static void bptForEachPageRec(bptree *bt, bptNode *n,
                              void (*cb)(void *, size_t, void *), void *ctx) {
    /* Overflow blobs attached to this node's cells. */
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = bptCellOff(bt, n, i);
        if (bptCellOverflow(n, off)) {
            bptBlob *b = bptCellBlob(bt, n, off);
            cb(b, zmalloc_usable_size(b), ctx);
        }
    }
    if (!n->isleaf) {
        bptForEachPageRec(bt, n->leftmost, cb, ctx);
        for (uint32_t i = 0; i < n->nkeys; i++)
            bptForEachPageRec(bt, (bptNode *)bptCellPayload(bt, n, bptCellOff(bt, n, i)),
                              cb, ctx);
    }
    cb(n, zmalloc_usable_size(n), ctx);
}

void bptForEachPage(bptree *bt,
                    void (*cb)(void *page, size_t size, void *ctx),
                    void *ctx) {
    if (!bt || !bt->root) return;
    bptForEachPageRec(bt, bt->root, cb, ctx);
}

/* ------------------------------------------------------------------------- */
/* Tests                                                                     */
/* ------------------------------------------------------------------------- */

#ifdef REDIS_TEST
#include <sys/time.h>
#include "rax.h"
#include "zmalloc.h"
#include "testhelp.h"

#define TEST(name) printf("test — %s\n", name);
#define UNUSED(x) (void)(x)

static long long bptUsTime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000 + tv.tv_usec;
}

/* A non-NULL, key-derived value so overwrite semantics can be checked. */
static void *bptTestVal(const unsigned char *k, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= k[i]; h *= 1099511628211ULL; }
    h |= 1; /* never NULL. */
    return (void *)(uintptr_t)h;
}

/* --- Structural validator ------------------------------------------------ */

static int bptValidateRec(bptree *bt, bptNode *n,
                          const unsigned char *low, uint32_t lowlen, int lowInf,
                          const unsigned char *high, uint32_t highlen, int highInf,
                          int depth, int *leafDepth, uint64_t *count) {
    int err = 0;

    /* Right-sizing invariants: the capacity never exceeds the page size, always
     * covers the live content, and always fits inside the real allocation; the
     * cell region and the directory must not overlap. */
    if (n->cap > bt->pageSize) { printf("validate: cap %u > pageSize %u\n", n->cap, bt->pageSize); err++; }
    if (bptUsedBytes(bt, n) > n->cap) { printf("validate: used %u > cap %u\n", bptUsedBytes(bt, n), n->cap); err++; }
    if (n->cap > bpt_malloc_usable_size(n)) { printf("validate: cap %u > usable %zu\n", n->cap, bpt_malloc_usable_size(n)); err++; }
    if (n->heapEnd > n->cap - n->nkeys * bptStride(bt, n)) { printf("validate: cells overlap directory\n"); err++; }
    if (n->heapEnd < bptDataStart(n)) { printf("validate: heapEnd below data start\n"); err++; }

    /* Keys sorted and within bounds. */
    unsigned char *prev = NULL;
    uint32_t prevlen = 0;
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t klen;
        unsigned char *k = bptFullKey(bt, n, i, &klen);
        if (prev && bptKeyCmp(prev, prevlen, k, klen) >= 0) {
            printf("validate: keys not sorted in node\n"); err++;
        }
        /* The directory replicates each key's head: a stale copy would silently
         * misroute searches, so check it against the materialized key. */
        unsigned char head[BPT_DIR_HEAD];
        bptDirHead(head, k + n->prefixLen, klen - n->prefixLen);
        if (memcmp(head, bptDeAt(bt, n, i)->head, BPT_DIR_HEAD) != 0) {
            printf("validate: directory head out of sync with cell\n"); err++;
        }
        if (!lowInf && bptKeyCmp(k, klen, low, lowlen) < 0) {
            printf("validate: key below lower bound\n"); err++;
        }
        if (!highInf && bptKeyCmp(k, klen, high, highlen) >= 0) {
            printf("validate: key at/above upper bound\n"); err++;
        }
        bpt_free(prev);
        prev = k; prevlen = klen;
    }

    if (n->isleaf) {
        if (*leafDepth < 0) *leafDepth = depth;
        else if (*leafDepth != depth) { printf("validate: leaf depth mismatch\n"); err++; }
        *count += n->nkeys;
        bpt_free(prev);
        return err;
    }

    /* Inner: recurse into children with tightened bounds. */
    if (!n->leftmost) { printf("validate: inner missing leftmost\n"); err++; bpt_free(prev); return err; }
    for (uint32_t c = 0; c <= n->nkeys; c++) {
        bptNode *child = bptChild(bt, n, c);
        const unsigned char *clow; uint32_t clowlen; int clowInf;
        const unsigned char *chigh; uint32_t chighlen; int chighInf;
        unsigned char *lk = NULL, *hk = NULL;
        if (c == 0) { clow = low; clowlen = lowlen; clowInf = lowInf; }
        else { lk = bptFullKey(bt, n, c - 1, &clowlen); clow = lk; clowInf = 0; }
        if (c == n->nkeys) { chigh = high; chighlen = highlen; chighInf = highInf; }
        else { hk = bptFullKey(bt, n, c, &chighlen); chigh = hk; chighInf = 0; }
        uint64_t before = *count;
        err += bptValidateRec(bt, child, clow, clowlen, clowInf,
                              chigh, chighlen, chighInf, depth + 1, leafDepth, count);
        uint64_t childCount = *count - before;
        if (bptChildCnt(bt, n, c) != childCount) {
            printf("validate: child count %u != actual %llu\n",
                   bptChildCnt(bt, n, c), (unsigned long long)childCount);
            err++;
        }
        bpt_free(lk); bpt_free(hk);
    }
    bpt_free(prev);
    return err;
}

static int bptValidate(bptree *bt) {
    int leafDepth = -1;
    uint64_t count = 0;
    int err = bptValidateRec(bt, bt->root, NULL, 0, 1, NULL, 0, 1, 0, &leafDepth, &count);
    if (count != bt->numele) {
        printf("validate: element count %llu != numele %llu\n",
               (unsigned long long)count, (unsigned long long)bt->numele);
        err++;
    }
    /* Leaf chain consistency: walk next/prev and compare against in-order. */
    bptNode *leaf = bt->root;
    while (!leaf->isleaf) leaf = leaf->leftmost;
    uint64_t chainCount = 0;
    unsigned char *prev = NULL; uint32_t prevlen = 0;
    bptNode *last = NULL;
    for (bptNode *cur = leaf; cur; cur = cur->next) {
        if (cur->prev != last) { printf("validate: broken prev link\n"); err++; }
        for (uint32_t i = 0; i < cur->nkeys; i++) {
            uint32_t klen;
            unsigned char *k = bptFullKey(bt, cur, i, &klen);
            if (prev && bptKeyCmp(prev, prevlen, k, klen) >= 0) {
                printf("validate: leaf chain not sorted\n"); err++;
            }
            bpt_free(prev); prev = k; prevlen = klen;
            chainCount++;
        }
        last = cur;
    }
    bpt_free(prev);
    if (chainCount != bt->numele) {
        printf("validate: leaf chain count %llu != numele %llu\n",
               (unsigned long long)chainCount, (unsigned long long)bt->numele);
        err++;
    }
    if (bt->tail != last) {
        printf("validate: cached tail is not the last leaf\n");
        err++;
    }
    return err;
}

/* --- Memory-accounting oracle -------------------------------------------- */

/* Sum the usable allocation size of a node, its overflow blobs and (for inner
 * nodes) its whole subtree -- the ground truth for the alloc_size counter. */
static size_t bptNodeAllocSize(bptree *bt, bptNode *n) {
    size_t total = zmalloc_usable_size(n);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = bptCellOff(bt, n, i);
        if (bptCellOverflow(n, off))
            total += zmalloc_usable_size(bptCellBlob(bt, n, off));
    }
    if (!n->isleaf) {
        total += bptNodeAllocSize(bt, n->leftmost);
        for (uint32_t i = 0; i < n->nkeys; i++)
            total += bptNodeAllocSize(bt, (bptNode *)bptCellPayload(bt, n, bptCellOff(bt, n, i)));
    }
    return total;
}

static size_t bptComputeAllocSize(bptree *bt) {
    return zmalloc_usable_size(bt) + bptNodeAllocSize(bt, bt->root);
}

/* Count the nodes (pages) actually present in the tree. */
static uint64_t bptCountNodes(bptree *bt, bptNode *n) {
    uint64_t c = 1;
    if (!n->isleaf) {
        c += bptCountNodes(bt, n->leftmost);
        for (uint32_t i = 0; i < n->nkeys; i++)
            c += bptCountNodes(bt, (bptNode *)bptCellPayload(bt, n, bptCellOff(bt, n, i)));
    }
    return c;
}

/* A relocation callback that always moves the block, used to stress bptDefrag:
 * every page, blob and the header ends up at a fresh address. */
static void *bptTestAlwaysMove(void *ptr) {
    size_t sz = zmalloc_usable_size(ptr);
    void *np = zmalloc(sz);
    memcpy(np, ptr, sz);
    zfree(ptr);
    return np;
}

/* --- Deterministic unit tests -------------------------------------------- */

static int bptUnitTests(void) {
    int err = 0;

    TEST("empty tree: find and seek") {
        bptree *bt = bptNewRanked(BPT_PAGE_DEFAULT, NULL);
        void *v;
        assert(bptFind(bt, (unsigned char *)"x", 1, &v) == 0);
        bptIterator it; bptStart(&it, bt);
        bptSeek(&it, "^", NULL, 0);
        assert(bptNext(&it) == 0);
        bptSeek(&it, ">=", (unsigned char *)"a", 1);
        assert(bptNext(&it) == 0);
        bptStop(&it);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("single key insert/find/overwrite/remove") {
        bptree *bt = bptNewRanked(BPT_PAGE_DEFAULT, NULL);
        void *old;
        assert(bptInsert(bt, (unsigned char *)"hello", 5, (void *)1, &old) == 1);
        void *v;
        assert(bptFind(bt, (unsigned char *)"hello", 5, &v) == 1 && v == (void *)1);
        assert(bptInsert(bt, (unsigned char *)"hello", 5, (void *)2, &old) == 0 && old == (void *)1);
        assert(bptFind(bt, (unsigned char *)"hello", 5, &v) == 1 && v == (void *)2);
        assert(bptTryInsert(bt, (unsigned char *)"hello", 5, (void *)3, &old) == 0 && old == (void *)2);
        assert(bptFind(bt, (unsigned char *)"hello", 5, &v) == 1 && v == (void *)2);
        assert(bptRemove(bt, (unsigned char *)"hello", 5, &old) == 1 && old == (void *)2);
        assert(bptFind(bt, (unsigned char *)"hello", 5, &v) == 0);
        assert(bptSize(bt) == 0);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("prefix keys and empty key") {
        bptree *bt = bptNewRanked(BPT_PAGE_DEFAULT, NULL);
        const char *keys[] = {"", "a", "ab", "abc", "abcd", "abce", "b"};
        int n = sizeof(keys) / sizeof(keys[0]);
        for (int i = 0; i < n; i++)
            assert(bptInsert(bt, (unsigned char *)keys[i], strlen(keys[i]),
                             (void *)(intptr_t)(i + 1), NULL) == 1);
        for (int i = 0; i < n; i++) {
            void *v;
            assert(bptFind(bt, (unsigned char *)keys[i], strlen(keys[i]), &v) == 1);
            assert(v == (void *)(intptr_t)(i + 1));
        }
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("long keys force overflow cells") {
        bptree *bt = bptNewRanked(256, NULL);
        char big[2000];
        for (size_t i = 0; i < sizeof(big); i++) big[i] = 'a' + (i % 5);
        for (int rep = 0; rep < 20; rep++) {
            big[0] = 'a' + rep;
            size_t len = 500 + rep * 50;
            assert(bptInsert(bt, (unsigned char *)big, len,
                             (void *)(intptr_t)(rep + 1), NULL) == 1);
        }
        for (int rep = 0; rep < 20; rep++) {
            big[0] = 'a' + rep;
            size_t len = 500 + rep * 50;
            void *v;
            assert(bptFind(bt, (unsigned char *)big, len, &v) == 1);
            assert(v == (void *)(intptr_t)(rep + 1));
        }
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("sequential inserts and deletes force split/merge/collapse") {
        bptree *bt = bptNewRanked(256, NULL);
        char kb[16];
        int N = 3000;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptInsert(bt, (unsigned char *)kb, len, (void *)(intptr_t)(i + 1), NULL) == 1);
        }
        assert((int)bptSize(bt) == N);
        assert(bt->height > 1);
        err += bptValidate(bt);
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1 && v == (void *)(intptr_t)(i + 1));
        }
        /* Delete evens, then odds. */
        for (int i = 0; i < N; i += 2) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptRemove(bt, (unsigned char *)kb, len, NULL) == 1);
        }
        err += bptValidate(bt);
        for (int i = 1; i < N; i += 2) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1);
        }
        for (int i = 1; i < N; i += 2) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptRemove(bt, (unsigned char *)kb, len, NULL) == 1);
        }
        assert(bptSize(bt) == 0);
        assert(bt->height == 1); /* fully collapsed. */
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("edge splits keep sequential insert/prepend leaves packed") {
        /* Ascending bptInsert must append-split like bptAppend: far fewer
         * pages than a middle cut (~55% fill). Descending insert must
         * prepend-split the same way. */
        char kb[16];
        int N = 20000;
        bptree *bt = bptNewRanked(512, NULL);
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptInsert(bt, (unsigned char *)kb, len,
                             (void *)(intptr_t)(i + 1), NULL) == 1);
        }
        err += bptValidate(bt);
        assert(bptNumNodes(bt) < (uint64_t)N / 12);
        bptFree(bt);

        bt = bptNewRanked(512, NULL);
        for (int i = N - 1; i >= 0; i--) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptInsert(bt, (unsigned char *)kb, len,
                             (void *)(intptr_t)(i + 1), NULL) == 1);
        }
        err += bptValidate(bt);
        assert(bptNumNodes(bt) < (uint64_t)N / 12);
        for (int i = 0; i < N; i += 97) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1 &&
                   v == (void *)(intptr_t)(i + 1));
        }
        bptFree(bt);
    }

    TEST("seek operators at boundaries and misses") {
        bptree *bt = bptNewRanked(BPT_PAGE_DEFAULT, NULL);
        const char *keys[] = {"a", "c", "e", "g"};
        for (int i = 0; i < 4; i++)
            bptInsert(bt, (unsigned char *)keys[i], 1, (void *)(intptr_t)(i + 1), NULL);
        bptIterator it; bptStart(&it, bt);

        bptSeek(&it, "^", NULL, 0);
        assert(bptNext(&it) && it.key_len == 1 && it.key[0] == 'a');
        bptSeek(&it, "$", NULL, 0);
        assert(bptPrev(&it) && it.key[0] == 'g');

        bptSeek(&it, "==", (unsigned char *)"e", 1);
        assert(bptNext(&it) && it.key[0] == 'e');
        bptSeek(&it, "==", (unsigned char *)"b", 1);
        assert(bptNext(&it) == 0);

        bptSeek(&it, ">=", (unsigned char *)"c", 1);
        assert(bptNext(&it) && it.key[0] == 'c');
        bptSeek(&it, ">", (unsigned char *)"c", 1);
        assert(bptNext(&it) && it.key[0] == 'e');
        bptSeek(&it, ">", (unsigned char *)"b", 1);
        assert(bptNext(&it) && it.key[0] == 'c');
        bptSeek(&it, ">", (unsigned char *)"g", 1);
        assert(bptNext(&it) == 0);

        bptSeek(&it, "<=", (unsigned char *)"e", 1);
        assert(bptPrev(&it) && it.key[0] == 'e');
        bptSeek(&it, "<", (unsigned char *)"e", 1);
        assert(bptPrev(&it) && it.key[0] == 'c');
        bptSeek(&it, "<", (unsigned char *)"f", 1);
        assert(bptPrev(&it) && it.key[0] == 'e');
        bptSeek(&it, "<", (unsigned char *)"a", 1);
        assert(bptPrev(&it) == 0);

        /* Full forward scan. */
        bptSeek(&it, "^", NULL, 0);
        int cnt = 0; char exp = 'a';
        while (bptNext(&it)) { assert(it.key[0] == exp); exp += 2; cnt++; }
        assert(cnt == 4);
        /* Full backward scan. */
        bptSeek(&it, "$", NULL, 0);
        cnt = 0; exp = 'g';
        while (bptPrev(&it)) { assert(it.key[0] == exp); exp -= 2; cnt++; }
        assert(cnt == 4);
        bptStop(&it);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("safe iterator: mutation between seek and first next") {
        bptree *bt = bptNewRanked(256, NULL);
        char kb[16];
        int N = 500;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%04d", i);
            bptInsert(bt, (unsigned char *)kb, len, (void *)(intptr_t)(i + 1), NULL);
        }
        bptIterator it; bptStart(&it, bt);
        it.flags |= BPT_ITER_SAFE;
        bptSeek(&it, ">=", (unsigned char *)"k0100", 5);
        assert(it.flags & BPT_ITER_SAFE); /* seek must preserve SAFE. */
        /* Delete everything: every leaf the iterator could point at is freed. */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%04d", i);
            assert(bptRemove(bt, (unsigned char *)kb, len, NULL) == 1);
        }
        /* First step returns the seek-time snapshot without touching pages. */
        assert(bptNext(&it) == 1);
        assert(it.key_len == 5 && memcmp(it.key, "k0100", 5) == 0);
        assert(it.data == (void *)(intptr_t)101);
        /* Second step re-seeks against the (now empty) tree. */
        assert(bptNext(&it) == 0);
        bptStop(&it);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("safe iterator survives structural mutations while scanning") {
        bptree *bt = bptNewRanked(256, NULL);
        char kb[16];
        int N = 400;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%04d", i);
            bptInsert(bt, (unsigned char *)kb, len, (void *)(intptr_t)(i + 1), NULL);
        }
        bptIterator it; bptStart(&it, bt);
        it.flags |= BPT_ITER_SAFE;
        bptSeek(&it, "^", NULL, 0);
        int seen = 0;
        while (bptNext(&it)) {
            assert(it.key_len == 5);
            assert(it.data == (void *)(intptr_t)(seen + 1));
            seen++;
            /* Mutate behind the cursor: delete the visited key (merges the
             * iterator's leaf eventually) and insert a smaller key (splits
             * in the 'a' region). Neither affects the remaining scan. */
            assert(bptRemove(bt, it.key, it.key_len, NULL) == 1);
            int len = snprintf(kb, sizeof(kb), "a%04d", seen);
            bptInsert(bt, (unsigned char *)kb, len, NULL, NULL);
        }
        assert(seen == N);
        assert(bptSize(bt) == (uint64_t)N); /* the inserted 'a' keys. */
        bptStop(&it);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("alloc_size and numnodes tracked across split/merge") {
        size_t acc = 0;
        bptree *bt = bptNewRanked(256, &acc);
        /* Freshly created tree: header + one root leaf. */
        assert(bptNumNodes(bt) == 1);
        assert(acc == bptComputeAllocSize(bt));
        char kb[16];
        int N = 2000;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            bptInsert(bt, (unsigned char *)kb, len, (void *)(intptr_t)(i + 1), NULL);
        }
        assert(bt->height > 1);
        assert(bptNumNodes(bt) == bptCountNodes(bt, bt->root));
        assert(acc == bptComputeAllocSize(bt));
        /* Long keys exercise overflow-blob accounting too. */
        char big[900];
        for (size_t i = 0; i < sizeof(big); i++) big[i] = 'z';
        for (int i = 0; i < 30; i++) {
            big[0] = (char)('A' + i);
            bptInsert(bt, (unsigned char *)big, sizeof(big), (void *)(intptr_t)i, NULL);
        }
        assert(acc == bptComputeAllocSize(bt));
        for (int i = 0; i < 30; i++) {
            big[0] = (char)('A' + i);
            assert(bptRemove(bt, (unsigned char *)big, sizeof(big), NULL) == 1);
        }
        assert(acc == bptComputeAllocSize(bt));
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            assert(bptRemove(bt, (unsigned char *)kb, len, NULL) == 1);
        }
        assert(bptSize(bt) == 0);
        assert(bptNumNodes(bt) == 1); /* collapsed back to the root leaf. */
        assert(bptNumNodes(bt) == bptCountNodes(bt, bt->root));
        assert(acc == bptComputeAllocSize(bt));
        err += bptValidate(bt);
        bptFree(bt);
        /* Every accounted byte was released. */
        assert(acc == 0);
    }

    TEST("defrag relocates every node/blob and keeps the tree consistent") {
        bptree *bt = bptNewRanked(256, NULL);
        char kb[16];
        int N = 1500;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            bptInsert(bt, (unsigned char *)kb, len, (void *)(intptr_t)(i + 1), NULL);
        }
        /* Add long keys so overflow blobs must be relocated as well. */
        char big[700];
        for (size_t i = 0; i < sizeof(big); i++) big[i] = 'm';
        for (int i = 0; i < 20; i++) {
            big[0] = (char)('a' + i);
            bptInsert(bt, (unsigned char *)big, sizeof(big),
                      (void *)(intptr_t)(100000 + i), NULL);
        }
        assert(bt->height > 1);
        err += bptValidate(bt);

        bptDefrag(&bt, bptTestAlwaysMove);
        err += bptValidate(bt);

        /* Every key still resolves to its value. */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1);
            assert(v == (void *)(intptr_t)(i + 1));
        }
        for (int i = 0; i < 20; i++) {
            big[0] = (char)('a' + i);
            void *v;
            assert(bptFind(bt, (unsigned char *)big, sizeof(big), &v) == 1);
            assert(v == (void *)(intptr_t)(100000 + i));
        }
        /* Forward scan is ordered and complete after relocation. */
        bptIterator it; bptStart(&it, bt);
        bptSeek(&it, "^", NULL, 0);
        uint64_t fwd = 0;
        unsigned char prev[16]; uint32_t prevlen = 0; int haveprev = 0;
        while (bptNext(&it)) {
            if (haveprev && it.key_len <= sizeof(prev))
                assert(bptKeyCmp(prev, prevlen, it.key, it.key_len) < 0);
            if (it.key_len <= sizeof(prev)) {
                memcpy(prev, it.key, it.key_len); prevlen = it.key_len; haveprev = 1;
            } else haveprev = 0;
            fwd++;
        }
        assert(fwd == bptSize(bt));
        /* Backward scan visits the same number of elements. */
        bptSeek(&it, "$", NULL, 0);
        uint64_t bwd = 0;
        while (bptPrev(&it)) bwd++;
        assert(bwd == bptSize(bt));
        bptStop(&it);
        bptFree(bt);
    }

    TEST("incremental defrag preserves ranked tree between steps") {
        bptree *bt = bptNewRanked(256, NULL);
        char kb[700];
        int N = 500;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            if (i % 25 == 0) {
                memset(kb + len, 'x', sizeof(kb) - len);
                len = sizeof(kb);
            }
            bptInsert(bt, (unsigned char *)kb, len,
                      (void *)(intptr_t)(i + 1), NULL);
        }
        assert(bt->height > 1);
        err += bptValidate(bt);

        uint64_t cursor = 0;
        int steps = 0;
        do {
            steps++;
            int more = bptDefragStep(&bt, &cursor, bptTestAlwaysMove);
            err += bptValidate(bt);
            if (!more) break;
        } while (1);
        assert(steps > 1);
        assert(cursor == 0);

        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            if (i % 25 == 0) {
                memset(kb + len, 'x', sizeof(kb) - len);
                len = sizeof(kb);
            }
            void *v;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1);
            assert(v == (void *)(intptr_t)(i + 1));
        }
        bptFree(bt);
    }

    TEST("append: sequential IDs, duplicates, out-of-order fallback") {
        bptree *bt = bptNewRanked(512, NULL);
        unsigned char kb[16];
        int N = 100000;
        /* Stream-ID-like keys: 16-byte big-endian, strictly increasing. */
        for (int i = 0; i < N; i++) {
            uint64_t ms = 1700000000000ULL + (uint64_t)(i / 100), seq = i % 100;
            for (int b = 0; b < 8; b++) {
                kb[b] = (ms >> (56 - 8*b)) & 0xff;
                kb[8+b] = (seq >> (56 - 8*b)) & 0xff;
            }
            assert(bptAppend(bt, kb, 16, (void *)(intptr_t)(i + 1), NULL) == 1);
        }
        assert(bptSize(bt) == (uint64_t)N);
        err += bptValidate(bt);
        /* Append-split keeps leaves ~full: sequential load must use far fewer
         * pages than middle-splitting (which converges to ~55% fill). */
        uint64_t nodes = bptNumNodes(bt);
        assert(nodes < (uint64_t)N / 12); /* >= ~12 keys/page on average. */
        /* Duplicate of the maximum: rejected, old value returned. */
        void *ov = NULL;
        assert(bptAppend(bt, kb, 16, (void *)0xdead, &ov) == 0);
        assert(ov == (void *)(intptr_t)N);
        /* Duplicate of an interior key: same, via the fallback path. */
        kb[15] = 0; /* seq 0 of the last ms: an existing interior key. */
        assert(bptAppend(bt, kb, 16, (void *)0xdead, &ov) == 0);
        assert(ov != NULL && ov != (void *)0xdead);
        /* Out-of-order new key: falls back to an ordered insert. */
        unsigned char small[16] = {0};
        assert(bptAppend(bt, small, 16, (void *)0xbeef, &ov) == 1);
        void *v;
        assert(bptFind(bt, small, 16, &v) == 1 && v == (void *)0xbeef);
        err += bptValidate(bt);
        /* Everything still resolves. */
        for (int i = 0; i < N; i += 997) {
            uint64_t ms = 1700000000000ULL + (uint64_t)(i / 100), seq = i % 100;
            for (int b = 0; b < 8; b++) {
                kb[b] = (ms >> (56 - 8*b)) & 0xff;
                kb[8+b] = (seq >> (56 - 8*b)) & 0xff;
            }
            assert(bptFind(bt, kb, 16, &v) == 1 && v == (void *)(intptr_t)(i + 1));
        }
        /* Deleting from the tail keeps the cached tail correct (merges). */
        for (int i = N - 1; i >= N - 5000; i--) {
            uint64_t ms = 1700000000000ULL + (uint64_t)(i / 100), seq = i % 100;
            for (int b = 0; b < 8; b++) {
                kb[b] = (ms >> (56 - 8*b)) & 0xff;
                kb[8+b] = (seq >> (56 - 8*b)) & 0xff;
            }
            assert(bptRemove(bt, kb, 16, NULL) == 1);
        }
        err += bptValidate(bt);
        /* And appending again after tail deletions works. */
        memset(kb, 0xff, 16);
        assert(bptAppend(bt, kb, 16, (void *)0xf00d, NULL) == 1);
        err += bptValidate(bt);
        bptFree(bt);
    }

    return err;
}

/* --- Randomized fuzz against rax as oracle -------------------------------- */

static size_t bptGenKey(unsigned char *buf, uint32_t pageSize) {
    int r = rand() % 100;
    size_t len;
    int alpha;
    if (r < 3) {            /* long key -> overflow. */
        len = pageSize / 4 + (size_t)(rand() % (pageSize * 2));
        alpha = 3;
    } else if (r < 6) {     /* empty or tiny. */
        len = rand() % 2;
        alpha = 4;
    } else {                /* short, heavily shared prefixes. */
        len = rand() % 14;
        alpha = 2 + rand() % 4;
    }
    for (size_t i = 0; i < len; i++) buf[i] = 'a' + (rand() % alpha);
    return len;
}

static int bptFuzz(int flags) {
    int err = 0;
    int rounds = (flags & REDIS_TEST_ACCURATE) ? 40 : 8;
    int batch = (flags & REDIS_TEST_ACCURATE) ? 4000 : 1500;
    uint32_t pageSizes[] = {256, 512, 4096};

    for (unsigned ps = 0; ps < sizeof(pageSizes) / sizeof(pageSizes[0]); ps++) {
        uint32_t pageSize = pageSizes[ps];
        srand(0xB17 + pageSize);
        bptree *bt = bptNewRanked(pageSize, NULL);
        rax *rt = raxNew();

        size_t poolCap = 4096, poolLen = 0;
        unsigned char **pool = zmalloc(sizeof(char *) * poolCap);
        size_t *poolLens = zmalloc(sizeof(size_t) * poolCap);
        unsigned char kbuf[16384];

        for (int round = 0; round < rounds; round++) {
            for (int step = 0; step < batch; step++) {
                size_t klen;
                int useExisting = poolLen > 0 && (rand() % 3 == 0);
                if (useExisting) {
                    size_t idx = rand() % poolLen;
                    klen = poolLens[idx];
                    memcpy(kbuf, pool[idx], klen);
                } else {
                    klen = bptGenKey(kbuf, pageSize);
                }
                int doInsert = (round < rounds / 2) ? (rand() % 4 != 0) : (rand() % 2);
                if (doInsert) {
                    void *val = (rand() % 16 == 0) ? NULL : bptTestVal(kbuf, klen);
                    void *ob = (void *)0x1, *orx = (void *)0x1;
                    int rb, rr;
                    int variant = rand() % 2;
                    if (variant == 0) {
                        /* bptAppend must behave exactly like a try-insert for
                         * arbitrary keys (fast paths for maximal keys, plain
                         * fallback otherwise). */
                        rb = bptAppend(bt, kbuf, klen, val, &ob);
                        rr = raxTryInsert(rt, kbuf, klen, val, &orx);
                    } else {
                        rb = bptInsert(bt, kbuf, klen, val, &ob);
                        rr = raxInsert(rt, kbuf, klen, val, &orx);
                    }
                    assert(rb == rr);
                    if (rb == 0) assert(ob == orx);
                    if (!useExisting && poolLen < poolCap) {
                        pool[poolLen] = zmalloc(klen ? klen : 1);
                        memcpy(pool[poolLen], kbuf, klen);
                        poolLens[poolLen] = klen;
                        poolLen++;
                    }
                } else {
                    void *ob = (void *)0x1, *orx = (void *)0x1;
                    int rb = bptRemove(bt, kbuf, klen, &ob);
                    int rr = raxRemove(rt, kbuf, klen, &orx);
                    assert(rb == rr);
                    if (rb) assert(ob == orx);
                }
            }

            /* Cross-check size, structure, and both scan directions. */
            assert(bptSize(bt) == raxSize(rt));
            err += bptValidate(bt);

            bptIterator bi; bptStart(&bi, bt); bptSeek(&bi, "^", NULL, 0);
            raxIterator ri; raxStart(&ri, rt); raxSeek(&ri, "^", NULL, 0);
            uint64_t seen = 0;
            while (raxNext(&ri)) {
                int more = bptNext(&bi);
                assert(more);
                assert(bi.key_len == ri.key_len);
                assert(bi.key_len == 0 || memcmp(bi.key, ri.key, bi.key_len) == 0);
                assert(bi.data == ri.data);
                seen++;
            }
            assert(bptNext(&bi) == 0);
            assert(seen == bptSize(bt));
            bptStop(&bi); raxStop(&ri);

            bptStart(&bi, bt); bptSeek(&bi, "$", NULL, 0);
            raxStart(&ri, rt); raxSeek(&ri, "$", NULL, 0);
            while (raxPrev(&ri)) {
                int more = bptPrev(&bi);
                assert(more);
                assert(bi.key_len == ri.key_len);
                assert(bi.key_len == 0 || memcmp(bi.key, ri.key, bi.key_len) == 0);
                assert(bi.data == ri.data);
            }
            assert(bptPrev(&bi) == 0);
            bptStop(&bi); raxStop(&ri);

            /* Random point lookups and range seeks. */
            for (int s = 0; s < 200 && poolLen; s++) {
                size_t idx = rand() % poolLen;
                void *vb, *vr;
                int fb = bptFind(bt, pool[idx], poolLens[idx], &vb);
                int fr = raxFind(rt, pool[idx], poolLens[idx], &vr);
                assert(fb == fr);
                if (fb) assert(vb == vr);

                const char *op = (rand() % 2) ? ">=" : "<=";
                bptIterator b2; bptStart(&b2, bt);
                raxIterator r2; raxStart(&r2, rt);
                bptSeek(&b2, op, pool[idx], poolLens[idx]);
                raxSeek(&r2, op, pool[idx], poolLens[idx]);
                int mb, mr;
                if (op[0] == '>') { mb = bptNext(&b2); mr = raxNext(&r2); }
                else { mb = bptPrev(&b2); mr = raxPrev(&r2); }
                assert(mb == mr);
                if (mb) {
                    assert(b2.key_len == r2.key_len);
                    assert(b2.key_len == 0 || memcmp(b2.key, r2.key, b2.key_len) == 0);
                }
                bptStop(&b2); raxStop(&r2);
            }
        }

        for (size_t i = 0; i < poolLen; i++) zfree(pool[i]);
        zfree(pool); zfree(poolLens);
        bptFree(bt);
        raxFree(rt);
        printf("  fuzz pageSize=%u ok\n", pageSize);
    }
    return err;
}

/* --- Benchmark vs rax ---------------------------------------------------- */

static void bptBenchOne(const char *label, int N, unsigned char **keys,
                        size_t *lens, int fixedLen, int useAppend) {
    printf("benchmark: %s — %d keys\n", label, N);

    /* Lookups follow a shuffled order (the same one for both structures):
     * probing in insertion order would keep hitting the leaf the previous probe
     * just loaded, hiding everything about how a node is laid out. */
    int *order = zmalloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) order[i] = i;
    uint64_t rs = 0x9E3779B97F4A7C15ULL;
    for (int i = N - 1; i > 0; i--) {
        rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
        int j = (int)(rs % (uint64_t)(i + 1));
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }

    for (int which = 0; which < 2; which++) {
        int isBpt = (which == 0);
        size_t memBefore = zmalloc_used_memory();
        bptree *bt = NULL; rax *rt = NULL;
        if (isBpt) bt = bptNewRanked(BPT_PAGE_DEFAULT, NULL);
        else rt = fixedLen ? raxNewEx(0, NULL, (uint32_t)fixedLen) : raxNew();

        long long t0 = bptUsTime();
        for (int i = 0; i < N; i++) {
            if (isBpt && useAppend) bptAppend(bt, keys[i], lens[i], (void *)(intptr_t)(i+1), NULL);
            else if (isBpt) bptInsert(bt, keys[i], lens[i], (void *)(intptr_t)(i+1), NULL);
            else raxInsert(rt, keys[i], lens[i], (void *)(intptr_t)(i+1), NULL);
        }
        long long t1 = bptUsTime();
        size_t memAfter = zmalloc_used_memory();

        volatile long hits = 0;
        for (int i = 0; i < N; i++) {
            void *v;
            int j = order[i];
            if (isBpt) { if (bptFind(bt, keys[j], lens[j], &v)) hits++; }
            else { if (raxFind(rt, keys[j], lens[j], &v)) hits++; }
        }
        long long t2 = bptUsTime();

        /* Seek-to-tail (XADD hot path): N times. */
        for (int i = 0; i < N; i++) {
            if (isBpt) {
                bptIterator it; bptStart(&it, bt); bptSeek(&it, "$", NULL, 0);
                hits += !bptEOF(&it); bptStop(&it);
            } else {
                raxIterator it; raxStart(&it, rt); raxSeek(&it, "$", NULL, 0);
                hits += !raxEOF(&it); raxStop(&it);
            }
        }
        long long t2b = bptUsTime();

        uint64_t scan = 0;
        if (isBpt) {
            bptIterator it; bptStart(&it, bt); bptSeek(&it, "^", NULL, 0);
            while (bptNext(&it)) scan++;
            bptStop(&it);
        } else {
            raxIterator it; raxStart(&it, rt); raxSeek(&it, "^", NULL, 0);
            while (raxNext(&it)) scan++;
            raxStop(&it);
        }
        long long t3 = bptUsTime();

        uint64_t nnodes = isBpt ? bptNumNodes(bt) : rt->numnodes;

        for (int i = 0; i < N; i++) {
            if (isBpt) bptRemove(bt, keys[i], lens[i], NULL);
            else raxRemove(rt, keys[i], lens[i], NULL);
        }
        long long t4 = bptUsTime();

        printf("  %-10s insert %6.2f  find %6.2f  seek$ %6.2f  scan %6.2f  "
               "delete %6.2f  Mops/s  mem %.2f MB  nodes %llu\n",
               isBpt ? "bptree" : (fixedLen ? "rax-fixed" : "rax"),
               N / (double)(t1 - t0),
               N / (double)(t2 - t1),
               N / (double)(t2b - t2),
               scan / (double)(t3 - t2b),
               N / (double)(t4 - t3),
               (memAfter - memBefore) / (1024.0 * 1024.0),
               (unsigned long long)nnodes);
        (void)hits; (void)scan;
        if (isBpt) bptFree(bt); else raxFree(rt);
    }
    zfree(order);
}

static void bptBench(int flags) {
    int N = (flags & REDIS_TEST_ACCURATE) ? 2000000 : 300000;

    /* 1) Random variable-length string keys (generic ordered map). */
    {
        unsigned char **keys = zmalloc(sizeof(char *) * N);
        size_t *lens = zmalloc(sizeof(size_t) * N);
        char kb[32];
        srand(12345);
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "key:%d:%d", rand(), i);
            keys[i] = zmalloc(len);
            memcpy(keys[i], kb, len);
            lens[i] = len;
        }
        bptBenchOne("random string keys (bptree vs rax)", N, keys, lens, 0, 0);
        for (int i = 0; i < N; i++) zfree(keys[i]);
        zfree(keys); zfree(lens);
    }

    /* 2) Stream-like: 16-byte big-endian sequential IDs (one key per listpack
     * node). Compare against fixed-length leaf-inlined rax — what streams used. */
    {
        int Ns = (flags & REDIS_TEST_ACCURATE) ? 500000 : 100000;
        unsigned char **keys = zmalloc(sizeof(char *) * Ns);
        size_t *lens = zmalloc(sizeof(size_t) * Ns);
        for (int i = 0; i < Ns; i++) {
            keys[i] = zmalloc(16);
            /* ms = i, seq = 0, big-endian like streamEncodeID. */
            uint64_t ms = (uint64_t)i, seq = 0;
            for (int b = 0; b < 8; b++) {
                keys[i][b] = (ms >> (56 - 8*b)) & 0xff;
                keys[i][8+b] = (seq >> (56 - 8*b)) & 0xff;
            }
            lens[i] = 16;
        }
        bptBenchOne("stream-like 16B sequential IDs vs rax-fixed(16)",
                    Ns, keys, lens, 16, 0);
        bptBenchOne("stream-like 16B sequential IDs (bptAppend) vs rax-fixed(16)",
                    Ns, keys, lens, 16, 1);
        for (int i = 0; i < Ns; i++) zfree(keys[i]);
        zfree(keys); zfree(lens);
    }

    /* 3) Same IDs but denser suffix collision (same ms, varying seq) — models
     * many XADDs within one millisecond / clustered prefixes. */
    {
        int Ns = (flags & REDIS_TEST_ACCURATE) ? 500000 : 100000;
        unsigned char **keys = zmalloc(sizeof(char *) * Ns);
        size_t *lens = zmalloc(sizeof(size_t) * Ns);
        for (int i = 0; i < Ns; i++) {
            keys[i] = zmalloc(16);
            uint64_t ms = 1700000000000ULL + (i / 1000); /* 1000 keys / ms */
            uint64_t seq = (uint64_t)(i % 1000);
            for (int b = 0; b < 8; b++) {
                keys[i][b] = (ms >> (56 - 8*b)) & 0xff;
                keys[i][8+b] = (seq >> (56 - 8*b)) & 0xff;
            }
            lens[i] = 16;
        }
        bptBenchOne("stream-like 16B clustered (1000 seq/ms) vs rax-fixed(16)",
                    Ns, keys, lens, 16, 0);
        bptBenchOne("stream-like 16B clustered (bptAppend) vs rax-fixed(16)",
                    Ns, keys, lens, 16, 1);
        for (int i = 0; i < Ns; i++) zfree(keys[i]);
        zfree(keys); zfree(lens);
    }
}

/* --- Rank/select tests --------------------------------------------------- */

/* Build a sorted reference of the tree's contents by in-order scan, then check
 * bptRankOf()/bptSelect() against it for every element (and one out-of-range
 * select). bptValidate() independently checks that the stored per-child counts
 * equal the real subtree sizes, so together they pin the ranked invariants. */
static int bptRankRefCheck(bptree *bt) {
    int err = 0;
    uint64_t n = bptSize(bt);
    unsigned char **ref = bpt_malloc(sizeof(char *) * (n ? n : 1));
    uint32_t *reflen = bpt_malloc(sizeof(uint32_t) * (n ? n : 1));

    bptIterator it;
    bptStart(&it, bt);
    bptSeek(&it, "^", NULL, 0);
    uint64_t i = 0;
    while (bptNext(&it)) {
        if (i >= n) { i++; break; }
        ref[i] = bpt_malloc(it.key_len ? it.key_len : 1);
        memcpy(ref[i], it.key, it.key_len);
        reflen[i] = (uint32_t)it.key_len;
        i++;
    }
    bptStop(&it);
    if (i != n) { printf("rank: scan count %llu != size %llu\n",
                         (unsigned long long)i, (unsigned long long)n); err++; }

    for (uint64_t j = 0; j < n && j < i; j++) {
        uint64_t rank;
        int f = bptRankOf(bt, ref[j], reflen[j], &rank);
        if (!f || rank != j) {
            printf("rank: key at %llu got found=%d rank=%llu\n",
                   (unsigned long long)j, f, (unsigned long long)rank); err++;
        }
        bptIterator sit;
        bptStart(&sit, bt);
        if (!bptSelect(&sit, j)) { printf("select: %llu failed\n",
                                          (unsigned long long)j); err++; }
        else if (bptKeyCmp(sit.key, sit.key_len, ref[j], reflen[j]) != 0) {
            printf("select: %llu wrong key\n", (unsigned long long)j); err++;
        }
        bptStop(&sit);
    }

    bptIterator oit;
    bptStart(&oit, bt);
    if (bptSelect(&oit, n)) { printf("select: out-of-range succeeded\n"); err++; }
    bptStop(&oit);

    for (uint64_t j = 0; j < i && j < n; j++) bpt_free(ref[j]);
    bpt_free(ref);
    bpt_free(reflen);
    return err;
}

static int bptRankedTests(int flags) {
    TEST("bptree ranked rank/select (insert/delete fuzz)");
    int err = 0;
    int rounds = (flags & REDIS_TEST_ACCURATE) ? 6 : 3;
    int M = (flags & REDIS_TEST_ACCURATE) ? 20000 : 4000;
    srand(9997);
    for (int r = 0; r < rounds && err == 0; r++) {
        /* A small page forces a tall tree with plenty of splits and merges. */
        bptree *bt = bptNewRanked(256, NULL);
        for (int i = 0; i < M; i++) {
            char kb[40];
            int len = snprintf(kb, sizeof(kb), "k:%d", rand() % (M * 2));
            bptInsert(bt, (unsigned char *)kb, len,
                      bptTestVal((unsigned char *)kb, len), NULL);
        }
        if (bptValidate(bt)) { printf("ranked: validate after insert failed\n"); err++; }
        err += bptRankRefCheck(bt);

        /* Delete a random ~half of the current contents. */
        uint64_t n = bptSize(bt);
        unsigned char **cur = bpt_malloc(sizeof(char *) * (n ? n : 1));
        uint32_t *curlen = bpt_malloc(sizeof(uint32_t) * (n ? n : 1));
        bptIterator it; bptStart(&it, bt); bptSeek(&it, "^", NULL, 0);
        uint64_t c = 0;
        while (bptNext(&it) && c < n) {
            cur[c] = bpt_malloc(it.key_len ? it.key_len : 1);
            memcpy(cur[c], it.key, it.key_len); curlen[c] = (uint32_t)it.key_len; c++;
        }
        bptStop(&it);
        for (uint64_t j = 0; j < c; j++) {
            if (rand() & 1) bptRemove(bt, cur[j], curlen[j], NULL);
            bpt_free(cur[j]);
        }
        bpt_free(cur); bpt_free(curlen);

        if (bptValidate(bt)) { printf("ranked: validate after delete failed\n"); err++; }
        err += bptRankRefCheck(bt);
        bptFree(bt);
    }

    /* Monotonic bptAppend into a ranked tree, then rank/select. */
    {
        bptree *bt = bptNewRanked(512, NULL);
        int Na = (flags & REDIS_TEST_ACCURATE) ? 100000 : 20000;
        for (int i = 0; i < Na; i++) {
            unsigned char kb[8];
            for (int b = 0; b < 8; b++) kb[b] = ((uint64_t)i >> (56 - 8 * b)) & 0xff;
            bptAppend(bt, kb, 8, bptTestVal(kb, 8), NULL);
        }
        if (bptValidate(bt)) { printf("ranked: validate after append failed\n"); err++; }
        err += bptRankRefCheck(bt);
        bptFree(bt);
    }
    return err;
}

int bptTest(int argc, char *argv[], int flags) {
    UNUSED(argc);
    UNUSED(argv);
    int err = 0;

    err += bptUnitTests();
    err += bptFuzz(flags);
    err += bptRankedTests(flags);
    if (!(flags & REDIS_TEST_VALGRIND)) bptBench(flags);

    if (err) printf("bptree: %d test failures\n", err);
    else printf("bptree: all tests passed\n");
    return err;
}

#endif
