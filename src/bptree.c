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
#define BPT_SLOT_SIZE ((uint32_t)sizeof(uint16_t))

/* Cell layout: [u32 suffixLen][u8 flags][inline suffix][blob* if overflow][payload].
 * The 3 non-inline, non-payload bytes of a cell header are 4 (len) + 1 (flags). */
#define BPT_CELL_HDR 5
#define BPT_PTR_SIZE ((uint32_t)sizeof(void *))
#define BPT_CFLAG_OVERFLOW 0x01

/* Deletion rebalance threshold: a non-root node underflows when its live byte
 * usage drops below this fraction (in percent) of the page. */
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
 * structural operations (split/merge/borrow/prefix-rewrite) that rebuild nodes
 * from scratch. `key` is heap-owned by the extracting caller. */
typedef struct bptEnt {
    unsigned char *key;
    uint32_t keylen;
    void *payload;           /* value (leaf) or child bptNode* (inner). */
} bptEnt;

static inline uint16_t *bptSlots(bptNode *n) {
    return (uint16_t *)((unsigned char *)n + BPT_HDR_SIZE);
}
static inline unsigned char *bptRaw(bptNode *n, uint32_t off) {
    return (unsigned char *)n + off;
}
static inline unsigned char *bptPrefix(bptNode *n) {
    return (unsigned char *)n + n->prefixOff;
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

/* Free bytes available for a new (slot + cell) without compaction. */
static inline uint32_t bptFreeBytes(bptNode *n) {
    uint32_t used = BPT_HDR_SIZE + n->nkeys * BPT_SLOT_SIZE;
    return n->heapStart - used;
}
/* Live bytes accounted to a node (header + slots + live heap). */
static inline uint32_t bptUsedBytes(bptree *bt, bptNode *n) {
    uint32_t heapLive = bt->pageSize - n->heapStart - n->deadBytes;
    return BPT_HDR_SIZE + n->nkeys * BPT_SLOT_SIZE + heapLive;
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
    bptNode *n = bptMallocAccounted(bt, bt->pageSize);
    if (!n) return NULL;
    bt->numnodes++;
    n->isleaf = isleaf ? 1 : 0;
    n->nkeys = 0;
    n->heapStart = bt->pageSize;
    n->deadBytes = 0;
    n->prefixOff = bt->pageSize;
    n->prefixLen = 0;
    n->next = NULL;
    n->prev = NULL;
    n->leftmost = NULL;
    return n;
}

/* Free the overflow blobs referenced by a node's cells (not the node itself). */
static void bptFreeNodeBlobs(bptree *bt, bptNode *n) {
    uint16_t *slots = bptSlots(n);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = slots[i];
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
    return (bptNode *)bptCellPayload(bt, n, bptSlots(n)[cidx - 1]);
}

/* ------------------------------------------------------------------------- */
/* Key materialization and comparison inside a node                          */
/* ------------------------------------------------------------------------- */

/* Reconstruct the full key of slot i into a freshly malloc'd buffer. */
static unsigned char *bptFullKey(bptree *bt, bptNode *n, uint32_t i, uint32_t *outlen) {
    uint32_t off = bptSlots(n)[i];
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

/* Lower-bound search within a node: returns the index of the first slot whose
 * key is >= the search key, and sets *found when an exact match exists. The
 * returned index is in [0, nkeys]. */
static uint32_t bptNodeSearch(bptree *bt, bptNode *n,
                              const unsigned char *key, uint32_t len, int *found) {
    *found = 0;
    bptPfxRel rel = bptPrefixRel(n, key, len);
    if (rel == BPT_PFX_BELOW) return 0;
    if (rel == BPT_PFX_ABOVE) return n->nkeys;

    const unsigned char *s = key + n->prefixLen;
    uint32_t slen = len - n->prefixLen;
    uint16_t *slots = bptSlots(n);
    uint32_t lo = 0, hi = n->nkeys;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = bptCmpSuffix(bt, n, slots[mid], s, slen); /* sign(search - stored) */
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

/* Reclaim dead bytes by repacking the prefix and all live cells to the top of
 * the page. Blob pointers stay valid (only the cell bytes move). */
static void bptCompact(bptree *bt, bptNode *n) {
    if (n->deadBytes == 0) return;
    unsigned char *tmp = bpt_malloc(bt->pageSize);
    uint32_t top = bt->pageSize;

    /* Move the prefix. */
    if (n->prefixLen) {
        top -= n->prefixLen;
        memcpy(tmp + top, bptPrefix(n), n->prefixLen);
        n->prefixOff = top;
    } else {
        n->prefixOff = bt->pageSize;
    }
    /* Cells start at a pointer-aligned offset (see bptCellSize). */
    top &= ~((uint32_t)BPT_PTR_SIZE - 1);
    /* Move each live cell (in slot order for locality). */
    uint16_t *slots = bptSlots(n);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = slots[i];
        uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off),
                                     bptCellOverflow(n, off));
        top -= csize;
        memcpy(tmp + top, bptRaw(n, off), csize);
        slots[i] = (uint16_t)top;
    }
    memcpy((unsigned char *)n + top, tmp + top, bt->pageSize - top);
    n->heapStart = top;
    n->deadBytes = 0;
    bpt_free(tmp);
}

/* ------------------------------------------------------------------------- */
/* Building a node from a sorted list of entries                             */
/* ------------------------------------------------------------------------- */

/* Bytes required to store entries[start..start+count) in a single node,
 * given they'd share the common prefix `cp`. Includes header and slots. */
static uint32_t bptEntriesBytes(bptree *bt, bptEnt *ents, uint32_t start,
                                uint32_t count, uint32_t cp) {
    /* When a prefix is present the first cell offset is rounded down to a
     * pointer boundary, wasting up to BPT_PTR_SIZE-1 bytes; reserve that so a
     * "fits" verdict never overflows the page after alignment. */
    uint32_t total = BPT_HDR_SIZE + cp + count * BPT_SLOT_SIZE +
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

static int bptEntriesFit(bptree *bt, bptEnt *ents, uint32_t start, uint32_t count) {
    uint32_t cp = bptEntriesCp(bt, ents, start, count);
    return bptEntriesBytes(bt, ents, start, count, cp) <= bt->pageSize;
}

/* (Re)build node `n` in place from entries[start..start+count) with an
 * explicit stored-prefix length `cp` (any value <= the entries' true common
 * prefix is valid; a shorter one just stores longer suffixes). The node's
 * next/prev/leftmost links are preserved (managed by the caller). Any existing
 * blobs must have been freed by the caller. Must fit. */
static void bptBuildNodeCp(bptree *bt, bptNode *n, int isleaf,
                           bptEnt *ents, uint32_t start, uint32_t count,
                           uint32_t cp) {
    n->isleaf = isleaf ? 1 : 0;
    n->nkeys = 0;
    n->heapStart = bt->pageSize;
    n->deadBytes = 0;

    if (cp) {
        n->heapStart -= cp;
        memcpy(bptRaw(n, n->heapStart), ents[start].key, cp);
        n->prefixOff = n->heapStart;
        n->prefixLen = cp;
    } else {
        n->prefixOff = bt->pageSize;
        n->prefixLen = 0;
    }
    /* Cells start at a pointer-aligned offset (see bptCellSize). */
    n->heapStart &= ~((uint32_t)BPT_PTR_SIZE - 1);

    uint16_t *slots = bptSlots(n);
    for (uint32_t i = 0; i < count; i++) {
        bptEnt *e = &ents[start + i];
        uint32_t slen = e->keylen - cp;
        uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
        /* Check before subtracting: heapStart is unsigned, so a wrapped
         * subtraction would sail past a post-decrement bound check. */
        assert(csize <= n->heapStart - (BPT_HDR_SIZE + count * BPT_SLOT_SIZE));
        n->heapStart -= csize;
        bptWriteCell(bt, n, n->heapStart, e->key + cp, slen, e->payload);
        slots[i] = (uint16_t)n->heapStart;
    }
    n->nkeys = count;
}

static void bptBuildNode(bptree *bt, bptNode *n, int isleaf,
                         bptEnt *ents, uint32_t start, uint32_t count) {
    bptBuildNodeCp(bt, n, isleaf, ents, start, count,
                   bptEntriesCp(bt, ents, start, count));
}

/* Extract all entries of a node into a fresh array of full keys. Each key is
 * heap-owned; free with bptFreeEnts(). Payloads are copied verbatim. */
static bptEnt *bptExtract(bptree *bt, bptNode *n, uint32_t *outcount) {
    uint32_t cnt = n->nkeys;
    bptEnt *ents = bpt_malloc(sizeof(bptEnt) * (cnt ? cnt : 1));
    for (uint32_t i = 0; i < cnt; i++) {
        ents[i].key = bptFullKey(bt, n, i, &ents[i].keylen);
        ents[i].payload = bptCellPayload(bt, n, bptSlots(n)[i]);
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
    if (!bptEntriesFit(bt, ents, 0, mid)) return 0;
    if (isleaf) return bptEntriesFit(bt, ents, mid, count - mid);
    return bptEntriesFit(bt, ents, mid + 1, count - mid - 1);
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
        if (!bptEntriesFit(bt, ents, 0, mid) && mid > 1) { mid--; continue; }
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

bptree *bptNewEx(uint32_t pageSize, size_t *alloc_size) {
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

bptree *bptNew(void) {
    return bptNewEx(BPT_PAGE_DEFAULT, NULL);
}

/* Recursive free. Exactly one of free_cb / free_cb_ctx is non-NULL (or both
 * NULL when the caller only wants the structure freed). */
static void bptFreeRec(bptree *bt, bptNode *n,
                       void (*free_cb)(void *),
                       void (*free_cb_ctx)(void *, void *), void *ctx) {
    if (!n->isleaf) {
        bptFreeRec(bt, n->leftmost, free_cb, free_cb_ctx, ctx);
        for (uint32_t i = 0; i < n->nkeys; i++)
            bptFreeRec(bt, (bptNode *)bptCellPayload(bt, n, bptSlots(n)[i]),
                       free_cb, free_cb_ctx, ctx);
    } else if (free_cb || free_cb_ctx) {
        for (uint32_t i = 0; i < n->nkeys; i++) {
            void *v = bptCellPayload(bt, n, bptSlots(n)[i]);
            if (!v) continue;
            if (free_cb) free_cb(v);
            else free_cb_ctx(v, ctx);
        }
    }
    bptFreeNode(bt, n);
}

void bptFreeWithCallback(bptree *bt, void (*free_cb)(void *)) {
    if (!bt) return;
    bptFreeRec(bt, bt->root, free_cb, NULL, NULL);
    bptFreeAccounted(bt, bt);
}

void bptFreeWithCbAndContext(bptree *bt,
                             void (*free_cb)(void *item, void *ctx),
                             void *ctx) {
    if (!bt) return;
    bptFreeRec(bt, bt->root, NULL, free_cb, ctx);
    bptFreeAccounted(bt, bt);
}

void bptFree(bptree *bt) {
    bptFreeWithCallback(bt, NULL);
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

int bptFind(bptree *bt, unsigned char *s, size_t len, void **value) {
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
    if (value) *value = bptCellPayload(bt, n, bptSlots(n)[idx]);
    return 1;
}

/* ------------------------------------------------------------------------- */
/* In-place slot insertion / removal helpers                                 */
/* ------------------------------------------------------------------------- */

/* Insert a new (suffix,payload) cell at sorted position `pos`, assuming the key
 * shares the node's full prefix and there is room (after compaction). */
static void bptInsertCellAt(bptree *bt, bptNode *n, uint32_t pos,
                            const unsigned char *suffix, uint32_t slen, void *payload) {
    uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
    if (bptFreeBytes(n) < csize + BPT_SLOT_SIZE) bptCompact(bt, n);
    assert(bptFreeBytes(n) >= csize + BPT_SLOT_SIZE);
    n->heapStart -= csize;
    bptWriteCell(bt, n, n->heapStart, suffix, slen, payload);
    uint16_t *slots = bptSlots(n);
    memmove(&slots[pos + 1], &slots[pos], (n->nkeys - pos) * BPT_SLOT_SIZE);
    slots[pos] = (uint16_t)n->heapStart;
    n->nkeys++;
}

/* Remove the cell at slot `pos` in place (frees its blob). */
static void bptRemoveCellAt(bptree *bt, bptNode *n, uint32_t pos) {
    uint16_t *slots = bptSlots(n);
    uint32_t off = slots[pos];
    uint32_t csize = bptCellSize(bt, bptCellSuffixLen(n, off),
                                 bptCellOverflow(n, off));
    if (bptCellOverflow(n, off)) bptFreeAccounted(bt, bptCellBlob(bt, n, off));
    n->deadBytes += csize;
    memmove(&slots[pos], &slots[pos + 1], (n->nkeys - pos - 1) * BPT_SLOT_SIZE);
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

/* Rebuild a leaf in place, or split it, from a sorted entry list. On split,
 * *sp is populated (caller must free sp->sep); otherwise sp->right is NULL. */
static void bptLeafRebuildOrSplit(bptree *bt, bptNode *n,
                                  bptEnt *ents, uint32_t count, bptSplit *sp) {
    sp->right = NULL;
    if (bptEntriesFit(bt, ents, 0, count)) {
        bptFreeNodeBlobs(bt, n);
        bptBuildNode(bt, n, 1, ents, 0, count);
        return;
    }
    uint32_t mid = bptChooseSplit(bt, ents, count, 1);
    bptNode *right = bptNewNode(bt, 1);
    /* Separator BEFORE we rebuild (keys still available in ents). */
    sp->sep = bptShortestSep(ents[mid - 1].key, ents[mid - 1].keylen,
                             ents[mid].key, ents[mid].keylen, &sp->seplen);
    sp->right = right;

    bptFreeNodeBlobs(bt, n);
    bptBuildNode(bt, n, 1, ents, 0, mid);
    bptBuildNode(bt, right, 1, ents, mid, count - mid);

    /* Link the new leaf after n. */
    right->next = n->next;
    right->prev = n;
    if (n->next) n->next->prev = right;
    n->next = right;
    if (bt->tail == n) bt->tail = right;
}

/* Rebuild an inner node in place, or split it, from a leftmost child plus a
 * sorted entry list (entry.payload == child to the right of entry.key). */
static void bptInnerRebuildOrSplit(bptree *bt, bptNode *n, bptNode *leftmost,
                                   bptEnt *ents, uint32_t count, bptSplit *sp) {
    sp->right = NULL;
    n->leftmost = leftmost;
    if (bptEntriesFit(bt, ents, 0, count)) {
        bptFreeNodeBlobs(bt, n);
        bptBuildNode(bt, n, 0, ents, 0, count);
        n->leftmost = leftmost;
        return;
    }
    uint32_t mid = bptChooseSplit(bt, ents, count, 0);
    /* Inner split moves the middle entry up: it is removed from both children. */
    sp->sep = bpt_malloc(ents[mid].keylen ? ents[mid].keylen : 1);
    memcpy(sp->sep, ents[mid].key, ents[mid].keylen);
    sp->seplen = ents[mid].keylen;
    bptNode *right = bptNewNode(bt, 0);
    right->leftmost = (bptNode *)ents[mid].payload;
    sp->right = right;

    bptFreeNodeBlobs(bt, n);
    bptBuildNode(bt, n, 0, ents, 0, mid);
    n->leftmost = leftmost;
    bptBuildNode(bt, right, 0, ents, mid + 1, count - mid - 1);
}

/* Insert (sep,child) into inner node `n` at child position `pos`; rebuild or
 * split as needed. */
static void bptInnerInsert(bptree *bt, bptNode *n, uint32_t pos,
                           unsigned char *sep, uint32_t seplen, bptNode *child,
                           bptSplit *sp) {
    uint32_t cnt;
    bptEnt *ents = bptExtract(bt, n, &cnt);
    bptEnt *merged = bpt_malloc(sizeof(bptEnt) * (cnt + 1));
    memcpy(merged, ents, sizeof(bptEnt) * pos);
    merged[pos].key = sep;
    merged[pos].keylen = seplen;
    merged[pos].payload = child;
    memcpy(merged + pos + 1, ents + pos, sizeof(bptEnt) * (cnt - pos));

    bptNode *leftmost = n->leftmost;
    bptInnerRebuildOrSplit(bt, n, leftmost, merged, cnt + 1, sp);

    bpt_free(ents); /* keys are freed below via merged (same pointers). */
    for (uint32_t i = 0; i < cnt + 1; i++)
        if (i != pos) bpt_free(merged[i].key);
    bpt_free(merged);
}

/* Propagate a split from level `level` (the parent that must absorb the split)
 * up toward the root, creating a new root if necessary. Allocation failures
 * abort inside the allocator (zmalloc), as everywhere else in this file: the
 * tree offers no graceful OOM recovery once a structural mutation started. */
static void bptPropagate(bptree *bt, bptNode **path, uint32_t *cidx,
                         int level, bptSplit sp) {
    while (level >= 0 && sp.right) {
        bptNode *parent = path[level];
        bptSplit up;
        bptInnerInsert(bt, parent, cidx[level], sp.sep, sp.seplen, sp.right, &up);
        bpt_free(sp.sep);
        sp = up;
        level--;
    }
    if (sp.right) {
        bptNode *newroot = bptNewNode(bt, 0);
        newroot->leftmost = bt->root;
        bptSplit dummy;
        bptInnerInsert(bt, newroot, 0, sp.sep, sp.seplen, sp.right, &dummy);
        assert(dummy.right == NULL);
        bpt_free(sp.sep);
        bt->root = newroot;
        bt->height++;
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
        if (old) *old = bptCellPayload(bt, n, bptSlots(n)[pos]);
        if (overwrite) bptCellSetPayload(bt, n, bptSlots(n)[pos], data);
        errno = 0;
        return 0; /* key already existed. */
    }
    if (old) *old = NULL;

    bptSplit sp;
    sp.right = NULL;

    /* Fast path: key shares the full node prefix and the cell fits. */
    bptPfxRel rel = bptPrefixRel(n, s, (uint32_t)len);
    uint32_t slen = (uint32_t)len - n->prefixLen;
    uint32_t csize = (rel == BPT_PFX_MATCH) ?
                     bptCellSize(bt, slen, slen > bt->maxInline) : 0;
    if (rel == BPT_PFX_MATCH &&
        bptFreeBytes(n) + n->deadBytes >= csize + BPT_SLOT_SIZE) {
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
        bptLeafRebuildOrSplit(bt, n, merged, cnt + 1, &sp);
        bpt_free(ents);
        for (uint32_t i = 0; i < cnt + 1; i++)
            if (i != pos) bpt_free(merged[i].key);
        bpt_free(merged);
    }

    if (sp.right) bptPropagate(bt, path, cidx, depth - 1, sp);

    bt->numele++;
    bt->version++;
    errno = 0;
    return 1;
}

int bptInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old) {
    return bptInsertGeneric(bt, s, len, data, old, 1);
}

int bptTryInsert(bptree *bt, unsigned char *s, size_t len, void *data, void **old) {
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
    /* Only the root leaf can be empty; treat as a plain insert. */
    if (t->nkeys == 0) return bptTryInsert(bt, s, len, data, old);

    /* Compare against the current maximum key (last cell of the tail leaf).
     * cmp = sign(new key - max key). */
    bptPfxRel rel = bptPrefixRel(t, s, (uint32_t)len);
    int cmp;
    if (rel == BPT_PFX_BELOW) cmp = -1;
    else if (rel == BPT_PFX_ABOVE) cmp = 1;
    else cmp = bptCmpSuffix(bt, t, bptSlots(t)[t->nkeys - 1],
                            s + t->prefixLen, (uint32_t)len - t->prefixLen);

    if (cmp == 0) { /* The key IS the current maximum: exists, don't touch. */
        if (old) *old = bptCellPayload(bt, t, bptSlots(t)[t->nkeys - 1]);
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
        if (bptFreeBytes(t) + t->deadBytes >= csize + BPT_SLOT_SIZE) {
            bptInsertCellAt(bt, t, t->nkeys, s + t->prefixLen, slen, data);
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
    bptBuildNodeCp(bt, right, 1, &e, 0, 1, cp);

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

    bt->numele++;
    bt->version++;
    errno = 0;
    return 1;
}

int bptFindLink(bptree *bt, unsigned char *s, size_t len, void **value,
                bptNodeLink *link) {
    assert(len == (uint32_t)len); /* Keys are limited to 4GB (u32 lengths). */
    bptNode *n = bt->root;
    while (!n->isleaf) {
        int found;
        uint32_t idx = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
        n = bptChild(bt, n, found ? idx + 1 : idx);
    }
    int found;
    uint32_t pos = bptNodeSearch(bt, n, s, (uint32_t)len, &found);
    link->leaf = n;
    link->pos = pos;
    link->found = found;
    link->version = bt->version;
    if (found) {
        if (value) *value = bptCellPayload(bt, n, bptSlots(n)[pos]);
        return 1;
    }
    if (value) *value = NULL;
    return 0;
}

int bptInsertAt(bptree *bt, unsigned char *s, size_t len, void *data,
                void **old, bptNodeLink *link) {
    assert(len == (uint32_t)len); /* Keys are limited to 4GB (u32 lengths). */
    /* The link must be fresh: no mutation of this tree since bptFindLink(). */
    assert(link->version == bt->version);

    if (link->found) {
        bptNode *n = link->leaf;
        uint16_t off = bptSlots(n)[link->pos];
        if (old) *old = bptCellPayload(bt, n, off);
        bptCellSetPayload(bt, n, off, data);
        errno = 0;
        return 0; /* key already existed. */
    }
    if (old) *old = NULL;

    /* Fast path: the located leaf still shares the key's full prefix and the
     * new cell fits (after reclaiming dead bytes). This mirrors the fast path
     * in bptInsertGeneric and avoids a second descent. */
    bptNode *n = link->leaf;
    bptPfxRel rel = bptPrefixRel(n, s, (uint32_t)len);
    if (rel == BPT_PFX_MATCH) {
        uint32_t slen = (uint32_t)len - n->prefixLen;
        uint32_t csize = bptCellSize(bt, slen, slen > bt->maxInline);
        if (bptFreeBytes(n) + n->deadBytes >= csize + BPT_SLOT_SIZE) {
            bptInsertCellAt(bt, n, link->pos, s + n->prefixLen, slen, data);
            bt->numele++;
            bt->version++;
            errno = 0;
            return 1;
        }
    }
    /* Slow path (prefix change or split): re-descend via the generic insert,
     * which handles rebuild/split/propagate. The key is known absent, so this
     * always inserts and returns 1. */
    return bptInsertGeneric(bt, s, len, data, NULL, 1);
}

/* ------------------------------------------------------------------------- */
/* Remove (with borrow / merge / root collapse)                              */
/* ------------------------------------------------------------------------- */

static int bptUnderflow(bptree *bt, bptNode *n) {
    if (n->nkeys == 0) return 1;
    return bptUsedBytes(bt, n) < (bt->pageSize * BPT_MIN_FILL_PCT) / 100;
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

/* Replace the separator at parent cell `sepIdx` with newsep, keeping its child
 * pointer. May grow the parent and thus split/propagate. */
static void bptReplaceSep(bptree *bt, bptNode **path, uint32_t *cidx, int plevel,
                          uint32_t sepIdx, unsigned char *newsep, uint32_t newlen) {
    bptNode *parent = path[plevel];
    uint32_t cnt;
    bptEnt *ents = bptExtract(bt, parent, &cnt);
    bpt_free(ents[sepIdx].key);
    ents[sepIdx].key = bpt_malloc(newlen ? newlen : 1);
    memcpy(ents[sepIdx].key, newsep, newlen);
    ents[sepIdx].keylen = newlen;

    bptNode *leftmost = parent->leftmost;
    bptSplit sp;
    bptInnerRebuildOrSplit(bt, parent, leftmost, ents, cnt, &sp);
    bptFreeEnts(ents, cnt);

    if (sp.right) {
        bptPropagate(bt, path, cidx, plevel - 1, sp);
    }
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
    if (old) *old = bptCellPayload(bt, n, bptSlots(n)[pos]);
    bptRemoveCellAt(bt, n, pos);
    bt->numele--;
    bt->version++;

    /* Rebalance upward. `cur` is the (possibly) underflowing node. */
    bptNode *cur = n;
    int plevel = depth - 1;

    while (1) {
        if (cur == bt->root) {
            if (!cur->isleaf && cur->nkeys == 0) {
                bptNode *only = cur->leftmost;
                bptFreeNode(bt, cur);
                bt->root = only;
                bt->height--;
            } else if (cur->isleaf && cur->nkeys == 0) {
                cur->prefixOff = bt->pageSize;
                cur->prefixLen = 0;
                cur->heapStart = bt->pageSize;
                cur->deadBytes = 0;
            }
            break;
        }
        if (!bptUnderflow(bt, cur)) break;

        bptNode *parent = path[plevel];
        uint32_t pcidx = cidx[plevel];
        bptNode *L, *R;
        uint32_t sepIdx;
        if (pcidx > 0) { L = bptChild(bt, parent, pcidx - 1); R = cur; sepIdx = pcidx - 1; }
        else { L = cur; R = bptChild(bt, parent, pcidx + 1); sepIdx = pcidx; }

        /* Parent separator between L and R. */
        uint32_t psepLen;
        unsigned char *psep = bptFullKey(bt, parent, sepIdx, &psepLen);

        uint32_t total;
        bptEnt *all;
        if (cur->isleaf)
            all = bptCombineLeaf(bt, L, R, &total);
        else
            all = bptCombineInner(bt, L, R, psep, psepLen, &total);

        if (bptEntriesFit(bt, all, 0, total)) {
            /* MERGE into L, free R, drop the parent separator + R pointer. */
            if (cur->isleaf) {
                bptFreeNodeBlobs(bt, L);
                bptBuildNode(bt, L, 1, all, 0, total);
                L->next = R->next;
                if (R->next) R->next->prev = L;
                if (bt->tail == R) bt->tail = L;
            } else {
                bptNode *lm = L->leftmost;
                bptFreeNodeBlobs(bt, L);
                bptBuildNode(bt, L, 0, all, 0, total);
                L->leftmost = lm;
            }
            bptFreeEnts(all, total);
            bptFreeNode(bt, R);
            bptRemoveCellAt(bt, parent, sepIdx);
            bpt_free(psep);
            cur = parent;
            plevel--;
            continue;
        } else {
            /* BORROW: redistribute evenly, update the parent separator. */
            uint32_t mid = bptChooseSplit(bt, all, total, cur->isleaf);
            unsigned char *newsep;
            uint32_t newlen;
            if (cur->isleaf) {
                newsep = bptShortestSep(all[mid - 1].key, all[mid - 1].keylen,
                                        all[mid].key, all[mid].keylen, &newlen);
                bptFreeNodeBlobs(bt, L);
                bptBuildNode(bt, L, 1, all, 0, mid);
                bptFreeNodeBlobs(bt, R);
                bptBuildNode(bt, R, 1, all, mid, total - mid);
            } else {
                newlen = all[mid].keylen;
                newsep = bpt_malloc(newlen ? newlen : 1);
                memcpy(newsep, all[mid].key, newlen);
                bptNode *lLeft = L->leftmost;
                bptNode *rLeft = (bptNode *)all[mid].payload;
                bptFreeNodeBlobs(bt, L);
                bptBuildNode(bt, L, 0, all, 0, mid);
                L->leftmost = lLeft;
                bptFreeNodeBlobs(bt, R);
                bptBuildNode(bt, R, 0, all, mid + 1, total - mid - 1);
                R->leftmost = rLeft;
            }
            bptFreeEnts(all, total);
            bpt_free(psep);
            bptReplaceSep(bt, path, cidx, plevel, sepIdx, newsep, newlen);
            bpt_free(newsep);
            break; /* cur is balanced now. */
        }
    }
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
    uint32_t off = bptSlots(n)[i];
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

/* Descend to the leaf that would contain `key` and return the lower-bound slot
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
        bptNode *leaf = bptFirstLeaf(bt);
        if (leaf->nkeys == 0) { it->flags = safe | BPT_ITER_EOF; return 1; }
        it->node = leaf; it->idx = 0;
        bptIterMaterialize(it);
        return 1;
    }
    if (op[0] == '$') {
        bptNode *leaf = bt->tail; /* O(1): the rightmost leaf is cached. */
        if (leaf->nkeys == 0) { it->flags = safe | BPT_ITER_EOF; return 1; }
        it->node = leaf; it->idx = (int)leaf->nkeys - 1;
        bptIterMaterialize(it);
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
     * key/value, so bptCompare() is valid right after a seek and the first
     * bptNext()/bptPrev() never touches tree pages (which a safe-iterator
     * caller may have mutated in the meantime). */
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

/* Update the value stored in the tree for the element the iterator is
 * currently positioned on (the bptree analogue of raxSetData). If the tree
 * was mutated since the iterator was positioned, the element is located again
 * by key; if it no longer exists, only the iterator's local copy is updated. */
void bptIteratorSetData(bptIterator *it, void *data) {
    it->data = data;
    if (it->flags & BPT_ITER_EOF) return;
    if (it->node == NULL) return; /* Never positioned. */
    bptree *bt = it->bt;
    if (it->version != bt->version) {
        uint32_t idx;
        int found;
        bptNode *leaf = bptDescendLB(bt, it->key, (uint32_t)it->key_len,
                                     &idx, &found);
        if (!found) return;
        bptCellSetPayload(bt, leaf, bptSlots(leaf)[idx], data);
        return;
    }
    bptCellSetPayload(bt, it->node, bptSlots(it->node)[it->idx], data);
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

int bptCompare(bptIterator *it, const char *op, unsigned char *key, size_t key_len) {
    int eq = 0, lt = 0, gt = 0;
    if (op[0] == '>') { gt = 1; if (op[1] == '=') eq = 1; }
    else if (op[0] == '<') { lt = 1; if (op[1] == '=') eq = 1; }
    else if (op[0] == '=') { eq = 1; }
    else return 0;
    int c = bptKeyCmp(it->key, (uint32_t)it->key_len, key, (uint32_t)key_len);
    if (c == 0) return eq;
    if (c > 0) return gt;
    return lt;
}

/* ------------------------------------------------------------------------- */
/* Active defragmentation                                                    */
/* ------------------------------------------------------------------------- */

/* Relocate one node's page and its overflow blobs via `fn`, then recurse into
 * children (fixing the parent's child pointers) or, for leaves, stitch the
 * doubly linked leaf chain in key order. Returns the node's (possibly new)
 * address. `*prevLeaf` threads the previously relocated leaf so its `next`
 * link can be repaired once the current leaf's new address is known.
 *
 * The traversal visits leaves left to right (leftmost child first, then cells
 * in slot order), i.e. in key order, so the chain is rebuilt correctly. */
static bptNode *bptDefragNode(bptree *bt, bptNode *n, bptReallocFn fn,
                              bptNode **prevLeaf) {
    bptNode *moved = fn(n);
    if (moved) n = moved;

    /* Repair overflow blob pointers stored inside this (relocated) page. */
    uint16_t *slots = bptSlots(n);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = slots[i];
        if (!bptCellOverflow(n, off)) continue;
        bptBlob *b = bptCellBlob(bt, n, off);
        bptBlob *nb = fn(b);
        if (nb) bptCellSetBlob(bt, n, off, nb);
    }

    if (n->isleaf) {
        n->prev = *prevLeaf;
        if (*prevLeaf) (*prevLeaf)->next = n;
        *prevLeaf = n;
        return n;
    }

    n->leftmost = bptDefragNode(bt, n->leftmost, fn, prevLeaf);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = slots[i];
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
 * Values stored against keys are NOT touched -- relocate those separately via
 * bptIteratorSetData while iterating. */
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

/* ------------------------------------------------------------------------- */
/* Debug print                                                               */
/* ------------------------------------------------------------------------- */

static void bptShowRec(bptree *bt, bptNode *n, int level) {
    for (int i = 0; i < level; i++) printf("  ");
    printf("%s nkeys=%u prefixLen=%u used=%u\n", n->isleaf ? "LEAF" : "INNER",
           n->nkeys, n->prefixLen, bptUsedBytes(bt, n));
    if (!n->isleaf) {
        bptShowRec(bt, n->leftmost, level + 1);
        for (uint32_t i = 0; i < n->nkeys; i++)
            bptShowRec(bt, (bptNode *)bptCellPayload(bt, n, bptSlots(n)[i]), level + 1);
    }
}

void bptShow(bptree *bt) {
    printf("=== bptree numele=%llu height=%u pageSize=%u ===\n",
           (unsigned long long)bt->numele, bt->height, bt->pageSize);
    bptShowRec(bt, bt->root, 0);
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
    /* Keys sorted and within bounds. */
    unsigned char *prev = NULL;
    uint32_t prevlen = 0;
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t klen;
        unsigned char *k = bptFullKey(bt, n, i, &klen);
        if (prev && bptKeyCmp(prev, prevlen, k, klen) >= 0) {
            printf("validate: keys not sorted in node\n"); err++;
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
        err += bptValidateRec(bt, child, clow, clowlen, clowInf,
                              chigh, chighlen, chighInf, depth + 1, leafDepth, count);
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
    uint16_t *slots = bptSlots(n);
    for (uint32_t i = 0; i < n->nkeys; i++) {
        uint32_t off = slots[i];
        if (bptCellOverflow(n, off))
            total += zmalloc_usable_size(bptCellBlob(bt, n, off));
    }
    if (!n->isleaf) {
        total += bptNodeAllocSize(bt, n->leftmost);
        for (uint32_t i = 0; i < n->nkeys; i++)
            total += bptNodeAllocSize(bt, (bptNode *)bptCellPayload(bt, n, slots[i]));
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
            c += bptCountNodes(bt, (bptNode *)bptCellPayload(bt, n, bptSlots(n)[i]));
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

static int bptCtxFreeCount = 0;
static void bptCtxFreeCb(void *item, void *ctx) {
    (void)item;
    (*(int *)ctx)++;
    bptCtxFreeCount++;
}

/* --- Deterministic unit tests -------------------------------------------- */

static int bptUnitTests(void) {
    int err = 0;

    TEST("empty tree: find and seek") {
        bptree *bt = bptNew();
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
        bptree *bt = bptNew();
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
        bptree *bt = bptNew();
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
        bptree *bt = bptNewEx(256, NULL);
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
        bptree *bt = bptNewEx(256, NULL);
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

    TEST("seek operators at boundaries and misses") {
        bptree *bt = bptNew();
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

    TEST("bptCompare valid immediately after seek") {
        bptree *bt = bptNew();
        const char *keys[] = {"a", "c", "e", "g"};
        for (int i = 0; i < 4; i++)
            bptInsert(bt, (unsigned char *)keys[i], 1, (void *)(intptr_t)(i + 1), NULL);
        bptIterator it; bptStart(&it, bt);
        bptSeek(&it, ">=", (unsigned char *)"b", 1);
        assert(bptCompare(&it, "==", (unsigned char *)"c", 1));
        assert(it.data == (void *)2);
        bptSeek(&it, "<=", (unsigned char *)"f", 1);
        assert(bptCompare(&it, "==", (unsigned char *)"e", 1));
        assert(it.data == (void *)3);
        bptStop(&it);
        bptFree(bt);
    }

    TEST("iterator set data writes through to the tree") {
        bptree *bt = bptNew();
        const char *keys[] = {"a", "c", "e"};
        for (int i = 0; i < 3; i++)
            bptInsert(bt, (unsigned char *)keys[i], 1, (void *)(intptr_t)(i + 1), NULL);
        bptIterator it; bptStart(&it, bt);
        bptSeek(&it, "==", (unsigned char *)"c", 1);
        assert(bptNext(&it));
        bptIteratorSetData(&it, (void *)0x42);
        void *v;
        assert(bptFind(bt, (unsigned char *)"c", 1, &v) == 1 && v == (void *)0x42);
        /* Stale-version path: mutate the tree, then set through the same
         * (safe) iterator; the element must be located again by key. */
        it.flags |= BPT_ITER_SAFE;
        bptInsert(bt, (unsigned char *)"b", 1, NULL, NULL);
        bptIteratorSetData(&it, (void *)0x43);
        assert(bptFind(bt, (unsigned char *)"c", 1, &v) == 1 && v == (void *)0x43);
        bptStop(&it);
        err += bptValidate(bt);
        bptFree(bt);
    }

    TEST("safe iterator: mutation between seek and first next") {
        bptree *bt = bptNewEx(256, NULL);
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
        bptree *bt = bptNewEx(256, NULL);
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
        bptree *bt = bptNewEx(256, &acc);
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

    TEST("free with callback and context") {
        bptree *bt = bptNew();
        int localCount = 0;
        bptCtxFreeCount = 0;
        int N = 500;
        char kb[16];
        int expected = 0;
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%05d", i);
            /* Mix in some NULL values: the free callback must skip those. */
            void *v = (i % 7 == 0) ? NULL : (void *)(intptr_t)(i + 1);
            if (v) expected++;
            bptInsert(bt, (unsigned char *)kb, len, v, NULL);
        }
        int ctxSentinel = 0;
        bptFreeWithCbAndContext(bt, bptCtxFreeCb, &ctxSentinel);
        assert(ctxSentinel == expected);
        assert(bptCtxFreeCount == expected);
        (void)localCount;
    }

    TEST("defrag relocates every node/blob and keeps the tree consistent") {
        bptree *bt = bptNewEx(256, NULL);
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

    TEST("append: sequential IDs, duplicates, out-of-order fallback") {
        bptree *bt = bptNewEx(512, NULL);
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

    TEST("find-link / insert-at: hit, miss, overwrite, split") {
        bptree *bt = bptNewEx(256, NULL);
        char kb[16];
        int N = 2000; /* enough to force splits at a 256-byte page. */

        /* Miss then insert-at for every key (single-walk upsert). */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v = (void *)0x1;
            bptNodeLink link;
            int found = bptFindLink(bt, (unsigned char *)kb, len, &v, &link);
            assert(found == 0);
            assert(v == NULL);
            int ins = bptInsertAt(bt, (unsigned char *)kb, len,
                                  (void *)(intptr_t)(i + 1), NULL, &link);
            assert(ins == 1);
        }
        assert(bptSize(bt) == (uint64_t)N);
        assert(bt->height > 1); /* splits happened, exercising the slow path. */
        err += bptValidate(bt);

        /* Hit: link reports found and yields the stored value. */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v = NULL;
            bptNodeLink link;
            int found = bptFindLink(bt, (unsigned char *)kb, len, &v, &link);
            assert(found == 1);
            assert(v == (void *)(intptr_t)(i + 1));
            assert(link.found == 1);
        }

        /* Overwrite via insert-at on a found link: returns 0, reports *old. */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v = NULL, *old = NULL;
            bptNodeLink link;
            assert(bptFindLink(bt, (unsigned char *)kb, len, &v, &link) == 1);
            int ins = bptInsertAt(bt, (unsigned char *)kb, len,
                                  (void *)(intptr_t)(i + 1000000), &old, &link);
            assert(ins == 0);
            assert(old == (void *)(intptr_t)(i + 1));
        }
        assert(bptSize(bt) == (uint64_t)N); /* overwrite must not change count. */
        for (int i = 0; i < N; i++) {
            int len = snprintf(kb, sizeof(kb), "k%08d", i);
            void *v = NULL;
            assert(bptFind(bt, (unsigned char *)kb, len, &v) == 1);
            assert(v == (void *)(intptr_t)(i + 1000000));
        }
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
        bptree *bt = bptNewEx(pageSize, NULL);
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
                    int variant = rand() % 3;
                    if (variant == 0) {
                        /* Drive the single-walk upsert (find-link + insert-at)
                         * on both structures, mirroring the stream call sites. */
                        void *fb = (void *)0x1, *fr = (void *)0x1;
                        bptNodeLink blink;
                        raxNodeLink rlink;
                        int found_b = bptFindLink(bt, kbuf, klen, &fb, &blink);
                        int found_r = raxFindLink(rt, kbuf, klen, &fr, &rlink);
                        assert(found_b == found_r);
                        if (found_b) assert(fb == fr);
                        rb = bptInsertAt(bt, kbuf, klen, val, &ob, &blink);
                        rr = raxInsertAt(rt, kbuf, klen, val, &orx, &rlink);
                    } else if (variant == 1) {
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
    for (int which = 0; which < 2; which++) {
        int isBpt = (which == 0);
        size_t memBefore = zmalloc_used_memory();
        bptree *bt = NULL; rax *rt = NULL;
        if (isBpt) bt = bptNewEx(512, NULL);
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
            if (isBpt) { if (bptFind(bt, keys[i], lens[i], &v)) hits++; }
            else { if (raxFind(rt, keys[i], lens[i], &v)) hits++; }
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
        bptBenchOne("random string keys (bptree page=512 vs rax)", N, keys, lens, 0, 0);
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

int bptTest(int argc, char *argv[], int flags) {
    UNUSED(argc);
    UNUSED(argv);
    int err = 0;

    err += bptUnitTests();
    err += bptFuzz(flags);
    if (!(flags & REDIS_TEST_VALGRIND)) bptBench(flags);

    if (err) printf("bptree: %d test failures\n", err);
    else printf("bptree: all tests passed\n");
    return err;
}

#endif
