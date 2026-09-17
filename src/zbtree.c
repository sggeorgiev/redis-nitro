/* zbtree.c -- Order-statistic B+ tree used as the large-encoding backend of
 * Redis sorted sets (ZSETs), replacing the previous skiplist.
 *
 * Design goals (see the ZSET encoding notes in server.h):
 *   - Elements are ordered by (score, member) exactly like the old skiplist.
 *   - Each member is a single heap object (zbtElem) with an embedded SDS.
 *   - Membership is an open-addressed table of (tag, leaf ID) plus
 *     table_to_leaf[], not a companion dict.
 *   - Leaves are packed arrays of zbtElem pointers, doubly linked so range
 *     scans walk contiguous memory instead of chasing skiplist pointers.
 *   - Internal nodes carry per-child subtree sizes, giving O(log N) rank and
 *     rank-based access (ZRANK / ZRANGE by index / ZREMRANGEBYRANK).
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "server.h"
#include <math.h>
#include <string.h>

/* Fanout of the tree. Nodes are allowed to temporarily hold one extra slot
 * (hence the "+1" sized arrays) before they are split.
 *
 * Every node holds at least its MIN, with three exceptions: the root, and the
 * head and tail leaves, which zbtSplitLeaf() deliberately starts under-filled
 * so that sorted insertion does not strand every leaf at half occupancy.
 * Interior overflows (interpolation into a packed tree) try a same-parent
 * sibling share before an even split, so the abandoned half is refilled
 * instead of sitting at MIN for the rest of its life. */

/* Which end of the leaf the element that overflowed it landed on. Sorted
 * insertion never returns to the side the split leaves behind, so an even
 * split would strand that side half full for the rest of its life. */
#define ZBT_SPLIT_EVEN    0
#define ZBT_SPLIT_APPEND  1  /* went in at the end of the tail leaf */
#define ZBT_SPLIT_PREPEND 2  /* went in at the front of the head leaf */
/* Leaf fanout. A smaller fanout (35, the largest that still fits the
 * jemalloc 320-byte size class) was tried and measured: it saved ~0.2% of
 * total memory on a 1M-element zset, but cost ~4% throughput on range scans
 * (more, smaller leaves means more leaf-to-leaf pointer chases per scan).
 * Prefetching the next leaf and a few upcoming elements in
 * zbtIterNext()/zbtIterPrev() recovered that loss, but adding the same
 * prefetching back here at fanout 64 only moved throughput by ~1% at best
 * (fewer, bigger leaves already mean fewer boundary crossings to hide the
 * latency of) - not worth the extra branches in a hot path, so left out. */
#define ZBT_LEAF_MAX   64
#define ZBT_LEAF_MIN   (ZBT_LEAF_MAX/2)
#define ZBT_INNER_MAX  64
#define ZBT_INNER_MIN  (ZBT_INNER_MAX/2)
/* Offsets up to this many elements are reached by stepping along the leaf
 * chain, which stays cheaper than the root-to-leaf descent zbtElemByRank()
 * needs to jump straight to a rank. Matches the search window the skiplist
 * used before the tree replaced it. */
#define ZBT_RANGE_WALK_MAX 10

/* Member lookup uses an open addressed table with eight slots per bucket. */
#define ZBT_INDEX_BUCKET_ITEMS 8
#define ZBT_INDEX_INITIAL_BUCKETS 4
#define ZBT_INDEX_MAX_LOAD_NUM 31
#define ZBT_INDEX_MAX_LOAD_DEN 32
#define ZBT_INDEX_MIN_LOAD_NUM 1
#define ZBT_INDEX_MIN_LOAD_DEN 8
#define ZBT_INDEX_MAX_FILLED_NUM 63
#define ZBT_INDEX_MAX_FILLED_DEN 64
#define ZBT_INDEX_DELETED_ID UINT32_MAX
#define ZBT_INDEX_WIDE_ID_AT (UINT16_MAX / 2)
#define ZBT_NEW_LEAF_ID UINT32_MAX
#define ZBT_NO_LEAF_ID UINT32_MAX
#define ZBT_SCAN_BUCKETS_PER_STEP 4

/* Common node header. Both leaf and inner nodes start with it so that a
 * zbtNode* can be inspected polymorphically. */
struct zbtNode {
    struct zbtNode *parent;
    uint32_t count;   /* used slots: elems (leaf) or children (inner) */
    uint32_t isleaf;
};

typedef struct zbtLeaf {
    zbtNode n;
    struct zbtLeaf *prev, *next;         /* sibling leaves (sorted order) */
    uint32_t id;
    uint16_t index_resize;               /* resize that copied this leaf, 0 if none */
    uint8_t tags[ZBT_LEAF_MAX + 1];      /* hash >> 24, not folded */
    zbtElem *elems[ZBT_LEAF_MAX + 1];
} zbtLeaf;

#define ZBT_FREE_LEAF_ID(next) \
    ((zbtLeaf *)((((uintptr_t)(next)) << 1) | 1))
#define ZBT_IS_FREE_LEAF_ID(ptr) (((uintptr_t)(ptr) & 1) != 0)
#define ZBT_NEXT_FREE_LEAF_ID(ptr) ((uint32_t)((uintptr_t)(ptr) >> 1))


typedef struct zbtInner {
    zbtNode n;
    struct zbtNode *child[ZBT_INNER_MAX + 1];
    unsigned long csize[ZBT_INNER_MAX + 1]; /* subtree element count of child[i] */
    zbtElem *sep[ZBT_INNER_MAX + 1];        /* minimum element of child[i] */
} zbtInner;

typedef struct zbtIndexBucket16 {
    uint64_t tags;
    uint64_t home_tags;
    uint16_t id[ZBT_INDEX_BUCKET_ITEMS];
} zbtIndexBucket16;

typedef struct zbtIndexBucket32 {
    uint64_t tags;
    uint64_t home_tags;
    uint32_t id[ZBT_INDEX_BUCKET_ITEMS];
} zbtIndexBucket32;

typedef union zbtIndexBucket {
    zbtIndexBucket16 narrow;
    zbtIndexBucket32 wide;
} zbtIndexBucket;

/*-----------------------------------------------------------------------------
 * Element allocation
 *----------------------------------------------------------------------------*/

/* moff is bounded by the header plus the widest score, an optional cached
 * member hash, and the largest sds header, so a single byte is enough. */
static_assert(offsetof(zbtElem, data) + 8 + sizeof(uint64_t) +
                  sizeof(struct sdshdr64) <= UINT8_MAX,
              "zbtElem member offset must fit in a byte");

static size_t zbtScoreEncSize(uint8_t enc) {
    switch (enc & ZBT_SCORE_MASK) {
    case ZBT_SCORE_I8:  return 1;
    case ZBT_SCORE_I16: return 2;
    case ZBT_SCORE_I24: return 3;
    case ZBT_SCORE_I32: return 4;
    case ZBT_SCORE_I48: return 6;
    default:            return 8;
    }
}

/* Pick the narrowest integer encoding that round-trips 'd', or a raw double.
 * Not on the compare hot path; zbtGetScore is. */
static void zbtScoreEncode(double d, uint8_t *enc, unsigned char *buf) {
    long long ll;

    /* double2ll(-0.0) succeeds with 0, but ZSCORE must still reply -0. */
    if (d == 0 && signbit(d)) {
        *enc = ZBT_SCORE_DBL;
        memcpy(buf, &d, sizeof(d));
        return;
    }
    if (!double2ll(d, &ll)) {
        *enc = ZBT_SCORE_DBL;
        memcpy(buf, &d, sizeof(d));
        return;
    }
    if (ll >= INT8_MIN && ll <= INT8_MAX) {
        *enc = ZBT_SCORE_I8;
        buf[0] = (unsigned char)(int8_t)ll;
    } else if (ll >= INT16_MIN && ll <= INT16_MAX) {
        int16_t v = (int16_t)ll;
        *enc = ZBT_SCORE_I16;
        memcpy(buf, &v, 2);
    } else if (ll >= -(1LL << 23) && ll <= ((1LL << 23) - 1)) {
        unsigned long long u = (unsigned long long)ll;
        *enc = ZBT_SCORE_I24;
        buf[0] = (unsigned char)u;
        buf[1] = (unsigned char)(u >> 8);
        buf[2] = (unsigned char)(u >> 16);
    } else if (ll >= INT32_MIN && ll <= INT32_MAX) {
        int32_t v = (int32_t)ll;
        *enc = ZBT_SCORE_I32;
        memcpy(buf, &v, 4);
    } else if (ll >= -(1LL << 47) && ll <= ((1LL << 47) - 1)) {
        unsigned long long u = (unsigned long long)ll;
        *enc = ZBT_SCORE_I48;
        buf[0] = (unsigned char)u;
        buf[1] = (unsigned char)(u >> 8);
        buf[2] = (unsigned char)(u >> 16);
        buf[3] = (unsigned char)(u >> 24);
        buf[4] = (unsigned char)(u >> 32);
        buf[5] = (unsigned char)(u >> 40);
    } else {
        *enc = ZBT_SCORE_DBL;
        memcpy(buf, &d, sizeof(d));
    }
}

/* Allocate an element with the member SDS embedded in the same allocation
 * (single block: zbtElem header + score bytes + sds header + data). The
 * member is copied from 'buf', which does not have to be an sds: callers
 * holding plain bytes (listpack entries, integer members) can build an
 * element without first materializing a temporary sds. When 'wide' is set
 * the score is stored as a raw double so a later in-place write of any
 * double (ZUNIONSTORE aggregation) cannot overflow the allocation. When
 * 'usable' is not NULL it receives the usable size of the allocation. */
zbtElem *zbtCreateElemWithHash(double score, const char *buf, size_t len,
                               int wide, size_t *usable, const uint64_t *known_hash)
{
    uint8_t enc;
    unsigned char sbuf[8];
    if (wide) {
        enc = ZBT_SCORE_DBL;
        memcpy(sbuf, &score, sizeof(score));
    } else {
        zbtScoreEncode(score, &enc, sbuf);
    }
    size_t score_sz = zbtScoreEncSize(enc);
    int cache_hash = len >= ZBT_CACHE_HASH_MIN_LEN;
    size_t hash_sz = cache_hash ? sizeof(uint64_t) : 0;
    char sds_type = sdsReqType(len);
    size_t sds_hdr_len = sdsHdrSize(sds_type);
    /* offsetof(), not sizeof(): the struct is padded out to its pointer
     * alignment, and that tail padding must not push the score bytes out. */
    size_t hdr = offsetof(zbtElem, data) + score_sz + hash_sz;
    size_t sds_buf_size = sds_hdr_len + len + 1;
    size_t total = hdr + sds_buf_size;

    zbtElem *e = zmalloc_usable(total, usable);
    e->leaf = NULL;   /* set when the element is placed in a leaf */
    e->enc = enc | (cache_hash ? ZBT_ELEM_CACHED_HASH : 0);
    memcpy(e->data, sbuf, score_sz);
    size_t sds_offset = hdr + sds_hdr_len;
    zbtSetOffset(e, (uint8_t)sds_offset);

    char *dst = (char *)e + hdr;
    sds emb = sdsnewplacement(dst, sds_buf_size, sds_type, buf, len);
    serverAssert(emb == (sds)((char *)e + sds_offset));
    if (cache_hash) {
        uint64_t hash = known_hash ? *known_hash : dictSdsHash(emb);
        memcpy(e->data + score_sz, &hash, sizeof(hash));
    }
    return e;
}

zbtElem *zbtCreateElem(double score, const char *buf, size_t len,
                       int wide, size_t *usable)
{
    return zbtCreateElemWithHash(score, buf, len, wide, usable, NULL);
}

/* Duplicate the complete packed representation. This preserves the source
 * score encoding and copies the member with a single memcpy. */
zbtElem *zbtDupElem(const zbtElem *elem, size_t *usable) {
    size_t size = zbtGetOffset(elem) + sdslen(zbtGetEle(elem)) + 1;
    zbtElem *copy = zmalloc_usable(size, usable);
    memcpy(copy, elem, size);
    copy->leaf = NULL;  /* the copy is not in a tree yet */
    return copy;
}

/* Free a detached element that is not owned by any tree. Used by callers that
 * allocate an element with zbtCreateElem() but fail before ownership is
 * transferred to a tree (e.g. duplicate detection on RDB load). */
void zbtFreeElem(zbtElem *e) {
    zfree_usable(e, NULL);  /* embedded sds is part of the allocation */
}

/* Compare {score, ele} with element 'e'. Returns 1 (bigger), 0 (equal),
 * -1 (smaller). NULL is treated as +infinity. Ordering: score, then member. */
int zbtCompare(double score, sds ele, const zbtElem *e) {
    if (e == NULL) return -1;
    double escore = zbtGetScore(e);
    if (score < escore) return -1;
    if (score > escore) return 1;
    return sdscmp(ele, zbtGetEle(e));
}

/* Compare two member byte ranges and report how far they agree. Bulk blocks
 * go through memcmp so the vectorized libc path still does the scanning; only
 * the block that differs is walked word- then byte-wise to pin the exact
 * offset. */
static int zbtMemcmpLcp(const unsigned char *a, const unsigned char *b,
                        size_t n, size_t *agree)
{
    size_t i = 0;
    while (n - i >= 256) {
        if (memcmp(a + i, b + i, 256) != 0) break;
        i += 256;
    }
    while (i + 8 <= n) {
        uint64_t x, y;
        memcpy(&x, a + i, 8);
        memcpy(&y, b + i, 8);
        if (x != y) break;
        i += 8;
    }
    while (i < n && a[i] == b[i]) i++;
    *agree = i;
    if (i == n) return 0;
    return a[i] < b[i] ? -1 : 1;
}

/* Like zbtCompare(), but the caller has already established that the first
 * 'skip' member bytes match on both sides, so they are not re-read. '*agree'
 * receives the number of leading member bytes the two members share, or 0
 * when the scores decided the order and nothing is known about the members.
 *
 * Skipping is only sound because the caller derives 'skip' from two bounds
 * that bracket 'e'; see zbtLcp. */
static int zbtCompareSkip(double score, sds ele, const zbtElem *e,
                          size_t skip, size_t *agree)
{
    double escore = zbtGetScore(e);
    if (score < escore) { *agree = 0; return -1; }
    if (score > escore) { *agree = 0; return 1; }

    sds b = zbtGetEle(e);
    size_t la = sdslen(ele), lb = sdslen(b);
    size_t n = la < lb ? la : lb;
    if (skip > n) skip = n;
    size_t extra;
    int c = zbtMemcmpLcp((const unsigned char *)ele + skip,
                         (const unsigned char *)b + skip, n - skip, &extra);
    *agree = skip + extra;
    if (c != 0) return c;
    if (la == lb) return 0;
    return la < lb ? -1 : 1;
}

/* How many leading member bytes the search target is known to share with the
 * two elements that currently bracket it: 'lo' for the greatest element not
 * after the target, 'hi' for the smallest one after it. Zero means that
 * bound was decided by score, which tells us nothing about its member.
 *
 * Both bounds agreeing with the target on the first m bytes forces them to
 * agree with each other on those bytes, so every element sorted between them
 * starts with the same m bytes too - which is what makes skipping them safe.
 * The bracket only ever tightens, so the state is carried down the whole
 * root-to-leaf descent rather than restarted at each node. */
typedef struct zbtLcp {
    size_t lo, hi;
} zbtLcp;

#define ZBT_LCP_INIT ((zbtLcp){0, 0})

static inline size_t zbtLcpSkip(const zbtLcp *l) {
    return l->lo < l->hi ? l->lo : l->hi;
}

/* dict keyFromStoredKey callback: recover the member SDS from a stored
 * zbtElem*. */
const void *zbtGetEleForDict(const void *elem) {
    return zbtGetEle((const zbtElem *)elem);
}

static uint32_t zbtElemIndexHash(const zbtElem *e) {
    return (uint32_t)zbtElemHash(e);
}

static uint8_t zbtElemLeafTag(const zbtElem *e) {
    return (uint8_t)(zbtElemIndexHash(e) >> 24);
}

static void zbtLeafSetElem(zbtLeaf *lf, int idx, zbtElem *e) {
    lf->elems[idx] = e;
    lf->tags[idx] = zbtElemLeafTag(e);
    e->leaf = lf;
}

static void *zbtAlloc(zbtree *t, size_t bytes) {
    size_t usable;
    void *ptr = zmalloc_usable(bytes, &usable);
    t->alloc_size += usable;
    return ptr;
}

static void zbtFreeAllocation(zbtree *t, void *ptr) {
    size_t usable;
    zfree_usable(ptr, &usable);
    t->alloc_size -= usable;
}

static uint32_t zbtIndexNextRevision(void) {
    static uint32_t revision = 0;
    if (++revision == 0) revision++;
    return revision;
}

static void zbtIndexInvalidateScan(zbtree *t) {
    if (t->member_index.size)
        t->member_index.scan_revision = zbtIndexNextRevision();
}

static void zbtRegisterLeaf(zbtree *t, zbtLeaf *leaf, uint32_t id);
static void zbtReleaseLeafId(zbtree *t, uint32_t id);
static int zbtIndexMove(zbtree *t, uint32_t hash, uint32_t old_leaf_id,
                        uint32_t new_leaf_id, int target_is_incomplete,
                        int *target_was_copied);
static void zbtIndexInsert(zbtree *t, uint32_t hash, uint32_t leaf_id,
                           const zbtreeInsertPosition *position);
static int zbtIndexRehashStep(zbtree *t, int steps);
static void zbtIndexExpandIfNeeded(zbtree *t, unsigned long add);
static void zbtIndexShrinkIfNeeded(zbtree *t);
static void zbtIndexTableRelease(zbtree *t, zbtIndexTable *table);
static void zbtIndexTableInsertRaw(zbtIndexTable *table, uint32_t hash,
                                   uint32_t id);
static int zbtIndexFindReference(zbtree *t, uint32_t hash, uint32_t leaf_id,
                                 zbtIndexBucket **found_bucket,
                                 unsigned int *found_pos);
static void zbtIndexDeleteAt(zbtree *t, zbtIndexBucket *bucket,
                             unsigned int pos);
static int zbtIndexLeafMigrated(const zbtree *t, const zbtLeaf *leaf);

static void zbtIndexMoveElem(zbtree *t, zbtElem *e, uint32_t old_id,
                             uint32_t new_id, int incomplete, int *copied)
{
    if (old_id == new_id) return;
    uint32_t hash = zbtElemIndexHash(e);
    if (!zbtIndexMove(t, hash, old_id, new_id, incomplete, copied)) {
        if (t->pending_insert == e) return;
        zbtIndexBucket *bucket;
        unsigned int pos;
        if (zbtIndexFindReference(t, hash, new_id, &bucket, &pos)) return;
        zbtIndexTable *table = t->member_rehash ?
            &t->member_rehash->table : &t->member_index;
        zbtIndexTableInsertRaw(table, hash, new_id);
    }
}

static void zbtIndexMoveRange(zbtree *t, zbtLeaf *dst, int from, int n,
                              uint32_t old_id, int incomplete, int *copied)
{
    for (int i = 0; i < n; i++)
        zbtIndexMoveElem(t, dst->elems[from + i], old_id, dst->id,
                         incomplete, copied);
}

static int zbtLeafFindMember(zbtLeaf *leaf, uint32_t hash,
                             const unsigned char *ele, size_t elelen,
                             unsigned int *score_pos)
{
    uint8_t tag = (uint8_t)(hash >> 24);
    uint8_t *tags = leaf->tags;
    unsigned int count = leaf->n.count;
    uint8_t *p = tags;
    uint8_t *end = tags + count;
    while (p < end && (p = memchr(p, tag, (size_t)(end - p))) != NULL) {
        unsigned int i = (unsigned int)(p - tags);
        zbtElem *e = leaf->elems[i];
        if (zbtHasCachedHash(e) && (uint32_t)zbtGetCachedHash(e) != hash) {
            p++;
            continue;
        }
        sds s = zbtGetEle(e);
        if (sdslen(s) == elelen && memcmp(s, ele, elelen) == 0) {
            if (score_pos) *score_pos = i;
            return 1;
        }
        p++;
    }
    return 0;
}

static unsigned long zbtIndexSlots(const zbtIndexTable *table) {
    return table->size * ZBT_INDEX_BUCKET_ITEMS;
}

static size_t zbtIndexBucketBytes(const zbtIndexTable *table) {
    return table->wide_ids ? sizeof(zbtIndexBucket32) : sizeof(zbtIndexBucket16);
}

static size_t zbtIndexTableBytes(const zbtIndexTable *table) {
    return table->size * zbtIndexBucketBytes(table);
}

static inline zbtIndexBucket *zbtIndexBucketAt(const zbtIndexTable *table,
                                               unsigned long index)
{
    return (zbtIndexBucket *)((unsigned char *)table->buckets +
           index * zbtIndexBucketBytes(table));
}

static inline uint64_t *zbtIndexHomeTagsAt(const zbtIndexTable *table,
                                           unsigned long index)
{
    return &zbtIndexBucketAt(table, index)->narrow.home_tags;
}

static inline uint32_t zbtIndexGetId(const zbtIndexTable *table,
                                     const zbtIndexBucket *bucket,
                                     unsigned int pos)
{
    if (table->wide_ids) return bucket->wide.id[pos];
    uint16_t id = bucket->narrow.id[pos];
    return id == UINT16_MAX ? ZBT_INDEX_DELETED_ID : id;
}

static inline uint64_t zbtIndexTags(const zbtIndexBucket *bucket) {
    return bucket->narrow.tags;
}

static inline void zbtIndexSetId(const zbtIndexTable *table,
                                 zbtIndexBucket *bucket, unsigned int pos,
                                 uint32_t id)
{
    if (table->wide_ids) {
        bucket->wide.id[pos] = id;
    } else {
        serverAssert(id == ZBT_INDEX_DELETED_ID || id < UINT16_MAX);
        bucket->narrow.id[pos] = id == ZBT_INDEX_DELETED_ID ?
                                 UINT16_MAX : (uint16_t)id;
    }
}

static inline uint8_t zbtIndexTag(uint32_t hash) {
    uint8_t tag = hash >> 24;
    return tag ? tag : 1;
}

static inline uint64_t zbtIndexTagBits(uint8_t tag) {
    uint64_t mixed = (uint64_t)tag * UINT64_C(0x9e3779b97f4a7c15);
    return (UINT64_C(1) << (tag & 63)) |
           (UINT64_C(1) << (mixed >> 58)) |
           (UINT64_C(1) << ((mixed >> 36) & 63));
}

static inline void zbtIndexRecordHomeTag(zbtIndexTable *table, uint32_t hash) {
    unsigned long home = hash & (table->size - 1);
    *zbtIndexHomeTagsAt(table, home) |= zbtIndexTagBits(zbtIndexTag(hash));
}

static inline int zbtIndexHomeMayContain(const zbtIndexTable *table,
                                         uint32_t hash)
{
    unsigned long home = hash & (table->size - 1);
    uint64_t bits = zbtIndexTagBits(zbtIndexTag(hash));
    return (*zbtIndexHomeTagsAt(table, home) & bits) == bits;
}

static inline uint64_t zbtIndexTagMask(uint64_t tags, uint8_t tag) {
    uint64_t x = tags ^ (UINT64_C(0x0101010101010101) * tag);
    return (x - UINT64_C(0x0101010101010101)) & ~x &
           UINT64_C(0x8080808080808080);
}

static inline unsigned int zbtIndexFirstTag(uint64_t mask) {
    return (unsigned int)(__builtin_ctzll(mask) >> 3);
}

static inline void zbtIndexSetTag(zbtIndexBucket *bucket, unsigned int pos,
                                  uint8_t tag)
{
    uint64_t shift = pos * 8;
    bucket->narrow.tags =
        (zbtIndexTags(bucket) & ~(UINT64_C(0xff) << shift)) |
        ((uint64_t)tag << shift);
}

static unsigned long zbtIndexNextPower(unsigned long size) {
    unsigned long result = ZBT_INDEX_INITIAL_BUCKETS;
    while (result < size) {
        if (result > ULONG_MAX / 2) return 0;
        result <<= 1;
    }
    return result;
}

static unsigned long zbtIndexBucketsForElements(unsigned long elements) {
    if (elements == 0) return ZBT_INDEX_INITIAL_BUCKETS;
    unsigned long quotient = elements / ZBT_INDEX_MAX_LOAD_NUM;
    unsigned long remainder = elements % ZBT_INDEX_MAX_LOAD_NUM;
    if (quotient > ULONG_MAX / ZBT_INDEX_MAX_LOAD_DEN) return 0;
    unsigned long slots = quotient * ZBT_INDEX_MAX_LOAD_DEN;
    unsigned long extra =
        (remainder * ZBT_INDEX_MAX_LOAD_DEN + ZBT_INDEX_MAX_LOAD_NUM - 1) /
        ZBT_INDEX_MAX_LOAD_NUM;
    if (slots > ULONG_MAX - extra) return 0;
    slots += extra;
    unsigned long buckets = slots / ZBT_INDEX_BUCKET_ITEMS +
                            (slots % ZBT_INDEX_BUCKET_ITEMS != 0);
    return zbtIndexNextPower(buckets);
}

static void zbtIndexTableInit(zbtree *t, zbtIndexTable *table,
                              unsigned long buckets, int wide_ids)
{
    memset(table, 0, sizeof(*table));
    table->size = zbtIndexNextPower(buckets);
    serverAssert(table->size != 0);
    table->wide_ids = wide_ids;
    table->scan_revision = zbtIndexNextRevision();
    size_t bytes = zbtIndexTableBytes(table);
    size_t usable;
    table->buckets = zcalloc_usable(bytes, &usable);
    t->alloc_size += usable;
}

static int zbtIndexTableTryInit(zbtree *t, zbtIndexTable *table,
                                unsigned long buckets, int wide_ids)
{
    memset(table, 0, sizeof(*table));
    table->size = zbtIndexNextPower(buckets);
    if (table->size == 0) return 0;
    table->wide_ids = wide_ids;
    size_t bucket_bytes = zbtIndexBucketBytes(table);
    if (table->size > SIZE_MAX / bucket_bytes) {
        memset(table, 0, sizeof(*table));
        return 0;
    }
    size_t usable;
    table->buckets = ztrycalloc_usable(table->size * bucket_bytes, &usable);
    if (table->buckets == NULL) {
        memset(table, 0, sizeof(*table));
        return 0;
    }
    table->scan_revision = zbtIndexNextRevision();
    t->alloc_size += usable;
    return 1;
}

static void zbtIndexTableRelease(zbtree *t, zbtIndexTable *table) {
    if (table->buckets) zbtFreeAllocation(t, table->buckets);
    memset(table, 0, sizeof(*table));
}

static void zbtRegisterLeaf(zbtree *t, zbtLeaf *leaf, uint32_t id) {
    if (id == ZBT_NEW_LEAF_ID) {
        if (t->free_score_leaf_id && t->member_rehash == NULL) {
            id = t->free_score_leaf_id - 1;
            serverAssert(ZBT_IS_FREE_LEAF_ID(t->table_to_leaf[id]));
            t->free_score_leaf_id =
                ZBT_NEXT_FREE_LEAF_ID(t->table_to_leaf[id]);
        } else {
            id = t->next_score_leaf_id++;
            if (id == t->score_leaf_cap) {
                uint32_t newcap = t->score_leaf_cap ? t->score_leaf_cap * 2 : 16;
                size_t usable, old_usable = 0;
                t->table_to_leaf = zrealloc_usable(
                    t->table_to_leaf,
                    newcap * sizeof(*t->table_to_leaf),
                    &usable, &old_usable);
                memset(t->table_to_leaf + t->score_leaf_cap, 0,
                       (newcap - t->score_leaf_cap) *
                       sizeof(*t->table_to_leaf));
                t->score_leaf_cap = newcap;
                t->alloc_size += usable - old_usable;
            }
        }
    }
    leaf->id = id;
    t->table_to_leaf[id] = leaf;
}

static void zbtReleaseLeafId(zbtree *t, uint32_t id) {
    serverAssert(id < t->next_score_leaf_id);
    t->table_to_leaf[id] = ZBT_FREE_LEAF_ID(t->free_score_leaf_id);
    t->free_score_leaf_id = id + 1;
}

static void zbtIndexTableInsertRaw(zbtIndexTable *table, uint32_t hash,
                                   uint32_t id)
{
    uint8_t tag = zbtIndexTag(hash);
    unsigned long mask = table->size - 1;
    unsigned long index = hash & mask;

    for (unsigned long probes = 0; probes < table->size; probes++) {
        zbtIndexBucket *bucket = zbtIndexBucketAt(table, index);
        for (unsigned int pos = 0; pos < ZBT_INDEX_BUCKET_ITEMS; pos++) {
            uint8_t oldtag = (uint8_t)(zbtIndexTags(bucket) >> (pos * 8));
            uint32_t oldid = zbtIndexGetId(table, bucket, pos);
            if (oldtag == 0 || oldid == ZBT_INDEX_DELETED_ID) {
                if (oldtag == 0) table->filled++;
                zbtIndexSetTag(bucket, pos, tag);
                zbtIndexSetId(table, bucket, pos, id);
                table->used++;
                zbtIndexRecordHomeTag(table, hash);
                return;
            }
        }
        index = (index + 1) & mask;
    }
    serverPanic("B+ tree member index has no free slot");
}

static zbtLeaf *zbtLeafFromId(zbtree *t, uint32_t id) {
    if (id >= t->next_score_leaf_id) return NULL;
    zbtLeaf *leaf = t->table_to_leaf[id];
    if (leaf == NULL || ZBT_IS_FREE_LEAF_ID(leaf)) return NULL;
    return leaf;
}

static int zbtIndexTableFind(zbtree *t, zbtIndexTable *table,
                             uint32_t hash, const unsigned char *ele,
                             size_t elelen,
                             zbtLeaf **found_leaf, unsigned int *found_leaf_pos,
                             zbtIndexBucket **found_bucket,
                             unsigned int *found_pos,
                             zbtIndexBucket **insert_bucket,
                             unsigned int *insert_pos)
{
    if (table->size == 0) return 0;
    if (!zbtIndexHomeMayContain(table, hash)) return 0;
    uint8_t tag = zbtIndexTag(hash);
    unsigned long mask = table->size - 1;
    unsigned long index = hash & mask;
    zbtIndexBucket *first_deleted = NULL;
    unsigned int first_deleted_pos = 0;

    for (unsigned long probes = 0; probes < table->size; probes++) {
        zbtIndexBucket *bucket = zbtIndexBucketAt(table, index);
        uint64_t matches = zbtIndexTagMask(zbtIndexTags(bucket), tag);
        while (matches) {
            unsigned int pos = zbtIndexFirstTag(matches);
            if ((uint8_t)(zbtIndexTags(bucket) >> (pos * 8)) != tag) {
                matches &= matches - 1;
                continue;
            }
            uint32_t id = zbtIndexGetId(table, bucket, pos);
            if (id != ZBT_INDEX_DELETED_ID) {
                zbtLeaf *leaf = zbtLeafFromId(t, id);
                if (leaf == NULL) {
                    serverAssert(t->member_rehash &&
                                 table == &t->member_index);
                    matches &= matches - 1;
                    continue;
                }
                serverAssert(leaf->id == id);
                if (t->member_rehash) {
                    int in_new = table == &t->member_rehash->table;
                    int migrated = zbtIndexLeafMigrated(t, leaf);
                    if (in_new != migrated) {
                        matches &= matches - 1;
                        continue;
                    }
                }
                unsigned int leaf_pos;
                if (zbtLeafFindMember(leaf, hash, ele, elelen, &leaf_pos)) {
                    if (found_leaf) *found_leaf = leaf;
                    if (found_leaf_pos) *found_leaf_pos = leaf_pos;
                    if (found_bucket) *found_bucket = bucket;
                    if (found_pos) *found_pos = pos;
                    return 1;
                }
            }
            matches &= matches - 1;
        }

        for (unsigned int pos = 0; pos < ZBT_INDEX_BUCKET_ITEMS; pos++) {
            uint8_t oldtag = (uint8_t)(zbtIndexTags(bucket) >> (pos * 8));
            if (oldtag == 0) {
                if (insert_bucket) {
                    *insert_bucket = first_deleted ? first_deleted : bucket;
                    *insert_pos = first_deleted ? first_deleted_pos : pos;
                }
                return 0;
            }
            if (first_deleted == NULL &&
                zbtIndexGetId(table, bucket, pos) == ZBT_INDEX_DELETED_ID)
            {
                first_deleted = bucket;
                first_deleted_pos = pos;
            }
        }
        index = (index + 1) & mask;
    }
    if (insert_bucket && first_deleted) {
        *insert_bucket = first_deleted;
        *insert_pos = first_deleted_pos;
    }
    return 0;
}

static int zbtIndexTableFindReference(zbtIndexTable *table, uint32_t hash,
                                      uint32_t id,
                                      zbtIndexBucket **found_bucket,
                                      unsigned int *found_pos)
{
    if (table->size == 0) return 0;
    if (!zbtIndexHomeMayContain(table, hash)) return 0;
    uint8_t tag = zbtIndexTag(hash);
    unsigned long mask = table->size - 1;
    unsigned long index = hash & mask;

    for (unsigned long probes = 0; probes < table->size; probes++) {
        zbtIndexBucket *bucket = zbtIndexBucketAt(table, index);
        uint64_t matches = zbtIndexTagMask(zbtIndexTags(bucket), tag);
        while (matches) {
            unsigned int pos = zbtIndexFirstTag(matches);
            if ((uint8_t)(zbtIndexTags(bucket) >> (pos * 8)) == tag &&
                zbtIndexGetId(table, bucket, pos) == id)
            {
                *found_bucket = bucket;
                *found_pos = pos;
                return 1;
            }
            matches &= matches - 1;
        }
        if (zbtIndexTagMask(zbtIndexTags(bucket), 0)) return 0;
        index = (index + 1) & mask;
    }
    return 0;
}

static int zbtIndexStartResize(zbtree *t, unsigned long elements,
                               int allow_same_size)
{
    if (t->member_rehash) return 0;
    unsigned long buckets = zbtIndexBucketsForElements(elements);
    serverAssert(buckets != 0);
    if (buckets == t->member_index.size && !allow_same_size) return 0;

    t->member_rehash = zbtAlloc(t, sizeof(*t->member_rehash));
    memset(t->member_rehash, 0, sizeof(*t->member_rehash));
    int wide_ids = t->member_index.wide_ids ||
                   t->next_score_leaf_id >= ZBT_INDEX_WIDE_ID_AT;
    zbtIndexTableInit(t, &t->member_rehash->table, buckets, wide_ids);
    zbtLeaf *first = (zbtLeaf *)t->head;
    serverAssert(first != NULL);
    t->member_rehash->next_leaf_id = first->id;
    uint16_t resize_id = first->index_resize + 1;
    if (resize_id == 0) resize_id = 1;
    t->member_rehash->resize_id = resize_id;
    t->member_revision = zbtIndexNextRevision();
    return 1;
}

static int zbtIndexLeafMigrated(const zbtree *t, const zbtLeaf *leaf) {
    return t->member_rehash &&
           leaf->index_resize == t->member_rehash->resize_id;
}

static void zbtIndexCopyLeaf(zbtree *t, zbtLeaf *leaf) {
    zbtIndexRehash *rehash = t->member_rehash;
    serverAssert(rehash != NULL);
    if (zbtIndexLeafMigrated(t, leaf)) return;

    /* Structural changes can move members into an uncopied leaf before the
     * rehash cursor reaches it. Remove those partial target references, then
     * publish one fresh reference for every current member in the leaf. */
    for (unsigned int i = 0; i < leaf->n.count; i++) {
        zbtIndexBucket *bucket;
        unsigned int pos;
        uint32_t hash = zbtElemIndexHash(leaf->elems[i]);
        while (zbtIndexTableFindReference(&rehash->table, hash, leaf->id,
                                           &bucket, &pos))
        {
            zbtIndexSetId(&rehash->table, bucket, pos,
                          ZBT_INDEX_DELETED_ID);
            rehash->table.used--;
        }
    }
    for (unsigned int i = 0; i < leaf->n.count; i++) {
        if (t->pending_insert == leaf->elems[i]) continue;
        zbtIndexTableInsertRaw(&rehash->table, zbtElemIndexHash(leaf->elems[i]),
                               leaf->id);
    }
    leaf->index_resize = rehash->resize_id;
    /* Entries may have moved to buckets already passed by an active scan. */
    zbtIndexInvalidateScan(t);
    t->member_revision = zbtIndexNextRevision();
}

static int zbtIndexRehashStep(zbtree *t, int steps) {
    zbtIndexRehash *rehash = t->member_rehash;
    if (rehash == NULL) return 0;

    while (steps--) {
        zbtLeaf *leaf = NULL;
        if (rehash->next_leaf_id != ZBT_NO_LEAF_ID &&
            rehash->next_leaf_id < t->next_score_leaf_id)
        {
            leaf = zbtLeafFromId(t, rehash->next_leaf_id);
        }
        if (leaf == NULL) leaf = (zbtLeaf *)t->head;
        while (leaf && zbtIndexLeafMigrated(t, leaf)) leaf = leaf->next;
        if (leaf == NULL) {
            serverAssert(rehash->table.used == t->length);
            zbtIndexTable old = t->member_index;
            t->member_index = rehash->table;
            zbtFreeAllocation(t, rehash);
            t->member_rehash = NULL;
            zbtIndexTableRelease(t, &old);
            t->member_revision = zbtIndexNextRevision();
            return 0;
        }

        rehash->next_leaf_id =
            leaf->next ? leaf->next->id : ZBT_NO_LEAF_ID;
        zbtIndexCopyLeaf(t, leaf);
    }
    return 1;
}

static void zbtIndexExpandIfNeeded(zbtree *t, unsigned long add) {
    if (t->member_index.size == 0) {
        unsigned long buckets = zbtIndexBucketsForElements(add);
        serverAssert(buckets != 0);
        zbtIndexTableInit(t, &t->member_index, buckets, 0);
        t->member_revision = zbtIndexNextRevision();
        return;
    }
    if (t->member_rehash) {
        zbtIndexTable *target = &t->member_rehash->table;
        unsigned long slots = zbtIndexSlots(target);

        if ((t->length + add) * ZBT_INDEX_MAX_LOAD_DEN <=
                slots * ZBT_INDEX_MAX_LOAD_NUM &&
            (target->filled + add) * ZBT_INDEX_MAX_FILLED_DEN <
                slots * ZBT_INDEX_MAX_FILLED_NUM)
            return;
        while (t->member_rehash)
            zbtIndexRehashStep(t, 64);
    }

    if (!t->member_index.wide_ids &&
        t->next_score_leaf_id >= ZBT_INDEX_WIDE_ID_AT)
    {
        zbtIndexStartResize(t, t->member_index.used + add, 1);
        return;
    }

    unsigned long slots = zbtIndexSlots(&t->member_index);
    unsigned long live = t->member_index.used + add;
    unsigned long filled = t->member_index.filled + add;
    if (live * ZBT_INDEX_MAX_LOAD_DEN >
            slots * ZBT_INDEX_MAX_LOAD_NUM ||
        filled * ZBT_INDEX_MAX_FILLED_DEN >=
            slots * ZBT_INDEX_MAX_FILLED_NUM)
    {
        zbtIndexStartResize(t, live, 1);
    }
}

static void zbtIndexShrinkIfNeeded(zbtree *t) {
    if (t->length == 0) {
        zbtIndexTableRelease(t, &t->member_index);
        if (t->member_rehash) {
            zbtIndexTableRelease(t, &t->member_rehash->table);
            zbtFreeAllocation(t, t->member_rehash);
            t->member_rehash = NULL;
        }
        t->member_revision = zbtIndexNextRevision();
        return;
    }
    if (t->member_rehash ||
        t->member_index.size <= ZBT_INDEX_INITIAL_BUCKETS)
        return;
    unsigned long used = t->member_index.used;
    if (used * ZBT_INDEX_MIN_LOAD_DEN <=
        zbtIndexSlots(&t->member_index) * ZBT_INDEX_MIN_LOAD_NUM)
    {
        zbtIndexStartResize(t, used ? used : 1, 0);
    }
}

static int zbtIndexOwnsBucket(zbtIndexTable *table, zbtIndexBucket *bucket) {
    if (table->buckets == NULL) return 0;
    uintptr_t address = (uintptr_t)bucket;
    uintptr_t first = (uintptr_t)table->buckets;
    uintptr_t last = first + table->size * zbtIndexBucketBytes(table);
    return address >= first && address < last;
}

static int zbtIndexFind(zbtree *t, uint32_t hash,
                        const unsigned char *ele, size_t elelen,
                        zbtLeaf **found_leaf, unsigned int *found_leaf_pos,
                        zbtIndexBucket **found_bucket,
                        unsigned int *found_bucket_position)
{
    zbtIndexBucket *bucket = NULL, *hint_bucket = NULL;
    unsigned int pos = 0, hint_pos = 0;
    zbtIndexRehashStep(t, 1);

    if (t->member_rehash &&
        zbtIndexTableFind(t, &t->member_rehash->table, hash, ele, elelen,
                          found_leaf, found_leaf_pos,
                          &bucket, &pos, &hint_bucket, &hint_pos))
        goto found;

    if (zbtIndexTableFind(t, &t->member_index, hash, ele, elelen,
                          found_leaf, found_leaf_pos,
                          &bucket, &pos,
                          t->member_rehash ? NULL : &hint_bucket,
                          t->member_rehash ? NULL : &hint_pos))
        goto found;

    if (found_bucket) *found_bucket = hint_bucket;
    if (found_bucket_position) *found_bucket_position = hint_pos;
    return 0;

found:
    if (found_bucket) *found_bucket = bucket;
    if (found_bucket_position) *found_bucket_position = pos;
    return 1;
}

static int zbtIndexFindReference(zbtree *t, uint32_t hash,
                                 uint32_t leaf_id,
                                 zbtIndexBucket **found_bucket,
                                 unsigned int *found_pos)
{
    zbtIndexBucket *bucket;
    if (t->member_rehash &&
        zbtIndexTableFindReference(&t->member_rehash->table, hash,
                                   leaf_id, &bucket, found_pos))
    {
        *found_bucket = bucket;
        return 1;
    }
    if (t->member_index.size &&
        zbtIndexTableFindReference(&t->member_index, hash, leaf_id,
                                   &bucket, found_pos))
    {
        *found_bucket = bucket;
        return 1;
    }
    return 0;
}

static int zbtIndexMove(zbtree *t, uint32_t hash,
                        uint32_t old_leaf_id, uint32_t new_leaf_id,
                        int target_is_incomplete, int *target_was_copied)
{
    zbtIndexBucket *bucket;
    unsigned int pos;
    if (t->member_rehash == NULL) {
        if (!zbtIndexTableFindReference(&t->member_index, hash,
                                        old_leaf_id, &bucket, &pos))
            return 0;
        zbtIndexSetId(&t->member_index, bucket, pos, new_leaf_id);
        return 1;
    }

    zbtLeaf *newleaf = zbtLeafFromId(t, new_leaf_id);
    serverAssert(newleaf);
    if (!zbtIndexLeafMigrated(t, newleaf) && target_is_incomplete) {
        if (!zbtIndexFindReference(t, hash, old_leaf_id, &bucket, &pos))
            return 0;
        zbtIndexTable *source = zbtIndexOwnsBucket(&t->member_index, bucket) ?
                                &t->member_index :
                                &t->member_rehash->table;
        zbtIndexSetId(source, bucket, pos, new_leaf_id);
        return 1;
    }
    int copied_target = target_was_copied && *target_was_copied;
    if (!zbtIndexLeafMigrated(t, newleaf)) {
        zbtIndexCopyLeaf(t, newleaf);
        copied_target = 1;
        if (target_was_copied) *target_was_copied = 1;
    }

    if (!zbtIndexFindReference(t, hash, old_leaf_id, &bucket, &pos))
        return 0;

    zbtIndexTable *source = &t->member_index;
    if (!zbtIndexOwnsBucket(source, bucket)) {
        serverAssert(zbtIndexOwnsBucket(&t->member_rehash->table, bucket));
        source = &t->member_rehash->table;
    }
    if (copied_target) {
        zbtIndexSetId(source, bucket, pos, ZBT_INDEX_DELETED_ID);
        source->used--;
    } else if (source == &t->member_rehash->table) {
        zbtIndexSetId(source, bucket, pos, new_leaf_id);
    } else {
        zbtIndexSetId(source, bucket, pos, ZBT_INDEX_DELETED_ID);
        source->used--;
        zbtIndexTableInsertRaw(&t->member_rehash->table, hash, new_leaf_id);
    }
    return 1;
}

static void zbtIndexDeleteAt(zbtree *t, zbtIndexBucket *bucket,
                             unsigned int pos)
{
    zbtIndexTable *table = &t->member_index;
    if (!zbtIndexOwnsBucket(table, bucket)) {
        serverAssert(t->member_rehash &&
                     zbtIndexOwnsBucket(&t->member_rehash->table, bucket));
        table = &t->member_rehash->table;
    }
    serverAssert(zbtIndexGetId(table, bucket, pos) != ZBT_INDEX_DELETED_ID);
    zbtIndexSetId(table, bucket, pos, ZBT_INDEX_DELETED_ID);
    table->used--;
}

static void zbtIndexInsert(zbtree *t, uint32_t hash, uint32_t leaf_id,
                           const zbtreeInsertPosition *position)
{
    if (t->member_rehash) {
        zbtLeaf *leaf = zbtLeafFromId(t, leaf_id);
        serverAssert(leaf);
        if (!zbtIndexLeafMigrated(t, leaf))
            zbtIndexCopyLeaf(t, leaf);
    }

    zbtIndexTable *table = t->member_rehash ?
        &t->member_rehash->table : &t->member_index;
    zbtIndexBucket *bucket = position ?
        (zbtIndexBucket *)position->bucket : NULL;
    unsigned int pos = position ? position->pos : 0;
    if (position && position->hash == hash &&
        position->revision == t->member_revision && bucket &&
        zbtIndexOwnsBucket(table, bucket) &&
        pos < ZBT_INDEX_BUCKET_ITEMS)
    {
        uint8_t oldtag = (uint8_t)(zbtIndexTags(bucket) >> (pos * 8));
        uint32_t oldid = zbtIndexGetId(table, bucket, pos);
        if (oldtag == 0 || oldid == ZBT_INDEX_DELETED_ID) {
            if (oldtag == 0) table->filled++;
            zbtIndexSetTag(bucket, pos, zbtIndexTag(hash));
            zbtIndexSetId(table, bucket, pos, leaf_id);
            table->used++;
            zbtIndexRecordHomeTag(table, hash);
            zbtIndexRehashStep(t, 1);
            return;
        }
    }

    table = t->member_rehash ?
        &t->member_rehash->table : &t->member_index;
    zbtIndexTableInsertRaw(table, hash, leaf_id);
    zbtIndexRehashStep(t, 1);
}

static void zbtIndexDeleteElem(zbtree *t, zbtElem *e) {
    uint32_t hash = zbtElemIndexHash(e);
    zbtIndexBucket *bucket;
    unsigned int pos;
    serverAssert(zbtIndexFindReference(t, hash, e->leaf->id, &bucket, &pos));
    zbtIndexDeleteAt(t, bucket, pos);
}

zbtElem *zbtFindMemberHash(zbtree *t, sds ele, uint32_t hash,
                           zbtreeInsertPosition *position)
{
    zbtLeaf *leaf = NULL;
    unsigned int leaf_pos = 0;
    zbtIndexBucket *bucket = NULL;
    unsigned int pos = 0;
    int found = zbtIndexFind(t, hash, (const unsigned char *)ele, sdslen(ele),
                             &leaf, &leaf_pos, &bucket, &pos);
    if (position) {
        position->bucket = bucket;
        position->pos = pos;
        position->hash = hash;
        position->revision = t->member_revision;
    }
    if (!found) return NULL;
    return leaf->elems[leaf_pos];
}

zbtElem *zbtFindMember(zbtree *t, sds ele, zbtreeInsertPosition *position) {
    uint32_t hash = (uint32_t)dictSdsHash(ele);
    return zbtFindMemberHash(t, ele, hash, position);
}

void zbtIndexReserve(zbtree *t, unsigned long count) {
    zbtIndexExpandIfNeeded(t, count);
}

int zbtIndexTryReserve(zbtree *t, unsigned long count) {
    serverAssert(t->length == 0);
    serverAssert(t->member_index.size == 0);
    serverAssert(t->member_rehash == NULL);
    unsigned long buckets = zbtIndexBucketsForElements(count);
    unsigned long leaves = count / ZBT_LEAF_MAX +
                           (count % ZBT_LEAF_MAX != 0);
    int wide_ids = leaves >= ZBT_INDEX_WIDE_ID_AT;
    if (buckets == 0 ||
        !zbtIndexTableTryInit(t, &t->member_index, buckets, wide_ids))
        return 0;
    t->member_revision = zbtIndexNextRevision();
    return 1;
}

void zbtIndexMaintenance(zbtree *t, unsigned int steps) {
    if (t->member_rehash)
        zbtIndexRehashStep(t, (int)steps);
    zbtIndexShrinkIfNeeded(t);
    if (t->member_rehash)
        zbtIndexRehashStep(t, (int)steps);
}

static unsigned long zbtRandomBelow(unsigned long limit) {
    serverAssert(limit != 0);
    unsigned long threshold = (0UL - limit) % limit;
    unsigned long value;
    do {
        value = randomULong();
    } while (value < threshold);
    return value % limit;
}

void zbtIndexIndexTree(zbtree *t) {
    if (t->length == 0) return;
    zbtIndexExpandIfNeeded(t, t->length);
    zbtIndexTable *table = t->member_rehash ?
        &t->member_rehash->table : &t->member_index;
    if (table->used >= t->length) {
        while (t->member_rehash) zbtIndexRehashStep(t, 64);
        return;
    }
    zbtLeaf *lf = (zbtLeaf *)t->head;
    while (lf) {
        if (t->member_rehash) {
            zbtIndexCopyLeaf(t, lf);
        } else {
            for (uint32_t i = 0; i < lf->n.count; i++)
                zbtIndexTableInsertRaw(&t->member_index,
                                       zbtElemIndexHash(lf->elems[i]), lf->id);
        }
        lf = lf->next;
    }
    while (t->member_rehash) zbtIndexRehashStep(t, 64);
}

zbtElem *zbtRandomElem(zbtree *t) {
    if (t->length == 0) return NULL;
    zbtIndexMaintenance(t, 1);
    /* Sampling an ID and a fixed-capacity leaf slot gives every element the
     * same chance. Reject holes and unused slots; under normal occupancy this
     * succeeds in O(1), while the rank fallback bounds pathological ID churn. */
    for (unsigned int attempts = 0; attempts < 64; attempts++) {
        uint32_t id = (uint32_t)zbtRandomBelow(t->next_score_leaf_id);
        zbtLeaf *leaf = zbtLeafFromId(t, id);
        if (leaf == NULL) continue;
        unsigned int pos = (unsigned int)zbtRandomBelow(ZBT_LEAF_MAX);
        if (pos < leaf->n.count) return leaf->elems[pos];
    }
    unsigned long rank = zbtRandomBelow(t->length) + 1;
    return zbtElemByRank(t, rank, NULL);
}

static unsigned long zbtIndexScanSlot(const zbtree *t, zbtIndexTable *table,
                                      zbtIndexBucket *bucket,
                                      unsigned int slot_pos,
                                      zbtScanFunction *fn, void *privdata)
{
    uint8_t tag = (uint8_t)(zbtIndexTags(bucket) >> (slot_pos * 8));
    uint32_t id = zbtIndexGetId(table, bucket, slot_pos);
    if (tag == 0 || id == ZBT_INDEX_DELETED_ID) return 0;
    zbtLeaf *leaf = zbtLeafFromId((zbtree *)t, id);
    if (leaf == NULL) {
        serverAssert(t->member_rehash && table == &t->member_index);
        return 0;
    }
    serverAssert(leaf->id == id);
    if (t->member_rehash) {
        int in_new = table == &t->member_rehash->table;
        if (in_new != zbtIndexLeafMigrated(t, leaf)) return 0;
    }

    unsigned int positions[ZBT_LEAF_MAX + 1];
    unsigned int count = 0;
    for (unsigned int i = 0; i < leaf->n.count; i++) {
        uint8_t leaf_tag = leaf->tags[i];
        if (leaf_tag == tag || (tag == 1 && leaf_tag == 0))
            positions[count++] = i;
    }
    serverAssert(count != 0);

    unsigned long emitted = 0;
    for (unsigned int i = 0; i < count; i++) {
        zbtElem *e = leaf->elems[positions[i]];
        uint32_t hash = zbtElemIndexHash(e);
        zbtIndexBucket *owner;
        unsigned int owner_pos;
        if (!zbtIndexTableFindReference(table, hash, id, &owner, &owner_pos) ||
            owner != bucket || owner_pos != slot_pos)
            continue;
        sds s = zbtGetEle(e);
        fn(privdata, (const unsigned char *)s, sdslen(s), zbtGetScore(e));
        emitted++;
    }
    return emitted;
}

void zbtDismissIndex(zbtree *t) {
    if (t->table_to_leaf)
        dismissMemory(t->table_to_leaf,
                      (size_t)t->score_leaf_cap * sizeof(void *));
    if (t->member_index.buckets)
        dismissMemory(t->member_index.buckets,
                      zbtIndexTableBytes(&t->member_index));
    if (t->member_rehash && t->member_rehash->table.buckets)
        dismissMemory(t->member_rehash->table.buckets,
                      zbtIndexTableBytes(&t->member_rehash->table));
}

uint64_t zbtScan(zbtree *t, uint64_t cursor, unsigned long count,
                 zbtScanFunction *fn, void *privdata)
{
    if (t->length == 0) return 0;
    uint32_t revision = (uint32_t)(cursor >> 32);
    uint64_t group = cursor ? (uint32_t)cursor - 1 : 0;
    if (cursor == 0 || revision != t->member_index.scan_revision) {
        revision = t->member_index.scan_revision;
        group = 0;
    }

    uint64_t first_buckets = t->member_index.size;
    uint64_t total_buckets = first_buckets;
    if (t->member_rehash)
        total_buckets += t->member_rehash->table.size;
    uint64_t bucket_index = group * ZBT_SCAN_BUCKETS_PER_STEP;
    if (bucket_index >= total_buckets) return 0;
    unsigned long emitted = 0;
    unsigned long max_groups =
        count > ULONG_MAX / 10 ? ULONG_MAX : count * 10;
    unsigned long scanned_groups = 0;
    while (bucket_index < total_buckets && emitted < count &&
           scanned_groups < max_groups)
    {
        uint64_t end = bucket_index + ZBT_SCAN_BUCKETS_PER_STEP;
        if (end > total_buckets) end = total_buckets;
        while (bucket_index < end) {
            zbtIndexTable *table;
            uint64_t local;
            if (bucket_index < first_buckets) {
                table = &t->member_index;
                local = bucket_index;
            } else {
                serverAssert(t->member_rehash != NULL);
                table = &t->member_rehash->table;
                local = bucket_index - first_buckets;
            }
            zbtIndexBucket *bucket = zbtIndexBucketAt(table, local);
            for (unsigned int pos = 0; pos < ZBT_INDEX_BUCKET_ITEMS; pos++)
                emitted += zbtIndexScanSlot(t, table, bucket, pos, fn, privdata);
            bucket_index++;
        }
        group++;
        scanned_groups++;
    }
    if (bucket_index >= total_buckets) return 0;
    serverAssert(group < UINT32_MAX);
    return ((uint64_t)revision << 32) | (uint32_t)(group + 1);
}

/*-----------------------------------------------------------------------------
 * Node allocation / tree lifecycle
 *----------------------------------------------------------------------------*/

static zbtLeaf *zbtNewLeaf(zbtree *t) {
    size_t usable;
    zbtLeaf *lf = zmalloc_usable(sizeof(*lf), &usable);
    lf->n.parent = NULL;
    lf->n.count = 0;
    lf->n.isleaf = 1;
    lf->prev = lf->next = NULL;
    lf->id = ZBT_NO_LEAF_ID;
    lf->index_resize = 0;
    t->alloc_size += usable;
    zbtRegisterLeaf(t, lf, ZBT_NEW_LEAF_ID);
    return lf;
}

static zbtInner *zbtNewInner(zbtree *t) {
    size_t usable;
    zbtInner *in = zmalloc_usable(sizeof(*in), &usable);
    in->n.parent = NULL;
    in->n.count = 0;
    in->n.isleaf = 0;
    t->alloc_size += usable;
    return in;
}

static void zbtFreeNodeShallow(zbtree *t, zbtNode *n) {
    size_t usable;
    zfree_usable(n, &usable);
    t->alloc_size -= usable;
}

static void zbtFreeLeaf(zbtree *t, zbtLeaf *lf) {
    zbtReleaseLeafId(t, lf->id);
    zbtFreeNodeShallow(t, (zbtNode *)lf);
}

zbtree *zbtCreate(void) {
    size_t usable;
    zbtree *t = zmalloc_usable(sizeof(*t), &usable);
    memset(t, 0, sizeof(*t));
    t->length = 0;
    t->alloc_size = usable;
    t->root = NULL;
    t->defrag_resume = NULL;
    t->defrag_resume_score = 0;
    t->pending_insert = NULL;
    zbtLeaf *lf = zbtNewLeaf(t);
    t->root = (zbtNode *)lf;
    t->head = t->tail = (zbtNode *)lf;
    return t;
}

static void zbtFreeSubtree(zbtree *t, zbtNode *n) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        for (uint32_t i = 0; i < n->count; i++) {
            size_t usable;
            zfree_usable(lf->elems[i], &usable);
            t->alloc_size -= usable;
        }
    } else {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < n->count; i++)
            zbtFreeSubtree(t, in->child[i]);
    }
    zbtFreeNodeShallow(t, n);
}

void zbtFree(zbtree *t) {
    zbtFreeSubtree(t, t->root);
    if (t->defrag_resume) sdsfree(t->defrag_resume);
    zbtIndexTableRelease(t, &t->member_index);
    if (t->member_rehash) {
        zbtIndexTableRelease(t, &t->member_rehash->table);
        zbtFreeAllocation(t, t->member_rehash);
    }
    if (t->table_to_leaf) zfree(t->table_to_leaf);
    zfree(t);
}

size_t zbtAllocSize(const zbtree *t) { return t->alloc_size; }

/*-----------------------------------------------------------------------------
 * Navigation helpers
 *----------------------------------------------------------------------------*/

/* Minimum element of a subtree rooted at 'n' (assumes non-empty). An inner node
 * records the minimum of every child in sep[], so sep[0] is already the minimum
 * of the whole subtree and there is no need to descend to the leftmost leaf.
 * Callers must therefore keep sep[] consistent bottom-up, which is what
 * zbtUpdateToRoot() and the split/merge paths do. */
static zbtElem *zbtNodeMin(zbtNode *n) {
    if (n->isleaf) return ((zbtLeaf *)n)->elems[0];
    return ((zbtInner *)n)->sep[0];
}

/* Number of elements contained in the subtree rooted at 'n'. */
static unsigned long zbtSubtreeSize(zbtNode *n) {
    if (n->isleaf) return n->count;
    zbtInner *in = (zbtInner *)n;
    unsigned long s = 0;
    for (uint32_t i = 0; i < in->n.count; i++) s += in->csize[i];
    return s;
}

/* Index of child 'c' inside inner node 'p'. */
static int zbtChildIdx(zbtInner *p, zbtNode *c) {
    for (uint32_t i = 0; i < p->n.count; i++)
        if (p->child[i] == c) return (int)i;
    serverPanic("zbtree: child not found in parent");
}

/* Record 'lf' as the owner of elems[from, to). Every path that puts an element
 * into a leaf slot it did not previously occupy has to call this, or the
 * element's back-pointer goes stale and zbtRankByElem() walks up from the
 * wrong leaf. zbtDebugVerify() checks the whole tree for exactly that. */
static void zbtLeafClaim(zbtLeaf *lf, int from, int to) {
    for (int i = from; i < to; i++) {
        lf->elems[i]->leaf = lf;
        lf->tags[i] = zbtElemLeafTag(lf->elems[i]);
    }
}

/* Choose the child of inner node 'in' whose key range contains (score,ele).
 * sep[] is sorted, so the predicate "target >= sep[i]" holds on a prefix and
 * the last such i can be bisected. Every evaluation dereferences a separator
 * element, which for large members is a cold line in its own page, so the
 * comparison count - not the arithmetic - is what this loop costs. */
static int zbtInnerChildIdx(zbtInner *in, double score, sds ele, zbtLcp *lcp) {
    int lo = 1, hi = (int)in->n.count - 1, res = 0;
    /* sep[0] is the subtree minimum, which is the separator the parent
     * already compared against, so lcp->lo describes index 0 on entry. */
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        size_t agree;
        if (zbtCompareSkip(score, ele, in->sep[mid], zbtLcpSkip(lcp), &agree) >= 0) {
            res = mid;
            lo = mid + 1;
            lcp->lo = agree;
        } else {
            hi = mid - 1;
            lcp->hi = agree;
        }
    }
    return res;
}

/* Descend from the root to the leaf that would contain (score,ele). '*lcp'
 * accumulates the prefix knowledge of the descent and is handed to the leaf
 * search, which continues bisecting inside the same bracket. */
static zbtLeaf *zbtFindLeaf(zbtree *t, double score, sds ele, zbtLcp *lcp) {
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        n = in->child[zbtInnerChildIdx(in, score, ele, lcp)];
    }
    return (zbtLeaf *)n;
}

/* Locate (score,ele) inside a leaf. Sets *found and returns the index where
 * the element is (if found) or where it should be inserted. */
static int zbtLeafSearch(zbtLeaf *lf, double score, sds ele, int *found,
                         zbtLcp *lcp)
{
    int lo = 0, hi = (int)lf->n.count, eq = -1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        size_t agree;
        int c = zbtCompareSkip(score, ele, lf->elems[mid], zbtLcpSkip(lcp), &agree);
        if (c > 0) {
            lo = mid + 1;
            lcp->lo = agree;
        } else {
            /* (score,ele) is unique in the tree, so an equal hit is already
             * the lower bound and survives as 'lo'. */
            if (c == 0) eq = mid;
            hi = mid;
            lcp->hi = agree;
        }
    }
    *found = (eq == lo);
    return lo;
}

/* Locate an element already known to live in this leaf. The caller holds the
 * exact pointer (it came out of the ZSET dict), so identity over the packed
 * slot array answers the question without dereferencing any element. */
static int zbtLeafFindPtr(zbtLeaf *lf, const zbtElem *e) {
    for (uint32_t i = 0; i < lf->n.count; i++)
        if (lf->elems[i] == e) return (int)i;
    return -1;
}

/* Refresh csize/sep for every ancestor of 'n' up to the root. Used after an
 * insertion, deletion or in-place element replacement changed a subtree.
 *
 * The caller has already made 'n' itself consistent, and 'n' is the only child
 * whose recorded size can be stale. It is stale by a fixed amount, and every
 * ancestor's total is off by that same amount, so the subtree is sized once at
 * the bottom and the difference is propagated upwards rather than re-summing
 * csize[] at every level. The walk stops at the first level that turns out to
 * be unchanged, because then no ancestor above it can change either: that makes
 * sibling borrows and defrag replacements O(1) instead of O(height). */
static void zbtUpdateToRoot(zbtree *t, zbtNode *n) {
    UNUSED(t);
    long delta = 0;
    int have_delta = 0;

    while (n->parent) {
        zbtInner *p = (zbtInner *)n->parent;
        int idx = zbtChildIdx(p, n);
        unsigned long oldsize = p->csize[idx];
        unsigned long newsize;

        if (have_delta) {
            newsize = (unsigned long)((long)oldsize + delta);
        } else {
            /* O(1) when the walk starts at a leaf, which is the common case;
             * only the split/merge paths start at an inner node. */
            newsize = zbtSubtreeSize(n);
            delta = (long)newsize - (long)oldsize;
            have_delta = 1;
        }

        zbtElem *newsep = zbtNodeMin(n);
        if (newsize == oldsize && newsep == p->sep[idx]) return;
        p->csize[idx] = newsize;
        p->sep[idx] = newsep;
        n = (zbtNode *)p;
    }
}

/*-----------------------------------------------------------------------------
 * Insertion
 *----------------------------------------------------------------------------*/

static void zbtSplitInner(zbtree *t, zbtInner *in);

/* Insert 'right' as a new child immediately after 'left' in their parent.
 * If 'p' is NULL, 'left' is the current root and a new root is created. */
static void zbtInsertChild(zbtree *t, zbtInner *p, zbtNode *left, zbtNode *right) {
    if (p == NULL) {
        zbtInner *root = zbtNewInner(t);
        root->n.count = 2;
        root->child[0] = left;  left->parent = (zbtNode *)root;
        root->child[1] = right; right->parent = (zbtNode *)root;
        root->csize[0] = zbtSubtreeSize(left);  root->sep[0] = zbtNodeMin(left);
        root->csize[1] = zbtSubtreeSize(right); root->sep[1] = zbtNodeMin(right);
        t->root = (zbtNode *)root;
        return;
    }

    int li = zbtChildIdx(p, left);
    int at = li + 1;
    int tail = (int)p->n.count - at;
    memmove(&p->child[at + 1], &p->child[at], tail * sizeof(zbtNode *));
    memmove(&p->csize[at + 1], &p->csize[at], tail * sizeof(unsigned long));
    memmove(&p->sep[at + 1], &p->sep[at], tail * sizeof(zbtElem *));
    p->child[at] = right;
    right->parent = (zbtNode *)p;
    p->n.count++;

    p->csize[li] = zbtSubtreeSize(left);  p->sep[li] = zbtNodeMin(left);
    p->csize[at] = zbtSubtreeSize(right); p->sep[at] = zbtNodeMin(right);

    if (p->n.count > ZBT_INNER_MAX)
        zbtSplitInner(t, p);
    else
        zbtUpdateToRoot(t, (zbtNode *)p);
}

static void zbtSplitInner(zbtree *t, zbtInner *in) {
    zbtInner *r = zbtNewInner(t);
    int total = (int)in->n.count; /* == ZBT_INNER_MAX + 1 */
    int keep = total / 2;
    int move = total - keep;
    memcpy(r->child, &in->child[keep], move * sizeof(zbtNode *));
    memcpy(r->csize, &in->csize[keep], move * sizeof(unsigned long));
    memcpy(r->sep, &in->sep[keep], move * sizeof(zbtElem *));
    r->n.count = move;
    in->n.count = keep;
    for (int i = 0; i < move; i++) r->child[i]->parent = (zbtNode *)r;
    zbtInsertChild(t, (zbtInner *)in->n.parent, (zbtNode *)in, (zbtNode *)r);
}

/* Split the overfull leaf 'lf' in two, biased by where the overflowing element
 * landed (see the ZBT_SPLIT_* constants).
 *
 * An even split is right in the general case but wrong for sorted insertion:
 * inserts never come back to the side left behind, so it stays half full for
 * the rest of its life and a sorted bulk ZADD needs twice the leaves its
 * contents call for. Peeling off just the overflowing element instead keeps the
 * other side full. Ascending insertion leaves the left side full and starts a
 * fresh tail; descending insertion is the mirror image, keeping one element in
 * the head leaf and handing the full load to the new right leaf.
 *
 * Interior overflows try zbtShareOverflow() first: a same-parent sibling with
 * slack takes the extra elements so an interpolation pass (1,2,3 then 1.1,2.1)
 * refills the half left behind by an even split, instead of stranding it at
 * MIN. Append/prepend never enter that path — the neighbour is already full.
 *
 * Since ZBT_LEAF_MIN is ZBT_LEAF_MAX/2, no skew is possible without letting the
 * under-filled side sit below the minimum -- which is why the head and tail
 * leaves are exempt from it (see zbtVerifyNode()). Further inserts fill them
 * again, and the delete paths need no special case: a full neighbour can always
 * cover a short end leaf's deficit, and rebalancing only merges when the pair
 * totals below 2 * ZBT_LEAF_MIN, so it can never overfill a leaf. */
static void zbtSplitLeaf(zbtree *t, zbtLeaf *lf, int bias) {
    zbtLeaf *r = zbtNewLeaf(t);
    int total = (int)lf->n.count; /* == ZBT_LEAF_MAX + 1 */
    int keep = total / 2;
    if (bias == ZBT_SPLIT_APPEND) keep = total - 1;
    else if (bias == ZBT_SPLIT_PREPEND) keep = 1;
    int move = total - keep;
    uint32_t old_id = lf->id;
    memcpy(r->elems, &lf->elems[keep], move * sizeof(zbtElem *));
    memcpy(r->tags, &lf->tags[keep], move);
    zbtLeafClaim(r, 0, move);
    r->n.count = move;
    r->index_resize = lf->index_resize;
    lf->n.count = keep;

    r->next = lf->next;
    r->prev = lf;
    if (lf->next) lf->next->prev = r;
    else t->tail = (zbtNode *)r;
    lf->next = r;

    int copied = 0;
    zbtIndexMoveRange(t, r, 0, move, old_id, 1, &copied);

    zbtInsertChild(t, (zbtInner *)lf->n.parent, (zbtNode *)lf, (zbtNode *)r);
}

/* Move n elements across the shared boundary of adjacent leaves 'left' and
 * 'right'. toRight != 0 moves the tail of left onto the front of right;
 * otherwise the head of right onto the end of left. */
static void zbtLeafShift(zbtree *t, zbtLeaf *left, zbtLeaf *right, int n, int toRight) {
    serverAssert(n > 0);
    int copied = 0;
    if (toRight) {
        uint32_t old_id = left->id;
        memmove(&right->elems[n], &right->elems[0],
                right->n.count * sizeof(zbtElem *));
        memmove(&right->tags[n], &right->tags[0], right->n.count);
        if (n == 1) {
            right->elems[0] = left->elems[left->n.count - 1];
            right->tags[0] = left->tags[left->n.count - 1];
        } else {
            memcpy(&right->elems[0], &left->elems[(int)left->n.count - n],
                    n * sizeof(zbtElem *));
            memcpy(&right->tags[0], &left->tags[(int)left->n.count - n], n);
        }
        zbtLeafClaim(right, 0, n);
        left->n.count -= (uint32_t)n;
        right->n.count += (uint32_t)n;
        zbtIndexMoveRange(t, right, 0, n, old_id, 0, &copied);
    } else {
        uint32_t old_id = right->id;
        int at = (int)left->n.count;
        if (n == 1) {
            left->elems[at] = right->elems[0];
            left->tags[at] = right->tags[0];
        } else {
            memcpy(&left->elems[at], &right->elems[0],
                    n * sizeof(zbtElem *));
            memcpy(&left->tags[at], &right->tags[0], n);
        }
        zbtLeafClaim(left, at, at + n);
        memmove(&right->elems[0], &right->elems[n],
                ((int)right->n.count - n) * sizeof(zbtElem *));
        memmove(&right->tags[0], &right->tags[n],
                (int)right->n.count - n);
        left->n.count += (uint32_t)n;
        right->n.count -= (uint32_t)n;
        zbtIndexMoveRange(t, left, at, n, old_id, 0, &copied);
    }
}

/* Rewrite parent slots for a pair of adjacent leaf children after a shift. */
static void zbtFixLeafPair(zbtInner *p, int leftIdx) {
    zbtNode *L = p->child[leftIdx];
    zbtNode *R = p->child[leftIdx + 1];
    p->csize[leftIdx] = L->count;     p->sep[leftIdx] = zbtNodeMin(L);
    p->csize[leftIdx + 1] = R->count; p->sep[leftIdx + 1] = zbtNodeMin(R);
}

/* Try to dump overflowing elements of 'lf' into a same-parent sibling that
 * has slack, instead of splitting. 'ins_idx' is where the new element landed.
 * Returns 1 if the overflow was absorbed. The insert's +1 still has to reach
 * ancestors: parent csize[] is rewritten from the leaves, then walked up. */
static int zbtShareOverflow(zbtree *t, zbtLeaf *lf, int ins_idx) {
    zbtInner *p = (zbtInner *)lf->n.parent;
    if (p == NULL) return 0;

    int idx = zbtChildIdx(p, (zbtNode *)lf);
    int prefer_left = ins_idx >= (int)lf->n.count / 2;
    int nch = (int)p->n.count;

    for (int attempt = 0; attempt < 2; attempt++) {
        int try_left = prefer_left ? (attempt == 0) : (attempt == 1);
        if (try_left) {
            if (idx == 0) continue;
            zbtLeaf *L = (zbtLeaf *)p->child[idx - 1];
            int slack = ZBT_LEAF_MAX - (int)L->n.count;
            int maxn = (int)lf->n.count - ZBT_LEAF_MIN;
            int n = slack < maxn ? slack : maxn;
            if (n <= 0) continue;
            zbtLeafShift(t, L, lf, n, 0); /* smallest of lf onto L's end */
            zbtFixLeafPair(p, idx - 1);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return 1;
        } else {
            if (idx >= nch - 1) continue;
            zbtLeaf *R = (zbtLeaf *)p->child[idx + 1];
            int slack = ZBT_LEAF_MAX - (int)R->n.count;
            int maxn = (int)lf->n.count - ZBT_LEAF_MIN;
            int n = slack < maxn ? slack : maxn;
            if (n <= 0) continue;
            zbtLeafShift(t, lf, R, n, 1); /* largest of lf onto R's front */
            zbtFixLeafPair(p, idx);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return 1;
        }
    }
    return 0;
}

/* Insert an already-allocated element. The caller must guarantee the member
 * is not already present. Ownership of 'e' transfers to the tree. */
static void zbtInsertElem(zbtree *t, zbtElem *e, size_t usable) {
    double score = zbtGetScore(e);
    sds ele = zbtGetEle(e);
    zbtLcp lcp = ZBT_LCP_INIT;
    zbtLeaf *lf = zbtFindLeaf(t, score, ele, &lcp);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found, &lcp);
    serverAssert(!found);

    /* Landing at either end of an end leaf means the tree is growing in sorted
     * order; zbtSplitLeaf() then keeps the other side of the split full. Both
     * cannot hold at once: that would need idx to be 0 and lf->n.count at the
     * same time, and an empty leaf never overflows. */
    int bias = ZBT_SPLIT_EVEN;
    if (lf->next == NULL && idx == (int)lf->n.count) bias = ZBT_SPLIT_APPEND;
    else if (lf->prev == NULL && idx == 0) bias = ZBT_SPLIT_PREPEND;

    memmove(&lf->elems[idx + 1], &lf->elems[idx],
            ((int)lf->n.count - idx) * sizeof(zbtElem *));
    memmove(&lf->tags[idx + 1], &lf->tags[idx], (int)lf->n.count - idx);
    zbtLeafSetElem(lf, idx, e);
    lf->n.count++;
    t->length++;
    t->alloc_size += usable;

    if (lf->n.count > ZBT_LEAF_MAX) {
        /* Append/prepend must split with the existing bias: the neighbour is
         * already full, so a share attempt would fail and even-split. */
        if (bias == ZBT_SPLIT_APPEND || bias == ZBT_SPLIT_PREPEND)
            zbtSplitLeaf(t, lf, bias);
        else if (!zbtShareOverflow(t, lf, idx))
            zbtSplitLeaf(t, lf, ZBT_SPLIT_EVEN);
    } else {
        zbtUpdateToRoot(t, (zbtNode *)lf);
    }
}

zbtElem *zbtInsertWithHashAt(zbtree *t, double score, sds ele,
                             const uint64_t *known_hash,
                             const zbtreeInsertPosition *position)
{
    size_t usable;
    zbtElem *e = zbtCreateElemWithHash(score, ele, sdslen(ele), 0, &usable, known_hash);
    zbtIndexExpandIfNeeded(t, 1);
    t->pending_insert = e;
    zbtInsertElem(t, e, usable);
    zbtIndexInsert(t, zbtElemIndexHash(e), e->leaf->id, position);
    t->pending_insert = NULL;
    zbtIndexInvalidateScan(t);
    return e;
}

zbtElem *zbtInsertWithHash(zbtree *t, double score, sds ele, const uint64_t *known_hash) {
    return zbtInsertWithHashAt(t, score, ele, known_hash, NULL);
}

zbtElem *zbtInsert(zbtree *t, double score, sds ele) {
    return zbtInsertWithHash(t, score, ele, NULL);
}

/* Build a packed, balanced tree over 'elems[0..n)' in O(n). The elements must
 * already be strictly ascending by (score, member) and ownership of each one
 * transfers to the tree. 't' must be freshly created and empty. This is much
 * cheaper than n independent zbtInsert() calls (used by RDB load, COPY and
 * listpack->tree conversion, where the source order is already known). */
void zbtBuildFromSortedWithSize(zbtree *t, zbtElem **elems, unsigned long n,
                                size_t elems_alloc_size)
{
    if (n == 0) return;
    serverAssert(t->length == 0);

    /* Discard the placeholder empty root leaf created by zbtCreate(). */
    zbtFreeLeaf(t, (zbtLeaf *)t->root);
    t->root = t->head = t->tail = NULL;

    /* Build the leaf level. Distribute elements as evenly as possible so that
     * every non-root leaf holds at least ZBT_LEAF_MIN elements (required by the
     * structural invariants). With nleaves = ceil(n / ZBT_LEAF_MAX) the even
     * split guarantees this for more than one leaf. */
    unsigned long nleaves = (n + ZBT_LEAF_MAX - 1) / ZBT_LEAF_MAX;
    unsigned long base = n / nleaves;
    unsigned long rem = n % nleaves;

    zbtNode **level = zmalloc(sizeof(zbtNode *) * nleaves);
    unsigned long pos = 0;
    zbtLeaf *prev = NULL;
    for (unsigned long i = 0; i < nleaves; i++) {
        zbtLeaf *lf = zbtNewLeaf(t);
        unsigned long cnt = base + (i < rem ? 1 : 0);
        memcpy(lf->elems, &elems[pos], cnt * sizeof(zbtElem *));
        zbtLeafClaim(lf, 0, (int)cnt);
        lf->n.count = (uint32_t)cnt;
        pos += cnt;
        lf->prev = prev;
        if (prev) prev->next = lf;
        else t->head = (zbtNode *)lf;
        prev = lf;
        level[i] = (zbtNode *)lf;
    }
    prev->next = NULL;
    t->tail = (zbtNode *)prev;

    /* Build inner levels bottom-up until a single root remains. */
    unsigned long count = nleaves;
    while (count > 1) {
        unsigned long nparents = (count + ZBT_INNER_MAX - 1) / ZBT_INNER_MAX;
        unsigned long pbase = count / nparents;
        unsigned long prem = count % nparents;
        zbtNode **parents = zmalloc(sizeof(zbtNode *) * nparents);
        unsigned long ci = 0;
        for (unsigned long p = 0; p < nparents; p++) {
            zbtInner *in = zbtNewInner(t);
            unsigned long nch = pbase + (p < prem ? 1 : 0);
            in->n.count = (uint32_t)nch;
            for (unsigned long k = 0; k < nch; k++) {
                zbtNode *c = level[ci++];
                in->child[k] = c;
                in->csize[k] = zbtSubtreeSize(c);
                in->sep[k] = zbtNodeMin(c);
                c->parent = (zbtNode *)in;
            }
            parents[p] = (zbtNode *)in;
        }
        zfree(level);
        level = parents;
        count = nparents;
    }

    t->root = level[0];
    t->root->parent = NULL;
    zfree(level);

    t->length = n;
    t->alloc_size += elems_alloc_size;
    zbtIndexIndexTree(t);
}

void zbtBuildFromSorted(zbtree *t, zbtElem **elems, unsigned long n) {
    size_t elems_alloc_size = 0;
    for (unsigned long i = 0; i < n; i++)
        elems_alloc_size += zmalloc_usable_size(elems[i]);
    zbtBuildFromSortedWithSize(t, elems, n, elems_alloc_size);
}

/*-----------------------------------------------------------------------------
 * Deletion
 *----------------------------------------------------------------------------*/

static void zbtRebalanceInner(zbtree *t, zbtInner *in);

/* Remove child at position 'pos' from inner node 'p' (does not free it). */
static void zbtRemoveChild(zbtInner *p, int pos) {
    int tail = (int)p->n.count - pos - 1;
    memmove(&p->child[pos], &p->child[pos + 1], tail * sizeof(zbtNode *));
    memmove(&p->csize[pos], &p->csize[pos + 1], tail * sizeof(unsigned long));
    memmove(&p->sep[pos], &p->sep[pos + 1], tail * sizeof(zbtElem *));
    p->n.count--;
}

/* Called when inner node 'p' became the single-child root, or a subtree
 * shrank: fix up parent slots or collapse the root as needed. */
static void zbtFixupInnerAfterShrink(zbtree *t, zbtInner *p, int slot) {
    p->csize[slot] = zbtSubtreeSize(p->child[slot]);
    p->sep[slot] = zbtNodeMin(p->child[slot]);
    if (p->n.parent && p->n.count < ZBT_INNER_MIN) {
        zbtRebalanceInner(t, p);
    } else if (!p->n.parent && p->n.count == 1) {
        zbtNode *c = p->child[0];
        c->parent = NULL;
        t->root = c;
        zbtFreeNodeShallow(t, (zbtNode *)p);
    } else {
        zbtUpdateToRoot(t, (zbtNode *)p);
    }
}

/* Inner-node counterpart of zbtRebalanceLeaf(). A single borrowed child is
 * enough here, unlike for leaves: every caller reaches this through one
 * zbtRemoveChild() in zbtFixupInnerAfterShrink(), so 'in' is always short by
 * exactly one child. */
static void zbtRebalanceInner(zbtree *t, zbtInner *in) {
    zbtInner *p = (zbtInner *)in->n.parent;
    int idx = zbtChildIdx(p, (zbtNode *)in);
    serverAssert(in->n.count == ZBT_INNER_MIN - 1);

    /* Borrow from left sibling. */
    if (idx > 0) {
        zbtInner *L = (zbtInner *)p->child[idx - 1];
        if (L->n.count > ZBT_INNER_MIN) {
            memmove(&in->child[1], &in->child[0], in->n.count * sizeof(zbtNode *));
            memmove(&in->csize[1], &in->csize[0], in->n.count * sizeof(unsigned long));
            memmove(&in->sep[1], &in->sep[0], in->n.count * sizeof(zbtElem *));
            int last = (int)L->n.count - 1;
            in->child[0] = L->child[last];
            in->csize[0] = L->csize[last];
            in->sep[0] = L->sep[last];
            in->child[0]->parent = (zbtNode *)in;
            in->n.count++;
            L->n.count--;
            p->csize[idx - 1] = zbtSubtreeSize((zbtNode *)L); p->sep[idx - 1] = zbtNodeMin((zbtNode *)L);
            p->csize[idx] = zbtSubtreeSize((zbtNode *)in);   p->sep[idx] = zbtNodeMin((zbtNode *)in);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }
    /* Borrow from right sibling. */
    if (idx < (int)p->n.count - 1) {
        zbtInner *R = (zbtInner *)p->child[idx + 1];
        if (R->n.count > ZBT_INNER_MIN) {
            in->child[in->n.count] = R->child[0];
            in->csize[in->n.count] = R->csize[0];
            in->sep[in->n.count] = R->sep[0];
            in->child[in->n.count]->parent = (zbtNode *)in;
            in->n.count++;
            zbtRemoveChild(R, 0);
            p->csize[idx] = zbtSubtreeSize((zbtNode *)in);   p->sep[idx] = zbtNodeMin((zbtNode *)in);
            p->csize[idx + 1] = zbtSubtreeSize((zbtNode *)R); p->sep[idx + 1] = zbtNodeMin((zbtNode *)R);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }

    /* Merge with a sibling. */
    zbtInner *a, *b;
    int ai;
    if (idx > 0) { a = (zbtInner *)p->child[idx - 1]; b = in; ai = idx - 1; }
    else { a = in; b = (zbtInner *)p->child[idx + 1]; ai = idx; }
    serverAssert(a->n.count + b->n.count <= ZBT_INNER_MAX);
    for (uint32_t i = 0; i < b->n.count; i++) {
        a->child[a->n.count] = b->child[i];
        a->csize[a->n.count] = b->csize[i];
        a->sep[a->n.count] = b->sep[i];
        b->child[i]->parent = (zbtNode *)a;
        a->n.count++;
    }
    zbtRemoveChild(p, ai + 1);
    zbtFreeNodeShallow(t, (zbtNode *)b);
    zbtFixupInnerAfterShrink(t, p, ai);
}

/* Restore the minimum-occupancy invariant for a leaf that just dropped below
 * ZBT_LEAF_MIN.
 *
 * The amount moved tracks how short the leaf is. The range-delete path removes
 * a whole leaf slice per step, so 'lf' can arrive here short by many elements,
 * or empty; shifting a single element would leave it below the minimum. Queries
 * stay correct either way (they read count/csize, which stay consistent), but
 * the occupancy invariant zbtVerifyNode() checks would not hold, and nothing
 * would bound how empty leaves can get under repeated ZREMRANGEBY* traffic. */
static void zbtRebalanceLeaf(zbtree *t, zbtLeaf *lf) {
    zbtInner *p = (zbtInner *)lf->n.parent;
    int idx = zbtChildIdx(p, (zbtNode *)lf);

    /* Move exactly the deficit, no more: an element-at-a-time delete then costs
     * the same single-element shift it always did, while a slice delete pulls
     * across however many it takes. A sibling can cover the deficit precisely
     * when the pair holds 2 * ZBT_LEAF_MIN between them. */
    int deficit = ZBT_LEAF_MIN - (int)lf->n.count;
    serverAssert(deficit > 0);

    /* Take the tail of the left sibling onto the front of 'lf'. */
    if (idx > 0) {
        zbtLeaf *L = (zbtLeaf *)p->child[idx - 1];
        if ((int)L->n.count - deficit >= ZBT_LEAF_MIN) {
            zbtLeafShift(t, L, lf, deficit, 1);
            zbtFixLeafPair(p, idx - 1);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }
    /* Take the head of the right sibling onto the end of 'lf'. */
    if (idx < (int)p->n.count - 1) {
        zbtLeaf *R = (zbtLeaf *)p->child[idx + 1];
        if ((int)R->n.count - deficit >= ZBT_LEAF_MIN) {
            zbtLeafShift(t, lf, R, deficit, 0);
            zbtFixLeafPair(p, idx);
            zbtUpdateToRoot(t, (zbtNode *)p);
            return;
        }
    }

    /* Merge with a sibling. Neither pair could spare enough to lift 'lf' to
     * the minimum, so both pairs total below 2 * ZBT_LEAF_MIN and the survivor
     * fits in a single leaf. */
    zbtLeaf *a, *b;
    int ai;
    if (idx > 0) { a = (zbtLeaf *)p->child[idx - 1]; b = lf; ai = idx - 1; }
    else { a = lf; b = (zbtLeaf *)p->child[idx + 1]; ai = idx; }
    serverAssert(a->n.count + b->n.count <= ZBT_LEAF_MAX);
    uint32_t right_id = b->id;
    int at = (int)a->n.count;
    memcpy(&a->elems[a->n.count], b->elems, b->n.count * sizeof(zbtElem *));
    memcpy(&a->tags[a->n.count], b->tags, b->n.count);
    zbtLeafClaim(a, at, at + (int)b->n.count);
    a->n.count += b->n.count;
    int copied = 0;
    zbtIndexMoveRange(t, a, at, (int)b->n.count, right_id, 0, &copied);
    a->next = b->next;
    if (b->next) b->next->prev = a;
    else t->tail = (zbtNode *)a;
    zbtRemoveChild(p, ai + 1);
    zbtFreeLeaf(t, b);
    zbtFixupInnerAfterShrink(t, p, ai);
}

/* Remove element 'e' from the tree and free it. */
void zbtDeleteElem(zbtree *t, zbtElem *e) {
    zbtLeaf *lf = e->leaf;
    int idx = zbtLeafFindPtr(lf, e);
    serverAssert(idx >= 0);

    zbtIndexDeleteElem(t, e);
    memmove(&lf->elems[idx], &lf->elems[idx + 1],
            ((int)lf->n.count - idx - 1) * sizeof(zbtElem *));
    memmove(&lf->tags[idx], &lf->tags[idx + 1],
            (int)lf->n.count - idx - 1);
    lf->n.count--;
    t->length--;
    size_t usable;
    zfree_usable(e, &usable);
    t->alloc_size -= usable;

    if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, lf);
    else
        zbtUpdateToRoot(t, (zbtNode *)lf);
    zbtIndexShrinkIfNeeded(t);
    zbtIndexInvalidateScan(t);
}

/* Whether the re-scored member at slot 'idx' of 'lf' still sorts inside it.
 * The member does not change and its key only moves one way, so only the end
 * of the leaf that 'up' selects can be crossed and one comparison settles it.
 *
 * That comparison is made against the leaf's own extreme rather than the
 * neighbour leaf's, which answers "stays" for everything short of the very
 * last slot on that side and reaches the element through a pointer the caller
 * already scanned past. Consulting the neighbour instead is exact, but it has
 * to walk into a second leaf for its count and its end slot - three dependent
 * misses where this has one, on a check that a score update far outside the
 * leaf pays for nothing. The neighbour is only asked when the moving element
 * is itself the extreme and the leaf has no answer. Being one slot short of
 * exact costs an occasional needless slow path, never a wrong placement. */
static int zbtLeafHolds(zbtLeaf *lf, int idx, double score, sds ele, int up) {
    int last = (int)lf->n.count - 1;
    if (up) {
        if (idx < last) return zbtCompare(score, ele, lf->elems[last]) < 0;
        zbtLeaf *n = lf->next;
        return n == NULL || zbtCompare(score, ele, n->elems[0]) < 0;
    }
    if (idx > 0) return zbtCompare(score, ele, lf->elems[0]) > 0;
    zbtLeaf *p = lf->prev;
    return p == NULL || zbtCompare(score, ele, p->elems[p->n.count - 1]) > 0;
}

/* Move the element occupying slot 'idx' of 'lf' to the slot its new key
 * (score,ele) calls for, storing 'e' there ('e' is the element itself, or its
 * relocated copy). The caller has established that the key stays in this leaf.
 *
 * The rest of the leaf is untouched and still sorted, and 'up' says which way
 * the one moved key went, so the destination is bisected on that side alone:
 * one comparison when the element keeps its slot, at most
 * log2(ZBT_LEAF_MAX) + 1 when it does not. Returns the new slot. */
static int zbtLeafMoveSlot(zbtLeaf *lf, int idx, zbtElem *e, double score,
                           sds ele, int up)
{
    int last = (int)lf->n.count - 1;

    if (up && idx < last && zbtCompare(score, ele, lf->elems[idx + 1]) > 0) {
        /* Largest slot in (idx, last] the key still sorts after. */
        int lo = idx + 1, hi = last;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (zbtCompare(score, ele, lf->elems[mid]) > 0) lo = mid;
            else hi = mid - 1;
        }
        memmove(&lf->elems[idx], &lf->elems[idx + 1],
                (lo - idx) * sizeof(zbtElem *));
        memmove(&lf->tags[idx], &lf->tags[idx + 1], lo - idx);
        zbtLeafSetElem(lf, lo, e);
        return lo;
    }
    if (!up && idx > 0 && zbtCompare(score, ele, lf->elems[idx - 1]) < 0) {
        /* Smallest slot in [0, idx) the key sorts before. */
        int lo = 0, hi = idx - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (zbtCompare(score, ele, lf->elems[mid]) < 0) hi = mid;
            else lo = mid + 1;
        }
        memmove(&lf->elems[lo + 1], &lf->elems[lo],
                (idx - lo) * sizeof(zbtElem *));
        memmove(&lf->tags[lo + 1], &lf->tags[lo], idx - lo);
        zbtLeafSetElem(lf, lo, e);
        return lo;
    }
    lf->elems[idx] = e;
    lf->tags[idx] = zbtElemLeafTag(e);
    e->leaf = lf;
    return idx;
}

/* Move an existing element to reflect a new score. Returns the (possibly
 * reallocated) element. */
zbtElem *zbtUpdateScore(zbtree *t, zbtElem *e, double newscore) {
    sds ele = zbtGetEle(e);
    zbtLeaf *lf = e->leaf;
    int idx = zbtLeafFindPtr(lf, e);
    serverAssert(idx >= 0);

    uint8_t newenc;
    unsigned char newbuf[8];
    zbtScoreEncode(newscore, &newenc, newbuf);
    size_t old_usable = zmalloc_usable_size(e);
    int same_width = zbtScoreEncSize(newenc) == zbtScoreEncSize(e->enc);
    int up = newscore > zbtGetScore(e);

    /* Fast path: the element does not leave the leaf it already names. A
     * score bump that stays inside one leaf's key window - what ZINCRBY on a
     * leaderboard does over and over - then costs a reshuffle of a packed
     * pointer array, with no root-to-leaf descent to find the new home, no
     * occupancy change, and so no rebalance or split. */
    if (zbtLeafHolds(lf, idx, newscore, ele, up)) {
        zbtElem *ne = e;
        if (same_width) {
            e->enc = (e->enc & ZBT_ELEM_CACHED_HASH) | newenc;
            memcpy(e->data, newbuf, zbtScoreEncSize(newenc));
        } else {
            size_t new_usable;
            ne = zbtCreateElem(newscore, ele, sdslen(ele), 0, &new_usable);
            ne->leaf = lf;
            zfree_with_size(e, old_usable);
            t->alloc_size = t->alloc_size - old_usable + new_usable;
            ele = zbtGetEle(ne);
        }
        int at = zbtLeafMoveSlot(lf, idx, ne, newscore, ele, up);
        /* Only the leaf minimum is visible to the ancestors, and only sep[]
         * can have changed: the element count did not. */
        if (idx == 0 || at == 0) zbtUpdateToRoot(t, (zbtNode *)lf);
        return ne;
    }

    /* Slow path: detach, rewrite the score, reinsert. Index slot is removed
     * first so a later rebalance cannot alias the leaf ID. */
    zbtIndexDeleteElem(t, e);
    memmove(&lf->elems[idx], &lf->elems[idx + 1],
            ((int)lf->n.count - idx - 1) * sizeof(zbtElem *));
    memmove(&lf->tags[idx], &lf->tags[idx + 1],
            (int)lf->n.count - idx - 1);
    lf->n.count--;
    t->length--;
    t->alloc_size -= old_usable;
    if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN)
        zbtRebalanceLeaf(t, lf);
    else
        zbtUpdateToRoot(t, (zbtNode *)lf);

    zbtElem *ne = e;
    if (same_width) {
        e->enc = (e->enc & ZBT_ELEM_CACHED_HASH) | newenc;
        memcpy(e->data, newbuf, zbtScoreEncSize(newenc));
    } else {
        size_t new_usable;
        ne = zbtCreateElem(newscore, ele, sdslen(ele), 0, &new_usable);
        zfree_with_size(e, old_usable);
        old_usable = new_usable;
    }
    zbtIndexExpandIfNeeded(t, 1);
    t->pending_insert = ne;
    zbtInsertElem(t, ne, old_usable);
    zbtIndexInsert(t, zbtElemIndexHash(ne), ne->leaf->id, NULL);
    t->pending_insert = NULL;
    zbtIndexInvalidateScan(t);
    return ne;
}

/*-----------------------------------------------------------------------------
 * Rank and rank-based access
 *----------------------------------------------------------------------------*/

/* 1-based rank of element 'e'.
 *
 * The element names its own leaf, so the rank is its offset in that leaf plus
 * every left sibling's subtree size on the way up to the root. Descending
 * instead (as zbtGetRank() must, having only a key) means bisecting sep[] at
 * every level, and each probe there dereferences a separator element sitting
 * in its own allocation -- a chain of dependent cache misses that dominates
 * ZRANK. Climbing touches only the nodes themselves. */
unsigned long zbtRankByElem(zbtree *t, zbtElem *e) {
    UNUSED(t);
    zbtLeaf *lf = e->leaf;
    int idx = zbtLeafFindPtr(lf, e);
    serverAssert(idx >= 0);

    unsigned long rank = (unsigned long)idx;
    for (zbtNode *n = (zbtNode *)lf; n->parent; n = n->parent) {
        zbtInner *in = (zbtInner *)n->parent;
        /* Summing csize[] up to our own slot needs the slot index anyway, so
         * the identity scan that finds it is not an extra pass. */
        uint32_t i = 0;
        while (in->child[i] != n) {
            rank += in->csize[i];
            if (++i == in->n.count) serverPanic("zbtree: child not found in parent");
        }
    }
    return rank + 1;
}

/* 1-based rank of (score,ele), or 0 when the element does not exist. */
unsigned long zbtGetRank(zbtree *t, double score, sds ele) {
    zbtLcp lcp = ZBT_LCP_INIT;
    unsigned long rank = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int ci = zbtInnerChildIdx(in, score, ele, &lcp);
        for (int i = 0; i < ci; i++) rank += in->csize[i];
        n = in->child[ci];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found, &lcp);
    if (!found) return 0;
    return rank + (unsigned long)idx + 1;
}

/* Return the element at the given 1-based rank, or NULL if out of range.
 * When 'it' is not NULL it is positioned at the returned element. */
zbtElem *zbtElemByRank(zbtree *t, unsigned long rank, zbtIter *it) {
    if (rank < 1 || rank > t->length) return NULL;
    unsigned long r = rank - 1; /* 0-based */
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        uint32_t i = 0;
        while (i < in->n.count && r >= in->csize[i]) { r -= in->csize[i]; i++; }
        n = in->child[i];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = (int)r; }
    return lf->elems[r];
}

/*-----------------------------------------------------------------------------
 * Iteration
 *----------------------------------------------------------------------------*/

/* 0-based offset from the low end (reverse=0) or high end (reverse=1).
 * Small offsets walk the leaf chain from head/tail; larger ones rank-jump. */
zbtElem *zbtElemByEndOffset(zbtree *t, unsigned long offset, int reverse,
                           zbtIter *it)
{
    if (offset > ZBT_RANGE_WALK_MAX) {
        unsigned long rank = reverse ? (t->length - offset) : (offset + 1);
        return zbtElemByRank(t, rank, it);
    }
    zbtElem *e = reverse ? zbtLast(t, it) : zbtFirst(t, it);
    for (unsigned long i = 0; i < offset && e != NULL; i++)
        e = reverse ? zbtIterPrev(it) : zbtIterNext(it);
    return e;
}

zbtElem *zbtFirst(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf = (zbtLeaf *)t->head;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = 0; }
    return lf->elems[0];
}

zbtElem *zbtSeekGE(zbtree *t, double score, sds ele, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLcp lcp = ZBT_LCP_INIT;
    zbtLeaf *lf = zbtFindLeaf(t, score, ele, &lcp);
    int found;
    int idx = zbtLeafSearch(lf, score, ele, &found, &lcp);
    UNUSED(found);
    if (idx == (int)lf->n.count) {
        lf = lf->next;
        idx = 0;
    }
    if (lf == NULL) return NULL;
    if (it) {
        it->leaf = (zbtNode *)lf;
        it->idx = idx;
    }
    return lf->elems[idx];
}

zbtElem *zbtLast(zbtree *t, zbtIter *it) {
    if (t->length == 0) return NULL;
    zbtLeaf *lf = (zbtLeaf *)t->tail;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = (int)lf->n.count - 1; }
    return lf->elems[lf->n.count - 1];
}

zbtElem *zbtIterNext(zbtIter *it) {
    if (!it->leaf) return NULL;
    zbtLeaf *lf = (zbtLeaf *)it->leaf;
    it->idx++;
    if (it->idx >= (int)lf->n.count) {
        it->leaf = (zbtNode *)lf->next;
        it->idx = 0;
        if (!it->leaf) return NULL;
        lf = (zbtLeaf *)it->leaf;
        if (lf->n.count == 0) return NULL;
    }
    return lf->elems[it->idx];
}

zbtElem *zbtIterPrev(zbtIter *it) {
    if (!it->leaf) return NULL;
    zbtLeaf *lf = (zbtLeaf *)it->leaf;
    it->idx--;
    if (it->idx < 0) {
        it->leaf = (zbtNode *)lf->prev;
        if (!it->leaf) return NULL;
        lf = (zbtLeaf *)it->leaf;
        it->idx = (int)lf->n.count - 1;
        if (it->idx < 0) return NULL;
    }
    return lf->elems[it->idx];
}

/*-----------------------------------------------------------------------------
 * Range queries
 *----------------------------------------------------------------------------*/

/* A predicate that is monotonic in tree order: true for a prefix of the
 * sorted order, then false. */
typedef int (*zbtBeforeFn)(const zbtElem *e, void *arg);

/* Where such a predicate flips: 'count' elements satisfy it, and 'leaf'/'idx'
 * hold the position of the first element that does not ('idx' reaches
 * leaf->n.count when every element of the leaf satisfies it).
 *
 * The descent only skips a child once the minimum of the following one
 * satisfies the predicate, and that minimum is inherited all the way down, so
 * a non-empty prefix always ends inside the leaf the descent lands on: count
 * > 0 implies idx > 0, and the last satisfying element sits at idx - 1 of
 * that same leaf. Both ends of a range are therefore one descent away. */
typedef struct zbtBoundary {
    unsigned long count;
    zbtLeaf *leaf;
    int idx;
} zbtBoundary;

static void zbtFindBoundary(zbtree *t, zbtBeforeFn before, void *arg,
                            zbtBoundary *b) {
    unsigned long cnt = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int i = 0;
        while (i < (int)in->n.count - 1 && before(in->sep[i + 1], arg)) {
            cnt += in->csize[i];
            i++;
        }
        n = in->child[i];
    }
    zbtLeaf *lf = (zbtLeaf *)n;
    uint32_t i = 0;
    while (i < lf->n.count && before(lf->elems[i], arg)) i++;
    b->count = cnt + i;
    b->leaf = lf;
    b->idx = (int)i;
}

/* Count the elements at the start of the sorted order for which before()
 * returns true. */
static unsigned long zbtCountBefore(zbtree *t, zbtBeforeFn before, void *arg) {
    zbtBoundary b;
    zbtFindBoundary(t, before, arg, &b);
    return b.count;
}

/* Last element of the prefix (rank b->count), or NULL when it is empty. */
static zbtElem *zbtBoundaryLast(zbtBoundary *b, zbtIter *it) {
    if (b->count == 0) return NULL;
    debugServerAssert(b->idx > 0);
    if (it) { it->leaf = (zbtNode *)b->leaf; it->idx = b->idx - 1; }
    return b->leaf->elems[b->idx - 1];
}

/* First element past the prefix (rank b->count + 1), or NULL when the prefix
 * covers the whole tree. */
static zbtElem *zbtBoundaryNext(zbtBoundary *b, zbtIter *it) {
    zbtLeaf *lf = b->leaf;
    int idx = b->idx;
    while (lf && idx >= (int)lf->n.count) { lf = lf->next; idx = 0; }
    if (!lf) return NULL;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = idx; }
    return lf->elems[idx];
}

/* Predicates for score ranges. */
static int beforeScoreLt(const zbtElem *e, void *arg) {
    return zbtGetScore(e) < *(double *)arg;
}
static int beforeScoreLe(const zbtElem *e, void *arg) {
    return zbtGetScore(e) <= *(double *)arg;
}

/* Locate the first element satisfying the lower score bound. Searching from
 * the left keeps the common low-minimum case short. The skipped subtree sizes
 * give the element's rank without a second descent. */
static zbtElem *zbtFirstInRange(zbtree *t, zrangespec *range,
                                unsigned long *out_rank, zbtIter *it) {
    unsigned long before = 0;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int i = 0;
        while (i < (int)in->n.count - 1 &&
               !zslValueGteMin(zbtGetScore(in->sep[i + 1]), range))
        {
            before += in->csize[i];
            i++;
        }
        n = in->child[i];
    }

    zbtLeaf *lf = (zbtLeaf *)n;
    int idx = 0;
    while (idx < (int)lf->n.count &&
           !zslValueGteMin(zbtGetScore(lf->elems[idx]), range))
    {
        before++;
        idx++;
    }
    if (idx == (int)lf->n.count) {
        lf = lf->next;
        idx = 0;
    }
    if (lf == NULL) return NULL;

    zbtElem *e = lf->elems[idx];
    if (!zslValueLteMax(zbtGetScore(e), range)) return NULL;
    if (out_rank) *out_rank = before + 1;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = idx; }
    return e;
}

/* Locate the last element satisfying the upper score bound. Searching from
 * the right is the reverse-range counterpart of zbtFirstInRange(): whole
 * subtrees beyond max are subtracted from the absolute rank, and the result
 * is already positioned for backward leaf iteration. */
static zbtElem *zbtLastInRange(zbtree *t, zrangespec *range,
                               unsigned long *out_rank, zbtIter *it) {
    unsigned long rank = t->length;
    zbtNode *n = t->root;
    while (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        int i = (int)in->n.count - 1;
        while (i > 0 &&
               !zslValueLteMax(zbtGetScore(in->sep[i]), range))
        {
            rank -= in->csize[i];
            i--;
        }
        n = in->child[i];
    }

    zbtLeaf *lf = (zbtLeaf *)n;
    int idx = (int)lf->n.count - 1;
    while (idx >= 0 &&
           !zslValueLteMax(zbtGetScore(lf->elems[idx]), range))
    {
        rank--;
        idx--;
    }
    if (idx < 0) return NULL;

    zbtElem *e = lf->elems[idx];
    if (!zslValueGteMin(zbtGetScore(e), range)) return NULL;
    if (out_rank) *out_rank = rank;
    if (it) { it->leaf = (zbtNode *)lf; it->idx = idx; }
    return e;
}

/* Predicates for lex ranges. */
static int beforeNotGteMin(const zbtElem *e, void *arg) {
    return !zslLexValueGteMin(zbtGetEle((zbtElem *)e), (zlexrangespec *)arg);
}
static int beforeLteMax(const zbtElem *e, void *arg) {
    return zslLexValueLteMax(zbtGetEle((zbtElem *)e), (zlexrangespec *)arg);
}

/* Shared implementation of the Nth-in-range lookups. 'before_lo' selects the
 * elements that precede the range, 'before_hi' those up to and including its
 * end. Mirrors the skiplist zslNthIn*Range semantics: n >= 0 counts forward
 * from the first in-range element, n < 0 counts back from the last.
 *
 * Only the end the walk starts from is located, by a single boundary descent.
 * Everything past that end already clears its side of the range, so one
 * predicate check on the element landed on decides whether the offset has
 * carried the result out through the opposite side. */
static zbtElem *zbtNthGeneric(zbtree *t, long n, unsigned long *out_rank,
                              zbtIter *it,
                              zbtBeforeFn before_lo, void *lo_arg,
                              zbtBeforeFn before_hi, void *hi_arg) {
    zbtBoundary b;
    zbtIter pos;
    zbtElem *e;
    unsigned long rank;

    if (n >= 0) {
        zbtFindBoundary(t, before_lo, lo_arg, &b);
        e = zbtBoundaryNext(&b, &pos);
        if (e == NULL) return NULL;

        unsigned long steps = (unsigned long)n;
        if (steps >= t->length - b.count) return NULL;
        rank = b.count + 1 + steps;
        if (n > 0) {
            if (n <= ZBT_RANGE_WALK_MAX) {
                for (long i = 0; i < n; i++) e = zbtIterNext(&pos);
            } else {
                e = zbtElemByRank(t, rank, &pos);
            }
            if (e == NULL) return NULL;
        }
        if (!before_hi(e, hi_arg)) return NULL;
    } else {
        zbtFindBoundary(t, before_hi, hi_arg, &b);
        e = zbtBoundaryLast(&b, &pos);
        if (e == NULL) return NULL;

        /* Add before negating so LONG_MIN remains representable. */
        unsigned long steps = (unsigned long)(-(n + 1));
        if (steps >= b.count) return NULL;
        rank = b.count - steps;
        if (steps > 0) {
            if (steps <= ZBT_RANGE_WALK_MAX) {
                for (unsigned long i = 0; i < steps; i++) e = zbtIterPrev(&pos);
            } else {
                e = zbtElemByRank(t, rank, &pos);
            }
            if (e == NULL) return NULL;
        }
        if (before_lo(e, lo_arg)) return NULL;
    }

    if (out_rank) *out_rank = rank;
    if (it) *it = pos;
    return e;
}

zbtElem *zbtNthInRange(zbtree *t, zrangespec *range, long n,
                       unsigned long *out_rank, zbtIter *it) {
    if (t->length == 0) return NULL;

    zbtIter pos;
    zbtElem *e;
    unsigned long rank;

    if (n >= 0) {
        e = zbtFirstInRange(t, range, &rank, &pos);
        if (e == NULL) return NULL;

        unsigned long steps = (unsigned long)n;
        if (steps > t->length - rank) return NULL;
        rank += steps;
        if (steps > 0) {
            if (steps <= ZBT_RANGE_WALK_MAX) {
                for (unsigned long i = 0; i < steps; i++)
                    e = zbtIterNext(&pos);
            } else {
                e = zbtElemByRank(t, rank, &pos);
            }
            if (e == NULL || !zslValueLteMax(zbtGetScore(e), range))
                return NULL;
        }
    } else {
        /* Add before negating so LONG_MIN remains representable. */
        unsigned long steps = (unsigned long)(-(n + 1));
        if (steps <= ZBT_RANGE_WALK_MAX) {
            e = zbtLastInRange(t, range, &rank, &pos);
            if (e == NULL || steps >= rank) return NULL;
            for (unsigned long i = 0; i < steps; i++)
                e = zbtIterPrev(&pos);
            rank -= steps;
            if (steps > 0 && !zslValueGteMin(zbtGetScore(e), range))
                return NULL;
        } else {
            /* For a rank jump, only the upper endpoint's rank is needed.
             * Counting from the left matches zbtElemByRank()'s traversal and
             * avoids positioning an iterator that would be discarded. */
            double maxv = range->max;
            rank = range->maxex ?
                zbtCountBefore(t, beforeScoreLt, &maxv) :
                zbtCountBefore(t, beforeScoreLe, &maxv);
            if (steps >= rank) return NULL;
            rank -= steps;
            e = zbtElemByRank(t, rank, &pos);
            if (e == NULL || !zslValueGteMin(zbtGetScore(e), range))
                return NULL;
        }
    }

    if (out_rank) *out_rank = rank;
    if (it) *it = pos;
    return e;
}

zbtElem *zbtNthInLexRange(zbtree *t, zlexrangespec *range, long n,
                          unsigned long *out_rank, zbtIter *it) {
    if (t->length == 0) return NULL;
    return zbtNthGeneric(t, n, out_rank, it,
                         beforeNotGteMin, range, beforeLteMax, range);
}

/*-----------------------------------------------------------------------------
 * Range deletion
 *----------------------------------------------------------------------------*/

/* Delete 'want' consecutive elements starting at slot 'idx' of leaf 'lf',
 * tombstoning each member in the open-addressed index.
 *
 * Instead of locating and rebalancing once per element (O(K log N)), this
 * removes a whole leaf slice per structural pass: one rebalance per touched
 * leaf. The slice that follows is not looked up by rank either - the first
 * survivor past the current slice is picked up before the removal and names
 * its own leaf afterwards. Rebalancing relocates elements between leaves but
 * never frees them, and re-stamps the back-pointer of every one it moves, so
 * that hand-off stays valid across the merge that may consume 'lf' itself.
 * The whole deletion is therefore O(K + K/leaf) with a single descent, done
 * by the caller, for the starting position. */
static unsigned long zbtDeleteSlices(zbtree *t, zbtLeaf *lf, int idx,
                                     unsigned long want) {
    unsigned long removed = 0;

    while (want > 0) {
        /* Delete the contiguous in-range slice contained in this leaf. */
        int avail = (int)lf->n.count - idx;
        int take = ((unsigned long)avail > want) ? (int)want : avail;

        /* The element that takes over the position we are deleting from. */
        zbtElem *next = NULL;
        if ((unsigned long)take < want) {
            if (idx + take < (int)lf->n.count) next = lf->elems[idx + take];
            else if (lf->next) next = lf->next->elems[0];
        }

        for (int k = 0; k < take; k++) {
            zbtElem *el = lf->elems[idx + k];
            zbtIndexDeleteElem(t, el);
            size_t usable;
            zfree_usable(el, &usable);
            t->alloc_size -= usable;
        }
        memmove(&lf->elems[idx], &lf->elems[idx + take],
                ((int)lf->n.count - idx - take) * sizeof(zbtElem *));
        memmove(&lf->tags[idx], &lf->tags[idx + take],
                (int)lf->n.count - idx - take);
        lf->n.count -= take;
        t->length -= take;
        removed += take;
        want -= take;

        if (lf->n.parent && lf->n.count < ZBT_LEAF_MIN)
            zbtRebalanceLeaf(t, lf);
        else
            zbtUpdateToRoot(t, (zbtNode *)lf);

        if (next == NULL) break;
        lf = next->leaf;
        idx = zbtLeafFindPtr(lf, next);
        serverAssert(idx >= 0);
    }
    zbtIndexMaintenance(t, 16);
    zbtIndexInvalidateScan(t);
    return removed;
}

/* Delete every element whose 1-based rank falls in [first, last] (inclusive). */
static unsigned long zbtDeleteRankRange(zbtree *t, unsigned long first,
                                        unsigned long last) {
    if (last > t->length) last = t->length;
    if (first > last) return 0;

    zbtIter it;
    if (zbtElemByRank(t, first, &it) == NULL) return 0;
    return zbtDeleteSlices(t, (zbtLeaf *)it.leaf, it.idx, last - first + 1);
}

/* Delete the elements between two boundaries of the sorted order. The
 * boundary descent already landed on the first element to remove, so it is
 * handed over directly instead of being looked up again by rank. */
static unsigned long zbtDeleteBoundaryRange(zbtree *t,
                                            zbtBeforeFn before_lo, void *lo_arg,
                                            zbtBeforeFn before_hi, void *hi_arg) {
    if (t->length == 0) return 0;
    zbtBoundary lo, hi;
    zbtFindBoundary(t, before_lo, lo_arg, &lo);
    zbtFindBoundary(t, before_hi, hi_arg, &hi);
    if (lo.count >= hi.count) return 0;

    zbtIter it;
    if (zbtBoundaryNext(&lo, &it) == NULL) return 0;
    return zbtDeleteSlices(t, (zbtLeaf *)it.leaf, it.idx, hi.count - lo.count);
}

unsigned long zbtDeleteRangeByScore(zbtree *t, zrangespec *range) {
    double minv = range->min, maxv = range->max;
    return zbtDeleteBoundaryRange(t,
        range->minex ? beforeScoreLe : beforeScoreLt, &minv,
        range->maxex ? beforeScoreLt : beforeScoreLe, &maxv);
}

unsigned long zbtDeleteRangeByLex(zbtree *t, zlexrangespec *range) {
    return zbtDeleteBoundaryRange(t, beforeNotGteMin, range,
                                  beforeLteMax, range);
}

/* Delete elements whose 1-based rank is in [start, end] (inclusive). */
unsigned long zbtDeleteRangeByRank(zbtree *t, unsigned long start,
                                    unsigned long end) {
    if (t->length == 0 || start > end) return 0;
    return zbtDeleteRankRange(t, start, end);
}

/*-----------------------------------------------------------------------------
 * Active defragmentation support
 *----------------------------------------------------------------------------*/

/* Replace element 'olde' with the (content-identical) relocated 'newe' in its
 * leaf slot and fix any separator pointers that referenced it. */
void zbtReplaceElem(zbtree *t, zbtElem *olde, zbtElem *newe) {
    zbtLeaf *lf = olde->leaf;
    int idx = zbtLeafFindPtr(lf, olde);
    serverAssert(idx >= 0);
    lf->elems[idx] = newe;
    lf->tags[idx] = zbtElemLeafTag(newe);
    newe->leaf = lf;
    /* Ancestors only ever reference a leaf through its minimum, so nothing
     * above can be holding 'olde' unless it sat in slot 0. */
    if (idx == 0) zbtUpdateToRoot(t, (zbtNode *)lf);
}

static zbtNode *zbtDefragNode(zbtNode *n, void *(*fn)(void *)) {
    zbtNode *nn = fn(n);
    if (nn) {
        n = nn;
        if (n->isleaf) zbtLeafClaim((zbtLeaf *)n, 0, (int)n->count);
    }
    if (!n->isleaf) {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < in->n.count; i++) {
            zbtNode *c = zbtDefragNode(in->child[i], fn);
            in->child[i] = c;
            c->parent = n;
        }
    }
    return n;
}

static void zbtCollectLeaves(zbtree *t, zbtNode *n, zbtLeaf **prev) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        lf->prev = *prev;
        if (*prev) (*prev)->next = lf;
        else t->head = (zbtNode *)lf;
        *prev = lf;
    } else {
        zbtInner *in = (zbtInner *)n;
        for (uint32_t i = 0; i < in->n.count; i++)
            zbtCollectLeaves(t, in->child[i], prev);
    }
}

/* Relocate every tree node using the provided defrag allocator, then rebuild
 * the leaf sibling chain and head/tail pointers. Element objects are handled
 * separately by the caller (via the ZSET dict scan + zbtReplaceElem). */
void zbtDefragNodes(zbtree *t, void *(*fn)(void *)) {
    t->root = zbtDefragNode(t->root, fn);
    t->root->parent = NULL;
    zbtLeaf *prev = NULL;
    zbtCollectLeaves(t, t->root, &prev);
    if (prev) prev->next = NULL;
    t->tail = (zbtNode *)prev;
    for (zbtLeaf *lf = (zbtLeaf *)t->head; lf; lf = lf->next)
        t->table_to_leaf[lf->id] = lf;
}

/* Relocate a single leaf (if the allocator decides to move it) and repair all
 * external references: the parent child slot (or the root), the sibling links
 * and the head/tail pointers. Returns the current (possibly new) leaf. */
static zbtLeaf *zbtDefragRelocLeaf(zbtree *t, zbtLeaf *lf, void *(*fn)(void *)) {
    zbtLeaf *nl = fn(lf);
    if (!nl) return lf;
    zbtLeafClaim(nl, 0, (int)nl->n.count);
    t->table_to_leaf[nl->id] = nl;
    if (nl->n.parent) {
        zbtInner *p = (zbtInner *)nl->n.parent;
        p->child[zbtChildIdx(p, (zbtNode *)lf)] = (zbtNode *)nl;
    } else {
        t->root = (zbtNode *)nl;
    }
    if (nl->prev) nl->prev->next = nl; else t->head = (zbtNode *)nl;
    if (nl->next) nl->next->prev = nl; else t->tail = (zbtNode *)nl;
    return nl;
}

/* Relocate a single inner node (if moved) and repair the grandparent child
 * slot (or the root) and every child's parent back-pointer. Returns the
 * current (possibly new) inner node. */
static zbtInner *zbtDefragRelocInner(zbtree *t, zbtInner *in, void *(*fn)(void *)) {
    zbtInner *ni = fn(in);
    if (!ni) return in;
    if (ni->n.parent) {
        zbtInner *p = (zbtInner *)ni->n.parent;
        p->child[zbtChildIdx(p, (zbtNode *)in)] = (zbtNode *)ni;
    } else {
        t->root = (zbtNode *)ni;
    }
    for (uint32_t i = 0; i < ni->n.count; i++)
        ni->child[i]->parent = (zbtNode *)ni;
    return ni;
}

/* Incremental variant of zbtDefragNodes(): relocate up to 'budget' tree nodes,
 * walking the leaves left to right and relocating each inner node right after
 * its last child is processed (post-order). The resume position is stored in
 * the tree as the (score, member) key of the first element of the next leaf to
 * process, so it survives element inserts/deletes/rebalances happening between
 * calls. Returns 1 if more work remains (bookmark saved) or 0 when the whole
 * tree has been relocated. Unlike zbtDefragNodes(), the leaf sibling chain and
 * head/tail are kept consistent after every single relocation, so the tree is
 * safe to query/modify between steps. */
int zbtDefragNodesIncremental(zbtree *t, void *(*fn)(void *), unsigned int budget) {
    zbtLeaf *lf;
    if (t->defrag_resume) {
        zbtLcp lcp = ZBT_LCP_INIT;
        lf = zbtFindLeaf(t, t->defrag_resume_score, t->defrag_resume, &lcp);
    } else {
        lf = (zbtLeaf *)t->head;
    }

    unsigned int work = 0;
    while (lf) {
        zbtLeaf *next = lf->next; /* value stays valid across relocation */
        lf = zbtDefragRelocLeaf(t, lf, fn);
        work++;

        /* Post-order: relocate ancestors whose last child we just completed. */
        zbtNode *c = (zbtNode *)lf;
        while (c->parent) {
            zbtInner *p = (zbtInner *)c->parent;
            if (p->child[p->n.count - 1] != c) break;
            c = (zbtNode *)zbtDefragRelocInner(t, p, fn);
            work++;
        }

        if (!next) {
            /* Processed the last leaf; the cascade above relocated the root. */
            if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
            return 0;
        }
        if (work >= budget) {
            zbtElem *e0 = next->elems[0];
            sds m = sdsdup(zbtGetEle(e0));
            if (t->defrag_resume) sdsfree(t->defrag_resume);
            t->defrag_resume = m;
            t->defrag_resume_score = zbtGetScore(e0);
            return 1;
        }
        lf = next;
    }

    /* Empty tree (single empty root leaf) or nothing left. */
    if (t->defrag_resume) { sdsfree(t->defrag_resume); t->defrag_resume = NULL; }
    return 0;
}

void zbtDefragIndex(zbtree *t, void *(*fn)(void *)) {
    if (t->table_to_leaf) {
        void *n = fn(t->table_to_leaf);
        if (n) t->table_to_leaf = n;
    }
    if (t->member_index.buckets) {
        void *n = fn(t->member_index.buckets);
        if (n) t->member_index.buckets = n;
    }
    if (t->member_rehash) {
        void *n = fn(t->member_rehash);
        if (n) t->member_rehash = n;
        if (t->member_rehash->table.buckets) {
            void *b = fn(t->member_rehash->table.buckets);
            if (b) t->member_rehash->table.buckets = b;
        }
    }
}

/*-----------------------------------------------------------------------------
 * Debugging / test verification
 *----------------------------------------------------------------------------*/

#ifdef REDIS_TEST
#include <assert.h>
#include "testhelp.h"

/* Subtree minimum obtained by descending to the leftmost leaf. zbtNodeMin()
 * trusts sep[0] instead, so the verifier needs this independent version to
 * actually check the separator invariant against the tree contents. */
static zbtElem *zbtNodeMinDescend(zbtNode *n) {
    while (!n->isleaf) n = ((zbtInner *)n)->child[0];
    return ((zbtLeaf *)n)->elems[0];
}

static unsigned long zbtVerifyNode(zbtree *t, zbtNode *n, int depth,
                                   int *leafdepth) {
    if (n->isleaf) {
        zbtLeaf *lf = (zbtLeaf *)n;
        /* The head and tail leaves are exempt: zbtSplitLeaf() starts one of
         * them with a single element so sorted insertion can leave the leaf on
         * the other side of the split full. */
        if (n->parent && n != t->tail && n != t->head)
            serverAssert(n->count >= ZBT_LEAF_MIN);
        if (n->parent) serverAssert(n->count >= 1);
        serverAssert(n->count <= ZBT_LEAF_MAX);
        if (*leafdepth == -1) *leafdepth = depth;
        else serverAssert(*leafdepth == depth); /* all leaves same depth */
        for (uint32_t i = 0; i < n->count; i++) {
            serverAssert(lf->elems[i]->leaf == lf);
            serverAssert(lf->tags[i] == zbtElemLeafTag(lf->elems[i]));
            serverAssert(t->table_to_leaf[lf->id] == lf);
        }
        for (uint32_t i = 1; i < n->count; i++) {
            zbtElem *a = lf->elems[i - 1], *b = lf->elems[i];
            serverAssert(zbtCompare(zbtGetScore(a), zbtGetEle(a), b) < 0);
        }
        return n->count;
    }
    zbtInner *in = (zbtInner *)n;
    if (n->parent) serverAssert(n->count >= ZBT_INNER_MIN);
    serverAssert(n->count >= 2 || !n->parent);
    serverAssert(n->count <= ZBT_INNER_MAX);
    unsigned long total = 0;
    for (uint32_t i = 0; i < n->count; i++) {
        serverAssert(in->child[i]->parent == n);
        serverAssert(zbtNodeMinDescend(in->child[i]) == in->sep[i]);
        unsigned long cs = zbtVerifyNode(t, in->child[i], depth + 1, leafdepth);
        serverAssert(cs == in->csize[i]);
        total += cs;
    }
    return total;
}

/* Panics if any structural invariant is violated. */
void zbtDebugVerify(zbtree *t) {
    int leafdepth = -1;
    unsigned long total = zbtVerifyNode(t, t->root, 0, &leafdepth);
    serverAssert(total == t->length);

    /* Verify the leaf chain matches in-order traversal and head/tail. */
    unsigned long chain = 0;
    zbtLeaf *lf = (zbtLeaf *)t->head;
    zbtLeaf *plf = NULL;
    zbtElem *prev_elem = NULL;
    while (lf) {
        serverAssert(lf->prev == plf);
        for (uint32_t i = 0; i < lf->n.count; i++) {
            if (prev_elem) {
                zbtElem *cur = lf->elems[i];
                serverAssert(zbtCompare(zbtGetScore(prev_elem), zbtGetEle(prev_elem), cur) < 0);
            }
            prev_elem = lf->elems[i];
            chain++;
        }
        plf = lf;
        lf = lf->next;
    }
    serverAssert(chain == t->length);
    serverAssert(t->tail == (zbtNode *)plf || (t->length == 0));

    if (t->member_index.size) {
        unsigned long live = t->member_index.used;
        if (t->member_rehash) live += t->member_rehash->table.used;
        lf = (zbtLeaf *)t->head;
        while (lf) {
            for (uint32_t i = 0; i < lf->n.count; i++) {
                zbtElem *e = lf->elems[i];
                zbtreeInsertPosition pos;
                zbtElem *found = zbtFindMember(t, zbtGetEle(e), &pos);
                serverAssert(found == e);
            }
            lf = lf->next;
        }
        UNUSED(live);

        for (uint32_t id = 0; id < t->next_score_leaf_id; id++) {
            zbtLeaf *mapped = t->table_to_leaf[id];
            if (ZBT_IS_FREE_LEAF_ID(mapped)) {
                uint32_t next = ZBT_NEXT_FREE_LEAF_ID(mapped);
                serverAssert(next == 0 || next - 1 < t->next_score_leaf_id);
            } else {
                serverAssert(mapped != NULL && mapped->id == id);
            }
        }

        zbtIndexTable *tables[2];
        int require_migrated[2];
        int ntables = 0;
        tables[ntables] = &t->member_index;
        require_migrated[ntables++] = 0; /* old/main: do not require */
        if (t->member_rehash) {
            tables[ntables] = &t->member_rehash->table;
            require_migrated[ntables++] = 1;
        }
        for (int ti = 0; ti < ntables; ti++) {
            zbtIndexTable *table = tables[ti];
            unsigned long slot_live = 0;
            for (unsigned long i = 0; i < table->size; i++) {
                zbtIndexBucket *bucket = zbtIndexBucketAt(table, i);
                uint64_t tags = zbtIndexTags(bucket);
                for (unsigned int pos = 0; pos < ZBT_INDEX_BUCKET_ITEMS; pos++) {
                    uint8_t tag = (uint8_t)(tags >> (pos * 8));
                    uint32_t id = zbtIndexGetId(table, bucket, pos);
                    if (tag == 0) continue;
                    if (id == ZBT_INDEX_DELETED_ID) continue;
                    slot_live++;
                    serverAssert(id < t->next_score_leaf_id);
                    zbtLeaf *leaf = t->table_to_leaf[id];
                    serverAssert(leaf != NULL && !ZBT_IS_FREE_LEAF_ID(leaf));
                    serverAssert(leaf->id == id);
                    if (require_migrated[ti])
                        serverAssert(zbtIndexLeafMigrated(t, leaf));
                    int found_tag = 0;
                    for (uint32_t k = 0; k < leaf->n.count; k++) {
                        uint8_t ltag = leaf->tags[k];
                        if (ltag == tag || (tag == 1 && ltag == 0)) {
                            found_tag = 1;
                            break;
                        }
                    }
                    serverAssert(found_tag);
                }
            }
            serverAssert(slot_live == table->used);
        }
    }
}

/* Test allocator that always relocates the block, to exercise the pointer
 * fix-ups in the incremental node-defrag path. */
static void *zbtTestReloc(void *ptr) {
    size_t sz = zmalloc_usable_size(ptr);
    void *n = zmalloc(sz);
    memcpy(n, ptr, sz);
    zfree(ptr);
    return n;
}

/* Reference zbtNthInRange(), by linear scan: the elements of 'range' in tree
 * order, indexed the same way (n >= 0 from the first, n < 0 from the last).
 * Deliberately ignores the tree structure so it cannot share a bug with the
 * boundary descent it checks. */
static zbtElem *zbtRefNthInRange(zbtree *t, zrangespec *range, long n,
                                 unsigned long *out_rank) {
    unsigned long first = 0, last = 0, rank = 0;
    zbtIter it;

    for (zbtElem *e = zbtFirst(t, &it); e; e = zbtIterNext(&it)) {
        double s = zbtGetScore(e);
        rank++;
        if (!zslValueGteMin(s, range) || !zslValueLteMax(s, range)) continue;
        if (first == 0) first = rank;
        last = rank;
    }
    if (first == 0) return NULL;

    unsigned long target;
    if (n >= 0) {
        unsigned long steps = (unsigned long)n;
        if (steps > last - first) return NULL;
        target = first + steps;
    } else {
        unsigned long steps = (unsigned long)(-(n + 1));
        if (steps > last - first) return NULL;
        target = last - steps;
    }
    if (out_rank) *out_rank = target;
    return zbtElemByRank(t, target, NULL);
}

/* Compare zbtNthInRange() against the linear reference for one (range, n),
 * including the rank it reports and the position of the iterator it left
 * behind. */
static void zbtCheckNthInRange(zbtree *t, zrangespec *range, long n) {
    unsigned long got_rank = 0, want_rank = 0;
    zbtIter it;
    zbtElem *got = zbtNthInRange(t, range, n, &got_rank, &it);
    zbtElem *want = zbtRefNthInRange(t, range, n, &want_rank);

    serverAssert(got == want);
    if (want == NULL) return;
    serverAssert(got_rank == want_rank);
    serverAssert(zbtElemByRank(t, got_rank, NULL) == got);

    /* The iterator has to be usable for the scan the callers run from here,
     * in either direction. */
    zbtIter fwd = it, bwd = it;
    serverAssert(zbtIterNext(&fwd) == zbtElemByRank(t, got_rank + 1, NULL));
    if (got_rank > 1)
        serverAssert(zbtIterPrev(&bwd) == zbtElemByRank(t, got_rank - 1, NULL));
    else
        serverAssert(zbtIterPrev(&bwd) == NULL);
}

typedef struct zbtTestScanData {
    dict *seen;
    unsigned long emitted;
    int duplicate;
} zbtTestScanData;

static void zbtTestScanCollect(void *privdata, const unsigned char *ele,
                               size_t len, double score)
{
    UNUSED(score);
    zbtTestScanData *data = privdata;
    sds member = sdsnewlen(ele, len);
    if (dictAdd(data->seen, member, NULL) != DICT_OK) {
        sdsfree(member);
        data->duplicate = 1;
    }
    data->emitted++;
}

int zbtreeTest(int argc, char **argv, int flags) {
    UNUSED(argc);
    UNUSED(argv);
    UNUSED(flags);

    printf("Testing B+ tree operations with structure verification\n");

    const int N = 2000;
    zbtree *t = zbtCreate();

    typedef struct {
        double score;
        sds ele;
        zbtElem *elem;
        int deleted;
    } Inserted;

    Inserted *elements = zmalloc(sizeof(Inserted) * N);
    srand(12345);

    for (int i = 0; i < N; i++) {
        double score = (double)(rand() % 137);
        char buf[32];
        snprintf(buf, sizeof(buf), "elem:%d", i);
        sds ele = sdsnew(buf);
        zbtElem *e = zbtInsert(t, score, ele);
        elements[i].score = score;
        elements[i].ele = ele;
        elements[i].elem = e;
        elements[i].deleted = 0;

        if (i % 97 == 0) zbtDebugVerify(t);

        unsigned long rank = zbtGetRank(t, score, ele);
        assert(rank != 0);
        assert(zbtElemByRank(t, rank, NULL) == e);
        assert(zbtRankByElem(t, e) == rank);
    }
    zbtDebugVerify(t);
    test_cond("Insert N elements", t->length == (unsigned long)N);

    /* End-offset positioning must match rank for both leaf-walk and rank-jump. */
    {
        zbtIter it;
        int ok = 1;
        for (unsigned long off = 0; off < 15 && off < t->length; off++) {
            zbtElem *walk = zbtElemByEndOffset(t, off, 0, &it);
            zbtElem *rank = zbtElemByRank(t, off + 1, NULL);
            if (walk != rank) ok = 0;
            walk = zbtElemByEndOffset(t, off, 1, &it);
            rank = zbtElemByRank(t, t->length - off, NULL);
            if (walk != rank) ok = 0;
        }
        test_cond("zbtElemByEndOffset matches rank", ok);
    }

    /* Full in-order scan is sorted and matches length. */
    {
        zbtIter it;
        zbtElem *e = zbtFirst(t, &it);
        unsigned long c = 0;
        zbtElem *prev = NULL;
        while (e) {
            if (prev) assert(zbtCompare(zbtGetScore(prev), zbtGetEle(prev), e) < 0);
            prev = e;
            c++;
            e = zbtIterNext(&it);
        }
        test_cond("Forward scan sorted and complete", c == (unsigned long)N);
    }

    /* A stable cursor scan must assign every member to exactly one index slot,
     * even when several members in a leaf share the same hash tag. */
    {
        dictType scanType = {
            dictSdsHash, NULL, NULL, dictSdsKeyCompare,
            dictSdsDestructor, NULL, NULL
        };
        zbtTestScanData data = {
            .seen = dictCreate(&scanType),
            .emitted = 0,
            .duplicate = 0,
        };
        uint64_t cursor = 0;
        do {
            cursor = zbtScan(t, cursor, 10, zbtTestScanCollect, &data);
        } while (cursor != 0);
        test_cond("Stable index scan is complete without duplicates",
                  !data.duplicate && data.emitted == (unsigned long)N &&
                  dictSize(data.seen) == (unsigned long)N);
        dictRelease(data.seen);
    }

    /* Delete half in random order. */
    for (int i = 0; i < N; i += 2) {
        zbtElem *e = elements[i].elem;
        assert(zbtGetRank(t, zbtGetScore(e), zbtGetEle(e)) != 0);
        zbtDeleteElem(t, e);
        elements[i].deleted = 1;
        if (i % 101 == 0) zbtDebugVerify(t);
    }
    zbtDebugVerify(t);
    test_cond("Delete half", t->length == (unsigned long)N / 2);

    /* Update scores of the survivors. */
    for (int i = 1; i < N; i += 2) {
        zbtElem *e = elements[i].elem;
        elements[i].elem = zbtUpdateScore(t, e, (double)(rand() % 300));
    }
    zbtDebugVerify(t);

    /* Delete the rest. */
    for (int i = 1; i < N; i += 2) {
        zbtDeleteElem(t, elements[i].elem);
        if (i % 103 == 0) zbtDebugVerify(t);
    }
    zbtDebugVerify(t);
    test_cond("Delete rest", t->length == 0);

    for (int i = 0; i < N; i++) sdsfree(elements[i].ele);
    zfree(elements);
    zbtFree(t);

    /* A prepend split can create a leaf behind the active rehash cursor.
     * Members moved there must remain visible before another rehash step
     * eventually reaches the new leaf. */
    {
        const int BASE = 256;
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * BASE);
        for (int i = 0; i < BASE; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "rh:%03d", i);
            sds member = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, member, sdslen(member), 0, NULL);
            sdsfree(member);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, BASE);
        zfree(arr);

        for (int i = BASE; i <= 496; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "rh:%03d", i);
            sds member = sdsnew(buf);
            zbtInsert(bt, (double)i, member);
            sdsfree(member);
        }
        sds newmin = sdsnew("rh:min");
        zbtInsert(bt, -1, newmin);
        sdsfree(newmin);

        sds first = sdsnew("rh:000");
        zbtElem *found = zbtFindMember(bt, first, NULL);
        test_cond("Rehash preserves members moved by a prepend split",
                  found != NULL && zbtGetScore(found) == 0 &&
                  bt->length == 498);
        sdsfree(first);
        zbtDebugVerify(bt);
        zbtFree(bt);
    }

    /* --- Bottom-up bulk build and batched range deletion --- */
    /* Cover boundary sizes around leaf/inner fan-out multiples. Every entry
     * must run: the multi-leaf sizes are the only ones that exercise range
     * deletes spanning whole leaves. */
    static const int sizes[] = {1, ZBT_LEAF_MAX, ZBT_LEAF_MAX + 1,
                                ZBT_LEAF_MAX * ZBT_INNER_MAX + 3, 5000};
    for (int trial = 0; trial < (int)(sizeof(sizes) / sizeof(sizes[0])); trial++) {
        int M = sizes[trial];
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "bm:%08d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, s, sdslen(s), 0, NULL);
            sdsfree(s);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        zfree(arr);
        zbtDebugVerify(bt);
        serverAssert(bt->length == (unsigned long)M);
        serverAssert(zbtGetScore(zbtElemByRank(bt, 1, NULL)) == 0);
        serverAssert(zbtGetScore(zbtElemByRank(bt, M, NULL)) == (double)(M - 1));

        if (M >= 10) {
            unsigned long lo = M / 4 + 1, hi = M / 2;
            unsigned long want = hi - lo + 1;
            unsigned long got = zbtDeleteRangeByRank(bt, lo, hi);
            zbtDebugVerify(bt);
            serverAssert(got == want);
            serverAssert(bt->length == (unsigned long)M - want);
            zrangespec rs = {.min = (double)(M * 3 / 4), .max = 1.0 / 0.0,
                             .minex = 0, .maxex = 0};
            zbtDeleteRangeByScore(bt, &rs);
            zbtDebugVerify(bt);
        }
        zbtDeleteRangeByRank(bt, 1, bt->length);
        zbtDebugVerify(bt);
        serverAssert(bt->length == 0);
        zbtFree(bt);
    }
    test_cond("Bulk build + range delete", 1);

    /* --- Cached hashes for large-member range deletion --- */
    {
        zbtree *bt = zbtCreate();
        char longbuf[ZBT_CACHE_HASH_MIN_LEN];
        memset(longbuf, 'x', sizeof(longbuf));
        sds large = sdsnewlen(longbuf, sizeof(longbuf));
        sds small = sdsnew("small");
        zbtElem *large_elem = zbtInsert(bt, 1, large);
        zbtElem *small_elem = zbtInsert(bt, 2, small);

        serverAssert(zbtHasCachedHash(large_elem));
        serverAssert(zbtGetCachedHash(large_elem) == dictSdsHash(large));
        serverAssert(!zbtHasCachedHash(small_elem));
        uint64_t known = dictSdsHash(large);
        zbtElem *again = zbtCreateElemWithHash(3, large, sdslen(large), 0, NULL, &known);
        serverAssert(zbtHasCachedHash(again));
        serverAssert(zbtGetCachedHash(again) == known);
        zbtFreeElem(again);

        serverAssert(zbtDeleteRangeByRank(bt, 1, 1) == 1);
        serverAssert(zbtFindMember(bt, large, NULL) == NULL);
        serverAssert(zbtFindMember(bt, small, NULL) != NULL);
        serverAssert(zbtDeleteRangeByRank(bt, 1, 1) == 1);
        serverAssert(bt->length == 0);

        sdsfree(large);
        sdsfree(small);
        zbtFree(bt);
    }
    test_cond("Large members use cached hashes for range deletion", 1);

    /* --- Random-window range deletion, checking occupancy --- */
    {
        const int M = 20000;
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "fz:%08d", i);
            sds sd = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, sd, sdslen(sd), 0, NULL);
            sdsfree(sd);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        zfree(arr);

        unsigned long seed = 12345;
        while (bt->length > 200) {
            seed = seed * 1103515245 + 12345;
            unsigned long span = 1 + (seed >> 16) % (ZBT_LEAF_MAX * 2);
            seed = seed * 1103515245 + 12345;
            unsigned long lo = 1 + (seed >> 16) % bt->length;
            unsigned long hi = lo + span;
            if (hi > bt->length) hi = bt->length;
            unsigned long before = bt->length;
            unsigned long got = zbtDeleteRangeByRank(bt, lo, hi);
            zbtDebugVerify(bt);
            serverAssert(got == hi - lo + 1);
            serverAssert(bt->length == before - got);
            serverAssert(zbtRankByElem(bt, zbtElemByRank(bt, 1, NULL)) == 1);
            serverAssert(zbtRankByElem(bt, zbtElemByRank(bt, bt->length, NULL))
                         == bt->length);
        }
        zbtDeleteRangeByRank(bt, 1, bt->length);
        serverAssert(bt->length == 0);
        zbtFree(bt);
    }
    test_cond("Random-window range delete keeps occupancy", 1);

    /* --- Sorted insertion packs leaves, in both directions --- */
    for (int desc = 0; desc < 2; desc++) {
        const int M = 20000;
        zbtree *at = zbtCreate();
        for (int k = 0; k < M; k++) {
            int i = desc ? M - 1 - k : k;   /* descending inserts prepend */
            char buf[32];
            snprintf(buf, sizeof(buf), "as:%08d", i);
            sds sd = sdsnew(buf);
            zbtInsert(at, (double)i, sd);
            sdsfree(sd);
        }
        zbtDebugVerify(at);
        serverAssert(at->length == (unsigned long)M);

        /* Every leaf but the under-filled end one should be full, so the leaf
         * count should match what a bottom-up bulk build of the same elements
         * would use. An even split would need roughly twice as many. */
        unsigned long leaves = 0, full = 0;
        for (zbtLeaf *lf = (zbtLeaf *)at->head; lf; lf = lf->next) {
            leaves++;
            if (lf->n.count == ZBT_LEAF_MAX) full++;
        }
        unsigned long ideal = (M + ZBT_LEAF_MAX - 1) / ZBT_LEAF_MAX;
        serverAssert(leaves == ideal);
        serverAssert(full == leaves - 1);
        /* The short leaf must be the end the inserts were arriving at. */
        zbtLeaf *shortlf = desc ? (zbtLeaf *)at->head : (zbtLeaf *)at->tail;
        serverAssert(shortlf->n.count < ZBT_LEAF_MAX);

        /* Deleting from a packed tree must still hold the invariant, including
         * when the short end leaf itself is the one that underflows. */
        for (int k = 0; k < 3 * ZBT_LEAF_MAX; k++) {
            unsigned long rank = desc ? 1 : at->length;
            zbtElem *e = zbtElemByRank(at, rank, NULL);
            zbtDeleteElem(at, e);
            zbtDebugVerify(at);
        }
        serverAssert(at->length == (unsigned long)M - 3 * ZBT_LEAF_MAX);
        zbtFree(at);
    }
    test_cond("Sorted insert packs leaves both directions", 1);

    /* --- Interpolation into a packed tree stays packed (B* share) --- */
    {
        const int M = 20000;
        zbtree *at = zbtCreate();
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "ip:%08d", i);
            sds sd = sdsnew(buf);
            zbtInsert(at, (double)i, sd);
            sdsfree(sd);
        }
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "i1:%08d", i);
            sds sd = sdsnew(buf);
            zbtInsert(at, (double)i + 0.1, sd);
            sdsfree(sd);
            if (i % 1024 == 0) zbtDebugVerify(at);
        }
        zbtDebugVerify(at);
        serverAssert(at->length == (unsigned long)M * 2);

        unsigned long leaves = 0, full = 0;
        for (zbtLeaf *lf = (zbtLeaf *)at->head; lf; lf = lf->next) {
            leaves++;
            if (lf->n.count == ZBT_LEAF_MAX) full++;
        }
        unsigned long n = (unsigned long)M * 2;
        unsigned long ideal = (n + ZBT_LEAF_MAX - 1) / ZBT_LEAF_MAX;
        serverAssert(leaves == ideal);
        serverAssert(full == leaves || full == leaves - 1);
        if (full == leaves - 1)
            serverAssert(((zbtLeaf *)at->tail)->n.count < ZBT_LEAF_MAX);
        /* Rank of the last original integer and its interpolation. */
        char buf[32];
        snprintf(buf, sizeof(buf), "ip:%08d", M - 1);
        sds last = sdsnew(buf);
        serverAssert(zbtGetRank(at, (double)(M - 1), last) != 0);
        sdsfree(last);
        zbtFree(at);
    }
    test_cond("Interpolation insert packs leaves", 1);

    /* --- Denser sequential interpolation stays near-full --- */
    {
        const int M = 20000;
        const int extra = 3;
        zbtree *at = zbtCreate();
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "dp:%08d", i);
            sds sd = sdsnew(buf);
            zbtInsert(at, (double)i, sd);
            sdsfree(sd);
        }
        for (int p = 1; p <= extra; p++) {
            for (int i = 0; i < M; i++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "d%d:%08d", p, i);
                sds sd = sdsnew(buf);
                zbtInsert(at, (double)i + p * 0.1, sd);
                sdsfree(sd);
            }
            zbtDebugVerify(at);
        }
        unsigned long n = (unsigned long)M * (1 + extra);
        serverAssert(at->length == n);

        unsigned long leaves = 0;
        for (zbtLeaf *lf = (zbtLeaf *)at->head; lf; lf = lf->next) leaves++;
        unsigned long ideal = (n + ZBT_LEAF_MAX - 1) / ZBT_LEAF_MAX;
        /* Share keeps occupancy well above even-split-only (~50-67%). Later
         * interpolation passes hit full inner nodes, so same-parent share
         * cannot refill every stranded half. 80% of packed is the floor. */
        serverAssert(leaves * 4 <= ideal * 5 + 4);
        zbtFree(at);
    }
    test_cond("Denser interpolation keeps occupancy", 1);

    /* --- Incremental node defragmentation --- */
    {
        const int M = 4000;
        zbtElem **arr = zmalloc(sizeof(zbtElem *) * M);
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "dm:%08d", i);
            sds s = sdsnew(buf);
            arr[i] = zbtCreateElem((double)i, s, sdslen(s), 0, NULL);
            sdsfree(s);
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, M);
        zfree(arr);
        zbtDebugVerify(bt);

        int steps = 0;
        while (zbtDefragNodesIncremental(bt, zbtTestReloc, 16)) {
            zbtDebugVerify(bt); /* tree must stay valid after every slice */
            serverAssert(++steps < 100000);
        }
        zbtDebugVerify(bt);

        zbtIter it;
        zbtElem *e = zbtFirst(bt, &it);
        unsigned long c = 0;
        zbtElem *prev = NULL;
        while (e) {
            if (prev)
                serverAssert(zbtCompare(zbtGetScore(prev), zbtGetEle(prev), e) < 0);
            prev = e;
            c++;
            e = zbtIterNext(&it);
        }
        serverAssert(c == (unsigned long)M);
        zbtFree(bt);
        test_cond("Incremental node defrag", 1);
    }

    /* Compact score encoding: round-trip every width and cross boundaries
     * in both directions, including inf and -0.0. */
    {
        zbtree *bt = zbtCreate();
        struct {
            double score;
            const char *name;
            uint8_t enc;
        } cases[] = {
            {0, "z0", ZBT_SCORE_I8},
            {127, "i8hi", ZBT_SCORE_I8},
            {-128, "i8lo", ZBT_SCORE_I8},
            {128, "i16", ZBT_SCORE_I16},
            {-129, "i16n", ZBT_SCORE_I16},
            {32767, "i16hi", ZBT_SCORE_I16},
            {32768, "i24", ZBT_SCORE_I24},
            {8388607, "i24hi", ZBT_SCORE_I24},
            {8388608, "i32", ZBT_SCORE_I32},
            {2147483647.0, "i32hi", ZBT_SCORE_I32},
            {2147483648.0, "i48", ZBT_SCORE_I48},
            {(double)((1LL << 47) - 1), "i48hi", ZBT_SCORE_I48},
            {(double)(1LL << 47), "dbl48", ZBT_SCORE_DBL},
            {1.5, "frac", ZBT_SCORE_DBL},
            {INFINITY, "inf", ZBT_SCORE_DBL},
            {-INFINITY, "ninf", ZBT_SCORE_DBL},
            {-0.0, "nzero", ZBT_SCORE_DBL},
        };
        int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
        zbtElem **elems = zmalloc(sizeof(zbtElem *) * ncases);
        for (int i = 0; i < ncases; i++) {
            sds s = sdsnew(cases[i].name);
            elems[i] = zbtInsert(bt, cases[i].score, s);
            sdsfree(s);
            serverAssert((elems[i]->enc & ZBT_SCORE_MASK) == cases[i].enc);
            double got = zbtGetScore(elems[i]);
            if (cases[i].enc == ZBT_SCORE_DBL && cases[i].score == 0) {
                serverAssert(got == 0 && signbit(got));
            } else if (isinf(cases[i].score)) {
                serverAssert(isinf(got) && !!signbit(got) == !!signbit(cases[i].score));
            } else {
                serverAssert(got == cases[i].score);
            }
        }
        zbtDebugVerify(bt);

        /* Width changes in both directions. */
        elems[1] = zbtUpdateScore(bt, elems[1], 128);           /* 127 I8 -> 128 I16 */
        serverAssert((elems[1]->enc & ZBT_SCORE_MASK) == ZBT_SCORE_I16 && zbtGetScore(elems[1]) == 128);
        elems[1] = zbtUpdateScore(bt, elems[1], 127);           /* back I16 -> I8 */
        serverAssert((elems[1]->enc & ZBT_SCORE_MASK) == ZBT_SCORE_I8 && zbtGetScore(elems[1]) == 127);

        zbtElem *one;
        {
            sds s = sdsnew("one");
            one = zbtInsert(bt, 1, s);
            sdsfree(s);
        }
        one = zbtUpdateScore(bt, one, 1.5);                     /* 1 I8 -> 1.5 DBL */
        serverAssert((one->enc & ZBT_SCORE_MASK) == ZBT_SCORE_DBL && zbtGetScore(one) == 1.5);
        one = zbtUpdateScore(bt, one, 1);                       /* 1.5 DBL -> 1 I8 */
        serverAssert((one->enc & ZBT_SCORE_MASK) == ZBT_SCORE_I8 && zbtGetScore(one) == 1);

        zbtElem *big;
        {
            sds s = sdsnew("i32x");
            big = zbtInsert(bt, 2147483647.0, s);
            sdsfree(s);
        }
        serverAssert((big->enc & ZBT_SCORE_MASK) == ZBT_SCORE_I32);
        big = zbtUpdateScore(bt, big, (double)(1LL << 47));     /* I32 -> DBL via 2^47 */
        serverAssert((big->enc & ZBT_SCORE_MASK) == ZBT_SCORE_DBL && zbtGetScore(big) == (double)(1LL << 47));
        big = zbtUpdateScore(bt, big, 2147483648.0);            /* DBL -> I48 */
        serverAssert((big->enc & ZBT_SCORE_MASK) == ZBT_SCORE_I48 && zbtGetScore(big) == 2147483648.0);

        zbtDebugVerify(bt);
        zfree(elems);
        zbtFree(bt);
        test_cond("Compact score encoding width boundaries", 1);
    }

    /* --- Score updates, in-leaf and leaf-crossing --- */
    {
        const int M = 5000;
        zbtElem **el = zmalloc(sizeof(zbtElem *) * M);
        zbtree *bt = zbtCreate();
        /* Spread the scores so a small delta keeps the element in its leaf
         * and a large one has somewhere else to land. */
        for (int i = 0; i < M; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "up:%08d", i);
            sds s = sdsnew(buf);
            el[i] = zbtInsert(bt, (double)i * 16, s);
            sdsfree(s);
        }
        zbtDebugVerify(bt);

        unsigned long seed = 987654321;
        for (int round = 0; round < 20000; round++) {
            seed = seed * 1103515245 + 12345;
            int i = (int)((seed >> 16) % M);
            seed = seed * 1103515245 + 12345;
            long span = (long)((seed >> 16) % 4001) - 2000;
            /* Mostly nudges that stay inside one leaf, with occasional jumps
             * across the tree, and fractions to change the score width. */
            double want = zbtGetScore(el[i]) + (double)(round % 8 ? span % 24 : span * 37);
            if (round % 7 == 0) want += 0.5;

            el[i] = zbtUpdateScore(bt, el[i], want);
            serverAssert(zbtGetScore(el[i]) == want);
            serverAssert(zbtRankByElem(bt, el[i]) ==
                         zbtGetRank(bt, want, zbtGetEle(el[i])));
            if (round % 512 == 0) zbtDebugVerify(bt);
        }
        zbtDebugVerify(bt);
        serverAssert(bt->length == (unsigned long)M);

        /* Ranks are dense and the iteration order matches them. */
        zbtIter it;
        unsigned long rank = 0;
        for (zbtElem *e = zbtFirst(bt, &it); e; e = zbtIterNext(&it)) {
            rank++;
            serverAssert(zbtRankByElem(bt, e) == rank);
            serverAssert(zbtElemByRank(bt, rank, NULL) == e);
        }
        serverAssert(rank == (unsigned long)M);

        zfree(el);
        zbtFree(bt);
        test_cond("Score update keeps order, ranks and structure", 1);
    }

    /* Range endpoint lookups, against a linear reference. Both directions of
     * Z[REV]RANGEBYSCORE start here, and the offset decides whether the
     * element is reached by walking the leaf chain or by a rank jump. */
    {
        /* Equal-score runs wider than a leaf, so a boundary lands inside a
         * run instead of at a leaf edge, with infinities at both ends. */
        static const struct { double score; int count; } runs[] = {
            {-INFINITY, 1},
            {0, 100},
            {10, 300},          /* several leaves of one score */
            {10.5, 1},
            {20, 100},
            {INFINITY, 1},
        };
        int nruns = (int)(sizeof(runs) / sizeof(runs[0]));
        int total = 0;
        for (int r = 0; r < nruns; r++) total += runs[r].count;

        zbtElem **arr = zmalloc(sizeof(zbtElem *) * total);
        int at = 0;
        for (int r = 0; r < nruns; r++) {
            for (int i = 0; i < runs[r].count; i++) {
                char buf[32];
                snprintf(buf, sizeof(buf), "r%d:%05d", r, i);
                sds s = sdsnew(buf);
                arr[at++] = zbtCreateElem(runs[r].score, s, sdslen(s), 0, NULL);
                sdsfree(s);
            }
        }
        zbtree *bt = zbtCreate();
        zbtBuildFromSorted(bt, arr, total);
        zfree(arr);
        zbtDebugVerify(bt);

        static const zrangespec ranges[] = {
            {-INFINITY, INFINITY, 0, 0},   /* everything */
            {-INFINITY, INFINITY, 1, 1},   /* everything but the infinities */
            {10, 10, 0, 0},                /* the multi-leaf run, exactly */
            {10, 10, 1, 0},                /* empty: excluded from below */
            {10, 10, 0, 1},                /* empty: excluded from above */
            {0, 20, 0, 0},
            {0, 20, 1, 1},
            {-INFINITY, 10, 0, 0},
            {10, INFINITY, 0, 0},
            {-INFINITY, -INFINITY, 0, 0},  /* single element at the head */
            {INFINITY, INFINITY, 0, 0},    /* single element at the tail */
            {10.5, 10.5, 0, 0},            /* single element mid-tree */
            {20, 0, 0, 0},                 /* inverted */
            {100, 200, 0, 0},              /* past the tail */
            {-200, -100, 0, 0},            /* before the head */
        };
        /* Offsets around ZBT_RANGE_WALK_MAX (leaf walk vs rank jump), around
         * the leaf fanout, and past both ends of every range. */
        static const long offsets[] = {
            0, 1, 2, 9, 10, 11, 12, 63, 64, 65, 99, 100, 299, 300, 301,
            (long)ZBT_LEAF_MAX * ZBT_INNER_MAX, 100000,
            -1, -2, -9, -10, -11, -12, -64, -100, -300, -301, -100000,
            LONG_MIN, LONG_MAX,
        };
        int nranges = (int)(sizeof(ranges) / sizeof(ranges[0]));
        int noffsets = (int)(sizeof(offsets) / sizeof(offsets[0]));
        for (int i = 0; i < nranges; i++) {
            zrangespec rs = ranges[i];
            for (int j = 0; j < noffsets; j++)
                zbtCheckNthInRange(bt, &rs, offsets[j]);
        }
        zbtFree(bt);

        /* Degenerate trees: nothing to descend into, and a single element
         * that is both ends of every range covering it. */
        zbtree *empty = zbtCreate();
        zrangespec all = {-INFINITY, INFINITY, 0, 0};
        serverAssert(zbtNthInRange(empty, &all, 0, NULL, NULL) == NULL);
        serverAssert(zbtNthInRange(empty, &all, -1, NULL, NULL) == NULL);
        zbtFree(empty);

        zbtree *one = zbtCreate();
        {
            sds s = sdsnew("only");
            zbtInsert(one, 5, s);
            sdsfree(s);
        }
        for (int j = 0; j < noffsets; j++) {
            zrangespec rs = all;
            zbtCheckNthInRange(one, &rs, offsets[j]);
        }
        zbtFree(one);
        test_cond("Range endpoint lookups match a linear scan", 1);
    }

    return 0;
}
#endif
