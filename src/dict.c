/* Hash Tables Implementation.
 *
 * This file implements in memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto resize if needed
 * tables of power of two in size are used. Entries are stored in 64-byte
 * buckets (one cache line) with a one byte hash tag per slot; a full bucket
 * overflows into a chain of child buckets. See the source code for more
 * information... :)
 *
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "fmacros.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <sys/time.h>
#include <stddef.h>

#include "dict.h"
#include "zmalloc.h"
#include "redisassert.h"
#include "monotonic.h"
#include "atomicvar.h"
#include "util.h"

/* Using dictSetResizeEnabled() we make possible to disable
 * resizing and rehashing of the hash table as needed. This is very important
 * for Redis, as we use copy-on-write and don't want to move too much memory
 * around when there is a child performing saving operations.
 *
 * Note that even when dict_can_resize is set to DICT_RESIZE_AVOID, not all
 * resizes are prevented:
 *  - A hash table is still allowed to expand if the ratio between the number
 *    of elements and the slot capacity >= dict_force_resize_ratio.
 *  - A hash table is still allowed to shrink if the ratio between the number
 *    of elements and the slot capacity <= 1 / (HASHTABLE_MIN_FILL * dict_force_resize_ratio). */
static redisAtomic dictResizeEnable dict_can_resize = DICT_RESIZE_ENABLE;
static const unsigned int dict_force_resize_ratio = 4;

/* -------------------------- types ----------------------------------------- */

/* An entry of a dict with values. The slot of a bucket points to it. For
 * no_value dicts nothing is allocated: the slot stores the tagged key. */
struct dictEntry {
    void *key;
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
};

/* Bucket layout (64 bytes):
 *
 *   chained(1 bit) presence(SLOTS bits) hashes[SLOTS] slots[SLOTS]
 *
 * presence tells which positions hold an entry. hashes[] holds the top byte of
 * the hash of each entry and is used to rule out most non-matching entries
 * without touching them. When chained is set, the last slot is not an entry
 * but a pointer to a child bucket and its presence bit is clear. */
#if DICT_BUCKET_SLOTS == 7
#define BUCKET_BITS_TYPE uint8_t
#define BUCKET_FACTOR 5
#else
#define BUCKET_BITS_TYPE uint16_t
#define BUCKET_FACTOR 3
#endif
#define BUCKET_DIVISOR 32
#define BUCKET_BITS_TYPE_EX __extension__ BUCKET_BITS_TYPE

typedef struct dictBucket {
    BUCKET_BITS_TYPE_EX chained : 1;
    BUCKET_BITS_TYPE_EX presence : DICT_BUCKET_SLOTS;
    uint8_t hashes[DICT_BUCKET_SLOTS];
    dictEntry *slots[DICT_BUCKET_SLOTS];
} dictBucket;

static_assert(sizeof(dictBucket) == DICT_BUCKET_BYTES, "dictBucket must be one cache line");

typedef BUCKET_BITS_TYPE bucketBits;

/* Scan callbacks can delete their entry and insert another one without
 * changing the dict size. Track compaction into the actual slot so it
 * cannot hide an entry from the scan. The stack also handles nested scans. */
typedef struct dictScanState {
    struct dictScanState *prev;
    dict *d;
    dictEntryLink link;
    int compacted;
} dictScanState;

static __thread dictScanState *activeScan;

static inline void scanSlotCompacted(dict *d, dictEntryLink link) {
    for (dictScanState *scan = activeScan; scan; scan = scan->prev) {
        if (scan->d == d && scan->link == link) scan->compacted = 1;
    }
}

/* -------------------------- private prototypes ---------------------------- */

static int _dictExpandIfNeeded(dict *d);
static void _dictShrinkIfNeeded(dict *d);
static void _dictRehashStepIfNeeded(dict *d, uint64_t hash);
static signed char nextBucketExp(size_t min_capacity);
static int _dictInit(dict *d, dictType *type);
static int dictDefaultCompare(dictCmpCache *cache, const void *key1, const void *key2);
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket);

/* -------------------------- misc inline functions -------------------------------- */

typedef int (*keyCmpFunc)(dictCmpCache *cache, const void *key1, const void *key2);
static inline keyCmpFunc dictGetCmpFunc(dict *d) {
    if (d->type->keyCompare)
        return d->type->keyCompare;
    return dictDefaultCompare;
}

static const void *dictStoredKey2Key(dict *d, const void *key __stored_key) {
    return (d->type->keyFromStoredKey) ? d->type->keyFromStoredKey(key) : key;
}

/* -------------------------- hash functions -------------------------------- */

static uint8_t dict_hash_function_seed[16];

void dictSetHashFunctionSeed(uint8_t *seed) {
    memcpy(dict_hash_function_seed,seed,sizeof(dict_hash_function_seed));
}

/* The default hashing function uses SipHash implementation
 * in siphash.c. */

uint64_t siphash(const uint8_t *in, const size_t inlen, const uint8_t *k);
uint64_t siphash_nocase(const uint8_t *in, const size_t inlen, const uint8_t *k);

uint64_t dictGenHashFunction(const void *key, size_t len) {
    return siphash(key, len, dict_hash_function_seed);
}

uint64_t dictGenCaseHashFunction(const unsigned char *buf, size_t len) {
    return siphash_nocase(buf,len,dict_hash_function_seed);
}

/* --------------------- dictEntry pointer bit tricks ----------------------  */

/* The 3 least significant bits in a pointer to a dictEntry determines what the
 * pointer actually points to. If the least bit is set, it's a key. Otherwise,
 * the bit pattern of the least 3 significant bits mark the kind of entry. */

#define ENTRY_PTR_MASK        7 /* 111 */
#define ENTRY_PTR_NORMAL      0 /* 000 : If a pointer to an entry with value. */
#define ENTRY_PTR_IS_ODD_KEY  1 /* XX1 : If a pointer to odd key address (must be 1). */
#define ENTRY_PTR_IS_EVEN_KEY 2 /* 010 : If a pointer to even key address. (must be 2 or 4). */
#define ENTRY_PTR_UNUSED      4 /* 100 : Unused. */

/* Returns 1 if the entry pointer is a pointer to a key, rather than to an
 * allocated entry. Returns 0 otherwise. */
static inline int entryIsKey(const dictEntry *de) {
    return ((uintptr_t)de & (ENTRY_PTR_IS_ODD_KEY | ENTRY_PTR_IS_EVEN_KEY));
}

/* Returns 1 if the pointer is actually a pointer to a dictEntry struct. Returns
 * 0 otherwise. */
static inline int entryIsNormal(const dictEntry *de) {
    return ((uintptr_t)(void *)de & ENTRY_PTR_MASK) == ENTRY_PTR_NORMAL;
}

static inline dictEntry *encodeMaskedPtr(const void *ptr, unsigned int bits) {
    assert(((uintptr_t)ptr & ENTRY_PTR_MASK) == 0);
    return (dictEntry *)(void *)((uintptr_t)ptr | bits);
}

static inline void *decodeMaskedPtr(const dictEntry *de) {
    return (void *)((uintptr_t)(void *)de & ~ENTRY_PTR_MASK);
}

/* Encode a key pointer for storage in a no_value dict slot.
 * For odd keys (like SDS strings), the key can be stored directly.
 * For even keys, we need to tag it with ENTRY_PTR_IS_EVEN_KEY. */
static inline dictEntry *encodeEntryKey(dict *d, void *key) {
    if (d->type->keys_are_odd) {
        debugAssert(((uintptr_t)key & ENTRY_PTR_IS_ODD_KEY) == ENTRY_PTR_IS_ODD_KEY);
        return key;
    } else {
        return encodeMaskedPtr(key, ENTRY_PTR_IS_EVEN_KEY);
    }
}

/* Returns 1 if the entry has a value field and 0 otherwise. */
static inline int entryHasValue(const dictEntry *de) {
    return entryIsNormal(de);
}

static inline void *entryStoredKey(const dictEntry *de) {
    if ((uintptr_t)de & ENTRY_PTR_IS_ODD_KEY) return (void *) de;
    if ((uintptr_t)de & ENTRY_PTR_IS_EVEN_KEY) return decodeMaskedPtr(de);
    return de->key; /* Regular entry */
}

/* The key used for lookups of the entry stored in a slot. */
static inline const void *slotLookupKey(dict *d, const dictEntry *de) {
    return dictStoredKey2Key(d, entryStoredKey(de));
}

/* ----------------------------- bucket helpers ----------------------------- */

static inline size_t numBuckets(int exp) {
    return exp == -1 ? 0 : (size_t)1 << exp;
}

static inline size_t expToMask(int exp) {
    return exp == -1 ? 0 : numBuckets(exp) - 1;
}

/* The tag stored per slot is the top byte of the hash; the bucket index uses
 * the low bits, so the two are independent for tables up to 2^56 buckets. */
static inline uint8_t hashTag(uint64_t hash) {
    return hash >> (CHAR_BIT * 7);
}

static inline int numBucketPositions(const dictBucket *b) {
    return DICT_BUCKET_SLOTS - (b->chained ? 1 : 0);
}

static inline int bucketIsFull(const dictBucket *b) {
    return b->presence == (bucketBits)((1u << numBucketPositions(b)) - 1);
}

static inline int isPositionFilled(const dictBucket *b, int pos) {
    return b->presence & (1u << pos);
}

static inline dictBucket *getChildBucket(const dictBucket *b) {
    return b->chained ? (dictBucket *)(void *)b->slots[DICT_BUCKET_SLOTS - 1] : NULL;
}

static inline void setChildBucket(dictBucket *b, dictBucket *child) {
    b->slots[DICT_BUCKET_SLOTS - 1] = (dictEntry *)(void *)child;
}

/* A bucket with nothing in it and no child. */
static inline int bucketIsEmpty(const dictBucket *b) {
    return b->presence == 0 && !b->chained;
}

static inline bucketBits findHashMatches(const uint8_t *hashes, uint8_t tag) {
    bucketBits m = 0;
    for (int i = 0; i < DICT_BUCKET_SLOTS; i++)
        m |= (bucketBits)((hashes[i] == tag) << i);
    return m;
}

/* Table memory accounting. All changes of the hash table memory go through
 * here, in bytes. */
static inline void tableMemChanged(dict *d, long long delta_bytes) {
    if (d->type->tableMemChanged)
        d->type->tableMemChanged(d, delta_bytes);
}

/* Child buckets are always allocated and freed here, which keeps child_buckets[]
 * and the memory accounting callback in sync. */
static dictBucket *allocChildBucket(dict *d, int table) {
    dictBucket *b = zcalloc(sizeof(*b));
    d->child_buckets[table]++;
    tableMemChanged(d, DICT_BUCKET_BYTES);
    return b;
}

static void freeChildBucket(dict *d, int table, dictBucket *b) {
    zfree(b);
    debugAssert(d->child_buckets[table] > 0);
    d->child_buckets[table]--;
    tableMemChanged(d, -(long long)DICT_BUCKET_BYTES);
}

static inline size_t tableBytes(int exp) {
    return numBuckets(exp) * DICT_BUCKET_BYTES;
}

/* Move an entry from one bucket position to another. */
static void moveEntry(dictBucket *to, int pos_to, dictBucket *from, int pos_from) {
    assert(!isPositionFilled(to, pos_to));
    assert(isPositionFilled(from, pos_from));
    to->slots[pos_to] = from->slots[pos_from];
    to->hashes[pos_to] = from->hashes[pos_from];
    to->presence |= (1u << pos_to);
    from->presence &= ~(1u << pos_from);
}

/* Converts a full bucket to a chained one, moving its last entry to a new
 * child bucket. */
static void bucketConvertToChained(dict *d, dictBucket *b, int table) {
    assert(!b->chained);
    int pos = DICT_BUCKET_SLOTS - 1;
    assert(isPositionFilled(b, pos));
    dictBucket *child = allocChildBucket(d, table);
    moveEntry(child, 0, b, pos);
    b->chained = 1;
    setChildBucket(b, child);
}

/* If the last bucket of a chain is empty or has a single entry, free it. The
 * before-last bucket becomes the new last bucket; a lone entry takes the place
 * of its child pointer. */
static void pruneLastBucket(dict *d, dictBucket *before_last, dictBucket *last, int table) {
    assert(before_last->chained && getChildBucket(before_last) == last);
    assert(!last->chained);
    assert(last->presence == 0 || __builtin_popcount(last->presence) == 1);
    before_last->chained = 0;
    assert(!isPositionFilled(before_last, DICT_BUCKET_SLOTS - 1));
    if (last->presence != 0) {
        int pos_in_last = __builtin_ctz(last->presence);
        moveEntry(before_last, DICT_BUCKET_SLOTS - 1, last, pos_in_last);
        scanSlotCompacted(d, &before_last->slots[DICT_BUCKET_SLOTS - 1]);
    }
    freeChildBucket(d, table, last);
}

/* After removing an entry from a chained bucket, fill the hole with an entry
 * from the end of the chain and free the last bucket if it becomes (nearly)
 * empty. */
static void fillBucketHole(dict *d, dictBucket *b, int pos, int table) {
    assert(b->chained && !isPositionFilled(b, pos));
    dictBucket *before_last = b;
    dictBucket *last = getChildBucket(b);
    while (last->chained) {
        before_last = last;
        last = getChildBucket(last);
    }
    if (last->presence != 0) {
        int pos_in_last = __builtin_ctz(last->presence);
        moveEntry(b, pos, last, pos_in_last);
        scanSlotCompacted(d, &b->slots[pos]);
    }
    if (last->presence == 0 || __builtin_popcount(last->presence) == 1) {
        pruneLastBucket(d, before_last, last, table);
    }
}

/* Deleting entries while compaction is paused leaves holes. This fills them by
 * moving entries from the end of the chain and frees empty buckets. */
static void compactBucketChain(dict *d, size_t bucket_index, int table) {
    dictBucket *b = &d->ht_table[table][bucket_index];
    while (b->chained) {
        dictBucket *next = getChildBucket(b);
        if (next->chained && next->presence == 0) {
            /* Empty bucket in the middle of the chain: unlink it. */
            setChildBucket(b, getChildBucket(next));
            freeChildBucket(d, table, next);
            continue;
        }
        if (!next->chained && (next->presence == 0 || __builtin_popcount(next->presence) == 1)) {
            pruneLastBucket(d, b, next, table);
            return;
        }
        if (__builtin_popcount(b->presence) < DICT_BUCKET_SLOTS - 1) {
            for (int pos = 0; pos < DICT_BUCKET_SLOTS - 1; pos++) {
                if (!isPositionFilled(b, pos)) {
                    fillBucketHole(d, b, pos, table);
                    if (!b->chained) return;
                }
            }
        }
        b = next;
    }
}

/* Find a free position for an entry with the given hash, in the table that
 * accepts inserts. Chains the bucket if it is full. */
static dictBucket *findBucketForInsert(dict *d, uint64_t hash, int *pos_out, int *table_out) {
    int table = dictIsRehashing(d) ? 1 : 0;
    assert(d->ht_table[table]);
    dictBucket *b = &d->ht_table[table][hash & expToMask(d->ht_size_exp[table])];
    while (bucketIsFull(b)) {
        if (!b->chained) bucketConvertToChained(d, b, table);
        b = getChildBucket(b);
    }
    int pos;
    for (pos = 0; pos < DICT_BUCKET_SLOTS; pos++) {
        if (!isPositionFilled(b, pos)) break;
    }
    assert(pos < DICT_BUCKET_SLOTS);
    *pos_out = pos;
    if (table_out) *table_out = table;
    return b;
}

/* Search the chain of one top-level bucket for the key. Returns the position
 * or -1. */
static inline int bucketFindKey(dict *d, dictBucket *b, uint8_t tag, const void *key,
                                keyCmpFunc cmp, dictCmpCache *cache)
{
    bucketBits candidates = b->presence & findHashMatches(b->hashes, tag);
    while (candidates) {
        int pos = __builtin_ctz(candidates);
        const void *visited = slotLookupKey(d, b->slots[pos]);
        if (key == visited || cmp(cache, key, visited)) return pos;
        candidates &= candidates - 1;
    }
    return -1;
}

/* Find the bucket and position of a key. Does not rehash or write anything. */
static dictBucket *findBucketForKey(dict *d, uint64_t hash, const void *key, int *pos_out, int *table_out) {
    dictCmpCache cache = {0};
    keyCmpFunc cmp = dictGetCmpFunc(d);
    uint8_t tag = hashTag(hash);
    for (int table = 0; table <= 1; table++) {
        if (d->ht_used[table] == 0) continue;
        size_t idx = hash & expToMask(d->ht_size_exp[table]);
        /* Buckets before rehashidx have already been moved to the new table. */
        if (table == 0 && d->rehashidx >= 0 && (long)idx < d->rehashidx) continue;
        dictBucket *b = &d->ht_table[table][idx];
        do {
            int pos = bucketFindKey(d, b, tag, key, cmp, &cache);
            if (pos >= 0) {
                *pos_out = pos;
                if (table_out) *table_out = table;
                return b;
            }
            b = getChildBucket(b);
        } while (b);
    }
    return NULL;
}

/* ----------------------------- pausing ------------------------------------ */

/* Run a shrink that was deferred because the dict was paused. */
static void dictResumeFinalize(dict *d) {
    if (d->pauserehash == 0 && d->pausecompact == 0 && d->pending_shrink) {
        d->pending_shrink = 0;
        _dictShrinkIfNeeded(d);
    }
}

/* Used by scan: stops the tables from being rehashed (entries moving between
 * tables) while the callbacks run, but still allows compaction of the chain
 * being scanned. */
static inline void pauseRehashOnly(dict *d) {
    d->pauserehash++;
}

static void resumeRehashOnly(dict *d) {
    debugAssert(d->pauserehash > 0);
    d->pauserehash--;
    dictResumeFinalize(d);
}

/* Used by safe iterators, two-phase delete and callers that hold a link:
 * entries must not move at all, so rehashing and bucket compaction are both
 * paused. */
void dictPauseRehashing(dict *d) {
    d->pauserehash++;
    d->pausecompact++;
}

void dictResumeRehashing(dict *d) {
    debugAssert(d->pauserehash > 0 && d->pausecompact > 0);
    d->pauserehash--;
    d->pausecompact--;
    dictResumeFinalize(d);
}

/* Clears the presence bit of a slot, adjusts the entry count and compacts the
 * chain unless that is paused. The entry itself is not touched. */
static void removeSlot(dict *d, dictBucket *b, int pos, int table) {
    b->presence &= ~(1u << pos);
    d->ht_used[table]--;
    if (b->chained && d->pausecompact == 0)
        fillBucketHole(d, b, pos, table);
}

/* ----------------------------- API implementation ------------------------- */

/* Reset hash table parameters already initialized with _dictInit()*/
static void _dictReset(dict *d, int htidx)
{
    d->ht_table[htidx] = NULL;
    d->ht_size_exp[htidx] = -1;
    d->ht_used[htidx] = 0;
    d->child_buckets[htidx] = 0;
}

/* Create a new hash table */
dict *dictCreate(dictType *type)
{
    size_t metasize = type->dictMetadataBytes ? type->dictMetadataBytes(NULL) : 0;
    dict *d = zmalloc(sizeof(*d)+metasize);
    if (metasize > 0) {
        memset(dictMetadata(d), 0, metasize);
    }
    _dictInit(d,type);
    return d;
}

/* Change dictType of dict to another one with metadata support
 * Rest of dictType's values must stay the same */
void dictTypeAddMeta(dict **d, dictType *typeWithMeta) {
    /* Verify new dictType is compatible with the old one */
    dictType toCmp = *typeWithMeta;
    /* Ignore 'dictMetadataBytes' and 'onDictRelease' in comparison */
    toCmp.dictMetadataBytes = (*d)->type->dictMetadataBytes;
    toCmp.onDictRelease = (*d)->type->onDictRelease;
    assert(memcmp((*d)->type, &toCmp, sizeof(dictType)) == 0); /* The rest of the dictType fields must be the same */

    *d = zrealloc(*d, sizeof(dict) + typeWithMeta->dictMetadataBytes(*d));
    (*d)->type = typeWithMeta;
}

/* Initialize the hash table */
int _dictInit(dict *d, dictType *type)
{
    _dictReset(d, 0);
    _dictReset(d, 1);
    d->type = type;
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pausecompact = 0;
    d->pauseAutoResize = 0;
    d->pending_shrink = 0;
    return DICT_OK;
}

/* Number of top-level buckets to hold `min_capacity` entries, as an exponent
 * (buckets = 1 << exp). We aim for a fill of at most 91% (32 / 5 / 7) which
 * lets us avoid a division: buckets = ceil(min_capacity * 5 / 32). */
static signed char nextBucketExp(size_t min_capacity) {
    if (min_capacity <= DICT_HT_INITIAL_SIZE) return DICT_HT_INITIAL_EXP;
    size_t min_buckets = (min_capacity * BUCKET_FACTOR - 1) / BUCKET_DIVISOR + 1;
    if (min_buckets >= SIZE_MAX / 2) return CHAR_BIT * sizeof(size_t) - 1;
    if (min_buckets == 1) return 0;
    return CHAR_BIT * sizeof(size_t) - __builtin_clzl(min_buckets - 1);
}

/* Is the whole old table rehashed? */
static inline int oldTableIsEmpty(dict *d) {
    return d->ht_used[0] == 0 && d->child_buckets[0] == 0;
}

/* Swap the new table in place of the old one once all entries moved. */
static int dictCheckRehashingCompleted(dict *d) {
    if (!dictIsRehashing(d) || !oldTableIsEmpty(d)) return 0;

    if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
    tableMemChanged(d, -(long long)tableBytes(d->ht_size_exp[0]));
    zfree(d->ht_table[0]);
    /* Copy the new ht onto the old one */
    d->ht_table[0] = d->ht_table[1];
    d->ht_used[0] = d->ht_used[1];
    d->child_buckets[0] = d->child_buckets[1];
    d->ht_size_exp[0] = d->ht_size_exp[1];
    _dictReset(d, 1);
    d->rehashidx = -1;
    return 1;
}

/* Is rehashing allowed by the global resize policy? */
static inline int rehashAllowedByPolicy(dict *d) {
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if (can_resize == DICT_RESIZE_FORBID) return 0;
    /* If dict_can_resize is DICT_RESIZE_AVOID, we want to avoid rehashing.
     * - If expanding, the threshold is dict_force_resize_ratio which is 4.
     * - If shrinking, the threshold is 1 / (HASHTABLE_MIN_FILL * dict_force_resize_ratio) which is 1/32. */
    if (can_resize == DICT_RESIZE_AVOID) {
        size_t s0 = numBuckets(d->ht_size_exp[0]);
        size_t s1 = numBuckets(d->ht_size_exp[1]);
        /* Bucket sizing leaves some spare capacity, so a forced shrink at
         * 1/32 occupancy can select a target only 16 times smaller. Allow
         * that shrink to progress using the same occupancy threshold. */
        if ((s1 > s0 && s1 < dict_force_resize_ratio * s0) ||
            (s1 < s0 && s0 < HASHTABLE_MIN_FILL * dict_force_resize_ratio * s1 &&
             dictSize(d) > s0 * DICT_BUCKET_SLOTS / (HASHTABLE_MIN_FILL * dict_force_resize_ratio)))
            return 0;
    }
    return 1;
}

/* Moves every entry of the bucket chain at index `idx` of the old table to
 * the new table, frees its child buckets and leaves the root bucket empty.
 * Does not change rehashidx. */
static void rehashChain(dict *d, size_t idx) {
    dictBucket *root = &d->ht_table[0][idx];
    int expanding = d->ht_size_exp[1] > d->ht_size_exp[0];
    for (dictBucket *b = root; b; b = getChildBucket(b)) {
        bucketBits present = b->presence;
        while (present) {
            int pos = __builtin_ctz(present);
            present &= present - 1;
            dictEntry *e = b->slots[pos];
            uint64_t hash;
            uint8_t tag;
            if (expanding) {
                hash = dictGetHash(d, slotLookupKey(d, e));
                tag = hashTag(hash);
            } else {
                /* Shrinking: table sizes are powers of two, so the bucket
                 * index in the smaller table is the masked old index, and the
                 * tag is unchanged. */
                hash = idx;
                tag = b->hashes[pos];
            }
            int dpos;
            dictBucket *dst = findBucketForInsert(d, hash, &dpos, NULL);
            dst->slots[dpos] = e;
            dst->hashes[dpos] = tag;
            dst->presence |= (1u << dpos);
            d->ht_used[0]--;
            d->ht_used[1]++;
        }
    }
    dictBucket *child = getChildBucket(root);
    while (child) {
        dictBucket *next = getChildBucket(child);
        freeChildBucket(d, 0, child);
        child = next;
    }
    root->chained = 0;
    root->presence = 0;
}

/* Resize or create the hash table,
 * when malloc_failed is non-NULL, it'll avoid panic if malloc fails (in which case it'll be set to 1).
 * `size` is the number of entries the new table must be able to hold.
 * Returns DICT_OK if resize was performed, and DICT_ERR if skipped. */
int _dictResize(dict *d, unsigned long size, int* malloc_failed)
{
    if (malloc_failed) *malloc_failed = 0;

    /* We can't rehash twice if rehashing is ongoing. */
    assert(!dictIsRehashing(d));

    signed char new_exp = nextBucketExp(size);

    /* Detect overflows */
    size_t newbuckets = numBuckets(new_exp);
    if (newbuckets * DICT_BUCKET_SLOTS < size || newbuckets * DICT_BUCKET_BYTES < newbuckets)
        return DICT_ERR;

    /* Rehashing to the same table size is not useful. */
    if (new_exp == d->ht_size_exp[0]) return DICT_ERR;

    /* Allocate the new hash table, all buckets empty. */
    dictBucket *new_table;
    if (malloc_failed) {
        new_table = ztrycalloc(newbuckets * DICT_BUCKET_BYTES);
        *malloc_failed = new_table == NULL;
        if (*malloc_failed)
            return DICT_ERR;
    } else
        new_table = zcalloc(newbuckets * DICT_BUCKET_BYTES);

    /* Prepare a second hash table for incremental rehashing.
     * We do this even for the first initialization, so that we can trigger the
     * rehashingStarted more conveniently, we will clean it up right after. */
    d->ht_size_exp[1] = new_exp;
    d->ht_used[1] = 0;
    d->child_buckets[1] = 0;
    d->ht_table[1] = new_table;
    d->rehashidx = 0;
    if (d->type->rehashingStarted) d->type->rehashingStarted(d);
    tableMemChanged(d, newbuckets * DICT_BUCKET_BYTES);

    /* Is this the first initialization or is the first hash table empty? If so
     * it's not really a rehashing, we can just set the first hash table so that
     * it can accept keys. An empty old table that someone may be iterating
     * (rehashing paused) must stay alive until rehashing is resumed. */
    if (d->ht_table[0] == NULL || (oldTableIsEmpty(d) && d->pauserehash == 0)) {
        if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
        tableMemChanged(d, -(long long)tableBytes(d->ht_size_exp[0]));
        if (d->ht_table[0]) zfree(d->ht_table[0]);
        d->ht_size_exp[0] = new_exp;
        d->ht_used[0] = 0;
        d->child_buckets[0] = 0;
        d->ht_table[0] = new_table;
        _dictReset(d, 1);
        d->rehashidx = -1;
        return DICT_OK;
    }

    /* Force a full rehashing of the dictionary */
    if (d->type->force_full_rehash && d->pauserehash == 0) {
        while (dictRehash(d, 1000)) {
            /* Continue rehashing */
        }
    }
    return DICT_OK;
}

int _dictExpand(dict *d, unsigned long size, int* malloc_failed) {
    /* the size is invalid if it is smaller than the capacity of the hash table
     * or smaller than the number of elements already inside the hash table */
    if (dictIsRehashing(d) || d->ht_used[0] > size ||
        numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS >= size)
        return DICT_ERR;
    return _dictResize(d, size, malloc_failed);
}

/* return DICT_ERR if expand was not performed */
int dictExpand(dict *d, unsigned long size) {
    return _dictExpand(d, size, NULL);
}

/* return DICT_ERR if expand failed due to memory allocation failure */
int dictTryExpand(dict *d, unsigned long size) {
    int malloc_failed = 0;
    _dictExpand(d, size, &malloc_failed);
    return malloc_failed? DICT_ERR : DICT_OK;
}

/* return DICT_ERR if shrink was not performed */
int dictShrink(dict *d, unsigned long size) {
    /* the size is invalid if it is bigger than the capacity of the hash table
     * or smaller than the number of elements already inside the hash table */
    if (dictIsRehashing(d) || d->ht_used[0] > size ||
        numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS <= size)
        return DICT_ERR;
    return _dictResize(d, size, NULL);
}

/* Performs N steps of incremental rehashing. Returns 1 if there are still
 * keys to move from the old to the new hash table, otherwise 0 is returned.
 *
 * Note that a rehashing step consists in moving a bucket chain (a top-level
 * bucket and its child buckets) from the old to the new hash table. Since part
 * of the hash table may be composed of empty buckets, it is not guaranteed that
 * this function will rehash even a single chain, since it will visit at max
 * N*10 empty buckets in total, otherwise the amount of work it does would be
 * unbound and the function may block for a long time. */
int dictRehash(dict *d, int n) {
    int empty_visits = n*10; /* Max number of empty buckets to visit. */
    if (!dictIsRehashing(d) || d->pauserehash != 0) return 0;
    if (!rehashAllowedByPolicy(d)) return 0;

    while (n-- && !oldTableIsEmpty(d)) {
        size_t size0 = numBuckets(d->ht_size_exp[0]);
        /* rehashidx can't overflow as there is still something to move */
        assert(size0 > (size_t)d->rehashidx);
        while (bucketIsEmpty(&d->ht_table[0][d->rehashidx])) {
            d->rehashidx++;
            assert(size0 > (size_t)d->rehashidx);
            if (--empty_visits == 0) return 1;
        }
        /* Move all the keys in this bucket chain from the old to the new HT */
        rehashChain(d, d->rehashidx);
        d->rehashidx++;
    }

    return !dictCheckRehashingCompleted(d);
}

long long timeInMilliseconds(void) {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

/* Rehash in us+"delta" microseconds. The value of "delta" is larger
 * than 0, and is smaller than 1000 in most cases. The exact upper bound
 * depends on the running time of dictRehash(d,100).*/
int dictRehashMicroseconds(dict *d, uint64_t us) {
    if (d->pauserehash > 0) return 0;

    monotime timer;
    elapsedStart(&timer);
    int rehashes = 0;

    while(dictRehash(d,100)) {
        rehashes += 100;
        if (elapsedUs(timer) >= us) break;
    }
    return rehashes;
}

/* This function performs just a step of rehashing, and only if hashing has
 * not been paused for our hash table. When we have iterators in the
 * middle of a rehashing we can't mess with the two hash tables otherwise
 * some elements can be missed or duplicated.
 *
 * This function is called by common lookup or update operations in the
 * dictionary so that the hash table automatically migrates from H1 to H2
 * while it is actively used. */
static void _dictRehashStep(dict *d) {
    if (d->pauserehash == 0) dictRehash(d,1);
}

/* Performs rehashing on a single bucket chain. */
static int _dictBucketRehash(dict *d, uint64_t idx) {
    if (d->pauserehash != 0) return 0;
    if (!dictIsRehashing(d) || !rehashAllowedByPolicy(d)) return 0;
    rehashChain(d, idx);
    dictCheckRehashingCompleted(d);
    return 1;
}

static void _dictRehashStepIfNeeded(dict *d, uint64_t hash) {
    if ((!dictIsRehashing(d)) || (d->pauserehash != 0))
        return;
    size_t idx = hash & expToMask(d->ht_size_exp[0]);
    /* rehashing not in progress if rehashidx == -1 */
    if ((long)idx >= d->rehashidx && d->ht_table[0] && !bucketIsEmpty(&d->ht_table[0][idx])) {
        /* If we have a valid hash entry at `idx` in ht0, we perform
         * rehash on the chain at `idx` (being more CPU cache friendly) */
        _dictBucketRehash(d, idx);
    } else {
        /* If the hash entry is not in ht0, we rehash the buckets based
         * on the rehashidx (not CPU cache friendly). */
        dictRehash(d,1);
    }
}

/* Add an element to the target hash table */
int dictAdd(dict *d, void *key __stored_key, void *val)
{
    dictEntry *entry = dictAddRaw(d,key,NULL);

    if (!entry) return DICT_ERR;
    if (!d->type->no_value) dictSetVal(d, entry, val);
    return DICT_OK;
}

int dictCompareKeys(dict *d, const void *key1, const void *key2) {
    dictCmpCache cache = {0};
    keyCmpFunc cmpFunc = dictGetCmpFunc(d);
    return cmpFunc(&cache, key1, key2);
}

/* Stores `key` in a free position and returns the slot content (the tagged
 * key for no_value dicts, a new dictEntry otherwise). If `slot` is not NULL it
 * is set to the address of the slot. The key must not exist in the dict and
 * the table must have been made ready (rehash step, expansion) by the caller. */
static dictEntry *dictInsertKey(dict *d, uint64_t hash, void *key __stored_key, dictEntryLink *slot) {
    dictEntry *entry;
    int pos, table;
    if (d->type->no_value) {
        entry = encodeEntryKey(d, key);
        assert(entryIsKey(entry));
    } else {
        entry = zmalloc(sizeof(*entry));
        assert(entryIsNormal(entry)); /* Check alignment of allocation */
        entry->key = key;
    }
    dictBucket *b = findBucketForInsert(d, hash, &pos, &table);
    b->slots[pos] = entry;
    b->hashes[pos] = hashTag(hash);
    b->presence |= (1u << pos);
    d->ht_used[table]++;
    if (slot) *slot = &b->slots[pos];
    return entry;
}

/* Low level add or find:
 * This function adds the entry but instead of setting a value returns the
 * dictEntry structure to the user, that will make sure to fill the value
 * field as they wish.
 *
 * This function is also directly exposed to the user API to be called
 * mainly in order to store non-pointers inside the hash value, example:
 *
 * entry = dictAddRaw(dict,mykey,NULL);
 * if (entry != NULL) dictSetSignedIntegerVal(entry,1000);
 *
 * Return values:
 *
 * If key already exists NULL is returned, and "*existing" is populated
 * with the existing entry if existing is not NULL.
 *
 * If key was added, the hash entry is returned to be manipulated by the caller.
 */
dictEntry *dictAddRaw(dict *d, void *key __stored_key, dictEntry **existing)
{
    const void *lookup_key = dictStoredKey2Key(d, key);
    uint64_t hash = dictGetHash(d, lookup_key);
    int pos, table;
    if (existing) *existing = NULL;

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, hash);
    /* Expand the hash table if needed */
    _dictExpandIfNeeded(d);
    if (unlikely(d->ht_table[0] == NULL)) dictExpand(d, DICT_HT_INITIAL_SIZE);

    dictBucket *b = findBucketForKey(d, hash, lookup_key, &pos, &table);
    if (b) {
        if (existing) *existing = b->slots[pos];
        return NULL;
    }

    /* Dup the key if necessary. */
    if (d->type->keyDup) key = d->type->keyDup(d, key);

    return dictInsertKey(d, hash, key, NULL);
}

/* Add or Overwrite:
 * Add an element, discarding the old value if the key already exists.
 * Return 1 if the key was added from scratch, 0 if there was already an
 * element with such key and dictReplace() just performed a value update
 * operation. */
int dictReplace(dict *d, void *key __stored_key, void *val)
{
    dictEntry *entry, *existing;

    /* Try to add the element. If the key
     * does not exists dictAdd will succeed. */
    entry = dictAddRaw(d,key,&existing);
    if (entry) {
        dictSetVal(d, entry, val);
        return 1;
    }

    /* Set the new value and free the old one. Note that it is important
     * to do that in this order, as the value may just be exactly the same
     * as the previous one. In this context, think to reference counting,
     * you want to increment (set), and then decrement (free), and not the
     * reverse. */
    void *oldval = dictGetVal(existing);
    dictSetVal(d, existing, val);
    if (d->type->valDestructor)
        d->type->valDestructor(d, oldval);
    return 0;
}

/* Add or Find:
 * dictAddOrFind() is simply a version of dictAddRaw() that always
 * returns the hash entry of the specified key, even if the key already
 * exists and can't be added (in that case the entry of the already
 * existing key is returned.)
 *
 * See dictAddRaw() for more information. */
dictEntry *dictAddOrFind(dict *d, void *key __stored_key) {
    dictEntry *entry, *existing;
    entry = dictAddRaw(d,key,&existing);
    return entry ? entry : existing;
}

/* Search and remove an element. This is a helper function for
 * dictDelete() and dictUnlink(), please check the top comment
 * of those functions. */
static dictEntry *dictGenericDelete(dict *d, const void *key, int nofree) {
    int pos, table;

    /* dict is empty */
    if (dictSize(d) == 0) return NULL;

    uint64_t hash = dictGetHash(d, key);

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, hash);

    dictBucket *b = findBucketForKey(d, hash, key, &pos, &table);
    if (!b) return NULL; /* not found */

    dictEntry *he = b->slots[pos];
    removeSlot(d, b, pos, table);
    if (!nofree) {
        /* Destructors run while the entry is still counted in dictSize(), as
         * they always did (t_hash.c template registry relies on it). */
        d->ht_used[table]++;
        dictFreeUnlinkedEntry(d, he);
        d->ht_used[table]--;
    }
    _dictShrinkIfNeeded(d);
    return he;
}

/* Remove an element, returning DICT_OK on success or DICT_ERR if the
 * element was not found. */
int dictDelete(dict *ht, const void *key) {
    return dictGenericDelete(ht,key,0) ? DICT_OK : DICT_ERR;
}

/* Remove an element from the table, but without actually releasing
 * the key, value and dictionary entry. The dictionary entry is returned
 * if the element was found (and unlinked from the table), and the user
 * should later call `dictFreeUnlinkedEntry()` with it in order to release it.
 * Otherwise if the key is not found, NULL is returned.
 *
 * This function is useful when we want to remove something from the hash
 * table but want to use its value before actually deleting the entry.
 * Without this function the pattern would require two lookups:
 *
 *  entry = dictFind(...);
 *  // Do something with entry
 *  dictDelete(dictionary,entry);
 *
 * Thanks to this function it is possible to avoid this, and use
 * instead:
 *
 * entry = dictUnlink(dictionary,entry);
 * // Do something with entry
 * dictFreeUnlinkedEntry(entry); // <- This does not need to lookup again.
 */
dictEntry *dictUnlink(dict *d, const void *key) {
    return dictGenericDelete(d,key,1);
}

/* You need to call this function to really free the entry after a call
 * to dictUnlink(). It's safe to call this function with 'he' = NULL. */
void dictFreeUnlinkedEntry(dict *d, dictEntry *he) {
    if (he == NULL) return;
    dictFreeKey(d, he);
    dictFreeVal(d, he);
    if (!entryIsKey(he)) zfree(decodeMaskedPtr(he));
}

/* Free the entries of a table (keys, values, dictEntry structs), its child
 * buckets and the top-level array. The memory accounting callback is NOT
 * called: the caller reports the freed bytes. */
static int _dictClear(dict *d, int htidx, void(callback)(dict*)) {
    size_t nb = numBuckets(d->ht_size_exp[htidx]);
    for (size_t i = 0; i < nb && (d->ht_used[htidx] > 0 || d->child_buckets[htidx] > 0); i++) {
        /* Callback will be called once for every 8192 buckets. */
        if (callback && i != 0 && (i & 8191) == 0) callback(d);

        dictBucket *root = &d->ht_table[htidx][i];
        dictBucket *b = root;
        while (b) {
            bucketBits present = b->presence;
            while (present) {
                int pos = __builtin_ctz(present);
                present &= present - 1;
                dictEntry *he = b->slots[pos];
                dictFreeKey(d, he);
                dictFreeVal(d, he);
                if (!entryIsKey(he)) zfree(decodeMaskedPtr(he));
                d->ht_used[htidx]--;
            }
            dictBucket *next = getChildBucket(b);
            if (b != root) {
                zfree(b);
                d->child_buckets[htidx]--;
            }
            b = next;
        }
    }
    /* Free the table and the allocated cache structure */
    zfree(d->ht_table[htidx]);
    /* Re-initialize the table */
    _dictReset(d, htidx);
    return DICT_OK; /* never fails */
}

/* Total memory of the hash tables (top-level + child buckets). */
size_t dictTableMemUsage(const dict *d) {
    return (dictBuckets(d) + d->child_buckets[0] + d->child_buckets[1]) * DICT_BUCKET_BYTES;
}

/* Clear & Release the hash table */
void dictRelease(dict *d)
{
    /* Someone may be monitoring a dict that started rehashing, before
     * destroying the dict fake completion. */
    if (dictIsRehashing(d) && d->type->rehashingCompleted)
        d->type->rehashingCompleted(d);

    /* Subtract the size of all buckets. */
    tableMemChanged(d, -(long long)dictTableMemUsage(d));

    if (d->type->onDictRelease)
        d->type->onDictRelease(d);

    _dictClear(d,0,NULL);
    _dictClear(d,1,NULL);
    zfree(d);
}

/* Finds a given key. Like dictFindLink(), yet search bucket even if dict is empty.
 *
 * Returns dictEntryLink reference if found. Otherwise, return NULL.
 *
 * bucket - return an opaque, non-NULL, token for the bucket where the key
 *          would be inserted, unless the dict has no hash table. The token
 *          is never dereferenced by dict.c: inserts recompute the position.
 *          Lookups never write to the table.
 */
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket) {
    int pos, table;

    if (bucket) {
        *bucket = NULL;
    } else {
        /* If dict is empty and no need to find bucket, return NULL */
        if (dictSize(d) == 0) return NULL;
    }

    const uint64_t hash = dictGetHash(d, key);

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, hash);

    if (bucket) {
        int itable = dictIsRehashing(d) ? 1 : 0;
        if (d->ht_table[itable])
            *bucket = (dictEntryLink)&d->ht_table[itable][hash & expToMask(d->ht_size_exp[itable])].slots[0];
    }

    dictBucket *b = findBucketForKey(d, hash, key, &pos, &table);
    return b ? (dictEntryLink)&b->slots[pos] : NULL;
}

dictEntry *dictFind(dict *d, const void *key)
{
    dictEntryLink link = dictFindLink(d, key, NULL);
    return (link) ? *link : NULL;
}

/* Finds the dictEntry using pointer and pre-calculated hash.
 * oldkey is a dead pointer and should not be accessed.
 * the hash value should be provided using dictGetHash.
 * no string / key comparison is performed.
 * return value is a pointer to the dictEntry if found, or NULL if not found. */
dictEntry *dictFindByHashAndPtr(dict *d, const void *oldptr, const uint64_t hash) {
    uint8_t tag = hashTag(hash);

    if (dictSize(d) == 0) return NULL; /* dict is empty */
    for (int table = 0; table <= 1; table++) {
        if (d->ht_used[table] == 0) continue;
        size_t idx = hash & expToMask(d->ht_size_exp[table]);
        if (table == 0 && (long)idx < d->rehashidx) continue;
        dictBucket *b = &d->ht_table[table][idx];
        do {
            bucketBits candidates = b->presence & findHashMatches(b->hashes, tag);
            while (candidates) {
                int pos = __builtin_ctz(candidates);
                candidates &= candidates - 1;
                if (oldptr == entryStoredKey(b->slots[pos]))
                    return b->slots[pos];
            }
            b = getChildBucket(b);
        } while (b);
    }
    return NULL;
}

/* Find a key and return its dictEntryLink reference. Otherwise, return NULL
 *
 * A dictEntryLink is the address of the slot in the hash table that holds the
 * entry. It is Useful for deletion, addition, unlinking and updating,
 * especially for dict configured with 'no_value'. In such cases returning only
 * `dictEntry` from a lookup may be insufficient since it might be opt-out to
 * be the object itself. By locating the slot (dictEntryLink) these ops can be
 * properly handled.
 *
 * A link is only valid until the dict is modified: any insert into the same
 * bucket chain (a full bucket moves its last slot to a child bucket), any
 * deletion that compacts the chain, any lookup or write that performs a
 * rehash step, and any resize invalidate it. Use dictPauseRehashing() to hold
 * a link across calls that may look up keys in the same dict; note that an
 * insert can still invalidate it.
 *
 * After calling link = dictFindLink(...), any necessary updates based on returned
 * link or bucket must be performed immediately after by calling dictSetKeyAtLink()
 * without any intervening operations on given dict. Example with kvobj of
 * replacing key with new key:
 *
 *      link = dictFindLink(d, key, &bucket);
 *      ... Do something, but don't modify the dict ...
 *      // assert(link != NULL);
 *      dictSetKeyAtLink(d, kv, &link, 0);
 *
 * To add new value (dictSetKeyAtLink() makes room if needed and finds the
 * insert position again, so `bucket` only needs to be non-NULL; on return it
 * is updated to the link of the inserted key):
 *
 *      link = dictFindLink(d, key, &bucket);
 *      ... Do something, but don't modify the dict ...
 *      // assert(link == NULL);
 *      dictSetKeyAtLink(d, kv, &bucket, 1);
 *
 *  bucket - an opaque token for the bucket that the key was mapped to; not
 *           set (NULL) if the dict is empty.
 */
dictEntryLink dictFindLink(dict *d, const void *key, dictEntryLink *bucket) {
    if (bucket) *bucket = NULL;
    if (unlikely(dictSize(d) == 0))
        return NULL;

    return dictFindLinkInternal(d, key, bucket);
}

/* Set the key with link
 *
 * link:    - When `newItem` is set, `link` is ignored (it may be NULL or the
 *            bucket token from dictFindLink()). The insert position is always
 *            computed again on the current table.
 *          - When `newItem` is not set, `link` points to the link of the key.
 *          - If *link is NULL, dictFindLink() will be called to locate the key.
 *          - On return, get updated to the link of the key.
 *
 * newItem: 1 = Add a key with a new dictEntry.
 *          0 = Set a key to an existing dictEntry.
 */
void dictSetKeyAtLink(dict *d, void *key __stored_key, dictEntryLink *link, int newItem) {
    dictEntryLink dummy = NULL;
    if (link == NULL) link = &dummy;
    void *addedKey = (d->type->keyDup) ? d->type->keyDup(d, key) : key;

    if (newItem) {
        uint64_t hash = dictGetHash(d, dictStoredKey2Key(d, key));

        /* Rehash a step and make room if needed for the new key */
        _dictRehashStepIfNeeded(d, hash);
        _dictExpandIfNeeded(d);
        if (unlikely(d->ht_table[0] == NULL)) dictExpand(d, DICT_HT_INITIAL_SIZE);

        dictInsertKey(d, hash, addedKey, link);
        return;
    }

    /* Setting key of existing dictEntry (newItem == 0)*/

    if (*link == NULL) {
        *link = dictFindLink(d, key, NULL);
        assert(*link != NULL);
    }

    dictEntry **de = *link;
    if (entryIsKey(*de)) {
        /* `de` opt-out to be actually a key. Replace key but keep the lsb flags */
        *de = encodeEntryKey(d, addedKey);
    } else {
        (*de)->key = addedKey;
    }
}

void *dictFetchValue(dict *d, const void *key) {
    dictEntry *he;

    he = dictFind(d,key);
    return he ? dictGetVal(he) : NULL;
}

/* Find an element from the table. A link is returned if the element is found, and
 * the user should later call `dictTwoPhaseUnlinkFree` with it in order to unlink
 * and release it. Otherwise if the key is not found, NULL is returned. These two
 * functions should be used in pair.
 * `dictTwoPhaseUnlinkFind` pauses rehash (and bucket compaction) and
 * `dictTwoPhaseUnlinkFree` resumes it.
 *
 * We can use like this:
 *
 * dictPosition pos;
 * dictEntryLink link = dictTwoPhaseUnlinkFind(db->dict,key->ptr, &pos);
 * // Do something, but we can't modify the dict
 * dictTwoPhaseUnlinkFree(db->dict, link, &pos); // We don't need to lookup again
 *
 * If we want to find an entry before delete this entry, this an optimization to avoid
 * dictFind followed by dictDelete. i.e. the first API is a find, and it gives some info
 * to the second one to avoid repeating the lookup
 */
dictEntryLink dictTwoPhaseUnlinkFind(dict *d, const void *key, dictPosition *pos) {
    int ipos, table;

    if (dictSize(d) == 0) return NULL; /* dict is empty */

    uint64_t hash = dictGetHash(d, key);
    _dictRehashStepIfNeeded(d, hash);

    dictBucket *b = findBucketForKey(d, hash, key, &ipos, &table);
    if (!b) return NULL;
    pos->bucket = b;
    pos->pos = ipos;
    pos->table = table;
    dictPauseRehashing(d);
    return &b->slots[ipos];
}

void dictTwoPhaseUnlinkFree(dict *d, dictEntryLink plink, dictPosition *pos) {
    if (plink == NULL || *plink == NULL) return;
    dictBucket *b = pos->bucket;
    int ipos = pos->pos;
    debugAssert(plink == &b->slots[ipos] && isPositionFilled(b, ipos) &&
                !(b->chained && ipos == DICT_BUCKET_SLOTS - 1));
    dictEntry *de = *plink;

    /* Unlink, free the entry, then resume. The key may have been replaced by
     * NULL (see dbDelete) so the key is freed without decoding it here. */
    b->presence &= ~(1u << ipos);
    d->ht_used[pos->table]--;
    dictFreeKey(d, de);
    dictFreeVal(d, de);
    if (!entryIsKey(de)) zfree(decodeMaskedPtr(de));

    debugAssert(d->pauserehash > 0 && d->pausecompact > 0);
    d->pauserehash--;
    d->pausecompact--;
    if (b->chained && d->pausecompact == 0)
        fillBucketHole(d, b, ipos, pos->table);
    _dictShrinkIfNeeded(d);
    dictResumeFinalize(d);
}

void dictSetKey(dict *d, dictEntry* de, void *key __stored_key) {
    assert(!d->type->no_value);
    if (d->type->keyDup)
        de->key = d->type->keyDup(d, key);
    else
        de->key = key;
}

void dictSetVal(dict *d, dictEntry *de, void *val) {
    assert(entryHasValue(de));
    de->v.val = d->type->valDup ? d->type->valDup(d, val) : val;
}

void dictSetSignedIntegerVal(dictEntry *de, int64_t val) {
    assert(entryHasValue(de));
    de->v.s64 = val;
}

void dictSetUnsignedIntegerVal(dictEntry *de, uint64_t val) {
    assert(entryHasValue(de));
    de->v.u64 = val;
}

void dictSetDoubleVal(dictEntry *de, double val) {
    assert(entryHasValue(de));
    de->v.d = val;
}

int64_t dictIncrSignedIntegerVal(dictEntry *de, int64_t val) {
    assert(entryHasValue(de));
    return de->v.s64 += val;
}

uint64_t dictIncrUnsignedIntegerVal(dictEntry *de, uint64_t val) {
    assert(entryHasValue(de));
    return de->v.u64 += val;
}

double dictIncrDoubleVal(dictEntry *de, double val) {
    assert(entryHasValue(de));
    return de->v.d += val;
}

int dictEntryIsKey(const dictEntry *de) {
    return entryIsKey(de);
}

void *dictGetKey(const dictEntry *de) {
    return entryStoredKey(de);
}

void *dictGetVal(const dictEntry *de) {
    assert(entryHasValue(de));
    return de->v.val;
}

int64_t dictGetSignedIntegerVal(const dictEntry *de) {
    assert(entryHasValue(de));
    return de->v.s64;
}

uint64_t dictGetUnsignedIntegerVal(const dictEntry *de) {
    assert(entryHasValue(de));
    return de->v.u64;
}

double dictGetDoubleVal(const dictEntry *de) {
    assert(entryHasValue(de));
    return de->v.d;
}

/* Returns a mutable reference to the value as a double within the entry. */
double *dictGetDoubleValPtr(dictEntry *de) {
    assert(entryHasValue(de));
    return &de->v.d;
}

/* Returns the memory usage in bytes of the dict, excluding the size of the keys
 * and values: the hash tables (including child buckets) plus, for dicts with
 * values, one dictEntry per element. no_value dicts store the key directly in
 * the bucket slot and allocate nothing per element. */
size_t dictMemUsage(const dict *d) {
    return dictSize(d) * dictEntryMemUsage(d->type->no_value) + dictTableMemUsage(d);
}

size_t dictEntryMemUsage(int noValueDict) {
    return noValueDict ? 0 : sizeof(dictEntry);
}

/* A fingerprint is a 64 bit number that represents the state of the dictionary
 * at a given time, it's just a few dict properties xored together.
 * When an unsafe iterator is initialized, we get the dict fingerprint, and check
 * the fingerprint again when the iterator is released.
 * If the two fingerprints are different it means that the user of the iterator
 * performed forbidden operations against the dictionary while iterating. */
unsigned long long dictFingerprint(dict *d) {
    unsigned long long integers[8], hash = 0;
    int j;

    integers[0] = (long) d->ht_table[0];
    integers[1] = d->ht_size_exp[0];
    integers[2] = d->ht_used[0];
    integers[3] = (long) d->ht_table[1];
    integers[4] = d->ht_size_exp[1];
    integers[5] = d->ht_used[1];
    integers[6] = d->child_buckets[0];
    integers[7] = d->child_buckets[1];

    /* We hash N integers by summing every successive integer with the integer
     * hashing of the previous sum. Basically:
     *
     * Result = hash(hash(hash(int1)+int2)+int3) ...
     *
     * This way the same set of integers in a different order will (likely) hash
     * to a different number. */
    for (j = 0; j < 8; j++) {
        hash += integers[j];
        /* For the hashing step we use Tomas Wang's 64 bit integer hash. */
        hash = (~hash) + (hash << 21); // hash = (hash << 21) - hash - 1;
        hash = hash ^ (hash >> 24);
        hash = (hash + (hash << 3)) + (hash << 8); // hash * 265
        hash = hash ^ (hash >> 14);
        hash = (hash + (hash << 2)) + (hash << 4); // hash * 21
        hash = hash ^ (hash >> 28);
        hash = hash + (hash << 31);
    }
    return hash;
}

void dictInitIterator(dictIterator *iter, dict *d)
{
    iter->d = d;
    iter->table = 0;
    iter->index = -1;
    iter->pos = 0;
    iter->bucket = NULL;
    iter->safe = 0;
    iter->done = 0;
    iter->ret_last_unchained = 0;
    iter->skip_pos0 = 0;
    iter->last_seen_size = 0;
}

void dictInitSafeIterator(dictIterator *iter, dict *d)
{
    dictInitIterator(iter, d);
    iter->safe = 1;
}

void dictResetIterator(dictIterator *iter)
{
    if (!(iter->index == -1 && iter->table == 0)) {
        if (iter->safe)
            dictResumeRehashing(iter->d);
        else
            assert(iter->fingerprint == dictFingerprint(iter->d));
    }
}

dictIterator *dictGetIterator(dict *d)
{
    dictIterator *iter = zmalloc(sizeof(*iter));
    dictInitIterator(iter, d);
    return iter;
}

dictIterator *dictGetSafeIterator(dict *d) {
    dictIterator *i = dictGetIterator(d);

    i->safe = 1;
    return i;
}

dictEntry *dictNext(dictIterator *iter)
{
    dict *d = iter->d;
    if (iter->done) return NULL;

    while (1) {
        dictBucket *b;
        if (iter->index == -1 && iter->table == 0) {
            /* First call. A safe iterator pauses rehashing and bucket
             * compaction, so that entries don't move. */
            if (iter->safe) {
                dictPauseRehashing(d);
                iter->last_seen_size = d->ht_used[0];
            } else {
                iter->fingerprint = dictFingerprint(d);
            }
            iter->index = 0;
            if (d->ht_table[0] == NULL) {
                iter->done = 1; /* no table, nothing to iterate */
                return NULL;
            }
            /* skip the rehashed buckets in table[0] */
            if (dictIsRehashing(d)) iter->index = d->rehashidx;
            if ((size_t)iter->index >= numBuckets(d->ht_size_exp[0])) {
                iter->done = 1;
                return NULL;
            }
            iter->bucket = &d->ht_table[0][iter->index];
            iter->pos = 0;
        } else {
            /* Next position in the bucket, or child bucket, or next index. */
            iter->pos++;
            b = iter->bucket;
            if (b->chained && iter->pos >= DICT_BUCKET_SLOTS - 1) {
                /* If we returned the last slot of this bucket when it was not
                 * chained yet, that entry now lives in slot 0 of the child. */
                iter->skip_pos0 = iter->ret_last_unchained;
                iter->pos = 0;
                iter->bucket = getChildBucket(b);
            } else if (iter->pos >= DICT_BUCKET_SLOTS) {
                /* Bucket chain done. A safe iterator compacts the chain it is
                 * leaving if it is the only reason compaction is paused. */
                if (iter->safe) {
                    if (d->pausecompact == 1 && d->ht_used[iter->table] < iter->last_seen_size)
                        compactBucketChain(d, iter->index, iter->table);
                    iter->last_seen_size = d->ht_used[iter->table];
                }
                iter->skip_pos0 = 0;
                iter->pos = 0;
                iter->index++;
                if ((size_t)iter->index >= numBuckets(d->ht_size_exp[iter->table])) {
                    if (dictIsRehashing(d) && iter->table == 0) {
                        iter->table++;
                        iter->index = 0;
                    } else {
                        iter->done = 1;
                        return NULL;
                    }
                }
                iter->bucket = &d->ht_table[iter->table][iter->index];
            }
        }
        b = iter->bucket;
        iter->ret_last_unchained = 0;
        if (!isPositionFilled(b, iter->pos)) {
            if (iter->skip_pos0 && iter->pos == 0) iter->skip_pos0 = 0;
            continue;
        }
        if (iter->skip_pos0) {
            iter->skip_pos0 = 0;
            if (iter->pos == 0) continue;
        }
        iter->ret_last_unchained = (iter->pos == DICT_BUCKET_SLOTS - 1 && !b->chained);
        return b->slots[iter->pos];
    }
}

void dictReleaseIterator(dictIterator *iter)
{
    dictResetIterator(iter);
    zfree(iter);
}

/* Number of entries in a bucket chain. */
static unsigned chainEntries(const dictBucket *b) {
    unsigned n = 0;
    for (; b; b = getChildBucket(b)) n += __builtin_popcount(b->presence);
    return n;
}

/* Return a random entry from the hash table. Useful to
 * implement randomized algorithms */
dictEntry *dictGetRandomKey(dict *d)
{
    dictBucket *b;
    unsigned long h;
    unsigned n;

    if (dictSize(d) == 0) return NULL;
    if (dictIsRehashing(d)) _dictRehashStep(d);
    do {
        if (dictIsRehashing(d)) {
            unsigned long s0 = numBuckets(d->ht_size_exp[0]);
            /* We are sure there are no elements in indexes from 0
             * to rehashidx-1 */
            h = d->rehashidx + (randomULong() % (dictBuckets(d) - d->rehashidx));
            b = (h >= s0) ? &d->ht_table[1][h - s0] : &d->ht_table[0][h];
        } else {
            h = randomULong() & expToMask(d->ht_size_exp[0]);
            b = &d->ht_table[0][h];
        }
        /* A chain may be non empty but have no entries (holes left by deletes). */
    } while ((n = chainEntries(b)) == 0);

    /* Now we found a non empty chain and we need to get a random element from
     * it. The only sane way to do so is counting the elements and select a
     * random index. */
    unsigned target = random() % n;
    for (; b; b = getChildBucket(b)) {
        unsigned c = __builtin_popcount(b->presence);
        if (target < c) {
            bucketBits present = b->presence;
            while (target--) present &= present - 1;
            return b->slots[__builtin_ctz(present)];
        }
        target -= c;
    }
    assert(0);
    return NULL;
}

/* This function samples the dictionary to return a few keys from random
 * locations.
 *
 * It does not guarantee to return all the keys specified in 'count', nor
 * it does guarantee to return non-duplicated elements, however it will make
 * some effort to do both things.
 *
 * Returned pointers to hash table entries are stored into 'des' that
 * points to an array of dictEntry pointers. The array must have room for
 * at least 'count' elements, that is the argument we pass to the function
 * to tell how many random elements we need.
 *
 * The function returns the number of items stored into 'des', that may
 * be less than 'count' if the hash table has less than 'count' elements
 * inside, or if not enough elements were found in a reasonable amount of
 * steps.
 *
 * Note that this function is not suitable when you need a good distribution
 * of the returned items, but only when you need to "sample" a given number
 * of continuous elements to run some kind of algorithm or to produce
 * statistics. However the function is much faster than dictGetRandomKey()
 * at producing N elements. */
unsigned int dictGetSomeKeys(dict *d, dictEntry **des, unsigned int count) {
    unsigned long j; /* internal hash table id, 0 or 1. */
    unsigned long tables; /* 1 or 2 tables? */
    unsigned long stored = 0, maxsizemask;
    unsigned long maxsteps;

    if (dictSize(d) < count) count = dictSize(d);
    maxsteps = count*10;

    /* Try to do a rehashing work proportional to 'count'. */
    for (j = 0; j < count; j++) {
        if (dictIsRehashing(d))
            _dictRehashStep(d);
        else
            break;
    }

    tables = dictIsRehashing(d) ? 2 : 1;
    maxsizemask = expToMask(d->ht_size_exp[0]);
    if (tables > 1 && maxsizemask < expToMask(d->ht_size_exp[1]))
        maxsizemask = expToMask(d->ht_size_exp[1]);

    /* Pick a random point inside the larger table. */
    unsigned long i = randomULong() & maxsizemask;
    unsigned long emptylen = 0; /* Continuous empty buckets so far. */
    while(stored < count && maxsteps--) {
        for (j = 0; j < tables; j++) {
            /* Invariant of the dict.c rehashing: up to the indexes already
             * visited in ht[0] during the rehashing, there are no populated
             * buckets, so we can skip ht[0] for indexes between 0 and idx-1. */
            if (tables == 2 && j == 0 && i < (unsigned long) d->rehashidx) {
                /* Moreover, if we are currently out of range in the second
                 * table, there will be no elements in both tables up to
                 * the current rehashing index, so we jump if possible.
                 * (this happens when going from big to small table). */
                if (i >= numBuckets(d->ht_size_exp[1]))
                    i = d->rehashidx;
                else
                    continue;
            }
            if (i >= numBuckets(d->ht_size_exp[j])) continue; /* Out of range for this table. */
            dictBucket *b = &d->ht_table[j][i];

            /* Count contiguous empty buckets, and jump to other
             * locations if they reach 'count' (with a minimum of 5). */
            if (bucketIsEmpty(b)) {
                emptylen++;
                if (emptylen >= 5 && emptylen > count) {
                    i = randomULong() & maxsizemask;
                    emptylen = 0;
                }
            } else {
                emptylen = 0;
                for (; b; b = getChildBucket(b)) {
                    bucketBits present = b->presence;
                    while (present) {
                        dictEntry *he = b->slots[__builtin_ctz(present)];
                        present &= present - 1;
                        /* Collect all the elements of the buckets found non empty while iterating.
                         * To avoid the issue of being unable to sample the end of a long chain,
                         * we utilize the Reservoir Sampling algorithm to optimize the sampling process.
                         * This means that even when the maximum number of samples has been reached,
                         * we continue sampling until we reach the end of the chain.
                         * See https://en.wikipedia.org/wiki/Reservoir_sampling. */
                        if (stored < count) {
                            des[stored] = he;
                        } else {
                            unsigned long r = randomULong() % (stored + 1);
                            if (r < count) des[r] = he;
                        }
                        stored++;
                    }
                }
                if (stored >= count) goto end;
            }
        }
        i = (i+1) & maxsizemask;
    }

end:
    return stored > count ? count : stored;
}

/* Reallocate the dictEntry, key and value allocations of a bucket chain, and
 * the child buckets, using the provided allocation functions in order to
 * defrag them. The top-level bucket array is defragged by dictDefragTables(). */
static void dictDefragChain(dict *d, dictBucket *root, dictDefragFunctions *defragfns) {
    dictDefragAllocFunction *defragalloc = defragfns->defragAlloc;
    dictDefragAllocFunction *defragkey = defragfns->defragKey;
    dictDefragAllocFunction *defragval = defragfns->defragVal;
    dictBucket *b = root;
    while (b) {
        bucketBits present = b->presence;
        while (present) {
            int pos = __builtin_ctz(present);
            present &= present - 1;
            dictEntry *de = b->slots[pos], *newde = NULL;
            void *newkey = defragkey ? defragkey(entryStoredKey(de)) : NULL;
            if (d->type->no_value) {
                if (newkey) b->slots[pos] = encodeEntryKey(d, newkey);
            } else {
                void *newval = defragval ? defragval(dictGetVal(de)) : NULL;
                assert(entryIsNormal(de));
                newde = defragalloc(de);
                if (newde) de = newde;
                if (newkey) de->key = newkey;
                if (newval) de->v.val = newval;
                if (newde) b->slots[pos] = newde;
            }
        }
        dictBucket *next = getChildBucket(b);
        if (next && defragalloc) {
            dictBucket *newnext = defragalloc(next);
            if (newnext) {
                setChildBucket(b, newnext);
                next = newnext;
            }
        }
        b = next;
    }
}

/* This is like dictGetRandomKey() from the POV of the API, but will do more
 * work to ensure a better distribution of the returned element.
 *
 * This function improves the distribution because the dictGetRandomKey()
 * problem is that it selects a random bucket, then it selects a random
 * element from the chain in the bucket. However elements being in different
 * chain lengths will have different probabilities of being reported. With
 * this function instead what we do is to consider a "linear" range of the table
 * that may be constituted of N buckets with chains of different lengths
 * appearing one after the other. Then we report a random element in the range.
 * In this way we smooth away the problem of different chain lengths. */
#define GETFAIR_NUM_ENTRIES 15
dictEntry *dictGetFairRandomKey(dict *d) {
    dictEntry *entries[GETFAIR_NUM_ENTRIES];
    unsigned int count = dictGetSomeKeys(d,entries,GETFAIR_NUM_ENTRIES);
    /* Note that dictGetSomeKeys() may return zero elements in an unlucky
     * run() even if there are actually elements inside the hash table. So
     * when we get zero, we call the true dictGetRandomKey() that will always
     * yield the element if the hash table has at least one. */
    if (count == 0) return dictGetRandomKey(d);
    unsigned int idx = rand() % count;
    return entries[idx];
}

/* Function to reverse bits. Algorithm from:
 * http://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel */
static unsigned long rev(unsigned long v) {
    unsigned long s = CHAR_BIT * sizeof(v); // bit size; must be power of 2
    unsigned long mask = ~0UL;
    while ((s >>= 1) > 0) {
        mask ^= (mask << s);
        v = ((v >> s) & mask) | ((v << s) & ~mask);
    }
    return v;
}

/* dictScan() is used to iterate over the elements of a dictionary.
 *
 * Iterating works the following way:
 *
 * 1) Initially you call the function using a cursor (v) value of 0.
 * 2) The function performs one step of the iteration, and returns the
 *    new cursor value you must use in the next call.
 * 3) When the returned cursor is 0, the iteration is complete.
 *
 * The function guarantees all elements present in the
 * dictionary get returned between the start and end of the iteration.
 * However it is possible some elements get returned multiple times.
 *
 * For every element returned, the callback argument 'fn' is
 * called with 'privdata' as first argument, the dictionary entry
 * 'de' as second argument and a link to the slot of the entry as third.
 *
 * Scan callback rules: the callback may delete the entry that was passed to it
 * (which compacts the bucket chain right away; the entry that takes its place
 * is then emitted too) and may insert or replace entries. Deleting entries
 * other than the one passed to the callback is not supported.
 *
 * HOW IT WORKS.
 *
 * The iteration algorithm was designed by Pieter Noordhuis.
 * The main idea is to increment a cursor starting from the higher order
 * bits. That is, instead of incrementing the cursor normally, the bits
 * of the cursor are reversed, then the cursor is incremented, and finally
 * the bits are reversed again.
 *
 * This strategy is needed because the hash table may be resized between
 * iteration calls.
 *
 * dict.c hash tables are always power of two in size (in top-level buckets),
 * and an entry lives in the chain of the bucket given by the bitwise AND
 * between Hash(key) and SIZE-1.
 *
 * For example if the current table has 16 buckets, the mask is
 * (in binary) 1111. The position of a key in the hash table will always be
 * the last four bits of the hash output, and so forth.
 *
 * WHAT HAPPENS IF THE TABLE CHANGES IN SIZE?
 *
 * If the hash table grows, elements can go anywhere in one multiple of
 * the old bucket: for example let's say we already iterated with
 * a 4 bit cursor 1100 (the mask is 1111 because hash table size = 16).
 *
 * If the hash table will be resized to 64 buckets, then the new mask will
 * be 111111. The new buckets you obtain by substituting in ??1100
 * with either 0 or 1 can be targeted only by keys we already visited
 * when scanning the bucket 1100 in the smaller hash table.
 *
 * By iterating the higher bits first, because of the inverted counter, the
 * cursor does not need to restart if the table size gets bigger. It will
 * continue iterating using cursors without '1100' at the end, and also
 * without any other combination of the final 4 bits already explored.
 *
 * Similarly when the table size shrinks over time, for example going from
 * 16 to 8, if a combination of the lower three bits (the mask for size 8
 * is 111) were already completely explored, it would not be visited again
 * because we are sure we tried, for example, both 0111 and 1111 (all the
 * variations of the higher bit) so we don't need to test it again.
 *
 * WAIT... YOU HAVE *TWO* TABLES DURING REHASHING!
 *
 * Yes, this is true, but we always iterate the smaller table first, then
 * we test all the expansions of the current cursor into the larger
 * table. For example if the current cursor is 101 and we also have a
 * larger table of size 16, we also test (0)101 and (1)101 inside the larger
 * table. This reduces the problem back to having only one table, where
 * the larger one, if it exists, is just an expansion of the smaller one.
 *
 * LIMITATIONS
 *
 * This iterator is completely stateless, and this is a huge advantage,
 * including no additional memory used.
 *
 * The disadvantages resulting from this design are:
 *
 * 1) It is possible we return elements more than once. However this is usually
 *    easy to deal with in the application level.
 * 2) The iterator must return multiple elements per call, as it needs to always
 *    return all the keys in a given bucket chain, and all the expansions, so
 *    we are sure we don't miss keys moving during rehashing.
 * 3) The reverse cursor is somewhat hard to understand at first, but this
 *    comment is supposed to help.
 */
unsigned long dictScan(dict *d,
                       unsigned long v,
                       dictScanFunction *fn,
                       void *privdata)
{
    return dictScanDefrag(d, v, fn, NULL, privdata);
}

/* Emit the entries of the chain at (table, idx) to the scan callback. */
static void dictScanChain(dict *d, int table, size_t idx, dictScanFunction *fn,
                          dictDefragFunctions *defragfns, void *privdata)
{
    dictBucket *root = &d->ht_table[table][idx];
    size_t used_before = d->ht_used[table];

    if (defragfns) dictDefragChain(d, root, defragfns);

    dictBucket *b = root;
    while (b) {
        /* numBucketPositions() is re-evaluated each time: a callback that
         * deletes its entry compacts the chain, which may move an entry from
         * the end of the chain into the slot just visited (and possibly turn
         * this bucket into an unchained one). Either way that entry hasn't
         * been emitted yet, so we look at the same position again. */
        for (int pos = 0; pos < numBucketPositions(b); pos++) {
            while (isPositionFilled(b, pos)) {
                dictEntry *de = b->slots[pos];
                dictScanState scan = {activeScan, d, &b->slots[pos], 0};
                activeScan = &scan;
                fn(privdata, de, &b->slots[pos]);
                activeScan = scan.prev;
                /* If the callback deleted the entry and the chain was compacted,
                 * the slot may hold a different, not yet emitted, entry. A
                 * callback that merely replaces the entry is not
                 * emitted again. */
                if (!scan.compacted || !isPositionFilled(b, pos)) break;
            }
        }
        b = getChildBucket(b);
    }

    /* If other entries were deleted, fill the holes. */
    if (d->ht_used[table] < used_before && d->pausecompact == 0)
        compactBucketChain(d, idx, table);
}

/* Like dictScan, but additionally reallocates the memory used by the dict
 * entries using the provided allocation function. This feature was added for
 * the active defrag feature.
 *
 * The 'defragfns' callbacks are called with a pointer to memory that callback
 * can reallocate. The callbacks should return a new memory address or NULL,
 * where NULL means that no reallocation happened and the old memory is still
 * valid. */
unsigned long dictScanDefrag(dict *d,
                             unsigned long v,
                             dictScanFunction *fn,
                             dictDefragFunctions *defragfns,
                             void *privdata)
{
    int htidx0, htidx1;
    unsigned long m0, m1;

    if (dictSize(d) == 0) return 0;

    /* This is needed in case the scan callback tries to do dictFind or alike. */
    pauseRehashOnly(d);

    if (!dictIsRehashing(d)) {
        htidx0 = 0;
        m0 = expToMask(d->ht_size_exp[htidx0]);
        dictScanChain(d, htidx0, v & m0, fn, defragfns, privdata);

        /* Set unmasked bits so incrementing the reversed cursor
         * operates on the masked bits */
        v |= ~m0;

        /* Increment the reverse cursor */
        v = rev(v);
        v++;
        v = rev(v);

    } else {
        htidx0 = 0;
        htidx1 = 1;

        /* Make sure t0 is the smaller and t1 is the bigger table */
        if (numBuckets(d->ht_size_exp[htidx0]) > numBuckets(d->ht_size_exp[htidx1])) {
            htidx0 = 1;
            htidx1 = 0;
        }

        m0 = expToMask(d->ht_size_exp[htidx0]);
        m1 = expToMask(d->ht_size_exp[htidx1]);

        dictScanChain(d, htidx0, v & m0, fn, defragfns, privdata);

        /* Iterate over indices in larger table that are the expansion
         * of the index pointed to by the cursor in the smaller table */
        do {
            dictScanChain(d, htidx1, v & m1, fn, defragfns, privdata);

            /* Increment the reverse cursor not covered by the smaller mask.*/
            v |= ~m1;
            v = rev(v);
            v++;
            v = rev(v);

            /* Continue while bits covered by mask difference is non-zero */
        } while (v & (m0 ^ m1));
    }

    resumeRehashOnly(d);

    return v;
}

/* ------------------------- private functions ------------------------------ */

/* Because we may need to allocate huge memory chunk at once when dict
 * resizes, we will check this allocation is allowed or not if the dict
 * type has resizeAllowed member function. */
static int dictTypeResizeAllowed(dict *d, size_t size) {
    if (d->type->resizeAllowed == NULL) return 1;
    return d->type->resizeAllowed(
                    tableBytes(nextBucketExp(size)),
                    (double)d->ht_used[0] / (numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS));
}

/* Returning DICT_OK indicates a successful expand or the dictionary is undergoing rehashing,
 * and there is nothing else we need to do about this dictionary currently. While DICT_ERR indicates
 * that expand has not been triggered (may be try shrinking?)*/
int dictExpandIfNeeded(dict *d) {
    /* Incremental rehashing already in progress. Return. */
    if (dictIsRehashing(d)) return DICT_OK;

    /* If the hash table is empty expand it to the initial size. */
    if (numBuckets(d->ht_size_exp[0]) == 0) {
        dictExpand(d, DICT_HT_INITIAL_SIZE);
        return DICT_OK;
    }

    /* If we reached the 1:1 ratio, and we are allowed to resize the hash
     * table (global setting) or we should avoid it but the ratio between
     * elements/slots is over the "safe" threshold, we resize making room
     * for one more element. */
    size_t capacity = numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS;
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if ((can_resize == DICT_RESIZE_ENABLE &&
         d->ht_used[0] >= capacity) ||
        (can_resize != DICT_RESIZE_FORBID &&
         d->ht_used[0] >= dict_force_resize_ratio * capacity))
    {
        if (dictTypeResizeAllowed(d, d->ht_used[0] + 1))
            dictExpand(d, d->ht_used[0] + 1);
        return DICT_OK;
    }
    return DICT_ERR;
}

/* Expand the hash table if needed (OK=Expanded, ERR=Not expanded) */
static int _dictExpandIfNeeded(dict *d) {
    /* Automatic resizing is disallowed. Return */
    if (d->pauseAutoResize > 0) return DICT_ERR;

    return dictExpandIfNeeded(d);
}

/* Returning DICT_OK indicates a successful shrinking or the dictionary is undergoing rehashing,
 * and there is nothing else we need to do about this dictionary currently. While DICT_ERR indicates
 * that shrinking has not been triggered (may be try expanding?)*/
int dictShrinkIfNeeded(dict *d) {
    /* Incremental rehashing already in progress. Return. */
    if (dictIsRehashing(d)) return DICT_OK;

    /* If the table is a single bucket, don't shrink it. */
    if (d->ht_size_exp[0] <= DICT_HT_INITIAL_EXP) return DICT_OK;

    /* If we reached below 1:8 elements/slots ratio, and we are allowed to resize
     * the hash table (global setting) or we should avoid it but the ratio is below 1:32,
     * we'll trigger a resize of the hash table. */
    size_t capacity = numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS;
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if ((can_resize == DICT_RESIZE_ENABLE &&
         d->ht_used[0] * HASHTABLE_MIN_FILL <= capacity) ||
        (can_resize != DICT_RESIZE_FORBID &&
         d->ht_used[0] * HASHTABLE_MIN_FILL * dict_force_resize_ratio <= capacity))
    {
        if (dictTypeResizeAllowed(d, d->ht_used[0]))
            dictShrink(d, d->ht_used[0]);
        return DICT_OK;
    }
    return DICT_ERR;
}

static void _dictShrinkIfNeeded(dict *d)
{
    /* Automatic resizing is disallowed. Return */
    if (d->pauseAutoResize > 0) return;

    /* While iterating, scanning or holding a link, remember that a delete
     * happened: the last resume shrinks if it is still needed. */
    if (d->pauserehash > 0 || d->pausecompact > 0) {
        d->pending_shrink = 1;
        return;
    }

    dictShrinkIfNeeded(d);
}

void dictEmpty(dict *d, void(callback)(dict*)) {
    /* Someone may be monitoring a dict that started rehashing, before
     * destroying the dict fake completion. */
    if (dictIsRehashing(d) && d->type->rehashingCompleted)
        d->type->rehashingCompleted(d);

    /* Subtract the size of all buckets. */
    tableMemChanged(d, -(long long)dictTableMemUsage(d));

    _dictClear(d,0,callback);
    _dictClear(d,1,callback);
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pausecompact = 0;
    d->pauseAutoResize = 0;
    d->pending_shrink = 0;
}

void dictSetResizeEnabled(dictResizeEnable enable) {
    atomicSet(dict_can_resize, enable);
}

/* Compiler inlines this for internal calls within dict.c (verified with -O3). */
uint64_t dictGetHash(dict *d, const void *key) {
    return d->type->hashFunction(key);
}

/* Provides the old and new ht size, in top-level buckets, for a given dictionary
 * during rehashing. This method should only be invoked during
 * initialization/rehashing. */
void dictRehashingInfo(dict *d, unsigned long long *from_size, unsigned long long *to_size) {
    /* Invalid method usage if rehashing isn't ongoing. */
    assert(dictIsRehashing(d));
    *from_size = numBuckets(d->ht_size_exp[0]);
    *to_size = numBuckets(d->ht_size_exp[1]);
}

/* ------------------------------- Debugging ---------------------------------*/
#define DICT_STATS_VECTLEN 50
void dictFreeStats(dictStats *stats) {
    zfree(stats->clvector);
    zfree(stats);
}

void dictCombineStats(dictStats *from, dictStats *into) {
    into->toplevelBuckets += from->toplevelBuckets;
    into->childBuckets += from->childBuckets;
    into->maxChainLen = (from->maxChainLen > into->maxChainLen) ? from->maxChainLen : into->maxChainLen;
    into->htSize += from->htSize;
    into->htUsed += from->htUsed;
    for (int i = 0; i < DICT_STATS_VECTLEN; i++) {
        into->clvector[i] += from->clvector[i];
    }
}

dictStats *dictGetStatsHt(dict *d, int htidx, int full) {
    unsigned long *clvector = zcalloc(sizeof(unsigned long) * DICT_STATS_VECTLEN);
    dictStats *stats = zcalloc(sizeof(dictStats));
    stats->htidx = htidx;
    stats->clvector = clvector;
    stats->toplevelBuckets = numBuckets(d->ht_size_exp[htidx]);
    stats->childBuckets = d->child_buckets[htidx];
    stats->htSize = numBuckets(d->ht_size_exp[htidx]) * DICT_BUCKET_SLOTS;
    stats->htUsed = d->ht_used[htidx];
    if (!full) return stats;
    /* Compute stats about bucket chain lengths (in child buckets). */
    for (size_t idx = 0; idx < numBuckets(d->ht_size_exp[htidx]); idx++) {
        const dictBucket *b = &d->ht_table[htidx][idx];
        unsigned long chainlen = 0;
        while (b->chained) {
            chainlen++;
            b = getChildBucket(b);
        }
        if (chainlen > stats->maxChainLen) stats->maxChainLen = chainlen;
        clvector[(chainlen < DICT_STATS_VECTLEN) ? chainlen : (DICT_STATS_VECTLEN-1)]++;
    }

    return stats;
}

/* Generates human readable stats. */
size_t dictGetStatsMsg(char *buf, size_t bufsize, dictStats *stats, int full) {
    if (stats->htUsed == 0) {
        return snprintf(buf,bufsize,
            "Hash table %d stats (%s):\n"
            "No stats available for empty dictionaries\n",
            stats->htidx, (stats->htidx == 0) ? "main hash table" : "rehashing target");
    }
    size_t l = 0;
    l += snprintf(buf + l, bufsize - l,
                  "Hash table %d stats (%s):\n"
                  " table size: %lu\n"
                  " number of elements: %lu\n",
                  stats->htidx, (stats->htidx == 0) ? "main hash table" : "rehashing target",
                  stats->htSize, stats->htUsed);
    if (full) {
        l += snprintf(buf + l, bufsize - l,
                      " top-level buckets: %lu\n"
                      " child buckets: %lu\n"
                      " max chain length: %lu\n"
                      " avg chain length: %.02f\n"
                      " Chain length distribution (child buckets per chain):\n",
                      stats->toplevelBuckets, stats->childBuckets, stats->maxChainLen,
                      (float) stats->childBuckets / stats->toplevelBuckets);

        for (unsigned long i = 0; i < DICT_STATS_VECTLEN - 1; i++) {
            if (stats->clvector[i] == 0) continue;
            if (l >= bufsize) break;
            l += snprintf(buf + l, bufsize - l,
                          "   %ld: %ld (%.02f%%)\n",
                          i, stats->clvector[i], ((float) stats->clvector[i] / stats->toplevelBuckets) * 100);
        }
    }

    /* Make sure there is a NULL term at the end. */
    buf[bufsize-1] = '\0';
    /* Unlike snprintf(), return the number of characters actually written. */
    return strlen(buf);
}

void dictGetStats(char *buf, size_t bufsize, dict *d, int full) {
    size_t l;
    char *orig_buf = buf;
    size_t orig_bufsize = bufsize;

    dictStats *mainHtStats = dictGetStatsHt(d, 0, full);
    l = dictGetStatsMsg(buf, bufsize, mainHtStats, full);
    dictFreeStats(mainHtStats);
    buf += l;
    bufsize -= l;
    if (dictIsRehashing(d) && bufsize > 0) {
        dictStats *rehashHtStats = dictGetStatsHt(d, 1, full);
        dictGetStatsMsg(buf, bufsize, rehashHtStats, full);
        dictFreeStats(rehashHtStats);
    }
    /* Make sure there is a NULL term at the end. */
    orig_buf[orig_bufsize-1] = '\0';
}

static int dictDefaultCompare(dictCmpCache *cache, const void *key1, const void *key2) {
    (void)(cache); /*unused*/
    return key1 == key2;
}

/* ------------------------------ Misc API ---------------------------------- */

/* Number of entry slots in the hash tables (both while rehashing). Used by
 * callers (e.g. expire sampling) to estimate how full a dict is. */
unsigned long dictSlots(const dict *d) {
    return dictBuckets(d) * DICT_BUCKET_SLOTS;
}

/* Defrag the dict struct and its hash tables with `defragfn`, which receives an
 * allocation and returns a new one, or NULL if it was not moved. Child buckets
 * are defragged by dictScanDefrag().
 *
 * Returns the new dict pointer if the dict struct itself was moved (the old
 * pointer was released and must not be accessed), NULL otherwise. */
dict *dictDefragTables(dict *d, void *(*defragfn)(void *)) {
    dict *ret = NULL;
    void *newtable;
    if ((ret = defragfn(d)))
        d = ret;
    if (!d->ht_table[0]) return ret; /* created but unused */
    if ((newtable = defragfn(d->ht_table[0])))
        d->ht_table[0] = newtable;
    if (d->ht_table[1] && (newtable = defragfn(d->ht_table[1])))
        d->ht_table[1] = newtable;
    return ret;
}

/* Bytes of the top-level array of the table being rehashed away from. Only
 * meaningful while rehashing (and valid inside the rehashingStarted/Completed
 * callbacks). Child buckets are not included: they are freed one chain at a
 * time as the rehash progresses. */
size_t dictRehashingMemUsage(const dict *d) {
    return tableBytes(d->ht_size_exp[0]);
}

/* Call fn(ptr, size) for every top-level hash table allocation of the dict. */
void dictDismissTables(dict *d, void (*fn)(void *ptr, size_t size)) {
    if (!d) return;
    fn(d->ht_table[0], tableBytes(d->ht_size_exp[0]));
    fn(d->ht_table[1], tableBytes(d->ht_size_exp[1]));
}

/* ------------------------- Prefetch state machine -------------------------- */

enum { PF_BUCKET, PF_MATCH, PF_CANDIDATE, PF_ENTRY_KEY, PF_ENTRY_VALUE, PF_DONE };

void dictPrefetchInit(dictPrefetchState *st, dict *d, const void *key) {
    st->d = d;
    st->key = key;
    st->ht_idx = -1;
    st->bucket = NULL;
    st->current_entry = NULL;
    st->candidates = 0;
    st->lone = 0;
    st->hash = 0;
    if (!d || dictSize(d) == 0) {
        st->stage = PF_DONE;
        st->done = 1;
        return;
    }
    /* Prefetch is skipped during loading, so ht_table[0] is never NULL when
     * dictSize() > 0 (that only happens mid-dictEmpty via _dictReset). */
    assert(d->ht_table[0]);
    st->hash = dictGetHash(d, key);
    st->stage = PF_BUCKET;
    st->done = 0;
}

/* Advance the lookup until a useful address to prefetch is produced.
 * Returns NULL, and sets st->done, when the lookup is complete. The stages
 * mirror a dictFind: bucket -> tag match (bucket is now in cache) -> candidate
 * entry -> entry key payload -> compare and entry value payload. */
void *dictPrefetchNext(dictPrefetchState *st) {
    dict *d = st->d;
    for (;;) {
        switch (st->stage) {
        case PF_BUCKET: {
            /* Pick the next table that may hold the key; none left means done. */
            int t = st->ht_idx + 1;
            size_t idx = 0;
            for (; t <= 1; t++) {
                if (t == 1 && !dictIsRehashing(d)) { t = 2; break; }
                if (d->ht_used[t] == 0) continue;
                idx = st->hash & expToMask(d->ht_size_exp[t]);
                if (t == 0 && d->rehashidx >= 0 && (long)idx < d->rehashidx) continue;
                break;
            }
            if (t > 1) {
                st->stage = PF_DONE;
                st->done = 1;
                return NULL;
            }
            st->ht_idx = t;
            st->bucket = &d->ht_table[t][idx];
            st->stage = PF_MATCH;
            return st->bucket;
        }
        case PF_MATCH: {
            dictBucket *b = st->bucket;
            st->candidates = b->presence & findHashMatches(b->hashes, hashTag(st->hash));
            st->lone = (st->candidates && (st->candidates & (st->candidates - 1)) == 0 &&
                        !b->chained && !dictIsRehashing(d));
            st->stage = PF_CANDIDATE;
            break;
        }
        case PF_CANDIDATE: {
            dictBucket *b = st->bucket;
            if (st->candidates) {
                int pos = __builtin_ctz(st->candidates);
                st->candidates &= st->candidates - 1;
                st->current_entry = b->slots[pos];
                st->stage = PF_ENTRY_KEY;
                return st->current_entry;
            }
            /* No (more) candidates here: next bucket in the chain, else the
             * next table. */
            dictBucket *child = getChildBucket(b);
            if (child) {
                st->bucket = child;
                st->stage = PF_MATCH;
                return child;
            }
            st->stage = PF_BUCKET;
            break;
        }
        case PF_ENTRY_KEY:
            st->stage = PF_ENTRY_VALUE;
            if (d->type->prefetchEntryKey) {
                void *addr = d->type->prefetchEntryKey(st->current_entry);
                if (addr) return addr;
            }
            break;
        case PF_ENTRY_VALUE: {
            const void *cmp_key = slotLookupKey(d, st->current_entry);
            /* A single tag match in an unchained, non-rehashing bucket is assumed
             * to be a hit without comparing the keys. Otherwise compare. */
            if (st->lone || dictCompareKeys(d, st->key, cmp_key)) {
                st->stage = PF_DONE;
                st->done = 1;
                if (d->type->prefetchEntryValue)
                    return d->type->prefetchEntryValue(st->current_entry);
                return NULL;
            }
            st->stage = PF_CANDIDATE;
            break;
        }
        default:
            st->done = 1;
            return NULL;
        }
    }
}

#ifdef REDIS_TEST
#include "testhelp.h"

#define UNUSED(V) ((void) V)
#define TEST(name) printf("test — %s\n", name); fflush(stdout);

uint64_t hashCallback(const void *key) {
    return dictGenHashFunction((unsigned char*)key, strlen((char*)key));
}

int compareCallback(dictCmpCache *cache, const void *key1, const void *key2) {
    int l1,l2;
    UNUSED(cache);

    l1 = strlen((char*)key1);
    l2 = strlen((char*)key2);
    if (l1 != l2) return 0;
    return memcmp(key1, key2, l1) == 0;
}

void freeCallback(dict *d, void *val) {
    UNUSED(d);

    zfree(val);
}

char *stringFromLongLong(long long value) {
    char buf[32];
    int len;
    char *s;

    len = snprintf(buf,sizeof(buf),"%lld",value);
    s = zmalloc(len+1);
    memcpy(s, buf, len);
    s[len] = '\0';
    return s;
}

char *stringFromSubstring(void) {
    #define LARGE_STRING_SIZE 10000
    #define MIN_STRING_SIZE 100
    #define MAX_STRING_SIZE 500
    static char largeString[LARGE_STRING_SIZE + 1];
    static int init = 0;
    if (init == 0) {
        /* Generate a large string */
        for (size_t i = 0; i < LARGE_STRING_SIZE; i++) {
            /* Random printable ASCII character (33 to 126) */
            largeString[i] = 33 + (rand() % 94);
        }
        /* Null-terminate the large string */
        largeString[LARGE_STRING_SIZE] = '\0';
        init = 1;
    }
    /* Randomly choose a size between minSize and maxSize */
    size_t substringSize = MIN_STRING_SIZE + (rand() % (MAX_STRING_SIZE - MIN_STRING_SIZE + 1));
    size_t startIndex = rand() % (LARGE_STRING_SIZE - substringSize + 1);
    /* Allocate memory for the substring (+1 for null terminator) */
    char *s = zmalloc(substringSize + 1);
    memcpy(s, largeString + startIndex, substringSize); // Copy the substring
    s[substringSize] = '\0'; // Null-terminate the string
    return s;
}

dictType BenchmarkDictType = {
    hashCallback,
    NULL,
    NULL,
    compareCallback,
    freeCallback,
    NULL,
    NULL
};

/* Same as BenchmarkDictType, but a no_value=1 (set-style) dict -- used to verify
 * that dictMemUsage() charges nothing per entry for dicts without values. */
static dictType BenchmarkDictTypeNoValue = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
};

/* ------------------------- verification helpers --------------------------- */

/* Sum of the tableMemChanged() deltas of the dict under test. */
static long long test_table_bytes = 0;

static void testTableMemChanged(dict *d, long long delta_bytes) {
    UNUSED(d);
    test_table_bytes += delta_bytes;
}

/* A hash function that maps every key to the same bucket chain (but still
 * produces a valid tag), to build long bucket chains. */
static uint64_t constHashCallback(const void *key) {
    UNUSED(key);
    return 0x5a5a5a5a5a5a5a5aULL;
}

/* Two keys per bucket group: low bits from a small range, so chains are
 * long but not unique. */
static uint64_t fewBucketsHashCallback(const void *key) {
    return (hashCallback(key) & 0x3) | (hashCallback(key) & 0xff00000000000000ULL);
}

static dictType verifyDictType = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
    .keys_are_odd = 0,
    .tableMemChanged = testTableMemChanged,
};

static dictType verifyDictTypeVal = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .tableMemChanged = testTableMemChanged,
};

static dictType chainDictType = {
    .hashFunction = constHashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
    .tableMemChanged = testTableMemChanged,
};

static dictType fewBucketsDictType = {
    .hashFunction = fewBucketsHashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .tableMemChanged = testTableMemChanged,
};

/* Checks the structural invariants of the dict: presence bits, tags, bucket
 * indexes, chains, counters and memory accounting. */
static void dictVerify(dict *d) {
    size_t counted[2] = {0, 0}, children[2] = {0, 0};
    for (int t = 0; t <= 1; t++) {
        if (!d->ht_table[t]) {
            assert(d->ht_size_exp[t] == -1);
            assert(d->ht_used[t] == 0 && d->child_buckets[t] == 0);
            continue;
        }
        size_t nb = numBuckets(d->ht_size_exp[t]);
        for (size_t idx = 0; idx < nb; idx++) {
            dictBucket *root = &d->ht_table[t][idx];
            if (t == 0 && dictIsRehashing(d) && (long)idx < d->rehashidx)
                assert(bucketIsEmpty(root)); /* already moved */
            for (dictBucket *b = root; b; b = getChildBucket(b)) {
                if (b != root) children[t]++;
                if (b->chained) {
                    assert(!isPositionFilled(b, DICT_BUCKET_SLOTS - 1));
                    assert(getChildBucket(b) != NULL);
                }
                for (int pos = 0; pos < DICT_BUCKET_SLOTS; pos++) {
                    if (!isPositionFilled(b, pos)) continue;
                    dictEntry *e = b->slots[pos];
                    if (d->type->no_value) assert(entryIsKey(e));
                    else assert(entryIsNormal(e));
                    uint64_t hash = dictGetHash(d, slotLookupKey(d, e));
                    assert(hashTag(hash) == b->hashes[pos]);
                    assert((hash & expToMask(d->ht_size_exp[t])) == idx);
                    counted[t]++;
                }
            }
        }
    }
    for (int t = 0; t <= 1; t++) {
        assert(counted[t] == d->ht_used[t]);
        assert(children[t] == d->child_buckets[t]);
    }
    if (dictIsRehashing(d)) {
        assert(d->ht_table[0] && d->ht_table[1]);
        assert(d->rehashidx >= 0 && (size_t)d->rehashidx <= numBuckets(d->ht_size_exp[0]));
    } else {
        assert(d->ht_table[1] == NULL && d->ht_used[1] == 0);
    }
    if (d->type->tableMemChanged == testTableMemChanged)
        assert(test_table_bytes == (long long)dictTableMemUsage(d));
}

/* Finish an ongoing rehashing. Rehashing is not forced under
 * DICT_RESIZE_AVOID, so make sure it's enabled. */
static void drainRehash(dict *d) {
    dictSetResizeEnabled(DICT_RESIZE_ENABLE);
    while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
}

/* Reference model for randomized tests: keys are "k<id>" strings. */
#define MODEL_KEYS 4000

typedef struct modelCtx {
    uint8_t present[MODEL_KEYS];
    uint32_t seen[MODEL_KEYS];
    unsigned long count;
    int delete_percent; /* in scan/iterator callbacks */
    dict *d;
} modelCtx;

static int keyId(const void *key) {
    return atoi((const char *)key + 1);
}

static char *keyForId(int id) {
    char buf[32];
    snprintf(buf, sizeof(buf), "k%d", id);
    size_t len = strlen(buf) + 1;
    char *s = zmalloc(len);
    memcpy(s, buf, len);
    return s;
}

static int modelAdd(modelCtx *m, dict *d, int id) {
    char *k = keyForId(id);
    int ret = dictAdd(d, k, (void *)(long)id);
    if (m->present[id]) { assert(ret == DICT_ERR); zfree(k); return 0; }
    assert(ret == DICT_OK);
    m->present[id] = 1;
    m->count++;
    return 1;
}

static int modelDel(modelCtx *m, dict *d, int id) {
    char *k = keyForId(id);
    int ret = dictDelete(d, k);
    zfree(k);
    if (!m->present[id]) { assert(ret == DICT_ERR); return 0; }
    assert(ret == DICT_OK);
    m->present[id] = 0;
    m->count--;
    return 1;
}

/* Verify content equals the model by iterating (safe) and by lookups. */
static void modelCheck(modelCtx *m, dict *d) {
    assert(dictSize(d) == m->count);
    memset(m->seen, 0, sizeof(m->seen));
    dictIterator it;
    dictEntry *de;
    dictInitSafeIterator(&it, d);
    while ((de = dictNext(&it))) {
        int id = keyId(dictGetKey(de));
        assert(m->present[id]);
        assert(++m->seen[id] == 1);
    }
    dictResetIterator(&it);
    for (int i = 0; i < MODEL_KEYS; i++) assert(m->seen[i] == m->present[i]);
    for (int i = 0; i < MODEL_KEYS; i += 7) {
        char *k = keyForId(i);
        assert((dictFind(d, k) != NULL) == m->present[i]);
        zfree(k);
    }
    dictVerify(d);
}

static void modelScanCb(void *privdata, const dictEntry *de, dictEntryLink plink) {
    modelCtx *m = privdata;
    UNUSED(plink);
    int id = keyId(dictGetKey(de));
    m->seen[id]++;
    if ((int)(rand() % 100) < m->delete_percent) {
        /* Deleting the entry passed to us is allowed. */
        char *k = keyForId(id);
        assert(dictDelete(m->d, k) == DICT_OK);
        zfree(k);
        m->present[id] = 0;
        m->count--;
    }
}

typedef struct scanMutationCtx {
    dict *d;
    int seen[DICT_BUCKET_SLOTS + 3];
    int deletion; /* dictDelete, dictUnlink, or two-phase unlink */
    int additions;
    int nested;
} scanMutationCtx;

static void scanMutationCb(void *privdata, const dictEntry *de, dictEntryLink plink) {
    scanMutationCtx *ctx = privdata;
    int id = keyId(dictGetKey(de));
    UNUSED(plink);
    assert(id >= 0 && id < DICT_BUCKET_SLOTS + 3);
    ctx->seen[id]++;
    if (ctx->deletion == 3) {
        /* Refilling an unchained hole is an insertion, not compaction. */
        assert(ctx->seen[id] == 1);
        assert(dictDelete(ctx->d, dictGetKey(de)) == DICT_OK);
        assert(dictAdd(ctx->d, keyForId(id), NULL) == DICT_OK);
        return;
    }
    if (id != 0) return;

    if (ctx->nested == 1) {
        ctx->nested = 2;
        assert(dictScan(ctx->d, 0, scanMutationCb, ctx) == 0);
        return;
    }
    if (ctx->deletion == 0) {
        assert(dictDelete(ctx->d, "k0") == DICT_OK);
    } else if (ctx->deletion == 1) {
        dictEntry *unlinked = dictUnlink(ctx->d, "k0");
        assert(unlinked != NULL);
        dictFreeUnlinkedEntry(ctx->d, unlinked);
    } else {
        dictPosition pos;
        dictEntryLink link = dictTwoPhaseUnlinkFind(ctx->d, "k0", &pos);
        assert(link != NULL);
        dictTwoPhaseUnlinkFree(ctx->d, link, &pos);
    }
    for (int i = 0; i < ctx->additions; i++)
        assert(dictAdd(ctx->d, keyForId(DICT_BUCKET_SLOTS + 1 + i), NULL) == DICT_OK);
}

#define start_benchmark() start = timeInMilliseconds()
#define end_benchmark(msg) do { \
    elapsed = timeInMilliseconds()-start; \
    printf(msg ": %ld items in %lld ms\n", count, elapsed); \
} while(0)

/* Randomized test against a reference model. */
static void dictRandomizedTest(dictType *type, int rounds, int ops_per_round) {
    modelCtx *m = zcalloc(sizeof(*m));
    test_table_bytes = 0;
    dict *d = dictCreate(type);
    m->d = d;
    dictSetResizeEnabled(DICT_RESIZE_ENABLE);

    for (int round = 0; round < rounds; round++) {
        /* Phase profile: mostly growth, then mostly shrink, etc. */
        int add_bias = (round % 4 < 2) ? 70 : 30;
        for (int op = 0; op < ops_per_round; op++) {
            int id = rand() % MODEL_KEYS;
            int r = rand() % 100;
            if (r < add_bias) modelAdd(m, d, id);
            else if (r < 92) modelDel(m, d, id);
            else if (r < 94) {
                char *k = keyForId(id);
                dictEntry *de = dictFind(d, k);
                assert((de != NULL) == m->present[id]);
                zfree(k);
            } else if (r < 96) {
                /* two-phase delete with nested lookups in between */
                char *k = keyForId(id);
                dictPosition pos;
                dictEntryLink link = dictTwoPhaseUnlinkFind(d, k, &pos);
                assert((link != NULL) == m->present[id]);
                if (link) {
                    char *k2 = keyForId(rand() % MODEL_KEYS);
                    dictFind(d, k2); /* may look up, must not move entries */
                    zfree(k2);
                    dictTwoPhaseUnlinkFree(d, link, &pos);
                    m->present[id] = 0;
                    m->count--;
                }
                zfree(k);
            } else if (r < 97) {
                dictRehash(d, 1 + rand() % 3);
            } else if (r < 98) {
                dictSetResizeEnabled(rand() % 2 ? DICT_RESIZE_AVOID : DICT_RESIZE_ENABLE);
            } else if (r < 99) {
                if (rand() % 2) dictExpand(d, dictSize(d) * 2 + 1);
                else dictShrink(d, dictSize(d) + 1);
            } else {
                drainRehash(d);
            }
            assert(dictSize(d) == m->count);
        }
        dictVerify(d);

        /* Safe iteration while deleting what was returned and inserting
         * keys from the part of the key space the iterator can't have
         * returned... (inserts may or may not be returned). */
        {
            memset(m->seen, 0, sizeof(m->seen));
            uint8_t initial[MODEL_KEYS];
            memcpy(initial, m->present, sizeof(initial));
            uint8_t deleted[MODEL_KEYS] = {0};
            dictIterator it;
            dictEntry *de;
            dictInitSafeIterator(&it, d);
            while ((de = dictNext(&it))) {
                int id = keyId(dictGetKey(de));
                assert(m->present[id]);
                m->seen[id]++;
                int r = rand() % 100;
                if (r < 30) {
                    modelDel(m, d, id); /* the returned entry */
                    deleted[id] = 1;
                } else if (r < 50) {
                    /* Only insert keys that were absent at the start and were
                     * not returned and deleted: the iterator may return
                     * those, but only once. */
                    int nid = rand() % MODEL_KEYS;
                    if (!initial[nid] && !deleted[nid]) modelAdd(m, d, nid);
                }
            }
            dictResetIterator(&it);
            for (int i = 0; i < MODEL_KEYS; i++) {
                assert(m->seen[i] <= 1);
                if (initial[i] && !deleted[i]) assert(m->seen[i] == 1);
            }
            assert(dictSize(d) == m->count);
            dictVerify(d);
        }

        /* Full scan with deletes of the current entry in the callback. */
        {
            memset(m->seen, 0, sizeof(m->seen));
            uint8_t initial[MODEL_KEYS];
            memcpy(initial, m->present, sizeof(initial));
            m->delete_percent = rand() % 40;
            unsigned long cursor = 0;
            do {
                cursor = dictScan(d, cursor, modelScanCb, m);
                /* interleave a bit of rehashing between scan calls */
                if (rand() % 4 == 0) dictRehash(d, 1);
            } while (cursor);
            m->delete_percent = 0;
            for (int i = 0; i < MODEL_KEYS; i++)
                if (initial[i]) assert(m->seen[i] >= 1);
            assert(dictSize(d) == m->count);
            dictVerify(d);
        }

        if (round % 3 == 0) modelCheck(m, d);
    }
    dictSetResizeEnabled(DICT_RESIZE_ENABLE);
    drainRehash(d);
    modelCheck(m, d);
    dictRelease(d);
    assert(test_table_bytes == 0);
    zfree(m);
}

/* ./redis-server test dict [<count> | --accurate] */
int dictTest(int argc, char **argv, int flags) {
    long j;
    long long start, elapsed;
    int retval;
    dict *d = dictCreate(&BenchmarkDictType);
    dictEntry* de = NULL;
    dictEntry* existing = NULL;
    long count = 0;
    unsigned long current_dict_used, remain_keys, capacity, new_buckets;
    int accurate = (flags & REDIS_TEST_ACCURATE);

    if (argc == 4) {
        if (accurate) {
            count = 5000000;
        } else {
            count = strtol(argv[3],NULL,10);
        }
    } else {
        count = 5000;
    }

    TEST("Initial table is a single bucket") {
        assert(dictBuckets(d) == 0 && dictSlots(d) == 0);
        retval = dictAdd(d, stringFromLongLong(1000000), (void*)1);
        assert(retval == DICT_OK);
        assert(dictBuckets(d) == 1);
        assert(dictSlots(d) == DICT_BUCKET_SLOTS);
        assert(dictTableMemUsage(d) == DICT_BUCKET_BYTES);
        assert(dictDelete(d, "1000000") == DICT_OK);
        assert(dictBuckets(d) == 1); /* never shrinks below one bucket */
        dictVerify(d);
    }

    TEST("Add 16 keys and verify dict resize is ok") {
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        for (j = 0; j < 16; j++) {
            retval = dictAdd(d,stringFromLongLong(j),(void*)j);
            assert(retval == DICT_OK);
        }
        drainRehash(d);
        assert(dictSize(d) == 16);
        assert(dictSlots(d) >= 16);
        /* A power of two number of buckets. */
        assert((dictBuckets(d) & (dictBuckets(d) - 1)) == 0);
        dictVerify(d);
    }

    TEST("Use DICT_RESIZE_AVOID to disable the dict resize and pad to (dict_force_resize_ratio * capacity)") {
        dictSetResizeEnabled(DICT_RESIZE_AVOID);
        capacity = dictSlots(d);
        for (j = 16; j < (long)(dict_force_resize_ratio * capacity); j++) {
            retval = dictAdd(d,stringFromLongLong(j),(void*)j);
            assert(retval == DICT_OK);
        }
        current_dict_used = dict_force_resize_ratio * capacity;
        assert(dictSize(d) == current_dict_used);
        assert(dictSlots(d) == capacity);
        assert(!dictIsRehashing(d));
        dictVerify(d);
    }

    TEST("Add one more key, trigger the dict resize") {
        retval = dictAdd(d,stringFromLongLong(current_dict_used),(void*)(current_dict_used));
        assert(retval == DICT_OK);
        current_dict_used++;
        new_buckets = numBuckets(nextBucketExp(current_dict_used));
        assert(dictSize(d) == current_dict_used);
        assert(dictIsRehashing(d));
        assert(numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS == capacity);
        assert(numBuckets(d->ht_size_exp[1]) == new_buckets);
        dictVerify(d);

        /* Wait for rehashing. */
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        drainRehash(d);
        assert(dictSize(d) == current_dict_used);
        assert(numBuckets(d->ht_size_exp[0]) == new_buckets);
        assert(d->ht_size_exp[1] == -1);
        dictVerify(d);
    }

    TEST("Delete keys until we can trigger shrink in next test") {
        /* Delete keys until we can satisfy (1 / HASHTABLE_MIN_FILL) in the next test. */
        capacity = dictSlots(d);
        for (j = capacity / HASHTABLE_MIN_FILL + 1; j < (long)current_dict_used; j++) {
            char *key = stringFromLongLong(j);
            retval = dictDelete(d, key);
            zfree(key);
            assert(retval == DICT_OK);
        }
        current_dict_used = capacity / HASHTABLE_MIN_FILL + 1;
        assert(dictSize(d) == current_dict_used);
        assert(dictSlots(d) == capacity);
        assert(!dictIsRehashing(d));
    }

    TEST("Delete one more key, trigger the dict resize") {
        current_dict_used--;
        char *key = stringFromLongLong(current_dict_used);
        retval = dictDelete(d, key);
        zfree(key);
        new_buckets = numBuckets(nextBucketExp(current_dict_used));
        assert(retval == DICT_OK);
        assert(dictSize(d) == current_dict_used);
        assert(dictIsRehashing(d));
        assert(numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS == capacity);
        assert(numBuckets(d->ht_size_exp[1]) == new_buckets);

        /* Wait for rehashing. */
        drainRehash(d);
        assert(dictSize(d) == current_dict_used);
        assert(numBuckets(d->ht_size_exp[0]) == new_buckets);
        assert(d->ht_size_exp[1] == -1);
        dictVerify(d);
    }

    TEST("Empty the dictionary and add 128 keys") {
        dictEmpty(d, NULL);
        for (j = 0; j < 128; j++) {
            retval = dictAdd(d,stringFromLongLong(j),(void*)j);
            assert(retval == DICT_OK);
        }
        drainRehash(d);
        assert(dictSize(d) == 128);
        dictVerify(d);
    }

    TEST("Use DICT_RESIZE_AVOID to disable the dict resize and reduce to a few keys") {
        /* Use DICT_RESIZE_AVOID to disable the dict reset, and reduce
         * the number of keys until we can trigger shrinking in next test. */
        dictSetResizeEnabled(DICT_RESIZE_AVOID);
        capacity = dictSlots(d);
        remain_keys = capacity / (HASHTABLE_MIN_FILL * dict_force_resize_ratio) + 1;
        for (j = remain_keys; j < 128; j++) {
            char *key = stringFromLongLong(j);
            retval = dictDelete(d, key);
            zfree(key);
            assert(retval == DICT_OK);
        }
        current_dict_used = remain_keys;
        assert(dictSize(d) == remain_keys);
        assert(dictSlots(d) == capacity);
        assert(!dictIsRehashing(d));
    }

    TEST("Delete one more key, trigger the dict resize (AVOID)") {
        current_dict_used--;
        char *key = stringFromLongLong(current_dict_used);
        retval = dictDelete(d, key);
        zfree(key);
        assert(retval == DICT_OK);
        assert(dictSize(d) == current_dict_used);
        assert(dictIsRehashing(d));
        assert(numBuckets(d->ht_size_exp[0]) * DICT_BUCKET_SLOTS == capacity);
        new_buckets = numBuckets(d->ht_size_exp[1]);

        /* Wait for rehashing. */
        for (int steps = 0; dictIsRehashing(d) && steps < 1000; steps++)
            dictRehash(d, 100);
        assert(!dictIsRehashing(d));
        assert(dictSize(d) == current_dict_used);
        assert(numBuckets(d->ht_size_exp[0]) == new_buckets);
        assert(d->ht_size_exp[1] == -1);
        dictVerify(d);
    }

    TEST("Forced shrink at 1/32 occupancy finishes under DICT_RESIZE_AVOID") {
        dictType type = BenchmarkDictType;
        type.hashFunction = constHashCallback;
        dict *ds = dictCreate(&type);
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        assert(dictExpand(ds, 800) == DICT_OK);
        for (int i = 0; i < 100; i++)
            assert(dictAdd(ds, stringFromLongLong(i), NULL) == DICT_OK);
        size_t threshold = dictSlots(ds) / (HASHTABLE_MIN_FILL * dict_force_resize_ratio);
        dictSetResizeEnabled(DICT_RESIZE_AVOID);
        for (int i = 99; i >= (int)threshold; i--) {
            char *key = stringFromLongLong(i);
            assert(dictDelete(ds, key) == DICT_OK);
            zfree(key);
            if (dictSize(ds) > threshold) assert(!dictIsRehashing(ds));
        }
        assert(dictIsRehashing(ds));
        assert(numBuckets(ds->ht_size_exp[0]) == 16 * numBuckets(ds->ht_size_exp[1]));
        for (int steps = 0; dictIsRehashing(ds) && steps < 1000; steps++)
            dictRehash(ds, 100);
        assert(!dictIsRehashing(ds));
        assert(dictSize(ds) == threshold);
        dictVerify(ds);
        dictRelease(ds);
    }

    TEST("Restore to original state") {
        dictEmpty(d, NULL);
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
    }

    TEST("dictMemUsage charges one dictEntry per element only for dicts with values") {
        long n = 1000;
        dict *dn = dictCreate(&BenchmarkDictType);          /* no_value = 0 */
        dict *dv = dictCreate(&BenchmarkDictTypeNoValue);   /* no_value = 1 */
        for (long i = 0; i < n; i++) {
            assert(dictAdd(dn, stringFromLongLong(i), (void *)i) == DICT_OK);
            assert(dictAdd(dv, stringFromLongLong(i), NULL) == DICT_OK);
        }

        /* Identical keys and resize policy => identical bucket geometry, so the
         * two dicts' reported memory differs only by the dictEntry of each of
         * the n elements that the no_value dict does not allocate. */
        assert(dictSize(dn) == (unsigned long)n && dictSize(dv) == (unsigned long)n);
        assert(dictBuckets(dn) == dictBuckets(dv));
        assert(dictEntryMemUsage(1) == 0 && dictEntryMemUsage(0) == sizeof(dictEntry));
        assert(dictMemUsage(dn) - dictMemUsage(dv) == (size_t)n * dictEntryMemUsage(0));
        assert(dictMemUsage(dv) == dictTableMemUsage(dv));

        dictRelease(dn);
        dictRelease(dv);
    }

    TEST("Long bucket chains: insert, delete, compaction and accounting") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&chainDictType);
        const int n = 300; /* all keys share one chain */
        for (int i = 0; i < n; i++) {
            assert(dictAdd(dc, stringFromLongLong(i), NULL) == DICT_OK);
            dictVerify(dc);
        }
        assert(dc->child_buckets[0] + dc->child_buckets[1] > 0);
        /* Delete in a scattered order, every deletion must keep the chain
         * valid and compact it. */
        for (int i = 0; i < n; i += 2) {
            char *k = stringFromLongLong(i);
            assert(dictDelete(dc, k) == DICT_OK);
            zfree(k);
            dictVerify(dc);
        }
        drainRehash(dc);
        for (int i = 1; i < n; i += 2) {
            char *k = stringFromLongLong(i);
            assert(dictFind(dc, k) != NULL);
            zfree(k);
        }
        for (int i = 0; i < n; i += 2) {
            char *k = stringFromLongLong(i);
            assert(dictFind(dc, k) == NULL);
            zfree(k);
        }
        /* Chain is fully compacted: only as many buckets as needed. */
        unsigned long used = dictSize(dc);
        unsigned long needed = (used + DICT_BUCKET_SLOTS - 2) / (DICT_BUCKET_SLOTS - 1);
        unsigned long children = dc->child_buckets[0];
        assert(children <= needed + 1);
        dictRelease(dc);
        assert(test_table_bytes == 0);
    }

    TEST("Safe iterator deleting the returned entry in a long chain returns each entry once") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&chainDictType);
        const int n = 200;
        for (int i = 0; i < n; i++) assert(dictAdd(dc, stringFromLongLong(i), NULL) == DICT_OK);
        uint8_t seen[200] = {0};
        dictIterator it;
        dictInitSafeIterator(&it, dc);
        dictEntry *e;
        int visited = 0;
        while ((e = dictNext(&it))) {
            int id = atoi(dictGetKey(e));
            assert(!seen[id]);
            seen[id] = 1;
            visited++;
            if (id % 3 != 0) {
                char *k = stringFromLongLong(id);
                assert(dictDelete(dc, k) == DICT_OK);
                zfree(k);
            }
        }
        dictResetIterator(&it);
        assert(visited == n);
        dictVerify(dc);
        assert(dictSize(dc) == (unsigned long)((n + 2) / 3));
        dictRelease(dc);
        assert(test_table_bytes == 0);
    }

    TEST("Safe iterator inserting while iterating never returns an entry twice") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&chainDictType);
        const int n = 40;
        for (int i = 0; i < n; i++) assert(dictAdd(dc, stringFromLongLong(i), NULL) == DICT_OK);
        uint8_t seen[400] = {0};
        dictIterator it;
        dictInitSafeIterator(&it, dc);
        dictEntry *e;
        int next_new = n;
        while ((e = dictNext(&it))) {
            int id = atoi(dictGetKey(e));
            assert(id < 400 && !seen[id]);
            seen[id] = 1;
            if (next_new < 399) {
                assert(dictAdd(dc, stringFromLongLong(next_new++), NULL) == DICT_OK);
            }
        }
        dictResetIterator(&it);
        for (int i = 0; i < n; i++) assert(seen[i]);
        dictVerify(dc);
        dictRelease(dc);
        assert(test_table_bytes == 0);
    }

    TEST("Scan: callback deletes the entry it is given, chains stay valid") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&fewBucketsDictType);
        modelCtx *m = zcalloc(sizeof(*m));
        m->d = dc;
        for (int i = 0; i < MODEL_KEYS; i++) modelAdd(m, dc, i);
        drainRehash(dc);
        dictVerify(dc);
        m->delete_percent = 60;
        unsigned long cursor = 0;
        do {
            cursor = dictScan(dc, cursor, modelScanCb, m);
        } while (cursor);
        for (int i = 0; i < MODEL_KEYS; i++) assert(m->seen[i] >= 1);
        assert(dictSize(dc) == m->count);
        dictVerify(dc);
        dictRelease(dc);
        assert(test_table_bytes == 0);
        zfree(m);
    }

    TEST("Scan preserves surviving keys when callbacks delete and insert, including nested scans") {
        for (int no_value = 0; no_value <= 1; no_value++) {
            dictType type = chainDictType;
            type.no_value = no_value;
            test_table_bytes = 0;
            dict *unchained = dictCreate(&type);
            dictSetResizeEnabled(DICT_RESIZE_FORBID);
            for (int i = 0; i < DICT_BUCKET_SLOTS; i++)
                assert(dictAdd(unchained, keyForId(i), NULL) == DICT_OK);
            scanMutationCtx replacement = {.d = unchained, .deletion = 3};
            assert(dictScan(unchained, 0, scanMutationCb, &replacement) == 0);
            for (int i = 0; i < DICT_BUCKET_SLOTS; i++) assert(replacement.seen[i] == 1);
            dictVerify(unchained);
            dictRelease(unchained);
            assert(test_table_bytes == 0);
            for (int deletion = 0; deletion < 3; deletion++) {
                for (int additions = 1; additions <= 2; additions++) {
                    for (int nested = 0; nested <= 1; nested++) {
                        dictType type = chainDictType;
                        type.no_value = no_value;
                        test_table_bytes = 0;
                        dict *dc = dictCreate(&type);
                        dictSetResizeEnabled(DICT_RESIZE_FORBID);
                        for (int i = 0; i <= DICT_BUCKET_SLOTS; i++)
                            assert(dictAdd(dc, keyForId(i), NULL) == DICT_OK);
                        scanMutationCtx ctx = {.d = dc, .deletion = deletion,
                                               .additions = additions, .nested = nested};
                        assert(dictScan(dc, 0, scanMutationCb, &ctx) == 0);
                        for (int i = 1; i <= DICT_BUCKET_SLOTS; i++) assert(ctx.seen[i] >= 1);
                        assert(dictFind(dc, "k0") == NULL);
                        assert(dictSize(dc) == DICT_BUCKET_SLOTS + (unsigned long)additions);
                        dictVerify(dc);
                        dictRelease(dc);
                        assert(test_table_bytes == 0);
                    }
                }
            }
        }
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
    }

    TEST("Randomized operations against a reference model (no_value)") {
        dictRandomizedTest(&verifyDictType, accurate ? 600 : 60, 1500);
    }

    TEST("Randomized operations against a reference model (with values)") {
        dictRandomizedTest(&verifyDictTypeVal, accurate ? 300 : 30, 1500);
    }

    TEST("Randomized operations with very long chains") {
        dictRandomizedTest(&fewBucketsDictType, accurate ? 200 : 20, 1000);
    }

    TEST("Rehash during safe iteration is paused and resumes afterwards") {
        test_table_bytes = 0;
        dict *dr = dictCreate(&verifyDictType);
        for (int i = 0; i < 500; i++) assert(dictAdd(dr, stringFromLongLong(i), NULL) == DICT_OK);
        drainRehash(dr);
        dictIterator it;
        dictInitSafeIterator(&it, dr);
        dictEntry *e = dictNext(&it);
        assert(e != NULL);
        /* Force a resize while iterating: rehashing must not progress. */
        dictExpand(dr, 100000);
        assert(dictIsRehashing(dr));
        assert(dictRehash(dr, 100) == 0);
        assert(dictIsRehashing(dr));
        int visited = 1;
        while (dictNext(&it)) visited++;
        dictResetIterator(&it);
        assert(visited == 500);
        drainRehash(dr);
        assert(dictSize(dr) == 500);
        dictVerify(dr);
        dictRelease(dr);
        assert(test_table_bytes == 0);
    }

    srand(12345);
    start_benchmark();
    for (j = 0; j < count; j++) {
        /* Create a dynamically allocated substring */
        char *key = stringFromSubstring();

        /* Insert the range directly from the large string */
        de = dictAddRaw(d, key, &existing);
        assert(de != NULL || existing != NULL);
        /* If key already exists NULL is returned so we need to free the temp key string */
        if (de == NULL) zfree(key);
    }
    end_benchmark("Inserting random substrings (100-500B) from large string with symbols");
    assert((long)dictSize(d) <= count);
    dictEmpty(d, NULL);

    start_benchmark();
    for (j = 0; j < count; j++) {
        retval = dictAdd(d,stringFromLongLong(j),(void*)j);
        assert(retval == DICT_OK);
    }
    end_benchmark("Inserting via dictAdd() non existing");
    assert((long)dictSize(d) == count);

    dictEmpty(d, NULL);

    start_benchmark();
    for (j = 0; j < count; j++) {
        de = dictAddRaw(d,stringFromLongLong(j),NULL);
        assert(de != NULL);
    }
    end_benchmark("Inserting via dictAddRaw() non existing");
    assert((long)dictSize(d) == count);

    start_benchmark();
    for (j = 0; j < count; j++) {
        void *key = stringFromLongLong(j);
        de = dictAddRaw(d,key,&existing);
        assert(existing != NULL);
        zfree(key);
    }
    end_benchmark("Inserting via dictAddRaw() existing (no insertion)");
    assert((long)dictSize(d) == count);

    /* Wait for rehashing. */
    while (dictIsRehashing(d)) {
        dictRehashMicroseconds(d,100*1000);
    }

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        dictEntry *de = dictFind(d,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        dictEntry *de = dictFind(d,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements (2nd round)");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        dictEntry *de = dictFind(d,key);
        assert(de != NULL);
        zfree(key);
    }
    end_benchmark("Random access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        dictEntry *de = dictGetRandomKey(d);
        assert(de != NULL);
    }
    end_benchmark("Accessing random keys");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        key[0] = 'X';
        dictEntry *de = dictFind(d,key);
        assert(de == NULL);
        zfree(key);
    }
    end_benchmark("Accessing missing");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        retval = dictDelete(d,key);
        assert(retval == DICT_OK);
        key[0] += 17; /* Change first number to letter. */
        retval = dictAdd(d,key,(void*)j);
        assert(retval == DICT_OK);
    }
    end_benchmark("Removing and adding");
    dictRelease(d);

    TEST("Use dict without values (no_value=1)") {
        dictType dt = BenchmarkDictType;
        dt.no_value = 1;

        /* Allocate array of size count and fill it with keys (stringFromLongLong(j) */
        char **lookupKeys = zmalloc(sizeof(char*) * count);
        for (long j = 0; j < count; j++)
            lookupKeys[j] = stringFromLongLong(j);


        /* Add keys without values. */
        dict *d = dictCreate(&dt);
        for (j = 0; j < count; j++) {
            retval = dictAdd(d,lookupKeys[j],NULL);
            assert(retval == DICT_OK);
        }

        /* Now, we should be able to find the keys. */
        for (j = 0; j < count; j++) {
            dictEntry *de = dictFind(d,lookupKeys[j]);
            assert(de != NULL);
        }

        /* Find non exists keys. */
        for (j = 0; j < count; j++) {
            /* Temporarily override first char of key */
            char tmp = lookupKeys[j][0];
            lookupKeys[j][0] = 'X';
            dictEntry *de = dictFind(d,lookupKeys[j]);
            lookupKeys[j][0] = tmp;
            assert(de == NULL);
        }

        dictRelease(d);
        zfree(lookupKeys);
    }

    TEST("Test dictFindLink() functionality") {
        dictType dt = BenchmarkDictType;
        dict *d = dictCreate(&dt);

        /* find in empty dict */
        dictEntryLink link = dictFindLink(d, "key", NULL);
        assert(link == NULL);

        /* Add keys to dict and test */
        for (j = 0; j < 10; j++) {
            /* Add another key to dict */
            char *key = stringFromLongLong(j);
            retval = dictAdd(d, key, (void*)j);
            assert(retval == DICT_OK);
            /* find existing keys with dictFindLink() */
            dictEntryLink link = dictFindLink(d, key, NULL);
            assert(link != NULL);
            assert(*link != NULL);
            assert(dictGetKey(*link) != NULL);

            /* Test that the key found is the correct one */
            void *foundKey = dictGetKey(*link);
            assert(compareCallback( NULL, foundKey, key));

            /* Test finding a non-existing key */
            char *nonExistingKey = stringFromLongLong(j + 10);
            link = dictFindLink(d, nonExistingKey, NULL);
            assert(link == NULL);

            /* Test with bucket parameter */
            dictEntryLink bucket = NULL;
            link = dictFindLink(d, key, &bucket);
            assert(link != NULL);
            assert(bucket != NULL);

            /* Test bucket parameter with non-existing key */
            link = dictFindLink(d, nonExistingKey, &bucket);
            assert(link == NULL);
            assert(bucket != NULL); /* Bucket should still be set even for non-existing keys */

            /* Clean up */
            zfree(nonExistingKey);
        }

        dictRelease(d);
    }

    TEST("dictFindLink() + dictSetKeyAtLink() insert and replace, with rehashing") {
        test_table_bytes = 0;
        dict *dl = dictCreate(&verifyDictType);
        for (int i = 0; i < 2000; i++) {
            char *key = stringFromLongLong(i);
            dictEntryLink bucket = NULL;
            dictEntryLink link = dictFindLink(dl, key, &bucket);
            assert(link == NULL);
            dictSetKeyAtLink(dl, key, &bucket, 1);
            /* bucket now points at the inserted entry */
            assert(bucket != NULL && dictGetKey(*bucket) == key);
            link = dictFindLink(dl, key, NULL);
            assert(link == bucket || dictGetKey(*link) == key);
            if (i % 5 == 0) {
                /* Replace the key object with an equal one. */
                char *key2 = stringFromLongLong(i);
                void *old = dictGetKey(*link);
                dictSetKeyAtLink(dl, key2, &link, 0);
                assert(dictGetKey(*link) == key2);
                zfree(old);
            }
        }
        assert(dictSize(dl) == 2000);
        drainRehash(dl);
        dictVerify(dl);
        dictRelease(dl);
        assert(test_table_bytes == 0);
    }

    return 0;
}
#endif
