/* Hash Tables Implementation.
 *
 * This file implements in memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto resize if needed
 * tables of power of two in size are used, collisions are handled by open
 * addressing with linear probing and Robin Hood insertion.
 *
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * TABLE LAYOUT
 * ------------
 * A hash table is a single allocation holding two arrays:
 *
 *   [ dictEntry *slots[nslots] ][ uint8_t meta[nslots] ]
 *
 * where nslots = DICTHT_PHYSICAL_SLOTS(exp) = 2^exp plus up to DICT_MAX_PROBE
 * guard slots. A key hashes to a *home* slot in [0, 2^exp) and is stored at
 * home + d, where the displacement d is smaller than DICT_MAX_PROBE. The
 * trailing guard slots mean a probe sequence never wraps around, which keeps
 * every probe loop a straight forward scan. A run that reaches the last
 * physical slot is treated as full, which just grows the table earlier; only
 * tables smaller than DICT_MAX_PROBE (where the guard is shortened to the
 * table size) can get there. meta[i] holds the displacement of the
 * entry (or tombstone) living in slot i, so the home of slot i is always
 * i - meta[i] without rehashing the key. d->ht_table[htidx] points at the
 * start of the allocation, so defrag can move the whole table by simply
 * reassigning that pointer.
 *
 * A slot is one of:
 *   - NULL:      never used, or freed by a compacting delete. Terminates probes.
 *   - TOMBSTONE: deleted while compaction was paused. Does not terminate probes.
 *   - anything else: a live entry (a dictEntry pointer, or a tagged key
 *     pointer, see the pointer tagging notes in dict.h).
 *
 * ROBIN HOOD INVARIANT
 * --------------------
 * Within a run of occupied slots, entries are ordered by home slot. An insert
 * that reaches a slot holding a "richer" entry (one closer to its home) steals
 * the slot and carries the evicted entry forward. Consequently a lookup can
 * stop as soon as it sees a slot whose displacement is smaller than the
 * distance it has already travelled: the key would have stolen that slot. Both
 * deletion strategies below preserve the ordering, so the early exit is always
 * valid and a miss costs only a couple of probes.
 *
 * DELETION
 * --------
 * Normally a delete compacts the run (Knuth's algorithm R backward shift) so
 * no tombstone is left behind. While d->pausecompact is set - during scans and
 * safe iterations, where moving entries backwards would make the caller miss
 * them - a delete leaves a TOMBSTONE instead. Tombstones are reclaimed by
 * later inserts, purged in bulk once they become a noticeable fraction of the
 * table, and dropped entirely by any resize.
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
 *  - A hash table is still allowed to expand if the fill ratio reaches
 *    DICT_GROW_RATIO_AVOID, and it is always allowed to expand when a probe
 *    sequence overflows, since there would be nowhere to put the key.
 *  - A hash table is still allowed to shrink if the ratio between the number
 *    of elements and the buckets <= 1 / (HASHTABLE_MIN_FILL * dict_force_resize_ratio). */
static redisAtomic dictResizeEnable dict_can_resize = DICT_RESIZE_ENABLE;
static const unsigned int dict_force_resize_ratio = 4;

/* Fill ratio (elements + tombstones over logical size) at which the table is
 * grown. Linear probing degrades sharply as the table fills up, so we grow far
 * earlier than a chained table would. */
#define DICT_GROW_PERCENT        75  /* DICT_RESIZE_ENABLE */
#define DICT_GROW_PERCENT_AVOID  90  /* DICT_RESIZE_AVOID */

/* Purge tombstones once they reach this fraction of the logical table size. */
#define DICT_TOMBSTONE_PURGE_SHIFT 3 /* size / 8 */

/* -------------------------- types ----------------------------------------- */

/* Open addressing stores at most one element per slot, so entries have no
 * 'next' field. A no_value dict never allocates a dictEntry at all: the key
 * pointer is tagged and stored directly in the slot. */
struct dictEntry {
    void *key;               /* Must be first */
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
};

/* Marks a slot whose entry was deleted while compaction was paused. It is a
 * real (never linked, never freed) dictEntry so that code reading a raw slot
 * without going through dict.c sees NULL fields instead of faulting. */
static dictEntry dictTombstoneEntry;
#define TOMBSTONE (&dictTombstoneEntry)
#define SLOT_IS_LIVE(de) ((de) != NULL && (de) != TOMBSTONE)

/* -------------------------- private prototypes ---------------------------- */

static int _dictExpandIfNeeded(dict *d);
static void _dictShrinkIfNeeded(dict *d);
static void _dictRehashStepIfNeeded(dict *d, uint64_t visitedIdx);
static signed char _dictNextExp(unsigned long size);
static int _dictInit(dict *d, dictType *type);
static int dictDefaultCompare(dictCmpCache *cache, const void *key1, const void *key2);
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket);
static void dictPurgeTombstones(dict *d, int htidx);
static void completeRehash(dict *d);
static int dictRehashForce(dict *d, int n);
dictEntryLink dictFindLinkForInsert(dict *d, const void *key, dictEntry **existing);
dictEntry *dictInsertKeyAtLink(dict *d, void *key __stored_key, dictEntryLink link);

/* -------------------------- unused  --------------------------- */
void dictSetSignedIntegerVal(dictEntry *de, int64_t val);
int64_t dictGetSignedIntegerVal(const dictEntry *de);
double dictIncrDoubleVal(dictEntry *de, double val);
void *dictEntryMetadata(dictEntry *de);
int64_t dictIncrSignedIntegerVal(dictEntry *de, int64_t val);

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

/* Compaction (backward shift on delete) must be paused whenever moving an
 * entry to an earlier slot could make the caller miss it, i.e. while walking
 * slots in order: dictScan() and safe iterators. */
#define dictPauseCompact(d) ((d)->pausecompact++)
#define dictResumeCompact(d) do {                                             \
    debugAssert((d)->pausecompact > 0);                                       \
    if (--(d)->pausecompact == 0) _dictPurgeTombstonesIfNeeded(d);            \
} while (0)

#ifdef DEBUG_ASSERTIONS
#define dictBumpLinkEpoch(d) ((d)->linkEpoch++)
#else
#define dictBumpLinkEpoch(d) ((void)0)
#endif

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

/* ----------------------------- slot helpers ------------------------------- */

/* Number of physical slots (logical size + guard slots) of table 'htidx'. */
static inline unsigned long htSlotCount(const dict *d, int htidx) {
    return DICTHT_PHYSICAL_SLOTS(d->ht_size_exp[htidx]);
}

/* Per slot displacement array, stored right after the slots in the same
 * allocation. Always derived from ht_table[] so the table can be relocated. */
static inline uint8_t *htMeta(const dict *d, int htidx) {
    return (uint8_t *)(d->ht_table[htidx] + htSlotCount(d, htidx));
}

static inline unsigned long htHome(const dict *d, int htidx, uint64_t hash) {
    return (unsigned long)(hash & DICTHT_SIZE_MASK(d->ht_size_exp[htidx]));
}

static inline size_t htAllocSize(signed char exp) {
    unsigned long nslots = DICTHT_PHYSICAL_SLOTS(exp);
    return nslots * (sizeof(dictEntry *) + sizeof(uint8_t));
}

/* Returns the slot holding 'key' in table 'htidx', or -1 if it is not there.
 * 'home' must be the home slot of the key in that table. */
static long htFindSlot(dict *d, int htidx, const void *key, unsigned long home,
                       keyCmpFunc cmpFunc, dictCmpCache *cmpCache)
{
    dictEntry **slots = d->ht_table[htidx];
    uint8_t *meta = htMeta(d, htidx);
    unsigned long pos, limit = home + DICT_MAX_PROBE;
    unsigned long nslots = htSlotCount(d, htidx);
    unsigned int dist = 0;

    if (limit > nslots) limit = nslots;
    for (pos = home; pos < limit; pos++, dist++) {
        dictEntry *de = slots[pos];
        if (de == NULL) return -1;
        /* Robin Hood: an entry poorer than us would have taken this slot. */
        if (meta[pos] < dist) return -1;
        if (de == TOMBSTONE) continue;
        const void *storedKey = dictStoredKey2Key(d, dictGetKey(de));
        if (key == storedKey || cmpFunc(cmpCache, key, storedKey)) return (long)pos;
    }
    return -1;
}

/* Walk the Robin Hood cascade an insert at 'home' would perform, without
 * touching the table, and tell whether every entry it moves stays within
 * DICT_MAX_PROBE of its own home. Only the displacement array is read, so
 * this is a scan of a handful of contiguous bytes. */
static int htProbeInsertFits(dict *d, int htidx, unsigned long home) {
    dictEntry **slots = d->ht_table[htidx];
    uint8_t *meta = htMeta(d, htidx);
    unsigned long nslots = htSlotCount(d, htidx);
    unsigned long pos = home;
    unsigned int dist = 0;

    while (pos < nslots) {
        dictEntry *cur = slots[pos];

        if (dist >= DICT_MAX_PROBE) return 0;
        if (cur == NULL) return 1;
        if (cur == TOMBSTONE ? meta[pos] <= dist : meta[pos] < dist) {
            if (cur == TOMBSTONE) return 1;
            dist = meta[pos]; /* carry the evicted entry forward instead */
        }
        pos++;
        dist++;
    }
    return 0;
}

/* Insert 'entry' into table 'htidx' with the given home slot, displacing
 * richer entries as needed (Robin Hood). Returns the slot the entry ended up
 * in, or -1 if it does not fit within DICT_MAX_PROBE, in which case the table
 * is left untouched and the caller has to make room and retry.
 *
 * Note this does not update ht_used, and does not check for duplicates. */
static long htProbeInsert(dict *d, int htidx, dictEntry *entry, unsigned long home) {
    dictEntry **slots = d->ht_table[htidx];
    uint8_t *meta = htMeta(d, htidx);
    unsigned long pos;
    long placed = -1;
    unsigned int dist;

    debugAssert(home < DICTHT_SIZE(d->ht_size_exp[htidx]));

    /* Decide up front, so that a failure never leaves a half applied insert
     * behind: the loop below makes exactly the same choices. */
    if (!htProbeInsertFits(d, htidx, home)) return -1;

    for (pos = home, dist = 0; ; pos++, dist++) {
        dictEntry *cur = slots[pos];

        if (cur == NULL) {
            slots[pos] = entry;
            meta[pos] = (uint8_t)dist;
            dictBumpLinkEpoch(d);
            return placed == -1 ? (long)pos : placed;
        }

        /* Steal the slot from a richer occupant. A tombstone is dead weight,
         * so it also loses ties, which is how tombstones get reclaimed. */
        if (cur == TOMBSTONE ? meta[pos] <= dist : meta[pos] < dist) {
            uint8_t evictedDist = meta[pos];
            slots[pos] = entry;
            meta[pos] = (uint8_t)dist;
            if (placed == -1) placed = (long)pos;
            if (cur == TOMBSTONE) {
                d->ht_tombstones[htidx]--;
                dictBumpLinkEpoch(d);
                return placed;
            }
            entry = cur;
            dist = evictedDist;
        }
    }
}

/* Remove the content of slot 'idx' and compact the run behind it so that no
 * probe sequence is left broken (Knuth 6.4 algorithm R). */
static void htBackwardShift(dict *d, int htidx, unsigned long idx) {
    dictEntry **slots = d->ht_table[htidx];
    uint8_t *meta = htMeta(d, htidx);
    unsigned long nslots = htSlotCount(d, htidx);
    unsigned long hole = idx, j;

    slots[hole] = NULL;
    meta[hole] = 0;

    for (j = hole + 1; j < nslots; j++) {
        dictEntry *de = slots[j];

        if (de == NULL) break;
        /* No entry further out can reach back to the hole: its displacement
         * would have to exceed the probe window. */
        if (j - hole >= DICT_MAX_PROBE) break;

        if (de == TOMBSTONE) {
            /* Slide the tombstone back into the hole. It keeps the run
             * connected for entries that probe across it, and moving the hole
             * forward gives later entries a chance to move back. */
            slots[hole] = TOMBSTONE;
            meta[hole] = 0;
            slots[j] = NULL;
            meta[j] = 0;
            hole = j;
            continue;
        }

        if (j - meta[j] <= hole) {
            slots[hole] = de;
            meta[hole] = (uint8_t)(hole - (j - meta[j]));
            slots[j] = NULL;
            meta[j] = 0;
            hole = j;
        }
    }
    dictBumpLinkEpoch(d);
}

/* Take the live entry at slot 'idx' of table 'htidx' out of the table. The
 * entry itself is not freed and ht_used is left alone: the caller decrements
 * it once it is done freeing the entry, so that a key or value destructor
 * still sees the entry it is being handed counted in dictSize(). */
static void htUnlinkAt(dict *d, int htidx, unsigned long idx) {
    debugAssert(SLOT_IS_LIVE(d->ht_table[htidx][idx]));

    if (d->pausecompact) {
        /* Keep meta[] as is: it still describes the home of the removed entry,
         * which keeps the Robin Hood ordering of the run intact. */
        d->ht_table[htidx][idx] = TOMBSTONE;
        d->ht_tombstones[htidx]++;
        dictBumpLinkEpoch(d);
    } else {
        htBackwardShift(d, htidx, idx);
    }
}

/* Drop every tombstone of table 'htidx', compacting the runs they sit in. */
static void dictPurgeTombstones(dict *d, int htidx) {
    unsigned long nslots = htSlotCount(d, htidx);
    unsigned long i = 0;

    while (i < nslots && d->ht_tombstones[htidx] > 0) {
        if (d->ht_table[htidx][i] == TOMBSTONE) {
            /* The shift may slide another tombstone into this slot, so only
             * advance once the slot is free of them. */
            htBackwardShift(d, htidx, i);
            d->ht_tombstones[htidx]--;
        } else {
            i++;
        }
    }
    debugAssert(d->ht_tombstones[htidx] == 0);
}

static void _dictPurgeTombstonesIfNeeded(dict *d) {
    if (d->pausecompact) return;
    for (int htidx = 0; htidx <= 1; htidx++) {
        if (d->ht_tombstones[htidx] == 0) continue;
        if (d->ht_tombstones[htidx] <
            (DICTHT_SIZE(d->ht_size_exp[htidx]) >> DICT_TOMBSTONE_PURGE_SHIFT))
            continue;
        dictPurgeTombstones(d, htidx);
    }
}

/* ----------------------------- API implementation ------------------------- */

/* Reset hash table parameters already initialized with _dictInit()*/
static void _dictReset(dict *d, int htidx)
{
    d->ht_table[htidx] = NULL;
    d->ht_size_exp[htidx] = -1;
    d->ht_used[htidx] = 0;
    d->ht_tombstones[htidx] = 0;
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
    d->allocated_entries = 0;
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pausecompact = 0;
    d->pauseAutoResize = 0;
#ifdef DEBUG_ASSERTIONS
    d->linkEpoch = 0;
#endif
    return DICT_OK;
}

/* Resize or create the hash table,
 * when malloc_failed is non-NULL, it'll avoid panic if malloc fails (in which case it'll be set to 1).
 * Returns DICT_OK if resize was performed, and DICT_ERR if skipped. */
int _dictResize(dict *d, unsigned long size, int* malloc_failed)
{
    if (malloc_failed) *malloc_failed = 0;

    /* We can't rehash twice if rehashing is ongoing. */
    assert(!dictIsRehashing(d));

    /* the new hash table */
    dictEntry **new_ht_table;
    signed char new_ht_size_exp = _dictNextExp(size);

    /* Detect overflows */
    size_t newsize = DICTHT_SIZE(new_ht_size_exp);
    if (newsize < size || htAllocSize(new_ht_size_exp) < newsize)
        return DICT_ERR;

    /* Rehashing to the same table size is not useful. */
    if (new_ht_size_exp == d->ht_size_exp[0]) return DICT_ERR;

    /* Allocate the new hash table. zcalloc() gives us NULL slots and zeroed
     * displacements, which is exactly an empty table. */
    if (malloc_failed) {
        new_ht_table = ztrycalloc(htAllocSize(new_ht_size_exp));
        *malloc_failed = new_ht_table == NULL;
        if (*malloc_failed)
            return DICT_ERR;
    } else
        new_ht_table = zcalloc(htAllocSize(new_ht_size_exp));

    /* Prepare a second hash table for incremental rehashing.
     * We do this even for the first initialization, so that we can trigger the
     * rehashingStarted more conveniently, we will clean it up right after. */
    d->ht_size_exp[1] = new_ht_size_exp;
    d->ht_used[1] = 0;
    d->ht_tombstones[1] = 0;
    d->ht_table[1] = new_ht_table;
    d->rehashidx = 0;
    dictBumpLinkEpoch(d);
    if (d->type->rehashingStarted) d->type->rehashingStarted(d);
    if (d->type->bucketChanged)
        d->type->bucketChanged(d, DICTHT_PHYSICAL_SLOTS(d->ht_size_exp[1]));

    /* Is this the first initialization or is the first hash table empty? If so
     * it's not really a rehashing, we can just set the first hash table so that
     * it can accept keys. */
    if (d->ht_table[0] == NULL || d->ht_used[0] == 0) {
        if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
        if (d->type->bucketChanged)
            d->type->bucketChanged(d, -(long long)DICTHT_PHYSICAL_SLOTS(d->ht_size_exp[0]));
        if (d->ht_table[0]) zfree(d->ht_table[0]);
        d->ht_size_exp[0] = new_ht_size_exp;
        d->ht_used[0] = 0;
        d->ht_tombstones[0] = 0;
        d->ht_table[0] = new_ht_table;
        _dictReset(d, 1);
        d->rehashidx = -1;
        return DICT_OK;
    }

    /* Force a full rehashing of the dictionary */
    if (d->type->force_full_rehash) {
        completeRehash(d);
    }
    return DICT_OK;
}

int _dictExpand(dict *d, unsigned long size, int* malloc_failed) {
    /* the size is invalid if it is smaller than the size of the hash table 
     * or smaller than the number of elements already inside the hash table */
    if (dictIsRehashing(d) || d->ht_used[0] > size || DICTHT_SIZE(d->ht_size_exp[0]) >= size)
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
    /* the size is invalid if it is bigger than the size of the hash table
     * or smaller than the number of elements already inside the hash table */
    if (dictIsRehashing(d) || d->ht_used[0] > size || DICTHT_SIZE(d->ht_size_exp[0]) <= size)
        return DICT_ERR;
    return _dictResize(d, size, NULL);
}

/* Double the table. The request has to cover what the table already holds:
 * the DICT_MAX_PROBE guard slots let ht_used grow past the logical size, and
 * _dictExpand() refuses a size that cannot fit the current elements. */
static int dictGrow(dict *d) {
    unsigned long used = d->ht_used[0];
    unsigned long target = DICTHT_SIZE(d->ht_size_exp[0]) + 1;
    unsigned long roomy = (used > ULONG_MAX / 2 - 1) ? used + 1 : used * 2 + 1;

    /* Aim for a table at most half full. A table that grew late (the resize
     * policy said no until a probe sequence overflowed) can hold more
     * elements than its logical size, and simply doubling would land it right
     * back near the fill ratio it just escaped. */
    if (target < roomy) target = roomy;
    return dictExpand(d, target);
}

/* Collapse both tables into a single, bigger one, leaving no rehash in
 * progress. This is the way out of the one situation the regular two table
 * resize cannot handle: a rehash whose destination filled up. It cannot be
 * grown (a resize may not start while a rehash is in flight) and it cannot
 * absorb what ht[0] still holds either, so a fresh table is built from both.
 *
 * Only reachable when the resize policy kept the rehash stalled long enough
 * for ht[1] to saturate, i.e. a long fork with a heavy write load. */
static void dictRebuildTables(dict *d) {
    unsigned long total = dictSize(d);
    unsigned long target;
    dictEntry **old[2] = { d->ht_table[0], d->ht_table[1] };
    signed char oldexp[2] = { d->ht_size_exp[0], d->ht_size_exp[1] };
    dict tmp;

    assert(dictIsRehashing(d));

    /* Size for a table at most half full, doubling until everything fits. */
    target = (total > ULONG_MAX / 4) ? total + 1 : total * 2 + 1;
    while (1) {
        signed char new_exp = _dictNextExp(target);
        int fits = 1;

        /* Build into a detached view of the dict so the real one keeps
         * describing the old tables until the move succeeded. */
        tmp = *d;
        tmp.ht_table[0] = zcalloc(htAllocSize(new_exp));
        tmp.ht_size_exp[0] = new_exp;
        tmp.ht_used[0] = 0;
        tmp.ht_tombstones[0] = 0;
        tmp.ht_table[1] = NULL;
        tmp.ht_size_exp[1] = -1;
        tmp.ht_used[1] = 0;
        tmp.ht_tombstones[1] = 0;
        tmp.rehashidx = -1;

        for (int src = 0; src <= 1 && fits; src++) {
            if (old[src] == NULL) continue;
            for (unsigned long i = 0; i < DICTHT_PHYSICAL_SLOTS(oldexp[src]); i++) {
                dictEntry *de = old[src][i];
                if (!SLOT_IS_LIVE(de)) continue;
                const void *key = dictStoredKey2Key(d, dictGetKey(de));
                if (htProbeInsert(&tmp, 0, de, htHome(&tmp, 0, dictGetHash(d, key))) < 0) {
                    fits = 0;
                    break;
                }
                tmp.ht_used[0]++;
            }
        }
        if (fits) break;

        zfree(tmp.ht_table[0]);
        if (target > ULONG_MAX / 2)
            panic("dict probe overflow: cannot size a table for %lu elements", total);
        target *= 2;
    }
    assert(tmp.ht_used[0] == total);

    /* The dict still describes the old tables, which is the state the
     * rehashingCompleted callback expects to observe. */
    if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
    if (d->type->bucketChanged) {
        d->type->bucketChanged(d, -(long long)DICTHT_PHYSICAL_SLOTS(oldexp[0]));
        d->type->bucketChanged(d, -(long long)DICTHT_PHYSICAL_SLOTS(oldexp[1]));
        d->type->bucketChanged(d, (long long)DICTHT_PHYSICAL_SLOTS(tmp.ht_size_exp[0]));
    }
    zfree(old[0]);
    zfree(old[1]);

    d->ht_table[0] = tmp.ht_table[0];
    d->ht_size_exp[0] = tmp.ht_size_exp[0];
    d->ht_used[0] = tmp.ht_used[0];
    d->ht_tombstones[0] = 0;
    _dictReset(d, 1);
    d->rehashidx = -1;
    dictBumpLinkEpoch(d);
}

/* Insert an entry that is being migrated between the two tables. Returns 0
 * when the entry does not fit within DICT_MAX_PROBE of its home, leaving the
 * table untouched, in which case the migration as a whole cannot go on: the
 * destination has a fixed size and cannot be grown while it is the target of
 * a rehash.
 *
 * With well spread hashes this does not happen: Robin Hood insertion keeps
 * the largest displacement growing like log(log(n)) and the destination is at
 * most ~3/4 full. It does happen when a table is shrunk while the keys it
 * holds share their low bits, since the destination home is the source home
 * masked down and the collisions pile up. */
static int htInsertMigrated(dict *d, int htidx, dictEntry *entry, unsigned long home) {
    if (htProbeInsert(d, htidx, entry, home) < 0) {
        /* The destination may just be clogged with tombstones left by a
         * paused compaction, those are the cheapest thing to reclaim. */
        if (d->ht_tombstones[htidx] == 0 || d->pausecompact) return 0;
        dictPurgeTombstones(d, htidx);
        if (htProbeInsert(d, htidx, entry, home) < 0) return 0;
    }
    d->ht_used[htidx]++;
    return 1;
}

/* Helper function for `dictRehash` and `_dictBucketRehash` which rehashes all
 * the keys whose home slot in ht[0] is `idx` into ht[1].
 *
 * Entries of a home bucket live in the run starting at `idx`, within
 * DICT_MAX_PROBE slots, mixed with entries of other buckets. Every removal
 * compacts that run, so the scan restarts after each move.
 *
 * Returns 0 if the migration had to be abandoned because an entry did not fit
 * in ht[1]: both tables were then merged into a single new one and there is
 * no rehash left in progress for the caller to step. */
static int rehashEntriesInBucketAtIndex(dict *d, unsigned long idx) {
    unsigned long nslots = htSlotCount(d, 0);
    unsigned long limit = idx + DICT_MAX_PROBE;
    int grow = d->ht_size_exp[1] > d->ht_size_exp[0];
    unsigned long pos = idx;

    if (limit > nslots) limit = nslots;

    while (pos < limit) {
        dictEntry *de = d->ht_table[0][pos];
        uint8_t *meta = htMeta(d, 0);

        if (de == NULL) break;

        if (de == TOMBSTONE) {
            /* Only the tombstones blocking the head of this bucket are worth
             * removing here, the rest belong to buckets we did not reach yet. */
            if (pos != idx) { pos++; continue; }
            htBackwardShift(d, 0, pos);
            d->ht_tombstones[0]--;
            continue;
        }

        if (pos - meta[pos] != idx) { pos++; continue; }

        /* Get the home slot in the new hash table. */
        unsigned long home;
        if (grow) {
            const void *key = dictStoredKey2Key(d, dictGetKey(de));
            home = htHome(d, 1, dictGetHash(d, key));
        } else {
            /* We're shrinking the table. The tables sizes are powers of two,
             * so we simply mask the home slot in the larger table to get the
             * home slot in the smaller one. */
            home = idx & DICTHT_SIZE_MASK(d->ht_size_exp[1]);
        }

        /* Place the entry in the destination before taking it out of the
         * source: a failed insert leaves ht[1] untouched, and the entry must
         * stay reachable in ht[0] for the rebuild below. */
        if (!htInsertMigrated(d, 1, de, home)) {
            dictRebuildTables(d);
            return 0;
        }
        htBackwardShift(d, 0, pos);
        d->ht_used[0]--;
        /* The shift may have pulled another entry of this bucket into 'pos'. */
        pos = idx;
    }
    return 1;
}

/* This checks if we already rehashed the whole table and if more rehashing is required */
static int dictCheckRehashingCompleted(dict *d) {
    if (d->ht_used[0] != 0) return 0;
    
    if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
    if (d->type->bucketChanged)
        d->type->bucketChanged(d, -(long long)DICTHT_PHYSICAL_SLOTS(d->ht_size_exp[0]));
    zfree(d->ht_table[0]);
    /* Copy the new ht onto the old one */
    d->ht_table[0] = d->ht_table[1];
    d->ht_used[0] = d->ht_used[1];
    d->ht_tombstones[0] = d->ht_tombstones[1];
    d->ht_size_exp[0] = d->ht_size_exp[1];
    _dictReset(d, 1);
    d->rehashidx = -1;
    dictBumpLinkEpoch(d);
    return 1;
}

/* Performs N steps of incremental rehashing, ignoring the global resize
 * policy. Returns 1 if there are still keys to move from the old to the new
 * hash table, otherwise 0 is returned.
 *
 * Note that a rehashing step consists in moving a home bucket from the old to
 * the new hash table, however since part of the hash table may be composed of
 * empty slots, it is not guaranteed that this function will rehash even a
 * single bucket, since it will visit at max N*10 empty slots in total,
 * otherwise the amount of work it does would be unbound and the function may
 * block for a long time. */
static int dictRehashForce(dict *d, int n) {
    int empty_visits = n*10; /* Max number of empty slots to visit. */

    if (!dictIsRehashing(d)) return 0;

    while(n-- && d->ht_used[0] != 0) {
        /* Note that rehashidx can't overflow as we are sure there are more
         * elements because ht[0].used != 0 */
        assert(DICTHT_SIZE(d->ht_size_exp[0]) > (unsigned long)d->rehashidx);
        while(d->ht_table[0][d->rehashidx] == NULL) {
            d->rehashidx++;
            if (--empty_visits == 0) return 1;
        }
        /* Move all the keys homed at `rehashidx` from the old to the new HT */
        if (!rehashEntriesInBucketAtIndex(d, d->rehashidx)) return 0;
        d->rehashidx++;
    }

    return !dictCheckRehashingCompleted(d);
}

/* Rehash until there is nothing left in ht[0], regardless of the global resize
 * policy. Used where we cannot afford to leave a rehash half done, e.g. before
 * growing a table whose probe window overflowed. */
static void completeRehash(dict *d) {
    while (dictRehashForce(d, 1000)) {
        /* Continue rehashing */
    }
}

/* Performs N steps of incremental rehashing, honoring the global resize
 * policy. See dictRehashForce().
 *
 * The policy decides whether a resize may *start* (see dictExpandIfNeeded()
 * and dictShrinkIfNeeded()); once one has started, its steps are only held
 * back by DICT_RESIZE_FORBID, which a forked child uses to not write to the
 * tables at all. DICT_RESIZE_AVOID (a child process is saving, so writes cost
 * a page copy) is deliberately not honored here: a half migrated table is the
 * expensive state to sit in, since both tables stay allocated, every lookup
 * probes both, and inserts keep filling a destination that cannot be grown
 * again until the migration ends. Callers also wait for dictIsRehashing() to
 * clear, which a rehash held back for the whole duration of a save never
 * does. */
int dictRehash(dict *d, int n) {
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if (can_resize == DICT_RESIZE_FORBID || !dictIsRehashing(d)) return 0;

    return dictRehashForce(d, n);
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

/* Performs rehashing on a single home bucket. */
int _dictBucketRehash(dict *d, uint64_t idx) {
    if (d->pauserehash != 0) return 0;
    unsigned long s0 = DICTHT_SIZE(d->ht_size_exp[0]);
    unsigned long s1 = DICTHT_SIZE(d->ht_size_exp[1]);
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if (can_resize == DICT_RESIZE_FORBID || !dictIsRehashing(d)) return 0;
    /* If dict_can_resize is DICT_RESIZE_AVOID, we want to avoid rehashing.
     * - If expanding, the threshold is dict_force_resize_ratio which is 4.
     * - If shrinking, the threshold is 1 / (HASHTABLE_MIN_FILL * dict_force_resize_ratio) which is 1/32. */
    if (can_resize == DICT_RESIZE_AVOID &&
        ((s1 > s0 && s1 < dict_force_resize_ratio * s0) ||
         (s1 < s0 && s0 < HASHTABLE_MIN_FILL * dict_force_resize_ratio * s1)))
    {
        return 0;
    }
    if (rehashEntriesInBucketAtIndex(d, idx))
        dictCheckRehashingCompleted(d);
    return 1;
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

/* Build the slot content for 'key': a tagged key pointer for a no_value dict
 * (which never allocates), an initialized dictEntry otherwise. */
static dictEntry *createEntry(dict *d, void *key __stored_key) {
    dictEntry *entry;

    if (d->type->no_value) {
        entry = encodeEntryKey(d, key);
        debugAssert(entryIsKey(entry));
        return entry;
    }
    entry = zmalloc(sizeof(*entry));
    assert(entryIsNormal(entry)); /* Check alignment of allocation */
    entry->key = key;
    entry->v.val = NULL;
    d->allocated_entries++;
    return entry;
}

/* Insert an entry known to be absent, growing the table as many times as it
 * takes to find room for it inside the probe window. Returns the slot the
 * entry was stored in.
 *
 * This is the only place that can conclude the table cannot host a key. Rather
 * than spinning forever when growing is refused (which is what a broken table
 * looks like from the outside: a hung server), it makes sure every iteration
 * either grows the table or panics. */
static dictEntryLink insertNewKey(dict *d, dictEntry *entry, uint64_t hash) {
    int purged = 0;

    while (1) {
        int htidx = dictIsRehashing(d) ? 1 : 0;
        unsigned long home = htHome(d, htidx, hash);
        long slot;

        if (d->ht_table[htidx] != NULL &&
            (slot = htProbeInsert(d, htidx, entry, home)) >= 0)
        {
            d->ht_used[htidx]++;
            return &d->ht_table[htidx][slot];
        }

        /* The key does not fit within DICT_MAX_PROBE of its home slot.
         * Tombstones left by a paused compaction are the cheapest thing to
         * reclaim, try that once before growing. */
        if (!purged && !d->pausecompact && d->ht_tombstones[htidx]) {
            purged = 1;
            dictPurgeTombstones(d, htidx);
            continue;
        }

        /* The table has to grow, whatever the resize policy says: there is
         * nowhere else to put this key. dictExpand() ignores the policy. */
        signed char prevExp = d->ht_size_exp[0];
        dictEntry **prevTable = d->ht_table[0];

        if (dictIsRehashing(d))
            dictRebuildTables(d);
        else
            dictGrow(d);

        /* Every iteration has to change the shape of the dict, either a
         * bigger table or a rehash to one. Spinning here without making
         * progress is what used to hang the server on startup. */
        if (!dictIsRehashing(d) && d->ht_size_exp[0] == prevExp &&
            d->ht_table[0] == prevTable)
        {
            panic("dict probe overflow: cannot grow hash table (size %lu, used %lu)",
                  DICTHT_SIZE(d->ht_size_exp[0]), d->ht_used[0]);
        }
    }
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
    /* Get the position for the new key or NULL if the key already exists. */
    dictEntryLink position = dictFindLinkForInsert(d, dictStoredKey2Key(d, key), existing);
    if (!position) return NULL;

    /* Dup the key if necessary. */
    if (d->type->keyDup) key = d->type->keyDup(d, key);

    return dictInsertKeyAtLink(d, key, position);
}

/* Adds a key in the dict's hashtable at the link returned by a preceding
 * call to dictFindLinkForInsert(). This is a low level function which allows
 * splitting dictAddRaw in two parts. Normally, dictAddRaw or dictAdd should be
 * used instead. It assumes that dictExpandIfNeeded() was called before.
 *
 * With open addressing the link is the *home* slot of the key: the entry may
 * well end up in one of the following slots, so 'slotOut', when given, is set
 * to the slot the key actually landed in. */
static dictEntry *dictInsertKeyAtLinkInternal(dict *d, void *key __stored_key,
                                              dictEntryLink link,
                                              dictEntryLink *slotOut)
{
    int htidx = dictIsRehashing(d) ? 1 : 0;
    dictEntry *entry;
    unsigned long home;
    long slot;

    /* If rehashing is ongoing, we insert in table 1, otherwise in table 0.
     * Assert that the provided link is a home slot of the right table. */
    assert(link >= &d->ht_table[htidx][0] &&
           link <= &d->ht_table[htidx][DICTHT_SIZE_MASK(d->ht_size_exp[htidx])]);
    home = (unsigned long)(link - d->ht_table[htidx]);

    entry = createEntry(d, key);
    if ((slot = htProbeInsert(d, htidx, entry, home)) >= 0) {
        d->ht_used[htidx]++;
        if (slotOut) *slotOut = &d->ht_table[htidx][slot];
        return entry;
    }
    /* The key does not fit within the probe window, so the table is about to
     * be resized and the home slot we were handed becomes meaningless: fall
     * back to the full hash, which is all insertNewKey() needs. */
    link = insertNewKey(d, entry, dictGetHash(d, dictStoredKey2Key(d, key)));
    if (slotOut) *slotOut = link;
    return entry;
}

dictEntry *dictInsertKeyAtLink(dict *d, void *key __stored_key, dictEntryLink link) {
    return dictInsertKeyAtLinkInternal(d, key, link, NULL);
}

/* Add a key that is known not to exist. Unlike dictAdd()/dictAddRaw() this
 * does not search for a duplicate. Returns the inserted entry. */
dictEntry *dictAddNonExisting(dict *d, void *key __stored_key) {
    const void *lookup_key = dictStoredKey2Key(d, key);
    uint64_t hash;
    dictEntry *entry;

    debugAssert(dictFind(d, lookup_key) == NULL);

    hash = dictGetHash(d, lookup_key);

    /* Rehash and expand exactly like a regular insert would. */
    _dictRehashStepIfNeeded(d, htHome(d, 0, hash));
    _dictExpandIfNeeded(d);

    /* Dup the key if necessary. */
    if (d->type->keyDup) key = d->type->keyDup(d, key);
    entry = createEntry(d, key);
    insertNewKey(d, entry, hash);
    return entry;
}

/* Batch form of dictAddNonExisting() for an array of known-absent keys.
 * Prefetches upcoming keys and destination slots, and expands the table
 * once so it does not resize mid-batch. */
#define DICT_ADD_BATCH_PREFETCH 8 /* Power of two: ring index is a mask. */
void dictAddNonExistingBatch(dict *d, void **keys __stored_key, size_t n) {
    uint64_t hashes[DICT_ADD_BATCH_PREFETCH], hash;
    size_t i, primed;
    int htidx;
    void *key;

    if (n == 0) return;

    /* dictFind() may advance rehashing, so check before we pick the table. */
#ifdef DEBUG_ASSERTIONS
    for (i = 0; i < n; i++)
        debugAssert(dictFind(d, dictStoredKey2Key(d, keys[i])) == NULL);
#endif

    /* Expand once, with headroom so the batch does not immediately push the
     * table past its fill ratio. */
    dictExpand(d, dictSize(d) + n + ((dictSize(d) + n) >> 1));
    htidx = dictIsRehashing(d) ? 1 : 0;

    primed = n < DICT_ADD_BATCH_PREFETCH ? n : DICT_ADD_BATCH_PREFETCH;
    for (i = 0; i < primed; i++) {
        hashes[i] = dictGetHash(d, dictStoredKey2Key(d, keys[i]));
        redis_prefetch_write(&d->ht_table[htidx][htHome(d, htidx, hashes[i])]);
    }

    for (i = 0; i < n; i++) {
        /* Prefetch the key two windows ahead so its bytes are warm when hashed. */
        if (i + 2 * DICT_ADD_BATCH_PREFETCH < n)
            redis_prefetch_read(keys[i + 2 * DICT_ADD_BATCH_PREFETCH]);

        key = keys[i];
        hash = hashes[i & (DICT_ADD_BATCH_PREFETCH - 1)];
        /* Dup the key if necessary. */
        if (d->type->keyDup) key = d->type->keyDup(d, key);
        /* insertNewKey() re-reads the table, so a growth triggered mid batch
         * (an overflowing probe window) stays correct. */
        insertNewKey(d, createEntry(d, key), hash);

        /* Hash the next window and prefetch its home slot. */
        if (i + DICT_ADD_BATCH_PREFETCH < n) {
            htidx = dictIsRehashing(d) ? 1 : 0;
            hash = dictGetHash(d, dictStoredKey2Key(d, keys[i + DICT_ADD_BATCH_PREFETCH]));
            hashes[(i + DICT_ADD_BATCH_PREFETCH) & (DICT_ADD_BATCH_PREFETCH - 1)] = hash;
            redis_prefetch_write(&d->ht_table[htidx][htHome(d, htidx, hash)]);
        }
    }
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
    dictCmpCache cmpCache = {0};
    uint64_t h;
    unsigned long idx;
    int table;

    /* dict is empty */
    if (dictSize(d) == 0) return NULL;

    h = dictGetHash(d, key);
    idx = htHome(d, 0, h);

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d,idx);

    keyCmpFunc cmpFunc = dictGetCmpFunc(d);

    for (table = 0; table <= 1; table++) {
        if (d->ht_table[table] == NULL) continue;
        idx = htHome(d, table, h);
        if (table == 0 && (long)idx < d->rehashidx) continue;

        long found = htFindSlot(d, table, key, idx, cmpFunc, &cmpCache);
        if (found >= 0) {
            dictEntry *he = d->ht_table[table][found];
            htUnlinkAt(d, table, (unsigned long)found);
            if (!nofree) {
                dictFreeUnlinkedEntry(d, he);
            }
            d->ht_used[table]--;
            _dictShrinkIfNeeded(d);
            return he;
        }
        if (!dictIsRehashing(d)) break;
    }
    return NULL; /* not found */
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
    if (!entryIsKey(he)) {
        zfree(decodeMaskedPtr(he));
        d->allocated_entries--;
    }
}

/* Destroy an entire dictionary */
int _dictClear(dict *d, int htidx, void(callback)(dict*)) {
    unsigned long i, nslots = htSlotCount(d, htidx);

    /* Free all the elements */
    for (i = 0; i < nslots && d->ht_used[htidx] > 0; i++) {
        dictEntry *he;
        /* Callback will be called once for every 65535 deletions. Beware,
         * if dict has less than 65535 items, it will not be called at all.*/
        if (callback && i != 0 && (i & 65535) == 0) callback(d);

        if (!SLOT_IS_LIVE((he = d->ht_table[htidx][i]))) continue;
        dictFreeKey(d, he);
        dictFreeVal(d, he);
        if (!entryIsKey(he)) {
            zfree(decodeMaskedPtr(he));
            d->allocated_entries--;
        }
        d->ht_used[htidx]--;
    }
    /* Free the table and the allocated cache structure */
    zfree(d->ht_table[htidx]);
    /* Re-initialize the table */
    _dictReset(d, htidx);
    dictBumpLinkEpoch(d);
    return DICT_OK; /* never fails */
}

/* Clear & Release the hash table */
void dictRelease(dict *d)
{
    /* Someone may be monitoring a dict that started rehashing, before
     * destroying the dict fake completion. */
    if (dictIsRehashing(d) && d->type->rehashingCompleted)
        d->type->rehashingCompleted(d);

    /* Subtract the size of all buckets. */
    if (d->type->bucketChanged)
        d->type->bucketChanged(d, -(long long)dictBuckets(d));

    if (d->type->onDictRelease)
        d->type->onDictRelease(d);

    _dictClear(d,0,NULL);
    _dictClear(d,1,NULL);
    zfree(d);
}

/* Finds a given key. Like dictFindLink(), yet search even if dict is empty.
 *
 * Returns dictEntryLink reference if found. Otherwise, return NULL.
 *
 * bucket - return pointer to the home slot the key maps to, in the table an
 *          insertion would go to. Unless the dict has no table at all.
 */
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket) {
    dictCmpCache cmpCache = {0};
    unsigned long idx;
    int table, tables;

    if (bucket) {
        *bucket = NULL;
    } else {
        /* If dict is empty and no need to find the home slot, return NULL */
        if (dictSize(d) == 0) return NULL;
    }

    const uint64_t hash = dictGetHash(d, key);
    keyCmpFunc cmpFunc = dictGetCmpFunc(d);

    /* Rehash the hash table if needed */
    if (d->ht_table[0]) _dictRehashStepIfNeeded(d, htHome(d, 0, hash));

    tables = (dictIsRehashing(d)) ? 2 : 1;
    if (bucket && d->ht_table[tables - 1])
        *bucket = &d->ht_table[tables - 1][htHome(d, tables - 1, hash)];

    for (table = 0; table < tables; table++) {
        if (d->ht_table[table] == NULL) continue;
        idx = htHome(d, table, hash);
        if (table == 0 && (long)idx < d->rehashidx) continue;

        long found = htFindSlot(d, table, key, idx, cmpFunc, &cmpCache);
        if (found >= 0) return &d->ht_table[table][found];
    }
    return NULL;
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
    unsigned long idx, pos, limit, nslots;
    int table;
    unsigned int dist;

    if (dictSize(d) == 0) return NULL; /* dict is empty */
    for (table = 0; table <= 1; table++) {
        if (d->ht_table[table] == NULL) continue;
        idx = htHome(d, table, hash);
        if (table == 0 && (long)idx < d->rehashidx) continue;

        nslots = htSlotCount(d, table);
        limit = idx + DICT_MAX_PROBE;
        if (limit > nslots) limit = nslots;
        uint8_t *meta = htMeta(d, table);
        for (pos = idx, dist = 0; pos < limit; pos++, dist++) {
            dictEntry *he = d->ht_table[table][pos];
            if (he == NULL) break;
            if (meta[pos] < dist) break;
            if (he == TOMBSTONE) continue;
            if (oldptr == dictGetKey(he)) return he;
        }
        if (!dictIsRehashing(d)) return NULL;
    }
    return NULL;
}

/* Find a key and return its dictEntryLink reference. Otherwise, return NULL
 * 
 * A dictEntryLink points to the slot holding the entry. It is useful for
 * deletion, addition, unlinking and updating, especially for dict configured
 * with 'no_value'. In such cases returning only `dictEntry` from a lookup may
 * be insufficient since it might be opt-out to be the object itself.
 * 
 * After calling link = dictFindLink(...), any necessary updates based on returned 
 * link or bucket must be performed immediately after by calling dictSetKeyAtLink() 
 * without any intervening operations on given dict. Otherwise, `dictEntryLink` may 
 * become invalid: with open addressing an insertion may move entries around, and
 * a resize invalidates every link. Example with kvobj of replacing key with new key:
 * 
 *      link = dictFindLink(d, key, &bucket);
 *      ... Do something, but don't modify the dict ...
 *      // assert(link != NULL);
 *      dictSetKeyAtLink(d, kv, &link, 0);
 *      
 * To add new value (If no space for the new key, dict will be expanded by
 * dictSetKeyAtLink() and the home slot will be looked up again.):
 *   
 *      link = dictFindLink(d, key, &bucket);
 *      ... Do something, but don't modify the dict ...
 *      // assert(link == NULL);
 *      dictSetKeyAtLink(d, kv, &bucket, 1);
 *  
 *  bucket - return link to the home slot that the key was mapped to, unless
 *           the dict has no table yet.
 */
dictEntryLink dictFindLink(dict *d, const void *key, dictEntryLink *bucket) {
    if (bucket) *bucket = NULL;
    if (unlikely(dictSize(d) == 0))
        return NULL;
    
    return dictFindLinkInternal(d, key, bucket);
}

/* Set the key with link 
 *
 * link:    - When `newItem` is set, `link` points to the home slot of the key.
 *          - When `newItem` is not set, `link` points to the slot of the key.
 *          - If *link is NULL, dictFindLink() will be called to locate the key.
 *          - On return, get updated, by need, to the inserted key. 
 *
 * newItem: 1 = Add a key as a new item.
 *          0 = Set the key of an existing item.
 */
void dictSetKeyAtLink(dict *d, void *key __stored_key, dictEntryLink *link, int newItem) {
    dictEntryLink dummy = NULL;
    if (link == NULL) link = &dummy;
    void *addedKey = (d->type->keyDup) ? d->type->keyDup(d, key) : key;
    
    if (newItem) {
        signed char snap[2] = {d->ht_size_exp[0], d->ht_size_exp[1] };

        /* Make room if needed for the new key */
        dictExpandIfNeeded(d);
        
        /* Lookup key's link if tables reallocated or if given link is set to NULL */
        if (snap[0] != d->ht_size_exp[0] || snap[1] != d->ht_size_exp[1] || *link == NULL) {
            dictEntryLink bucket;
            /* Bypass dictFindLink() to search the home slot even if dict is empty!!! */
            *link = dictFindLinkInternal(d, dictStoredKey2Key(d, key), &bucket);
            assert(bucket != NULL);
            assert(*link == NULL);
            *link = bucket; /* On newItem the link should be the home slot */
        }
        /* Report back the slot the key landed in: it is rarely the home slot
         * and the caller may keep using the link (to attach a TTL, say). */
        dictInsertKeyAtLinkInternal(d, addedKey, *link, link);
        return;
    } 
    
    /* Setting key of an existing item (newItem == 0) */
    
    if (*link == NULL) {
        *link = dictFindLink(d, key, NULL);
        assert(*link != NULL);
    }
    
    dictEntry **slot = *link;
    if (entryIsKey(*slot)) {
        /* The slot holds the key itself. Replace it, keeping the lsb flags. */
        *slot = encodeEntryKey(d, addedKey);
    } else {
        (*slot)->key = addedKey;
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
 * `dictTwoPhaseUnlinkFind` pauses rehash and `dictTwoPhaseUnlinkFree` resumes rehash.
 *
 * We can use like this:
 *
 * dictEntryLink link = dictTwoPhaseUnlinkFind(db->dict,key->ptr, &table);
 * // Do something, but we can't modify the dict
 * dictTwoPhaseUnlinkFree(db->dict, link, table); // We don't need to lookup again
 *
 * If we want to find an entry before delete this entry, this an optimization to avoid
 * dictFind followed by dictDelete. i.e. the first API is a find, and it gives some info
 * to the second one to avoid repeating the lookup
 */
dictEntryLink dictTwoPhaseUnlinkFind(dict *d, const void *key, int *table_index) {
    dictCmpCache cmpCache = {0};
    uint64_t h;
    unsigned long idx;
    int table;

    if (dictSize(d) == 0) return NULL; /* dict is empty */
    if (dictIsRehashing(d)) _dictRehashStep(d);

    h = dictGetHash(d, key);
    keyCmpFunc cmpFunc = dictGetCmpFunc(d);

    for (table = 0; table <= 1; table++) {
        if (d->ht_table[table] == NULL) continue;
        idx = htHome(d, table, h);
        if (table == 0 && (long)idx < d->rehashidx) continue;

        long found = htFindSlot(d, table, key, idx, cmpFunc, &cmpCache);
        if (found >= 0) {
            *table_index = table;
            dictPauseRehashing(d);
            return &d->ht_table[table][found];
        }
        if (!dictIsRehashing(d)) return NULL;
    }
    return NULL;
}

void dictTwoPhaseUnlinkFree(dict *d, dictEntryLink plink, int table_index) {
    if (plink == NULL || *plink == NULL) return;
    dictEntry *de = *plink;
    unsigned long idx = (unsigned long)(plink - d->ht_table[table_index]);

    htUnlinkAt(d, table_index, idx);
    dictFreeKey(d, de);
    dictFreeVal(d, de);
    if (!entryIsKey(de)) {
        zfree(decodeMaskedPtr(de));
        d->allocated_entries--;
    }
    d->ht_used[table_index]--;
    _dictShrinkIfNeeded(d);
    dictResumeRehashing(d);
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
    /* if entryIsKey() */
    if ((uintptr_t)de & ENTRY_PTR_IS_ODD_KEY) return (void *) de;
    if ((uintptr_t)de & ENTRY_PTR_IS_EVEN_KEY) return decodeMaskedPtr(de);    
    /* Regular entry */
    return de->key;
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

/* Entries have no chain in an open addressed table. Kept for API
 * compatibility: it always reports "no next entry". */
dictEntry *dictGetNext(const dictEntry *de) {
    (void)de;
    return NULL;
}

/* Returns the memory usage in bytes of the dict, excluding the size of the keys
 * and values. */
size_t dictMemUsage(const dict *d) {
    size_t tables = 0;

    for (int htidx = 0; htidx <= 1; htidx++) {
        if (d->ht_table[htidx] == NULL) continue;
        tables += htAllocSize(d->ht_size_exp[htidx]);
    }
    /* A no_value dict stores its keys directly in the slots and never
     * allocates an entry, so allocated_entries is 0 for those. */
    return d->allocated_entries * dictEntryMemUsage(d->type->no_value) + tables;
}

size_t dictEntryMemUsage(int noValueDict) {
    return (noValueDict) ? 0 : sizeof(dictEntry);
}

/* A fingerprint is a 64 bit number that represents the state of the dictionary
 * at a given time, it's just a few dict properties xored together.
 * When an unsafe iterator is initialized, we get the dict fingerprint, and check
 * the fingerprint again when the iterator is released.
 * If the two fingerprints are different it means that the user of the iterator
 * performed forbidden operations against the dictionary while iterating. */
unsigned long long dictFingerprint(dict *d) {
    unsigned long long integers[6], hash = 0;
    int j;

    integers[0] = (long) d->ht_table[0];
    integers[1] = d->ht_size_exp[0];
    integers[2] = d->ht_used[0];
    integers[3] = (long) d->ht_table[1];
    integers[4] = d->ht_size_exp[1];
    integers[5] = d->ht_used[1];

    /* We hash N integers by summing every successive integer with the integer
     * hashing of the previous sum. Basically:
     *
     * Result = hash(hash(hash(int1)+int2)+int3) ...
     *
     * This way the same set of integers in a different order will (likely) hash
     * to a different number. */
    for (j = 0; j < 6; j++) {
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
    iter->safe = 0;
    iter->entry = NULL;
    iter->nextEntry = NULL;
}

void dictInitSafeIterator(dictIterator *iter, dict *d)
{
    dictInitIterator(iter, d);
    iter->safe = 1;
}

void dictResetIterator(dictIterator *iter)
{
    if (!(iter->index == -1 && iter->table == 0)) {
        if (iter->safe) {
            dictResumeRehashing(iter->d);
            dictResumeCompact(iter->d);
        } else {
            assert(iter->fingerprint == dictFingerprint(iter->d));
        }
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

    while (1) {
        if (iter->index == -1 && iter->table == 0) {
            if (iter->safe) {
                dictPauseRehashing(d);
                /* A backward shift could move a not yet visited entry into a
                 * slot we already passed, so deletions must leave tombstones
                 * for as long as we walk the slots. */
                dictPauseCompact(d);
            } else {
                iter->fingerprint = dictFingerprint(d);
            }

            /* skip the rehashed slots in table[0] */
            if (dictIsRehashing(d))
                iter->index = d->rehashidx - 1;
        }

        iter->index++;
        if (iter->index >= (long) htSlotCount(d, iter->table)) {
            if (dictIsRehashing(d) && iter->table == 0) {
                iter->table++;
                iter->index = 0;
            } else {
                break;
            }
        }

        dictEntry *de = d->ht_table[iter->table][iter->index];
        if (SLOT_IS_LIVE(de)) {
            iter->entry = de;
            iter->nextEntry = NULL; /* no chains: nothing to save */
            return de;
        }
    }
    iter->entry = NULL;
    return NULL;
}

void dictReleaseIterator(dictIterator *iter)
{
    dictResetIterator(iter);
    zfree(iter);
}

/* Return a random entry from the hash table. Useful to
 * implement randomized algorithms */
dictEntry *dictGetRandomKey(dict *d)
{
    unsigned long s0, total, h, i;

    if (dictSize(d) == 0) return NULL;
    if (dictIsRehashing(d)) _dictRehashStep(d);

    s0 = htSlotCount(d, 0);
    total = s0 + htSlotCount(d, 1);

    /* Rejection sampling: every element sits in a slot of its own, so a
     * uniformly picked live slot is a uniformly picked element. The expected
     * number of tries is the inverse of the load factor, so budget a generous
     * multiple of it: overshooting it that far is so unlikely that the walk
     * below costs nothing on average, however empty the table is. */
    unsigned long tries = 8 * (total / dictSize(d) + 1);
    for (i = 0; i < tries; i++) {
        h = randomULong() % total;
        dictEntry *de = (h >= s0) ? d->ht_table[1][h - s0] : d->ht_table[0][h];
        if (SLOT_IS_LIVE(de)) return de;
    }

    /* Out of budget: walk the table and keep a reservoir sample, which
     * terminates and stays uniform where picking the first live slot found
     * would favour the entries that follow a long empty stretch. */
    dictEntry *chosen = NULL;
    unsigned long seen = 0;
    for (h = 0; h < total; h++) {
        dictEntry *de = (h >= s0) ? d->ht_table[1][h - s0] : d->ht_table[0][h];
        if (!SLOT_IS_LIVE(de)) continue;
        if (randomULong() % ++seen == 0) chosen = de;
    }
    return chosen;
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
    unsigned long stored = 0, maxsteps;
    unsigned long s0, total, i;
    unsigned long emptylen = 0; /* Continuous empty slots so far. */

    if (dictSize(d) < count) count = dictSize(d);
    if (count == 0) return 0;
    maxsteps = count*10;

    /* Try to do a rehashing work proportional to 'count'. */
    for (j = 0; j < count; j++) {
        if (dictIsRehashing(d))
            _dictRehashStep(d);
        else
            break;
    }

    s0 = htSlotCount(d, 0);
    total = s0 + htSlotCount(d, 1);

    /* Pick a random point and walk forward over the physical slots. */
    i = randomULong() % total;
    while (stored < count && maxsteps--) {
        dictEntry *he = (i >= s0) ? d->ht_table[1][i - s0] : d->ht_table[0][i];

        if (!SLOT_IS_LIVE(he)) {
            /* Count contiguous empty slots, and jump to another location if
             * they reach 'count' (with a minimum of 5). */
            emptylen++;
            if (emptylen >= 5 && emptylen > count) {
                i = randomULong() % total;
                emptylen = 0;
                continue;
            }
        } else {
            emptylen = 0;
            des[stored++] = he;
        }
        if (++i >= total) i = 0;
    }
    /* Coming back empty handed from a dict that has keys is not just a poor
     * sample, it is indistinguishable from an empty dict: eviction, for one,
     * gives up and reports OOM. The walk above is bounded, so a table that is
     * mostly empty (one that could not be shrunk while memory was tight, for
     * example) can miss everything. Pay for one uniform pick in that case. */
    if (stored == 0) {
        dictEntry *de = dictGetRandomKey(d);
        if (de) des[stored++] = de;
    }
    return stored > count ? count : stored;
}

/* Reallocate the dictEntry, key and value of a single slot using the provided
 * allocation functions in order to defrag them. */
static void dictDefragSlot(dict *d, dictEntry **slotref, dictDefragFunctions *defragfns) {
    dictDefragAllocFunction *defragalloc = defragfns->defragAlloc;
    dictDefragAllocFunction *defragkey = defragfns->defragKey;
    dictDefragAllocFunction *defragval = defragfns->defragVal;
    dictEntry *de = *slotref, *newde = NULL;
    void *newkey;

    if (!SLOT_IS_LIVE(de)) return;

    newkey = defragkey ? defragkey(dictGetKey(de)) : NULL;

    if (d->type->no_value) {
        /* The key is stored inline in the slot, there is no entry to move. */
        if (newkey) *slotref = encodeEntryKey(d, newkey);
        return;
    }

    void *newval = defragval ? defragval(dictGetVal(de)) : NULL;
    assert(entryIsNormal(de));
    newde = defragalloc(de);
    if (newde) de = newde;
    if (newkey) de->key = newkey;
    if (newval) de->v.val = newval;
    if (newde) *slotref = newde;
}

/* This is like dictGetRandomKey() from the POV of the API, but guarantees a
 * good distribution of the returned element.
 *
 * With chaining, dictGetRandomKey() had to pick a bucket and then an element
 * of its chain, which over-sampled the keys of short chains and is why this
 * function had to sample a range of buckets and choose from it. Every element
 * now occupies one slot of its own, so picking a random live slot is already
 * uniform and there is nothing left to correct. */
dictEntry *dictGetFairRandomKey(dict *d) {
    return dictGetRandomKey(d);
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
 * called with 'privdata' as first argument and the dictionary entry
 * 'de' as second argument.
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
 * dict.c hash tables are always power of two in size, and the cursor
 * enumerates *home* buckets: the position of an element in a given table is
 * the bitwise AND between Hash(key) and SIZE-1, plus a displacement that never
 * leaves the DICT_MAX_PROBE slots that follow. Inserting or deleting other
 * keys may shuffle an element within those slots but never changes its home
 * bucket, so a scan that walks home buckets is stable under concurrent
 * modification, exactly like the chained implementation was.
 *
 * WHAT HAPPENS IF THE TABLE CHANGES IN SIZE?
 *
 * If the hash table grows, elements can go anywhere in one multiple of
 * the old bucket: for example let's say we already iterated with
 * a 4 bit cursor 1100 (the mask is 1111 because hash table size = 16).
 *
 * If the hash table will be resized to 64 elements, then the new mask will
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
 *    return all the keys homed at a given bucket, and all the expansions, so
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

/* Emit (and optionally defrag) every entry whose home bucket in table 'htidx'
 * is 'home'. Those entries live in the run of occupied slots starting at
 * 'home', within DICT_MAX_PROBE slots.
 *
 * The callback may delete the entry it is given (compaction is paused, so the
 * slot just becomes a tombstone) and may even insert into the dict, which can
 * relocate the table. Therefore the slot address is recomputed on every
 * iteration rather than cached. */
static void dictScanDefragBucket(dict *d, int htidx, unsigned long home,
                                 dictScanFunction *fn,
                                 dictDefragFunctions *defragfns,
                                 void *privdata)
{
    unsigned long pos;

    for (pos = home; pos < home + DICT_MAX_PROBE; pos++) {
        if (pos >= htSlotCount(d, htidx)) break;

        dictEntry **slot = &d->ht_table[htidx][pos];
        unsigned int dist = (unsigned int)(pos - home);

        if (*slot == NULL) break;
        /* Robin Hood ordering: past this point no entry can be homed here. */
        if (htMeta(d, htidx)[pos] < dist) break;
        if (*slot == TOMBSTONE) continue;
        if (htMeta(d, htidx)[pos] != dist) continue; /* homed elsewhere */

        if (defragfns) dictDefragSlot(d, slot, defragfns);
        fn(privdata, *slot, slot);
    }
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
    dictPauseRehashing(d);
    /* Deletions from the callback must not shift entries backwards, or we
     * would walk past entries we have not emitted yet. */
    dictPauseCompact(d);

    if (!dictIsRehashing(d)) {
        htidx0 = 0;
        m0 = DICTHT_SIZE_MASK(d->ht_size_exp[htidx0]);
        dictScanDefragBucket(d, htidx0, v & m0, fn, defragfns, privdata);

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
        if (DICTHT_SIZE(d->ht_size_exp[htidx0]) > DICTHT_SIZE(d->ht_size_exp[htidx1])) {
            htidx0 = 1;
            htidx1 = 0;
        }

        m0 = DICTHT_SIZE_MASK(d->ht_size_exp[htidx0]);
        m1 = DICTHT_SIZE_MASK(d->ht_size_exp[htidx1]);

        dictScanDefragBucket(d, htidx0, v & m0, fn, defragfns, privdata);

        /* Iterate over indices in larger table that are the expansion
         * of the index pointed to by the cursor in the smaller table */
        do {
            dictScanDefragBucket(d, htidx1, v & m1, fn, defragfns, privdata);

            /* Increment the reverse cursor not covered by the smaller mask.*/
            v |= ~m1;
            v = rev(v);
            v++;
            v = rev(v);

            /* Continue while bits covered by mask difference is non-zero */
        } while (v & (m0 ^ m1));
    }

    dictResumeCompact(d);
    dictResumeRehashing(d);

    return v;
}

/* ------------------------- private functions ------------------------------ */

/* Because we may need to allocate huge memory chunk at once when dict
 * resizes, we will check this allocation is allowed or not if the dict
 * type has resizeAllowed member function. */
static int dictTypeResizeAllowed(dict *d, size_t size) {
    if (d->type->resizeAllowed == NULL) return 1;
    if (DICTHT_SIZE(d->ht_size_exp[0]) == 0) return 1;
    return d->type->resizeAllowed(
                    htAllocSize(_dictNextExp(size)),
                    (double)d->ht_used[0] / DICTHT_SIZE(d->ht_size_exp[0]));
}

/* Returning DICT_OK indicates a successful expand or the dictionary is undergoing rehashing, 
 * and there is nothing else we need to do about this dictionary currently. While DICT_ERR indicates
 * that expand has not been triggered (may be try shrinking?)*/
int dictExpandIfNeeded(dict *d) {
    /* If the hash table is empty expand it to the initial size. This must
     * succeed: every insert path relies on having a table to probe into. */
    if (!dictIsRehashing(d) && DICTHT_SIZE(d->ht_size_exp[0]) == 0) {
        int ret = dictExpand(d, DICT_HT_INITIAL_SIZE);
        assert(ret == DICT_OK && d->ht_table[0] != NULL);
        return DICT_OK;
    }

    /* Grow well before the table is full: with linear probing the probe
     * sequences get long (and the DICT_MAX_PROBE window tight) as the fill
     * ratio approaches 1. Tombstones count as fill since they lengthen
     * probes just like live entries do.
     *
     * While rehashing, the table to watch is ht[1]: new keys go there and
     * everything ht[0] still holds has to fit there as well. Letting it
     * saturate would corner us, since a resize cannot start before the
     * pending rehash finishes, and that rehash needs room in ht[1]. */
    int htidx = dictIsRehashing(d) ? 1 : 0;
    unsigned long size = DICTHT_SIZE(d->ht_size_exp[htidx]);
    unsigned long fill = dictSize(d) + d->ht_tombstones[htidx];
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);

    /* size - size/4 == 75%, size - size/10 == 90%, computed this way to avoid
     * overflowing on huge tables. */
    if ((can_resize == DICT_RESIZE_ENABLE &&
         fill >= size - size / (100 / (100 - DICT_GROW_PERCENT))) ||
        (can_resize != DICT_RESIZE_FORBID &&
         fill >= size - size / (100 / (100 - DICT_GROW_PERCENT_AVOID))))
    {
        /* A table clogged with tombstones only needs a cleanup. */
        if (d->ht_tombstones[htidx] > d->ht_used[htidx] && !d->pausecompact) {
            dictPurgeTombstones(d, htidx);
            return DICT_OK;
        }
        /* A resize cannot start while a rehash is in flight, and ht[1] is
         * running out of room, so finishing it now is not optional. */
        if (dictIsRehashing(d)) completeRehash(d);
        if (dictTypeResizeAllowed(d, DICTHT_SIZE(d->ht_size_exp[0]) + 1))
            dictGrow(d); /* Next power of two: double the table. */
        return DICT_OK;
    }
    return dictIsRehashing(d) ? DICT_OK : DICT_ERR;
}

/* Expand the hash table if needed (OK=Expanded, ERR=Not expanded) */
static int _dictExpandIfNeeded(dict *d) {
    /* Automatic resizing is disallowed. Return. Creating the very first table
     * is not optional though: the insert paths need something to probe. */
    if (d->pauseAutoResize > 0 && DICTHT_SIZE(d->ht_size_exp[0]) != 0)
        return DICT_ERR;

    return dictExpandIfNeeded(d);
}

/* Returning DICT_OK indicates a successful shrinking or the dictionary is undergoing rehashing, 
 * and there is nothing else we need to do about this dictionary currently. While DICT_ERR indicates
 * that shrinking has not been triggered (may be try expanding?)*/
int dictShrinkIfNeeded(dict *d) {
    /* Incremental rehashing already in progress. Return. */
    if (dictIsRehashing(d)) return DICT_OK;
    
    /* If the size of hash table is DICT_HT_INITIAL_SIZE, don't shrink it. */
    if (DICTHT_SIZE(d->ht_size_exp[0]) <= DICT_HT_INITIAL_SIZE) return DICT_OK;

    /* If we reached below 1:8 elements/buckets ratio, and we are allowed to resize
     * the hash table (global setting) or we should avoid it but the ratio is below 1:32,
     * we'll trigger a resize of the hash table. */
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if ((can_resize == DICT_RESIZE_ENABLE &&
         d->ht_used[0] * HASHTABLE_MIN_FILL <= DICTHT_SIZE(d->ht_size_exp[0])) ||
        (can_resize != DICT_RESIZE_FORBID &&
         d->ht_used[0] * HASHTABLE_MIN_FILL * dict_force_resize_ratio <= DICTHT_SIZE(d->ht_size_exp[0])))
    {
        /* Leave headroom so the shrunk table does not start out near the
         * fill ratio that triggers a growth. */
        unsigned long target = d->ht_used[0] * 2 + 1;
        if (dictTypeResizeAllowed(d, target))
            dictShrink(d, target);
        return DICT_OK;
    }
    return DICT_ERR;
}

/* Shrink the hash table if needed and complete the rehash immediately.
 * Unlike the keyspace dicts, which the cron steps through kvstore, a dict
 * owned by a single object is only stepped by commands that touch it. A bulk
 * deletion would otherwise leave the old table allocated until later. */
void dictShrinkIfNeededAndComplete(dict *d) {
    /* Incremental rehashing already in progress. Return. */
    if (dictIsRehashing(d) || d->pauserehash != 0) return;
    if (dictShrinkIfNeeded(d) != DICT_OK) return;

    /* If resizing is not fully enabled (e.g. a child save is running)
     * leave the rehash incremental. */
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if (can_resize != DICT_RESIZE_ENABLE) return;

    while (dictRehash(d, 1000)) {
        /* Continue rehashing */
    }
}

static void _dictShrinkIfNeeded(dict *d)
{
    /* Automatic resizing is disallowed. Return */
    if (d->pauseAutoResize > 0) return;

    dictShrinkIfNeeded(d);
}

static void _dictRehashStepIfNeeded(dict *d, uint64_t visitedIdx) {
    if ((!dictIsRehashing(d)) || (d->pauserehash != 0))
        return;
    /* rehashing not in progress if rehashidx == -1 */
    if ((long)visitedIdx >= d->rehashidx && d->ht_table[0][visitedIdx]) {
        /* If we have a valid hash entry at `idx` in ht0, we perform
         * rehash on the bucket at `idx` (being more CPU cache friendly) */
        _dictBucketRehash(d, visitedIdx);
    } else {
        /* If the hash entry is not in ht0, we rehash the buckets based
         * on the rehashidx (not CPU cache friendly). */
        dictRehash(d,1);
    }
}

/* Our hash table capability is a power of two */
static signed char _dictNextExp(unsigned long size)
{
    if (size <= DICT_HT_INITIAL_SIZE) return DICT_HT_INITIAL_EXP;
    if (size >= LONG_MAX) return (8*sizeof(long)-1);

    return 8*sizeof(long) - __builtin_clzl(size-1);
}

/* Finds and returns the link within the dict where the provided key should
 * be inserted using dictInsertKeyAtLink() if the key does not already exist in
 * the dict. If the key exists in the dict, NULL is returned and the optional
 * 'existing' entry pointer is populated, if provided.
 *
 * The returned link is the home slot of the key, the insertion itself decides
 * which slot of the probe window the key ends up in. */
dictEntryLink dictFindLinkForInsert(dict *d, const void *key, dictEntry **existing) {
    unsigned long idx;
    int table, htidx;
    dictCmpCache cmpCache = {0};
    uint64_t hash = dictGetHash(d, key);
    if (existing) *existing = NULL;

    /* Rehash the hash table if needed */
    if (d->ht_table[0]) _dictRehashStepIfNeeded(d, htHome(d, 0, hash));

    /* Expand the hash table if needed */
    _dictExpandIfNeeded(d);
    keyCmpFunc cmpFunc = dictGetCmpFunc(d);

    for (table = 0; table <= 1; table++) {
        if (d->ht_table[table] == NULL) continue;
        idx = htHome(d, table, hash);
        if (table == 0 && (long)idx < d->rehashidx) continue;

        long found = htFindSlot(d, table, key, idx, cmpFunc, &cmpCache);
        if (found >= 0) {
            if (existing) *existing = d->ht_table[table][found];
            return NULL;
        }
        if (!dictIsRehashing(d)) break;
    }

    /* If we are in the process of rehashing the hash table, the home slot is
     * always returned in the context of the second (new) hash table. */
    htidx = dictIsRehashing(d) ? 1 : 0;
    assert(d->ht_table[htidx] != NULL);
    return &d->ht_table[htidx][htHome(d, htidx, hash)];
}

void dictEmpty(dict *d, void(callback)(dict*)) {
    /* Someone may be monitoring a dict that started rehashing, before
     * destroying the dict fake completion. */
    if (dictIsRehashing(d) && d->type->rehashingCompleted)
        d->type->rehashingCompleted(d);

    /* Subtract the size of all buckets. */
    if (d->type->bucketChanged)
        d->type->bucketChanged(d, -(long long)dictBuckets(d));

    _dictClear(d,0,callback);
    _dictClear(d,1,callback);
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pausecompact = 0;
    d->pauseAutoResize = 0;
}

void dictSetResizeEnabled(dictResizeEnable enable) {
    atomicSet(dict_can_resize, enable);
}

/* Compiler inlines this for internal calls within dict.c (verified with -O3). */
uint64_t dictGetHash(dict *d, const void *key) {
    return d->type->hashFunction(key);
}

/* Provides the old and new ht size for a given dictionary during rehashing. This method
 * should only be invoked during initialization/rehashing. */
void dictRehashingInfo(dict *d, unsigned long long *from_size, unsigned long long *to_size) {
    /* Invalid method usage if rehashing isn't ongoing. */
    assert(dictIsRehashing(d));
    *from_size = DICTHT_SIZE(d->ht_size_exp[0]);
    *to_size = DICTHT_SIZE(d->ht_size_exp[1]);
}

/* ------------------------------- Debugging ---------------------------------*/
#define DICT_STATS_VECTLEN 50
void dictFreeStats(dictStats *stats) {
    zfree(stats->clvector);
    zfree(stats);
}

void dictCombineStats(dictStats *from, dictStats *into) {
    into->buckets += from->buckets;
    into->maxChainLen = (from->maxChainLen > into->maxChainLen) ? from->maxChainLen : into->maxChainLen;
    into->totalChainLen += from->totalChainLen;
    into->htSize += from->htSize;
    into->htUsed += from->htUsed;
    for (int i = 0; i < DICT_STATS_VECTLEN; i++) {
        into->clvector[i] += from->clvector[i];
    }
}

/* Stats of an open addressed table are reported in terms of home buckets: a
 * "chain" is the set of entries sharing a home bucket, which is the closest
 * analogue to the chains of the previous implementation. */
dictStats *dictGetStatsHt(dict *d, int htidx, int full) {
    unsigned long *clvector = zcalloc(sizeof(unsigned long) * DICT_STATS_VECTLEN);
    dictStats *stats = zcalloc(sizeof(dictStats));
    stats->htidx = htidx;
    stats->clvector = clvector;
    stats->htSize = DICTHT_SIZE(d->ht_size_exp[htidx]);
    stats->htUsed = d->ht_used[htidx];
    if (!full || d->ht_table[htidx] == NULL) return stats;

    unsigned long nslots = htSlotCount(d, htidx);
    uint8_t *meta = htMeta(d, htidx);
    unsigned long curHome = 0, curLen = 0;

    for (unsigned long i = 0; i <= nslots; i++) {
        dictEntry *he = (i < nslots) ? d->ht_table[htidx][i] : NULL;
        unsigned long home = 0;
        int live = SLOT_IS_LIVE(he);

        if (live) home = i - meta[i];
        if (curLen && (!live || home != curHome)) {
            /* Close the group of the previous home bucket. */
            clvector[(curLen < DICT_STATS_VECTLEN) ? curLen : (DICT_STATS_VECTLEN-1)]++;
            if (curLen > stats->maxChainLen) stats->maxChainLen = curLen;
            stats->totalChainLen += curLen;
            stats->buckets++;
            curLen = 0;
        }
        if (live) {
            curHome = home;
            curLen++;
        }
    }
    clvector[0] = stats->htSize > stats->buckets ? stats->htSize - stats->buckets : 0;

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
                      " different slots: %lu\n"
                      " max chain length: %lu\n"
                      " avg chain length (counted): %.02f\n"
                      " avg chain length (computed): %.02f\n"
                      " Chain length distribution:\n",
                      stats->buckets, stats->maxChainLen,
                      (float) stats->totalChainLen / stats->buckets, (float) stats->htUsed / stats->buckets);

        for (unsigned long i = 0; i < DICT_STATS_VECTLEN - 1; i++) {
            if (stats->clvector[i] == 0) continue;
            if (l >= bufsize) break;
            l += snprintf(buf + l, bufsize - l,
                          "   %ld: %ld (%.02f%%)\n",
                          i, stats->clvector[i], ((float) stats->clvector[i] / stats->htSize) * 100);
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

/* ------------------------------- Benchmark ---------------------------------*/

#ifdef REDIS_TEST
#include "testhelp.h"

#define UNUSED(V) ((void) V)
#define TEST(name) printf("test — %s\n", name);

/* Validate the structural invariants of every table of 'd':
 *  - each entry sits within DICT_MAX_PROBE slots of its home bucket and the
 *    recorded displacement matches the real one,
 *  - the run between an entry and its home bucket has no empty slot, so a
 *    probe starting at the home bucket always reaches it,
 *  - entries in a run are ordered by home bucket (Robin Hood invariant), which
 *    is what makes the early exit in htFindSlot() correct,
 *  - the used/tombstone counters match what the slots actually hold. */
static void dictVerify(dict *d) {
    for (int htidx = 0; htidx <= 1; htidx++) {
        if (d->ht_table[htidx] == NULL) {
            assert(d->ht_size_exp[htidx] == -1);
            assert(d->ht_used[htidx] == 0);
            assert(d->ht_tombstones[htidx] == 0);
            continue;
        }

        unsigned long nslots = htSlotCount(d, htidx);
        unsigned long size = DICTHT_SIZE(d->ht_size_exp[htidx]);
        uint8_t *meta = htMeta(d, htidx);
        unsigned long used = 0, tombs = 0;
        unsigned long prevHome = 0;
        int inRun = 0;

        for (unsigned long i = 0; i < nslots; i++) {
            dictEntry *de = d->ht_table[htidx][i];

            if (de == NULL) {
                inRun = 0;
                continue;
            }

            assert(meta[i] < DICT_MAX_PROBE);
            unsigned long home = i - meta[i];
            assert(home < size);
            /* Every slot between the home bucket and this one is occupied. */
            for (unsigned long j = home; j < i; j++)
                assert(d->ht_table[htidx][j] != NULL);
            /* Runs are ordered by home bucket. */
            if (inRun) assert(home >= prevHome);
            prevHome = home;
            inRun = 1;

            if (de == TOMBSTONE) {
                tombs++;
                continue;
            }
            used++;

            /* The key must be reachable through the regular lookup path. */
            const void *key = dictStoredKey2Key(d, dictGetKey(de));
            dictCmpCache cache = {0};
            long found = htFindSlot(d, htidx, key, htHome(d, htidx, dictGetHash(d, key)),
                                    dictGetCmpFunc(d), &cache);
            assert(found == (long)i);
        }
        assert(used == d->ht_used[htidx]);
        assert(tombs == d->ht_tombstones[htidx]);
    }
}

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

/* Same as BenchmarkDictType, with no_value=1. */
static dictType BenchmarkDictTypeNoValue = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
};

/* Count the live entries by walking the raw slots of both tables. */
static unsigned long dictWalkLiveEntries(dict *d) {
    unsigned long live = 0;

    for (int htidx = 0; htidx <= 1; htidx++) {
        if (d->ht_table[htidx] == NULL) continue;
        for (unsigned long i = 0; i < htSlotCount(d, htidx); i++)
            if (SLOT_IS_LIVE(d->ht_table[htidx][i])) live++;
    }
    return live;
}

/* Scan callback: marks every key it sees in a bitmap of 'count' longs. */
typedef struct {
    char *seen;
    long count;
} scanTestData;

static void scanTestCallback(void *privdata, const dictEntry *de, dictEntryLink plink) {
    scanTestData *data = privdata;
    long v = strtol(dictGetKey(de), NULL, 10);
    UNUSED(plink);
    assert(v >= 0 && v < data->count);
    data->seen[v] = 1;
}

/* Scan callback deleting every key it is given, to exercise tombstones. */
static void scanDeleteCallback(void *privdata, const dictEntry *de, dictEntryLink plink) {
    dict *d = privdata;
    UNUSED(plink);
    assert(dictDelete(d, dictGetKey(de)) == DICT_OK);
}

#define start_benchmark() start = timeInMilliseconds()
#define end_benchmark(msg) do { \
    elapsed = timeInMilliseconds()-start; \
    printf(msg ": %ld items in %lld ms\n", count, elapsed); \
} while(0)

/* ./redis-server test dict [<count> | --accurate] */
int dictTest(int argc, char **argv, int flags) {
    long j;
    long long start, elapsed;
    int retval;
    dict *d = dictCreate(&BenchmarkDictType);
    dictEntry* de = NULL;
    dictEntry* existing = NULL;
    long count = 0;
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

    dictSetResizeEnabled(DICT_RESIZE_ENABLE);

    TEST("An empty dict grows to the initial size on first insert") {
        assert(d->ht_size_exp[0] == -1 && d->ht_table[0] == NULL);
        assert(dictExpandIfNeeded(d) == DICT_OK);
        assert(DICTHT_SIZE(d->ht_size_exp[0]) == DICT_HT_INITIAL_SIZE);
        assert(d->ht_table[0] != NULL);
        assert(dictAdd(d, stringFromLongLong(0), (void*)0) == DICT_OK);
        assert(dictSize(d) == 1);
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Add, find and delete keys") {
        for (j = 0; j < 1000; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        assert((long)dictSize(d) == 1000);
        dictVerify(d);

        for (j = 0; j < 1000; j++) {
            char *key = stringFromLongLong(j);
            de = dictFind(d, key);
            assert(de != NULL);
            assert(dictGetVal(de) == (void*)j);
            /* Adding an existing key fails and reports the existing entry. */
            assert(dictAddRaw(d, key, &existing) == NULL && existing == de);
            zfree(key);
        }

        /* Misses must not find anything, whatever the probe window holds. */
        for (j = 0; j < 1000; j++) {
            char *key = stringFromLongLong(j);
            key[0] = 'X';
            assert(dictFind(d, key) == NULL);
            zfree(key);
        }

        for (j = 0; j < 1000; j += 2) {
            char *key = stringFromLongLong(j);
            assert(dictDelete(d, key) == DICT_OK);
            assert(dictDelete(d, key) == DICT_ERR);
            zfree(key);
        }
        assert((long)dictSize(d) == 500);
        assert(dictWalkLiveEntries(d) == dictSize(d));
        dictVerify(d);

        /* The surviving keys are still reachable after the backward shifts. */
        for (j = 1; j < 1000; j += 2) {
            char *key = stringFromLongLong(j);
            assert(dictFind(d, key) != NULL);
            zfree(key);
        }
        dictEmpty(d, NULL);
    }

    TEST("Insertion grows the table before it fills up") {
        unsigned long size;
        for (j = 0; ; j++) {
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
            while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
            size = DICTHT_SIZE(d->ht_size_exp[0]);
            assert(d->ht_used[0] * 100 <= size * DICT_GROW_PERCENT);
            if (size >= 1024) break;
        }
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Deletion shrinks the table") {
        for (j = 0; j < 1024; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        unsigned long grown = DICTHT_SIZE(d->ht_size_exp[0]);
        assert(grown >= 1024);

        for (j = 0; j < 1000; j++) {
            char *key = stringFromLongLong(j);
            assert(dictDelete(d, key) == DICT_OK);
            zfree(key);
        }
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        assert(dictSize(d) == 24);
        assert(DICTHT_SIZE(d->ht_size_exp[0]) < grown);
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Resizing is forced when a probe window overflows") {
        /* DICT_RESIZE_FORBID makes dictExpandIfNeeded() a no-op. Inserting
         * regardless must still terminate: the insert path grows the table on
         * its own rather than spinning (which used to hang the server). */
        dictSetResizeEnabled(DICT_RESIZE_FORBID);
        for (j = 0; j < 10000; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        assert((long)dictSize(d) == 10000);
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        for (j = 0; j < 10000; j++) {
            char *key = stringFromLongLong(j);
            assert(dictFind(d, key) != NULL);
            zfree(key);
        }
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Keys survive an incremental rehash") {
        for (j = 0; j < 1000; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        assert(dictExpand(d, 8192) == DICT_OK);
        assert(dictIsRehashing(d));

        /* Look ups, inserts and deletes while both tables are live. */
        for (j = 1000; j < 1500; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        for (j = 0; j < 1500; j += 3) {
            char *key = stringFromLongLong(j);
            assert(dictDelete(d, key) == DICT_OK);
            zfree(key);
        }
        dictVerify(d);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        dictVerify(d);

        for (j = 0; j < 1500; j++) {
            char *key = stringFromLongLong(j);
            assert((dictFind(d, key) != NULL) == (j % 3 != 0));
            zfree(key);
        }
        dictEmpty(d, NULL);
    }

    TEST("An ongoing rehash completes even while resizing is avoided") {
        /* DICT_RESIZE_AVOID (a child process is saving) must not strand a
         * rehash that already started: callers do wait for dictIsRehashing()
         * to clear, and a table stuck half migrated is the worst of both
         * worlds. Grow, then avoid resizing, then empty most of the dict. */
        long n = 3000;
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        assert(dictExpand(d, DICTHT_SIZE(d->ht_size_exp[0]) * 2) == DICT_OK);
        assert(dictIsRehashing(d));

        dictSetResizeEnabled(DICT_RESIZE_AVOID);
        for (j = 0; j < n - 5; j++) {
            char *key = stringFromLongLong(j);
            assert(dictDelete(d, key) == DICT_OK);
            zfree(key);
        }
        assert(dictSize(d) == 5);

        for (j = 0; j < 1000 && dictIsRehashing(d); j++)
            dictRehashMicroseconds(d, 100);
        assert(!dictIsRehashing(d));

        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        dictVerify(d);
        for (j = n - 5; j < n; j++) {
            char *key = stringFromLongLong(j);
            assert(dictFind(d, key) != NULL);
            zfree(key);
        }
        dictEmpty(d, NULL);
    }

    TEST("Iterators visit every key") {
        long n = 2000;
        char *seen = zcalloc(n);
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);

        dictIterator *iter = dictGetIterator(d);
        long visited = 0;
        while ((de = dictNext(iter)) != NULL) {
            long v = strtol(dictGetKey(de), NULL, 10);
            assert(v >= 0 && v < n && !seen[v]);
            seen[v] = 1;
            visited++;
        }
        dictReleaseIterator(iter);
        assert(visited == n);

        /* A safe iterator may delete what it is given. */
        iter = dictGetSafeIterator(d);
        while ((de = dictNext(iter)) != NULL)
            assert(dictDelete(d, dictGetKey(de)) == DICT_OK);
        dictReleaseIterator(iter);
        assert(dictSize(d) == 0);
        assert(d->pausecompact == 0);
        dictVerify(d);
        zfree(seen);
        dictEmpty(d, NULL);
    }

    TEST("dictScan() returns every key") {
        long n = 3000;
        scanTestData data;
        data.count = n;
        data.seen = zcalloc(n);
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);

        unsigned long cursor = 0;
        do {
            cursor = dictScan(d, cursor, scanTestCallback, &data);
        } while (cursor != 0);
        for (j = 0; j < n; j++) assert(data.seen[j]);

        /* Same, while rehashing. */
        memset(data.seen, 0, n);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
        assert(dictExpand(d, 1 << 16) == DICT_OK);
        assert(dictIsRehashing(d));
        dictPauseRehashing(d); /* keep both tables alive for the whole scan */
        cursor = 0;
        do {
            cursor = dictScan(d, cursor, scanTestCallback, &data);
        } while (cursor != 0);
        dictResumeRehashing(d);
        for (j = 0; j < n; j++) assert(data.seen[j]);

        zfree(data.seen);
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Deleting from a scan callback leaves a usable table") {
        long n = 3000;
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);

        unsigned long cursor = 0;
        do {
            cursor = dictScan(d, cursor, scanDeleteCallback, d);
        } while (cursor != 0 && dictSize(d) != 0);
        assert(dictSize(d) == 0);
        assert(d->pausecompact == 0);
        /* Leftover tombstones are fine, they are reclaimed lazily, but the
         * table must still be consistent and usable. */
        dictVerify(d);
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        assert(dictSize(d) == (unsigned long)n);
        dictVerify(d);
        dictEmpty(d, NULL);
    }

    TEST("Tombstones are reclaimed once compaction resumes") {
        long n = 2000;
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);

        dictPauseCompact(d);
        for (j = 0; j < n; j += 2) {
            char *key = stringFromLongLong(j);
            assert(dictDelete(d, key) == DICT_OK);
            zfree(key);
        }
        assert(d->ht_tombstones[0] > 0);
        dictVerify(d);
        /* Everything still reachable across the tombstones. */
        for (j = 1; j < n; j += 2) {
            char *key = stringFromLongLong(j);
            assert(dictFind(d, key) != NULL);
            zfree(key);
        }
        dictResumeCompact(d);
        assert(d->ht_tombstones[0] == 0);
        dictVerify(d);
        for (j = 1; j < n; j += 2) {
            char *key = stringFromLongLong(j);
            assert(dictFind(d, key) != NULL);
            zfree(key);
        }
        dictEmpty(d, NULL);
    }

    TEST("dictGetRandomKey() and dictGetSomeKeys() only return live entries") {
        long n = 1000;
        for (j = 0; j < n; j++)
            assert(dictAdd(d, stringFromLongLong(j), (void*)j) == DICT_OK);
        for (j = 0; j < 1000; j++) {
            de = dictGetRandomKey(d);
            assert(de != NULL && dictFind(d, dictGetKey(de)) == de);
            de = dictGetFairRandomKey(d);
            assert(de != NULL && dictFind(d, dictGetKey(de)) == de);
        }
        dictEntry *some[16];
        unsigned int got = dictGetSomeKeys(d, some, 16);
        assert(got > 0 && got <= 16);
        for (unsigned int i = 0; i < got; i++)
            assert(dictFind(d, dictGetKey(some[i])) == some[i]);
        dictEmpty(d, NULL);
    }

    TEST("dictMemUsage() accounts for entries and tables") {
        long n = 1000;
        dict *dn = dictCreate(&BenchmarkDictType);          /* no_value = 0 */
        dict *dv = dictCreate(&BenchmarkDictTypeNoValue);   /* no_value = 1 */
        for (long i = 0; i < n; i++) {
            assert(dictAdd(dn, stringFromLongLong(i), (void *)i) == DICT_OK);
            assert(dictAdd(dv, stringFromLongLong(i), NULL) == DICT_OK);
        }
        assert(dictSize(dn) == (unsigned long)n && dictSize(dv) == (unsigned long)n);

        /* A no_value dict stores the keys inline: no entry is ever allocated. */
        assert(dn->allocated_entries == (unsigned long)n);
        assert(dv->allocated_entries == 0);
        assert(dictMemUsage(dv) < dictMemUsage(dn));
        dictVerify(dn);
        dictVerify(dv);

        dictRelease(dn);
        dictRelease(dv);
    }

    TEST("Use dict without values (no_value=1)") {
        dictType dt = BenchmarkDictType;
        dt.no_value = 1;

        char **lookupKeys = zmalloc(sizeof(char*) * count);
        for (long i = 0; i < count; i++)
            lookupKeys[i] = stringFromLongLong(i);

        dict *dnv = dictCreate(&dt);
        for (j = 0; j < count; j++)
            assert(dictAdd(dnv, lookupKeys[j], NULL) == DICT_OK);

        for (j = 0; j < count; j++) {
            de = dictFind(dnv, lookupKeys[j]);
            assert(de != NULL);
            assert(dictEntryIsKey(de));
            assert(dictGetNext(de) == NULL);
        }

        /* Find non existing keys. Probe with a copy: this dict stores the key
         * pointers themselves, so editing one in place would edit the key
         * that is actually in the table. */
        for (j = 0; j < count; j++) {
            char *probe = stringFromLongLong(j);
            probe[0] = 'X';
            assert(dictFind(dnv, probe) == NULL);
            zfree(probe);
        }

        dictVerify(dnv);
        dictRelease(dnv);
        zfree(lookupKeys);
    }

    TEST("Test dictFindLink() functionality") {
        dictType dt = BenchmarkDictType;
        dict *dl = dictCreate(&dt);

        /* find in empty dict */
        assert(dictFindLink(dl, "key", NULL) == NULL);

        for (j = 0; j < 100; j++) {
            char *key = stringFromLongLong(j);
            assert(dictAdd(dl, key, (void*)j) == DICT_OK);

            dictEntryLink link = dictFindLink(dl, key, NULL);
            assert(link != NULL && *link != NULL);
            assert(compareCallback(NULL, dictGetKey(*link), key));

            char *nonExistingKey = stringFromLongLong(j + 100);
            assert(dictFindLink(dl, nonExistingKey, NULL) == NULL);

            dictEntryLink bucket = NULL;
            link = dictFindLink(dl, key, &bucket);
            assert(link != NULL && bucket != NULL);

            /* The home slot is reported even for a key that is not there. */
            link = dictFindLink(dl, nonExistingKey, &bucket);
            assert(link == NULL && bucket != NULL);

            /* Insert it at the home slot we were just given. */
            dictSetKeyAtLink(dl, nonExistingKey, &bucket, 1);
            assert(dictFind(dl, nonExistingKey) != NULL);
            assert(dictDelete(dl, nonExistingKey) == DICT_OK);
        }
        dictVerify(dl);
        dictRelease(dl);
    }

    TEST("dictAddNonExisting() adds a known-absent key") {
        dictType dt = BenchmarkDictType;
        dt.no_value = 1;
        dict *dne = dictCreate(&dt);

        long n = 1000;
        for (long i = 0; i < n; i++)
            assert(dictAddNonExisting(dne, stringFromLongLong(i)) != NULL);
        assert((long)dictSize(dne) == n);
        for (long i = 0; i < n; i++) {
            char *probe = stringFromLongLong(i);
            assert(dictFind(dne, probe) != NULL);
            zfree(probe);
        }
        dictVerify(dne);
        dictRelease(dne); /* freeCallback releases the stored keys */
    }

    TEST("dictAddNonExistingBatch() inserts a batch into a fresh dict") {
        dictType dt = BenchmarkDictType;
        dt.no_value = 1;
        dict *db1 = dictCreate(&dt);

        long n = 5000;
        void **keys = zmalloc(sizeof(void *) * n);
        for (long i = 0; i < n; i++) keys[i] = stringFromLongLong(i);

        dictAddNonExistingBatch(db1, keys, n);
        assert((long)dictSize(db1) == n);
        for (long i = 0; i < n; i++) {
            char *probe = stringFromLongLong(i);
            assert(dictFind(db1, probe) != NULL);
            zfree(probe);
        }
        dictVerify(db1);
        zfree(keys);
        dictRelease(db1);
    }

    TEST("dictAddNonExistingBatch() stays correct across a rehash") {
        /* Leave rehashing unfinished so the batch runs against two tables. */
        dictType dt = BenchmarkDictType;
        dt.no_value = 1;
        dict *db2 = dictCreate(&dt);

        long seed = 1024;
        for (long i = 0; i < seed; i++)
            assert(dictAdd(db2, stringFromLongLong(i), NULL) == DICT_OK);
        while (dictIsRehashing(db2)) dictRehashMicroseconds(db2, 1000);
        assert(dictExpand(db2, seed * 8) == DICT_OK);
        assert(dictIsRehashing(db2));

        long n = 2000;
        void **keys = zmalloc(sizeof(void *) * n);
        for (long i = 0; i < n; i++) keys[i] = stringFromLongLong(seed + i);
        dictAddNonExistingBatch(db2, keys, n);
        assert((long)dictSize(db2) == seed + n);

        for (long i = 0; i < seed + n; i++) {
            char *probe = stringFromLongLong(i);
            assert(dictFind(db2, probe) != NULL);
            zfree(probe);
        }
        dictVerify(db2);
        zfree(keys);
        dictRelease(db2);
    }

    /* ------------------------------ benchmarks ---------------------------- */

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
        dictEntry *found = dictFind(d,key);
        assert(found != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(j);
        dictEntry *found = dictFind(d,key);
        assert(found != NULL);
        zfree(key);
    }
    end_benchmark("Linear access of existing elements (2nd round)");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        dictEntry *found = dictFind(d,key);
        assert(found != NULL);
        zfree(key);
    }
    end_benchmark("Random access of existing elements");

    start_benchmark();
    for (j = 0; j < count; j++) {
        dictEntry *found = dictGetRandomKey(d);
        assert(found != NULL);
    }
    end_benchmark("Accessing random keys");

    start_benchmark();
    for (j = 0; j < count; j++) {
        char *key = stringFromLongLong(rand() % count);
        key[0] = 'X';
        dictEntry *found = dictFind(d,key);
        assert(found == NULL);
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
    dictVerify(d);
    dictRelease(d);

    return 0;
}
#endif
