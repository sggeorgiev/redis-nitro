/* Hash Tables Implementation.
 *
 * This file implements in-memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto-resize if needed
 * tables of power of two in size are used, collisions are handled by
 * open addressing over groups of slots (a Swiss table). See the source code
 * for more information... :)
 *
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * Dict usage of pointer tagging
 * -----------------------------
 * In the "normal" case (no_value=0), a dict slot contains only a pointer to a
 * dictEntry, and dictEntry holds untagged pointers to key and value. But when a
 * dict is used as a set (no_value=1), every slot stores the key pointer directly,
 * avoiding dictEntry allocation. Redis dicts uses pointer
 * tagging, to identify direct key pointers from dictEntry pointers, i.e embedding
 * metadata in the lowest three bits of pointers. This requires 8-byte alignment, 
 * which zmalloc() guarantees on both 32-bit and 64-bit systems (via jemalloc/tcmalloc, 
 * or standard malloc with explicit PREFIX_SIZE=8).
 * 
 * Besides of distinguishing direct key pointers from dictEntry pointers, we also 
 * need to distinguish between even and odd key pointers that being stored in the 
 * dict. Therefore, we use the following tagging scheme:
 * - dictEntry pointer: Points to a dictEntry structure (8-byte aligned). Left intact: 
 *   ENTRY_PTR_NORMAL=000
 * - Odd-address key (keys_are_odd=1): Direct pointer to a 
 *   key with odd address (e.g., all SDS strings), Left intact: 
 *   ENTRY_PTR_IS_ODD_KEY=XX1
 * - Even-address key  (keys_are_odd=0): Direct pointer to a key with 
 *   even address. Since 8-byte alignment yields bits = 000, same as dictEntry, 
 *   we tag it by setting bit 1 which results with: 
 *   ENTRY_PTR_IS_EVEN_KEY=010.
 */

#ifndef __DICT_H
#define __DICT_H

#include "mt19937-64.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#define DICT_OK 0
#define DICT_ERR 1

/* Hash table parameters */
#define HASHTABLE_MIN_FILL        8      /* Minimal hash table fill 12.5%(100/8) */

/* stored-key vs. key
 * ------------------
 * If dictType.keyFromStoredKey is non-NULL, then dict distinguishes between the
 * lookup key and the actual stored-key object. In this case, "key" is used to 
 * locate entries, while "storedKey" is the actual element stored in the dict.
 * If dictType.keyFromStoredKey is NULL, the lookup "key" and the stored-key are the
 * same. This API is primarily relevant for no_value=1 dicts, where the key and value
 * might be packed together. When values are stored separately, this identity 
 * distinction does not arise. The marker __stored_key is used to indicate that 
 * the pointer refers to the stored-key rather than the lookup key.
 */
#define __stored_key

typedef struct dictEntry dictEntry; /* opaque */
struct dictStashEntry; /* opaque, defined in dict.c */
typedef struct dict dict;
typedef dictEntry **dictEntryLink; /* See description of dictFindLink() */

/* Searching for a key in a dict may involve few comparisons.
 * If extracting the looked-up key is expensive (e.g., sdslen(), kvobjGetKey()),  
 * caching can be used to reduce those repetitive computations.  
 *  
 * This struct, passed to the comparison function as temporary caching, if 
 * needed by the function across comparison of a given lookup. 
 * for the looked-up key and resets before each new lookup. */
typedef struct dictCmpCache {
    int useCache;
    
    union {
        uint64_t u64;
        int64_t i64;
        int i;
        size_t sz;
        void *p;
    } data[2];
} dictCmpCache;

typedef struct dictType {
    /* Callbacks */
    uint64_t (*hashFunction)(const void *key);
    void *(*keyDup)(dict *d, const void *key __stored_key);
    void *(*valDup)(dict *d, const void *obj);
    int (*keyCompare)(dictCmpCache *cache, const void *key1, const void *key2);
    void (*keyDestructor)(dict *d, void *key __stored_key);
    void (*valDestructor)(dict *d, void *obj);
    int (*resizeAllowed)(size_t moreMem, double usedRatio);
    /* Invoked at the start of dict initialization/rehashing (old and new ht are already created) */
    void (*rehashingStarted)(dict *d);
    /* Invoked at the end of dict initialization/rehashing of all the entries from old to new ht. Both ht still exists
     * and are cleaned up after this callback.  */
    void (*rehashingCompleted)(dict *d);
    /* Invoked when the memory used by the dict's hash tables changes.
     * `delta_bytes` can be positive (growth) or negative (shrink). */
    void (*tableMemChanged)(dict *d, long long delta_bytes);
    /* Allow a dict to carry extra caller-defined metadata. The
     * extra memory is initialized to 0 when a dict is allocated. */
    size_t (*dictMetadataBytes)(dict *d);

    /* Data */
    void *userdata;

    /* Flags */
    /* The 'no_value' flag, if set, indicates that values are not used, i.e. the
     * dict is a set. When this flag is set, it's not possible to access the
     * value of a dictEntry and it's also impossible to use dictSetKey(). It 
     * enables an optimization to store a key directly without an allocating 
     * dictEntry in between. */
    unsigned int no_value:1;
    /* This flag is required for `no_value` optimization since the optimization
     * reuses LSB bits as metadata */ 
    unsigned int keys_are_odd:1;

    /* Ensures that the entire hash table is rehashed at once if set. */
    unsigned int force_full_rehash:1;
    
    /* Callback to extract key from stored-key object. When set, the dict can
     * store keys in one format (e.g., a structure) but look them up using a
     * different format, extracted from the stored-key. (e.g., sds or integer). 
     * Set to NULL if key and stored-key object are the same. Relevant only for
     * no_value=1 dicts. */
    const void *(*keyFromStoredKey)(const void *key __stored_key);

    /* Optional callback called when the dict is destroyed. */
    void (*onDictRelease)(dict *d);

    /* Optional prefetch hooks used by the memory_prefetch state machine.
     * Both default to NULL; when both are NULL the state machine just
     * prefetches the group, slot and dictEntry and stops there.
     *
     * prefetchEntryKey: called after a dictEntry has been brought into
     *   cache. Returns an address to issue redis_prefetch_read on (so the
     *   key payload behind the entry is warm before keyCompare runs), or
     *   NULL if nothing extra is needed (e.g. the key is co-located with
     *   the entry).
     * prefetchEntryValue: called when the entry is the *presumed* match
     *   for the lookup key — either keyCompare returned equal, or the
     *   state machine took the "lone tag match, not rehashing"
     *   shortcut and is betting on a hit without comparing. Callbacks
     *   must therefore not assume the key has been verified equal; the
     *   prefetch is advisory. Returns an address to prefetch for the
     *   value-side payload, or NULL. */
    void *(*prefetchEntryKey)(const dictEntry *de);
    void *(*prefetchEntryValue)(const dictEntry *de);
    /* Invoked when overflow stash allocations change. Deleted stash nodes
     * remain charged until rehashing resumes and their allocations are freed. */
    void (*stashMemChanged)(dict *d, long long delta_bytes);
} dictType;

#define DICTHT_SIZE(exp) ((exp) == -1 ? 0 : (unsigned long)1<<(exp))

/* Opaque position handed from dictTwoPhaseUnlinkFind() to
 * dictTwoPhaseUnlinkFree(). Callers must not look inside. */
typedef struct dictPosition {
    size_t slot;   /* slot index within the table, or the stash entry */
    uint64_t hash; /* mixed hash of the key, see dict.c */
    int table;     /* 0, 1, or 2 for the stash */
} dictPosition;

struct dict {
    dictType *type;

    /* Each table is one allocation of 1<<ht_size_exp groups, each holding
     * its control bytes followed by its entry slots (see dict.c). */
    uint8_t *ht_table[2];
    unsigned long ht_used[2]; /* number of entries in each table (ht_used[1] includes stashed ones) */
    /* Inserts that didn't fit while rehashing was paused (see dict.c). */
    struct dictStashEntry *stash;
    unsigned long stash_count; /* allocated nodes, including deleted entries */

    long rehashidx; /* rehashing not in progress if rehashidx == -1 */

    /* Note: the pause counter is a full unsigned so iterator increments
     * don't perform RMW on the same storage unit as other bitfields. */
    unsigned pauserehash; /* If >0 rehashing is paused */

    /* Keep small vars at end for optimal (minimal) struct padding */
    signed char ht_size_exp[2]; /* exponent of group count (groups = 1<<exp) */
    int16_t pauseAutoResize;  /* If >0 automatic resizing is disallowed (<0 indicates coding error) */
    uint8_t pending_shrink;   /* A delete wanted to shrink while paused */
    void *metadata[];
};

/* If safe is set to 1 this is a safe iterator, that means, you can call
 * dictAdd, dictFind, and other functions against the dictionary even while
 * iterating. Otherwise it is a non safe iterator, and only dictNext()
 * should be called while iterating. */
typedef struct dictIterator {
    dict *d;
    long index;     /* group index, -1 before the first call */
    int table, pos; /* table (2 = the stash) and slot position within the group */
    struct dictStashEntry *stash_next;
    uint8_t safe;
    uint8_t done;
    /* unsafe iterator fingerprint for misuse detection. */
    unsigned long long fingerprint;
} dictIterator;

typedef struct dictStats {
    int htidx;
    unsigned long groups;
    unsigned long overflowGroups;      /* groups that some entry probed past */
    unsigned long displaced;           /* entries stored outside their home group */
    unsigned long maxProbeLen;         /* in groups */
    unsigned long long totalProbeLen;  /* sum over entries, in groups */
    unsigned long htSize;              /* capacity in entry slots */
    unsigned long htUsed;
    unsigned long *clvector;           /* entry i counts entries found after probing i groups */
} dictStats;

typedef void (dictScanFunction)(void *privdata, const dictEntry *de, dictEntry **plink);
typedef void *(dictDefragAllocFunction)(void *ptr);
typedef struct {
    dictDefragAllocFunction *defragAlloc; /* Used for entries etc. */
    dictDefragAllocFunction *defragKey;   /* Defrag-realloc keys (optional) */
    dictDefragAllocFunction *defragVal;   /* Defrag-realloc values (optional) */
} dictDefragFunctions;

/* Swiss table: entries live in groups of DICT_GROUP_SLOTS slots with a control
 * byte (hash tag) each, plus one overflow counter byte per group. A full group
 * overflows into the next group of the key's probe sequence. dictBuckets()
 * counts groups. A group is 16 control bytes plus the slots, a power of two
 * (128 or 64 bytes), so tables match allocator size classes exactly. */
#if SIZE_MAX == UINT64_MAX
#define DICT_GROUP_SLOTS         14
#else
#define DICT_GROUP_SLOTS         12
#endif
#define DICT_GROUP_BYTES         (16 + DICT_GROUP_SLOTS * sizeof(void *))

/* Every table starts as a single group, which is also the shrink floor.
 * DICT_HT_INITIAL_SIZE is the number of entries it takes before growing
 * (tables grow at 7/8 of their slots). */
#define DICT_HT_INITIAL_EXP      0
#define DICT_HT_INITIAL_SIZE     (DICT_GROUP_SLOTS * 7 / 8)

/* ------------------------------- Macros ------------------------------------*/
#define dictFreeVal(d, entry) do {                     \
    if ((d)->type->valDestructor)                      \
        (d)->type->valDestructor((d), dictGetVal(entry)); \
   } while(0)

#define dictFreeKey(d, entry) \
    if ((d)->type->keyDestructor) \
        (d)->type->keyDestructor((d), dictGetKey(entry))

#define dictMetadata(d) (&(d)->metadata)
#define dictMetadataSize(d) ((d)->type->dictMetadataBytes \
                             ? (d)->type->dictMetadataBytes(d) : 0)

#define dictBuckets(d) (DICTHT_SIZE((d)->ht_size_exp[0])+DICTHT_SIZE((d)->ht_size_exp[1]))
#define dictSize(d) ((d)->ht_used[0]+(d)->ht_used[1])
#define dictIsEmpty(d) ((d)->ht_used[0] == 0 && (d)->ht_used[1] == 0)
#define dictIsRehashing(d) ((d)->rehashidx != -1)
#define dictIsRehashingPaused(d) ((d)->pauserehash > 0)
#define dictPauseAutoResize(d) ((d)->pauseAutoResize++)
#define dictResumeAutoResize(d) ((d)->pauseAutoResize--)

/* If our unsigned long type can store a 64 bit number, use a 64 bit PRNG. */
#if ULONG_MAX >= 0xffffffffffffffff
#define randomULong() ((unsigned long) genrand64_int64())
#else
#define randomULong() random()
#endif

typedef enum {
    DICT_RESIZE_ENABLE,
    DICT_RESIZE_AVOID,
    DICT_RESIZE_FORBID,
} dictResizeEnable;

/* API */
dict *dictCreate(dictType *type);
void dictTypeAddMeta(dict **d, dictType *typeWithMeta);
int dictExpand(dict *d, unsigned long size);
int dictTryExpand(dict *d, unsigned long size);
int dictShrink(dict *d, unsigned long size);
int dictAdd(dict *d, void *key __stored_key, void *val);
dictEntry *dictAddRaw(dict *d, void *key __stored_key, dictEntry **existing);
dictEntry *dictAddOrFind(dict *d, void *key __stored_key);
int dictReplace(dict *d, void *key __stored_key, void *val);
int dictDelete(dict *d, const void *key);
dictEntry *dictUnlink(dict *d, const void *key);
void dictFreeUnlinkedEntry(dict *d, dictEntry *he);
dictEntryLink dictTwoPhaseUnlinkFind(dict *d, const void *key, dictPosition *pos);
void dictTwoPhaseUnlinkFree(dict *d, dictEntryLink llink, dictPosition *pos);
void dictPauseRehashing(dict *d);
void dictResumeRehashing(dict *d);
unsigned long dictSlots(const dict *d);
size_t dictTableMemUsage(const dict *d);
size_t dictStashMemUsage(const dict *d);
size_t dictRehashingMemUsage(const dict *d);
dict *dictDefragTables(dict *d, void *(*defragfn)(void *));
void dictDismissTables(dict *d, void (*fn)(void *ptr, size_t size));
void dictRelease(dict *d);
dictEntry * dictFind(dict *d, const void *key);
dictEntry *dictFindByHashAndPtr(dict *d, const void *oldptr, const uint64_t hash);
int dictShrinkIfNeeded(dict *d);
int dictExpandIfNeeded(dict *d);
void *dictGetKey(const dictEntry *de);
int dictEntryIsKey(const dictEntry *de);
int dictCompareKeys(dict *d, const void *key1, const void *key2);
size_t dictMemUsage(const dict *d);
size_t dictEntryMemUsage(int noValueDict);
dictIterator *dictGetIterator(dict *d);
dictIterator *dictGetSafeIterator(dict *d);
void dictInitIterator(dictIterator *iter, dict *d);
void dictInitSafeIterator(dictIterator *iter, dict *d);
void dictResetIterator(dictIterator *iter);
dictEntry *dictNext(dictIterator *iter);
void dictReleaseIterator(dictIterator *iter);
dictEntry *dictGetRandomKey(dict *d);
dictEntry *dictGetFairRandomKey(dict *d);
unsigned int dictGetSomeKeys(dict *d, dictEntry **des, unsigned int count);
void dictGetStats(char *buf, size_t bufsize, dict *d, int full);
uint64_t dictGenHashFunction(const void *key, size_t len);
uint64_t dictGenCaseHashFunction(const unsigned char *buf, size_t len);
void dictEmpty(dict *d, void(callback)(dict*));
void dictSetResizeEnabled(dictResizeEnable enable);
int dictRehash(dict *d, int n);
int dictRehashMicroseconds(dict *d, uint64_t us);
void dictSetHashFunctionSeed(uint8_t *seed);
unsigned long dictScan(dict *d, unsigned long v, dictScanFunction *fn, void *privdata);
unsigned long dictScanDefrag(dict *d, unsigned long v, dictScanFunction *fn, dictDefragFunctions *defragfns, void *privdata);
uint64_t dictGetHash(dict *d, const void *key);
void dictRehashingInfo(dict *d, unsigned long long *from_size, unsigned long long *to_size);

size_t dictGetStatsMsg(char *buf, size_t bufsize, dictStats *stats, int full);
dictStats* dictGetStatsHt(dict *d, int htidx, int full);
void dictCombineStats(dictStats *from, dictStats *into);
void dictFreeStats(dictStats *stats);

dictEntryLink dictFindLink(dict *d, const void *key, dictEntryLink *bucket);
void dictSetKeyAtLink(dict *d, void *key __stored_key, dictEntryLink *link, int newItem);

/* API relevant only when dict is used as a hash-map (no_value=0) */ 
void dictSetKey(dict *d, dictEntry* de, void *key __stored_key);
void dictSetVal(dict *d, dictEntry *de, void *val);
void *dictGetVal(const dictEntry *de);
void dictSetDoubleVal(dictEntry *de, double val);
double dictGetDoubleVal(const dictEntry *de);
double *dictGetDoubleValPtr(dictEntry *de);
void *dictFetchValue(dict *d, const void *key);
void dictSetUnsignedIntegerVal(dictEntry *de, uint64_t val);
uint64_t dictIncrUnsignedIntegerVal(dictEntry *de, uint64_t val);
uint64_t dictGetUnsignedIntegerVal(const dictEntry *de);

/* Per-key state of a software-pipelined dictFind used by the memory prefetcher
 * (memory_prefetch.c). The caller owns the storage; the contents are private
 * to dict.c. Usage:
 *     dictPrefetchInit(&st, d, key);
 *     while (!st.done) { void *a = dictPrefetchNext(&st); if (a) prefetch(a); ... }
 * dictPrefetchNext() returns the next address worth prefetching, or NULL once
 * the lookup is complete (st.done is then set). Interleave several states to
 * overlap their memory stalls. */
typedef struct dictPrefetchState {
    dict *d;
    const void *key;
    uint64_t hash;       /* mixed hash */
    dictEntry **slot;
    dictEntry *current_entry;
    size_t group;
    uint32_t step;       /* probe step within the current table */
    uint16_t candidates; /* tag-matching, not yet visited positions */
    int8_t ht_idx;
    uint8_t stage;
    uint8_t done;
    uint8_t lone;        /* single candidate in a group nothing probed past */
} dictPrefetchState;

void dictPrefetchInit(dictPrefetchState *st, dict *d, const void *key);
void *dictPrefetchNext(dictPrefetchState *st);

#ifdef REDIS_TEST
int dictTest(int argc, char *argv[], int flags);
#endif

#endif /* __DICT_H */
