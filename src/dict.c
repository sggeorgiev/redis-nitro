/* Hash Tables Implementation.
 *
 * This file implements in memory hash tables with insert/del/replace/find/
 * get-random-element operations. Hash tables will auto resize if needed
 * tables of power of two in size are used. The tables are Swiss tables: open
 * addressing over groups of 14 slots (12 on 32-bit builds), with one control
 * byte (a 7 bit hash tag) per slot so that a whole group is matched with a
 * single vector compare. See the source code for more information... :)
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

#if defined(__SSE2__) && !defined(DICT_NO_SIMD)
#include <emmintrin.h>
#define DICT_GROUP_SSE2 1
#endif

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
 *  - A hash table always grows when it reaches the hard fill limit (31/32 of
 *    its slots, see hardFill()): open addressing can't hold more entries than
 *    it has slots. This is true even under DICT_RESIZE_FORBID.
 *  - A hash table is still allowed to shrink if the ratio between the number
 *    of elements and the slot capacity <= 1 / (HASHTABLE_MIN_FILL * dict_force_resize_ratio). */
static redisAtomic dictResizeEnable dict_can_resize = DICT_RESIZE_ENABLE;
static const unsigned int dict_force_resize_ratio = 4;

/* -------------------------- types ----------------------------------------- */

/* An entry of a dict with values. The slot of a group points to it. For
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

/* Table layout
 * ------------
 * A table is a power of two number of groups, allocated as one block. Each
 * group is DICT_GROUP_BYTES (128, or 64 on 32 bit builds) holding its 16
 * control bytes followed by its DICT_GROUP_SLOTS (S) entry slots, so a lookup
 * finds the tags and the slot of a match in the same aligned 128 bytes (which
 * the adjacent line prefetcher fetches together). The block size is a power of
 * two, exactly an allocator size class.
 *
 *   group[0..S-1]   one control byte per slot
 *   group[S..14]    unused, always 0
 *   group[15]       overflow counter
 *   group[16..]     the S entry slots
 *
 * Control byte: 0 means the slot is empty. Otherwise the low 7 bits are the tag
 * of the entry (1..127, the top bits of its mixed hash) and the high bit is set
 * if the entry is displaced, that is, not stored in its home group.
 *
 * The home group of a key is given by the low bits of its mixed hash (see
 * mixHash()). An insert takes the first group of the key's probe sequence with
 * a free slot. The probe sequence is triangular (home, home+1, home+3,
 * home+6, ...), which visits every group of a power of two table exactly once.
 *
 * Overflow counter: the number of entries whose probe sequence went through the
 * group and that are stored in a later group of it, as in Folly's F14. A lookup
 * stops after the first group whose counter is zero. Deleting a displaced entry
 * decrements the counters again, so there are no tombstones and deletes never
 * call for a cleanup rehash. A counter that reaches 255 stays there; lookups
 * then always continue past that group, which is safe.
 *
 * Entries only move when they are rehashed into the other table. A slot
 * address (a dictEntryLink) therefore stays valid across inserts and deletes of
 * other keys. */
#define GROUP_WIDTH     16          /* control bytes per group */
#define GROUP_SLOTS     DICT_GROUP_SLOTS
#define CTRL_EMPTY      0x00
#define CTRL_DISPLACED  0x80
#define CTRL_TAG_MASK   0x7f
#define OVF_POS         (GROUP_WIDTH - 1) /* control byte holding the overflow counter */
#define OVF_MAX         255

typedef uint16_t groupMask; /* bit i stands for slot i of a group */
#define GROUP_SLOT_BITS ((groupMask)((1u << GROUP_SLOTS) - 1))

static_assert(GROUP_SLOTS < GROUP_WIDTH, "a control byte per slot plus the counter");
static_assert(DICT_GROUP_BYTES == GROUP_WIDTH + GROUP_SLOTS * sizeof(dictEntry *), "group size");
static_assert((DICT_GROUP_BYTES & (DICT_GROUP_BYTES - 1)) == 0, "a group is a power of two");

/* -------------------------- private prototypes ---------------------------- */

static int _dictExpandIfNeeded(dict *d);
static void _dictShrinkIfNeeded(dict *d);
static void _dictRehashStepIfNeeded(dict *d, uint64_t mixed);
static signed char nextGroupExp(size_t n);
static int _dictInit(dict *d, dictType *type);
static int dictDefaultCompare(dictCmpCache *cache, const void *key1, const void *key2);
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket);
static int rehashSteps(dict *d, int n, int force);
static void dictForceRehashIfNeeded(dict *d);

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

/* Compiler inlines this for internal calls within dict.c (verified with -O3). */
uint64_t dictGetHash(dict *d, const void *key) {
    return d->type->hashFunction(key);
}

/* Placement uses a mixed form of the hash: the home group comes from its low
 * bits and the tag from its top 7 bits. Mixing spreads identity-like hashes
 * (client ids, pointers whose low bits are always zero) over groups and tags.
 * It is a bijection, so good hashes stay good. Like the raw hash, the home
 * group in a table twice as large is one of the two groups the home group of
 * the smaller table splits into, which SCAN relies on. */
static inline uint64_t mixHash(uint64_t hash) {
    hash *= 0x9e3779b97f4a7c15ULL;
    return hash ^ (hash >> 32);
}

static inline uint8_t hashTag(uint64_t mixed) {
    uint8_t tag = mixed >> 57;
    return tag ? tag : 1; /* 0 is the empty control byte */
}

static inline uint64_t keyMixedHash(dict *d, const void *key) {
    return mixHash(dictGetHash(d, key));
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

static inline uint64_t entryMixedHash(dict *d, const dictEntry *de) {
    return keyMixedHash(d, slotLookupKey(d, de));
}

/* ----------------------------- group matching ----------------------------- */

/* Each function returns a mask of the slots (not the counter) of a group whose
 * control byte matches. */
#if defined(DICT_GROUP_SSE2)

static inline __m128i groupLoad(const uint8_t *ctrl) {
    return _mm_loadu_si128((const __m128i *)(const void *)ctrl);
}

static inline groupMask groupMatchTag(const uint8_t *ctrl, uint8_t tag) {
    __m128i tags = _mm_and_si128(groupLoad(ctrl), _mm_set1_epi8(CTRL_TAG_MASK));
    __m128i eq = _mm_cmpeq_epi8(tags, _mm_set1_epi8((char)tag));
    return (groupMask)_mm_movemask_epi8(eq) & GROUP_SLOT_BITS;
}

static inline groupMask groupMatchEmpty(const uint8_t *ctrl) {
    __m128i eq = _mm_cmpeq_epi8(groupLoad(ctrl), _mm_setzero_si128());
    return (groupMask)_mm_movemask_epi8(eq) & GROUP_SLOT_BITS;
}

static inline groupMask groupMatchDisplaced(const uint8_t *ctrl) {
    return (groupMask)_mm_movemask_epi8(groupLoad(ctrl)) & GROUP_SLOT_BITS;
}

#elif !defined(DICT_SCALAR_GROUPS) && defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

/* Portable version working on the 16 control bytes as two 64-bit words. */
#define SWAR_LO7  0x7f7f7f7f7f7f7f7fULL
#define SWAR_ONES 0x0101010101010101ULL

static inline uint64_t swarLoad(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Bit 7 of each byte of the result is set if that byte of v is zero. Exact:
 * no carries cross byte boundaries. */
static inline uint64_t swarZeroBytes(uint64_t v) {
    return ~(((v & SWAR_LO7) + SWAR_LO7) | v | SWAR_LO7);
}

/* Gathers bit 7 of byte i into bit i. */
static inline groupMask swarPack(uint64_t high_bits) {
    return (groupMask)((((high_bits >> 7) & SWAR_ONES) * 0x0102040810204080ULL) >> 56);
}

static inline groupMask swarPack2(uint64_t lo, uint64_t hi) {
    return (groupMask)(swarPack(lo) | (swarPack(hi) << 8)) & GROUP_SLOT_BITS;
}

static inline groupMask groupMatchTag(const uint8_t *ctrl, uint8_t tag) {
    uint64_t t = SWAR_ONES * tag;
    return swarPack2(swarZeroBytes((swarLoad(ctrl) & SWAR_LO7) ^ t),
                     swarZeroBytes((swarLoad(ctrl + 8) & SWAR_LO7) ^ t));
}

static inline groupMask groupMatchEmpty(const uint8_t *ctrl) {
    return swarPack2(swarZeroBytes(swarLoad(ctrl)), swarZeroBytes(swarLoad(ctrl + 8)));
}

static inline groupMask groupMatchDisplaced(const uint8_t *ctrl) {
    return swarPack2(swarLoad(ctrl) & ~SWAR_LO7, swarLoad(ctrl + 8) & ~SWAR_LO7);
}

#else

static inline groupMask groupMatchTag(const uint8_t *ctrl, uint8_t tag) {
    groupMask m = 0;
    for (int i = 0; i < GROUP_SLOTS; i++)
        m |= (groupMask)(((ctrl[i] & CTRL_TAG_MASK) == tag) << i);
    return m;
}

static inline groupMask groupMatchEmpty(const uint8_t *ctrl) {
    groupMask m = 0;
    for (int i = 0; i < GROUP_SLOTS; i++)
        m |= (groupMask)((ctrl[i] == CTRL_EMPTY) << i);
    return m;
}

static inline groupMask groupMatchDisplaced(const uint8_t *ctrl) {
    groupMask m = 0;
    for (int i = 0; i < GROUP_SLOTS; i++)
        m |= (groupMask)(((ctrl[i] & CTRL_DISPLACED) != 0) << i);
    return m;
}

#endif

static inline groupMask groupMatchFull(const uint8_t *ctrl) {
    return ~groupMatchEmpty(ctrl) & GROUP_SLOT_BITS;
}

/* Entries stored in their home group. */
static inline groupMask groupMatchNative(const uint8_t *ctrl) {
    return groupMatchFull(ctrl) & ~groupMatchDisplaced(ctrl);
}

/* ----------------------------- table helpers ------------------------------ */

static inline size_t numGroups(int exp) {
    return exp == -1 ? 0 : (size_t)1 << exp;
}

static inline size_t expToMask(int exp) {
    return exp == -1 ? 0 : numGroups(exp) - 1;
}

static inline size_t tableCapacity(int exp) {
    return numGroups(exp) * GROUP_SLOTS;
}

static inline size_t tableBytes(int exp) {
    return numGroups(exp) * DICT_GROUP_BYTES;
}

/* A table grows when it reaches 7/8 of its slots. */
static inline size_t maxFill(int exp) {
    return tableCapacity(exp) * 7 / 8;
}

/* While resizing is avoided (fork child), refused (resizeAllowed, e.g. when
 * over maxmemory) or auto-resize is paused, a table still takes inserts up to
 * 31/32 of its slots, then grows anyway. */
static inline size_t hardFill(int exp) {
    return tableCapacity(exp) * 31 / 32;
}

static inline uint8_t *groupCtrl(const dict *d, int table, size_t g) {
    return d->ht_table[table] + g * DICT_GROUP_BYTES;
}

static inline dictEntry **groupSlots(const dict *d, int table, size_t g) {
    return (dictEntry **)(void *)(groupCtrl(d, table, g) + GROUP_WIDTH);
}

/* Next group of the triangular probe sequence. `step` counts the groups
 * visited before this one; once it reaches the table mask every group has been
 * visited. */
static inline size_t probeNext(size_t g, size_t *step, size_t mask) {
    return (g + ++*step) & mask;
}

/* Table memory accounting. All changes of the hash table memory go through
 * here, in bytes. */
static inline void tableMemChanged(dict *d, long long delta_bytes) {
    if (d->type->tableMemChanged)
        d->type->tableMemChanged(d, delta_bytes);
}

/* Stores an entry in the first free slot of its probe sequence in the given
 * table, and counts it in the overflow counter of every full group it passed.
 * Returns the slot. The table must have a free slot. */
static dictEntry **tableInsert(dict *d, int table, uint64_t mixed, dictEntry *entry) {
    size_t mask = expToMask(d->ht_size_exp[table]);
    size_t home = mixed & mask, g = home, step = 0;
    for (;;) {
        uint8_t *ctrl = groupCtrl(d, table, g);
        groupMask empty = groupMatchEmpty(ctrl);
        if (empty) {
            int pos = __builtin_ctz(empty);
            ctrl[pos] = hashTag(mixed) | (g != home ? CTRL_DISPLACED : 0);
            dictEntry **slot = &groupSlots(d, table, g)[pos];
            *slot = entry;
            d->ht_used[table]++;
            return slot;
        }
        assert(step < mask); /* a table never fills up completely */
        if (ctrl[OVF_POS] != OVF_MAX) ctrl[OVF_POS]++;
        g = probeNext(g, &step, mask);
    }
}

/* Empties slot `pos` of group `g`. For a displaced entry, `mixed` must be its
 * mixed hash: the overflow counters of the groups it passed are decremented.
 * The entry itself is not touched. */
static void tableRemove(dict *d, int table, size_t g, int pos, uint64_t mixed) {
    uint8_t *ctrl = groupCtrl(d, table, g);
    debugAssert(ctrl[pos] != CTRL_EMPTY);
    if (ctrl[pos] & CTRL_DISPLACED) {
        size_t mask = expToMask(d->ht_size_exp[table]);
        size_t q = mixed & mask, step = 0;
        while (q != g) {
            uint8_t *c = groupCtrl(d, table, q);
            if (c[OVF_POS] != OVF_MAX) {
                debugAssert(c[OVF_POS] > 0);
                c[OVF_POS]--;
            }
            assert(step < mask);
            q = probeNext(q, &step, mask);
        }
    }
    ctrl[pos] = CTRL_EMPTY;
    d->ht_used[table]--;
}

/* Searches the probe sequence of a key in one table. Returns the slot, and its
 * group and position, or NULL. Does not write anything. */
static inline dictEntry **tableFind(dict *d, int table, uint64_t mixed, const void *key,
                                    keyCmpFunc cmp, dictCmpCache *cache,
                                    size_t *g_out, int *pos_out)
{
    size_t mask = expToMask(d->ht_size_exp[table]);
    size_t g = mixed & mask, step = 0;
    uint8_t tag = hashTag(mixed);
    for (;;) {
        const uint8_t *ctrl = groupCtrl(d, table, g);
        groupMask m = groupMatchTag(ctrl, tag);
        if (m) {
            dictEntry **slots = groupSlots(d, table, g);
            do {
                int pos = __builtin_ctz(m);
                const void *visited = slotLookupKey(d, slots[pos]);
                if (key == visited || cmp(cache, key, visited)) {
                    *g_out = g;
                    *pos_out = pos;
                    return &slots[pos];
                }
                m &= m - 1;
            } while (m);
        }
        if (ctrl[OVF_POS] == 0 || step == mask) return NULL;
        g = probeNext(g, &step, mask);
    }
}

/* ----------------------------- overflow stash ----------------------------- */

/* While rehashing is paused no entry may move, so a table can't be grown by
 * rehashing into a bigger one. If the new table can't take an insert then (it
 * must also be able to take every entry of the old table once rehashing
 * resumes), the entry goes to the stash: a list of nodes whose slot is a
 * stable dictEntryLink. Stashed entries are counted in ht_used[1] (there is
 * only a stash while rehashing) and are moved into the table when rehashing
 * resumes. A deleted node stays in the list, with a NULL slot, until then, so
 * that the position of a safe iterator in the stash stays valid. */
typedef struct dictStashEntry {
    dictEntry *slot;
    uint64_t mixed;
    struct dictStashEntry *next;
} dictStashEntry;

static inline void stashMemChanged(dict *d, long long delta_bytes) {
    if (d->type->stashMemChanged)
        d->type->stashMemChanged(d, delta_bytes);
}

static inline int dictNeedsStash(dict *d) {
    return dictIsRehashing(d) && d->pauserehash > 0 &&
           dictSize(d) + 1 >= tableCapacity(d->ht_size_exp[1]);
}

static dictEntry **stashInsert(dict *d, uint64_t mixed, dictEntry *entry) {
    dictStashEntry *se = zmalloc(sizeof(*se));
    se->slot = entry;
    se->mixed = mixed;
    se->next = d->stash;
    d->stash = se;
    d->stash_count++;
    d->ht_used[1]++;
    stashMemChanged(d, sizeof(*se));
    return &se->slot;
}

static dictStashEntry *stashFind(dict *d, uint64_t mixed, const void *key,
                                 keyCmpFunc cmp, dictCmpCache *cache)
{
    for (dictStashEntry *se = d->stash; se; se = se->next) {
        if (!se->slot || se->mixed != mixed) continue;
        const void *visited = slotLookupKey(d, se->slot);
        if (key == visited || cmp(cache, key, visited)) return se;
    }
    return NULL;
}

static void stashRemove(dict *d, dictStashEntry *se) {
    debugAssert(se->slot != NULL);
    se->slot = NULL;
    d->ht_used[1]--;
}

static size_t stashLive(dict *d) {
    size_t n = 0;
    for (dictStashEntry *se = d->stash; se; se = se->next) n += se->slot != NULL;
    return n;
}

/* A random stashed entry. The stash must not be empty. */
static dictEntry *stashRandom(dict *d, size_t live) {
    size_t target = randomULong() % live;
    for (dictStashEntry *se = d->stash; se; se = se->next) {
        if (se->slot && target-- == 0) return se->slot;
    }
    assert(0);
    return NULL;
}

static void dictMakeRoomForInsert(dict *d);

/* Moves the stashed entries into the table, now that rehashing may run. */
static void stashDrain(dict *d) {
    dictStashEntry *se = d->stash;
    d->stash = NULL;
    /* They are inserted again one by one, growing the table as needed. */
    for (dictStashEntry *s = se; s; s = s->next) d->ht_used[1] -= s->slot != NULL;
    while (se) {
        dictStashEntry *next = se->next;
        if (se->slot) {
            dictMakeRoomForInsert(d);
            tableInsert(d, dictIsRehashing(d) ? 1 : 0, se->mixed, se->slot);
        }
        zfree(se);
        d->stash_count--;
        stashMemChanged(d, -(long long)sizeof(*se));
        se = next;
    }
}

/* Frees the stashed entries and the stash. The caller reports the freed bytes,
 * as for _dictClear(). */
static void stashClear(dict *d) {
    dictStashEntry *se = d->stash;
    d->stash = NULL;
    while (se) {
        dictStashEntry *next = se->next;
        if (se->slot) {
            dictFreeKey(d, se->slot);
            dictFreeVal(d, se->slot);
            if (!entryIsKey(se->slot)) zfree(decodeMaskedPtr(se->slot));
            d->ht_used[1]--;
        }
        zfree(se);
        d->stash_count--;
        se = next;
    }
}

/* Finds a key in the dict. Returns its slot, and its table, group and position,
 * or NULL. Does not rehash or write anything. For a stashed entry the table is
 * 2 and `g_out` is the stash entry. */
static dictEntry **findKey(dict *d, uint64_t mixed, const void *key,
                           int *table_out, size_t *g_out, int *pos_out)
{
    dictCmpCache cache = {0};
    keyCmpFunc cmp = dictGetCmpFunc(d);
    size_t g;
    int pos;
    for (int table = 0; table <= 1; table++) {
        if (d->ht_used[table] == 0) continue;
        /* Entries whose home group precedes rehashidx moved to the new table. */
        if (table == 0 && d->rehashidx >= 0 &&
            (long)(mixed & expToMask(d->ht_size_exp[0])) < d->rehashidx)
            continue;
        dictEntry **slot = tableFind(d, table, mixed, key, cmp, &cache, &g, &pos);
        if (slot) {
            if (table_out) *table_out = table;
            if (g_out) *g_out = g;
            if (pos_out) *pos_out = pos;
            return slot;
        }
    }
    if (unlikely(d->stash != NULL)) {
        dictStashEntry *se = stashFind(d, mixed, key, cmp, &cache);
        if (se) {
            if (table_out) *table_out = 2;
            if (g_out) *g_out = (size_t)(uintptr_t)se;
            if (pos_out) *pos_out = 0;
            return &se->slot;
        }
    }
    return NULL;
}

/* ----------------------------- pausing ------------------------------------ */

/* Once nothing pauses rehashing any more, move stashed entries into the table
 * and run a shrink that was deferred because the dict was paused. */
static void dictResumeFinalize(dict *d) {
    if (d->pauserehash != 0) return;
    if (d->stash) stashDrain(d);
    if (d->pending_shrink) {
        d->pending_shrink = 0;
        _dictShrinkIfNeeded(d);
    }
    dictForceRehashIfNeeded(d);
}

/* Used by safe iterators, scan, two-phase delete and callers that hold a link:
 * entries must not move to the other table while paused. Inserts and deletes
 * never move other entries, so they are allowed. */
void dictPauseRehashing(dict *d) {
    d->pauserehash++;
}

void dictResumeRehashing(dict *d) {
    debugAssert(d->pauserehash > 0);
    d->pauserehash--;
    dictResumeFinalize(d);
}

/* ----------------------------- API implementation ------------------------- */

/* Reset hash table parameters already initialized with _dictInit()*/
static void _dictReset(dict *d, int htidx)
{
    d->ht_table[htidx] = NULL;
    d->ht_size_exp[htidx] = -1;
    d->ht_used[htidx] = 0;
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
    d->stash = NULL;
    d->stash_count = 0;
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pauseAutoResize = 0;
    d->pending_shrink = 0;
    return DICT_OK;
}

/* The smallest table, as a group exponent (groups = 1 << exp), that holds `n`
 * entries without growing: groups = ceil(n / (GROUP_SLOTS * 7/8)). */
static signed char nextGroupExp(size_t n) {
    if (n <= maxFill(DICT_HT_INITIAL_EXP)) return DICT_HT_INITIAL_EXP;
    if (n >= SIZE_MAX / 8) return CHAR_BIT * sizeof(size_t) - 1;
    size_t groups = (n * 8 + GROUP_SLOTS * 7 - 1) / (GROUP_SLOTS * 7);
    if (groups <= 1) return 0;
    return CHAR_BIT * sizeof(size_t) - __builtin_clzl(groups - 1);
}

/* Swap the new table in place of the old one once all entries moved. */
static int dictCheckRehashingCompleted(dict *d) {
    if (!dictIsRehashing(d) || d->ht_used[0] != 0) return 0;
    debugAssert(d->stash == NULL); /* drained before rehashing can run */

    if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
    tableMemChanged(d, -(long long)tableBytes(d->ht_size_exp[0]));
    zfree(d->ht_table[0]);
    /* Copy the new ht onto the old one */
    d->ht_table[0] = d->ht_table[1];
    d->ht_used[0] = d->ht_used[1];
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
        size_t s0 = numGroups(d->ht_size_exp[0]);
        size_t s1 = numGroups(d->ht_size_exp[1]);
        /* Shrinking leaves spare capacity, so a forced shrink at 1/32
         * occupancy selects a target less than 32 times smaller. Allow that
         * shrink to progress using the same occupancy threshold. */
        if ((s1 > s0 && s1 < dict_force_resize_ratio * s0) ||
            (s1 < s0 && s0 < HASHTABLE_MIN_FILL * dict_force_resize_ratio * s1 &&
             dictSize(d) > tableCapacity(d->ht_size_exp[0]) / (HASHTABLE_MIN_FILL * dict_force_resize_ratio)))
            return 0;
    }
    return 1;
}

/* No entry of the old table has this home group. Conservative: a group that
 * other entries probed past is not reported empty. */
static inline int homeIsEmpty(dict *d, size_t g) {
    const uint8_t *ctrl = groupCtrl(d, 0, g);
    return ctrl[OVF_POS] == 0 && groupMatchNative(ctrl) == 0;
}

/* Moves every entry of the old table whose home group is `home` to the new
 * table. Native entries sit in the home group itself; entries displaced from it
 * are further along its probe sequence, which ends at the first group nothing
 * probed past. Does not change rehashidx. */
static void rehashHome(dict *d, size_t home) {
    int expanding = d->ht_size_exp[1] > d->ht_size_exp[0];
    size_t mask = expToMask(d->ht_size_exp[0]);
    size_t g = home, step = 0;
    for (;;) {
        uint8_t *ctrl = groupCtrl(d, 0, g);
        dictEntry **slots = groupSlots(d, 0, g);
        /* In the home group only native entries have this home; elsewhere
         * only displaced ones may. */
        groupMask m = (g == home) ? groupMatchNative(ctrl) : groupMatchDisplaced(ctrl);
        while (m) {
            int pos = __builtin_ctz(m);
            m &= m - 1;
            dictEntry *e = slots[pos];
            uint64_t mixed;
            if (g == home && !expanding) {
                /* Shrinking: the home group in the smaller table is the
                 * masked old one and the tag is unchanged, no need to hash. */
                mixed = ((uint64_t)(ctrl[pos] & CTRL_TAG_MASK) << 57) | home;
            } else {
                mixed = entryMixedHash(d, e);
                if ((mixed & mask) != home) continue; /* displaced from another home */
            }
            tableRemove(d, 0, g, pos, mixed);
            tableInsert(d, 1, mixed, e);
        }
        /* Removing entries of this group only changes the counters of the
         * groups before it, so the counter of `g` still tells whether entries
         * of this home may follow. */
        if (ctrl[OVF_POS] == 0 || step == mask) break;
        g = probeNext(g, &step, mask);
    }
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

    signed char new_exp = nextGroupExp(size);

    /* Detect overflows */
    size_t newgroups = numGroups(new_exp);
    if (newgroups * GROUP_SLOTS < size || newgroups > SIZE_MAX / DICT_GROUP_BYTES)
        return DICT_ERR;

    /* Rehashing to the same table size is not useful. */
    if (new_exp == d->ht_size_exp[0]) return DICT_ERR;

    /* Allocate the new hash table, all slots empty (control byte 0). */
    size_t newbytes = newgroups * DICT_GROUP_BYTES;
    uint8_t *new_table;
    if (malloc_failed) {
        new_table = ztrycalloc(newbytes);
        *malloc_failed = new_table == NULL;
        if (*malloc_failed)
            return DICT_ERR;
    } else
        new_table = zcalloc(newbytes);

    /* Prepare a second hash table for incremental rehashing.
     * We do this even for the first initialization, so that we can trigger the
     * rehashingStarted more conveniently, we will clean it up right after. */
    d->ht_size_exp[1] = new_exp;
    d->ht_used[1] = 0;
    d->ht_table[1] = new_table;
    d->rehashidx = 0;
    if (d->type->rehashingStarted) d->type->rehashingStarted(d);
    tableMemChanged(d, newbytes);

    /* Is this the first initialization or is the first hash table empty? If so
     * it's not really a rehashing, we can just set the first hash table so that
     * it can accept keys. An empty old table that someone may be iterating
     * (rehashing paused) must stay alive until rehashing is resumed. */
    if (d->ht_table[0] == NULL || (d->ht_used[0] == 0 && d->pauserehash == 0)) {
        if (d->type->rehashingCompleted) d->type->rehashingCompleted(d);
        tableMemChanged(d, -(long long)tableBytes(d->ht_size_exp[0]));
        if (d->ht_table[0]) zfree(d->ht_table[0]);
        d->ht_size_exp[0] = new_exp;
        d->ht_used[0] = 0;
        d->ht_table[0] = new_table;
        _dictReset(d, 1);
        d->rehashidx = -1;
        return DICT_OK;
    }

    dictForceRehashIfNeeded(d);
    return DICT_OK;
}

int _dictExpand(dict *d, unsigned long size, int* malloc_failed) {
    /* the size is invalid if the table already holds it without growing,
     * or if it is smaller than the number of elements already in the table */
    if (dictIsRehashing(d) || d->ht_used[0] > size ||
        maxFill(d->ht_size_exp[0]) >= size)
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
    /* the size is invalid if the table can't hold it without growing,
     * or if it is smaller than the number of elements already in the table */
    if (dictIsRehashing(d) || d->ht_used[0] > size ||
        maxFill(d->ht_size_exp[0]) <= size)
        return DICT_ERR;
    return _dictResize(d, size, NULL);
}

/* Performs N steps of incremental rehashing. Returns 1 if there are still
 * keys to move from the old to the new hash table, otherwise 0 is returned.
 *
 * Note that a rehashing step moves the entries of one home group (wherever
 * along its probe sequence they are) from the old to the new hash table. Since
 * part of the hash table may be composed of empty groups, it is not guaranteed
 * that this function will rehash even a single group, since it will visit at
 * max N*10 empty groups in total, otherwise the amount of work it does would be
 * unbound and the function may block for a long time.
 *
 * `force` ignores the resize policy: the new table is filling up and the old
 * one must be emptied into it. */
static int rehashSteps(dict *d, int n, int force) {
    int empty_visits = n*10; /* Max number of empty groups to visit. */
    if (!dictIsRehashing(d) || d->pauserehash != 0) return 0;
    if (!force && !rehashAllowedByPolicy(d)) return 0;

    while (n-- && d->ht_used[0] != 0) {
        size_t size0 = numGroups(d->ht_size_exp[0]);
        /* Every remaining entry has its home at or after rehashidx, so
         * rehashidx can't overflow. */
        assert(size0 > (size_t)d->rehashidx);
        while (homeIsEmpty(d, d->rehashidx)) {
            d->rehashidx++;
            assert(size0 > (size_t)d->rehashidx);
            if (--empty_visits == 0) return 1;
        }
        rehashHome(d, d->rehashidx);
        d->rehashidx++;
    }

    return !dictCheckRehashingCompleted(d);
}

int dictRehash(dict *d, int n) {
    return rehashSteps(d, n, 0);
}

/* Command dictionaries must finish resizing before concurrent readers use
 * them. Ignore the resize policy, including when hard-limit growth ran during
 * a fork. An iterator or held link still takes precedence; the final resume
 * completes any rehash that was deferred while paused. */
static void dictForceRehashIfNeeded(dict *d) {
    if (d->type->force_full_rehash && d->pauserehash == 0) {
        while (rehashSteps(d, 1000, 1)) {
            /* Continue rehashing */
        }
    }
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

/* Rehashes the entries of a single home group. */
static int _dictBucketRehash(dict *d, uint64_t idx) {
    if (d->pauserehash != 0) return 0;
    if (!dictIsRehashing(d) || !rehashAllowedByPolicy(d)) return 0;
    rehashHome(d, idx);
    dictCheckRehashingCompleted(d);
    return 1;
}

static void _dictRehashStepIfNeeded(dict *d, uint64_t mixed) {
    if ((!dictIsRehashing(d)) || (d->pauserehash != 0))
        return;
    size_t idx = mixed & expToMask(d->ht_size_exp[0]);
    if ((long)idx >= d->rehashidx && !homeIsEmpty(d, idx)) {
        /* If the old table may hold entries with the home group of the key,
         * rehash that group (being more CPU cache friendly). */
        _dictBucketRehash(d, idx);
    } else {
        /* Otherwise rehash at rehashidx (not CPU cache friendly). */
        dictRehash(d,1);
    }
}

/* Called before every insert, after the rehash step. Grows the table by the
 * resize policy, and makes sure that the table taking the insert has room.
 * Open addressing can't hold more entries than slots, so the hard fill limit
 * overrides the resize policy, paused auto-resize and resizeAllowed(). */
static void dictMakeRoomForInsert(dict *d) {
    if (!dictIsRehashing(d)) {
        if (d->ht_table[0] == NULL) {
            dictExpand(d, 1);
            return;
        }
        _dictExpandIfNeeded(d);
        if (!dictIsRehashing(d) && d->ht_used[0] >= hardFill(d->ht_size_exp[0]))
            _dictExpand(d, d->ht_used[0] + 1, NULL);
        if (!dictIsRehashing(d)) return;
    }

    /* While rehashing, inserts go to the new table, which in the end must hold
     * every entry. If it is filling up, rehashing can't wait for the policy. */
    if (dictSize(d) >= maxFill(d->ht_size_exp[1]) && d->pauserehash == 0) {
        rehashSteps(d, 1, 1);
        if (!dictIsRehashing(d)) {
            dictMakeRoomForInsert(d);
            return;
        }
    }
    if (dictSize(d) >= hardFill(d->ht_size_exp[1])) {
        if (d->pauserehash == 0) {
            while (rehashSteps(d, 1000, 1)) {
                /* Finish rehashing now, then grow. */
            }
            dictMakeRoomForInsert(d);
            return;
        }
        /* Entries can't move while rehashing is paused: the new table takes
         * the insert as it is, or if it is full it goes to the stash. */
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

/* Stores `key` in a free slot and returns the slot content (the tagged key for
 * no_value dicts, a new dictEntry otherwise). If `slot` is not NULL it is set
 * to the address of the slot. The key must not exist in the dict and the table
 * must have been made ready (rehash step, dictMakeRoomForInsert()) by the
 * caller. */
static dictEntry *dictInsertKey(dict *d, uint64_t mixed, void *key __stored_key, dictEntryLink *slot) {
    dictEntry *entry;
    if (d->type->no_value) {
        entry = encodeEntryKey(d, key);
        assert(entryIsKey(entry));
    } else {
        entry = zmalloc(sizeof(*entry));
        assert(entryIsNormal(entry)); /* Check alignment of allocation */
        entry->key = key;
    }
    dictEntry **s;
    if (unlikely(dictNeedsStash(d)))
        s = stashInsert(d, mixed, entry);
    else
        s = tableInsert(d, dictIsRehashing(d) ? 1 : 0, mixed, entry);
    if (slot) *slot = s;
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
    uint64_t mixed = keyMixedHash(d, lookup_key);
    if (existing) *existing = NULL;

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, mixed);
    /* Expand the hash table if needed */
    dictMakeRoomForInsert(d);

    dictEntry **slot = findKey(d, mixed, lookup_key, NULL, NULL, NULL);
    if (slot) {
        if (existing) *existing = *slot;
        return NULL;
    }

    /* Dup the key if necessary. */
    if (d->type->keyDup) key = d->type->keyDup(d, key);

    return dictInsertKey(d, mixed, key, NULL);
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
    size_t g;

    /* dict is empty */
    if (dictSize(d) == 0) return NULL;

    uint64_t mixed = keyMixedHash(d, key);

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, mixed);

    dictEntry **slot = findKey(d, mixed, key, &table, &g, &pos);
    if (!slot) return NULL; /* not found */

    dictEntry *he = *slot;
    if (table == 2) {
        stashRemove(d, (dictStashEntry *)(uintptr_t)g);
        table = 1; /* stashed entries are counted in ht_used[1] */
    } else {
        tableRemove(d, table, g, pos, mixed);
    }
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

/* Free the entries of a table (keys, values, dictEntry structs) and the table
 * itself. The memory accounting callback is NOT called: the caller reports
 * the freed bytes. */
static int _dictClear(dict *d, int htidx, void(callback)(dict*)) {
    size_t ng = numGroups(d->ht_size_exp[htidx]);
    for (size_t g = 0; g < ng && d->ht_used[htidx] > 0; g++) {
        /* Callback will be called once for every 4096 groups. */
        if (callback && g != 0 && (g & 4095) == 0) callback(d);

        groupMask full = groupMatchFull(groupCtrl(d, htidx, g));
        dictEntry **slots = groupSlots(d, htidx, g);
        while (full) {
            int pos = __builtin_ctz(full);
            full &= full - 1;
            dictEntry *he = slots[pos];
            dictFreeKey(d, he);
            dictFreeVal(d, he);
            if (!entryIsKey(he)) zfree(decodeMaskedPtr(he));
            d->ht_used[htidx]--;
        }
    }
    /* Free the table and the allocated cache structure */
    zfree(d->ht_table[htidx]);
    /* Re-initialize the table */
    _dictReset(d, htidx);
    return DICT_OK; /* never fails */
}

/* Total memory of the hash tables. */
size_t dictTableMemUsage(const dict *d) {
    return dictBuckets(d) * DICT_GROUP_BYTES;
}

size_t dictStashMemUsage(const dict *d) {
    return d->stash_count * sizeof(dictStashEntry);
}

/* Clear & Release the hash table */
void dictRelease(dict *d)
{
    /* Someone may be monitoring a dict that started rehashing, before
     * destroying the dict fake completion. */
    if (dictIsRehashing(d) && d->type->rehashingCompleted)
        d->type->rehashingCompleted(d);

    /* Subtract the size of all tables. */
    tableMemChanged(d, -(long long)dictTableMemUsage(d));
    stashMemChanged(d, -(long long)dictStashMemUsage(d));

    if (d->type->onDictRelease)
        d->type->onDictRelease(d);

    stashClear(d);
    _dictClear(d,0,NULL);
    _dictClear(d,1,NULL);
    zfree(d);
}

/* Finds a given key. Like dictFindLink(), yet search bucket even if dict is empty.
 *
 * Returns dictEntryLink reference if found. Otherwise, return NULL.
 *
 * bucket - return an opaque, non-NULL, token for the group where the key
 *          would be inserted, unless the dict has no hash table. The token
 *          is never dereferenced by dict.c: inserts recompute the position.
 *          Lookups never write to the table.
 */
static dictEntryLink dictFindLinkInternal(dict *d, const void *key, dictEntryLink *bucket) {
    if (bucket) {
        *bucket = NULL;
    } else {
        /* If dict is empty and no need to find bucket, return NULL */
        if (dictSize(d) == 0) return NULL;
    }

    const uint64_t mixed = keyMixedHash(d, key);

    /* Rehash the hash table if needed */
    _dictRehashStepIfNeeded(d, mixed);

    if (bucket) {
        int itable = dictIsRehashing(d) ? 1 : 0;
        if (d->ht_table[itable])
            *bucket = groupSlots(d, itable, mixed & expToMask(d->ht_size_exp[itable]));
    }

    return findKey(d, mixed, key, NULL, NULL, NULL);
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
    uint64_t mixed = mixHash(hash);
    uint8_t tag = hashTag(mixed);

    if (dictSize(d) == 0) return NULL; /* dict is empty */
    for (int table = 0; table <= 1; table++) {
        if (d->ht_used[table] == 0) continue;
        size_t mask = expToMask(d->ht_size_exp[table]);
        size_t g = mixed & mask, step = 0;
        if (table == 0 && (long)g < d->rehashidx) continue;
        for (;;) {
            const uint8_t *ctrl = groupCtrl(d, table, g);
            groupMask m = groupMatchTag(ctrl, tag);
            dictEntry **slots = groupSlots(d, table, g);
            while (m) {
                int pos = __builtin_ctz(m);
                m &= m - 1;
                if (oldptr == entryStoredKey(slots[pos]))
                    return slots[pos];
            }
            if (ctrl[OVF_POS] == 0 || step == mask) break;
            g = probeNext(g, &step, mask);
        }
    }
    for (dictStashEntry *se = d->stash; se; se = se->next) {
        if (se->slot && se->mixed == mixed && oldptr == entryStoredKey(se->slot))
            return se->slot;
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
 * Entries never move within a table, so a link stays valid across inserts and
 * deletes of other keys. It is invalidated by deleting the entry, and by any
 * rehash step (a lookup or write may perform one) or resize that moves the
 * entry to the other table. Use dictPauseRehashing() to hold a link across
 * calls that may look up or modify keys in the same dict.
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
 *  bucket - an opaque token for the group that the key was mapped to; not
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
        uint64_t mixed = keyMixedHash(d, dictStoredKey2Key(d, key));

        /* Rehash a step and make room if needed for the new key */
        _dictRehashStepIfNeeded(d, mixed);
        dictMakeRoomForInsert(d);

        dictInsertKey(d, mixed, addedKey, link);
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
 * `dictTwoPhaseUnlinkFind` pauses rehash and `dictTwoPhaseUnlinkFree` resumes it.
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
    size_t g;

    if (dictSize(d) == 0) return NULL; /* dict is empty */

    uint64_t mixed = keyMixedHash(d, key);
    _dictRehashStepIfNeeded(d, mixed);

    dictEntry **slot = findKey(d, mixed, key, &table, &g, &ipos);
    if (!slot) return NULL;
    pos->slot = table == 2 ? g : g * GROUP_SLOTS + ipos;
    pos->hash = mixed;
    pos->table = table;
    dictPauseRehashing(d);
    return slot;
}

void dictTwoPhaseUnlinkFree(dict *d, dictEntryLink plink, dictPosition *pos) {
    if (plink == NULL || *plink == NULL) return;
    size_t g = pos->slot / GROUP_SLOTS;
    int ipos = pos->slot % GROUP_SLOTS;
    dictEntry *de = *plink;

    /* Unlink, free the entry, then resume. The key may have been replaced by
     * NULL (see dbDelete), so the key is freed without decoding it here and
     * the hash saved by the find is used. */
    if (pos->table == 2) {
        dictStashEntry *se = (dictStashEntry *)(uintptr_t)pos->slot;
        debugAssert(plink == &se->slot);
        stashRemove(d, se);
    } else {
        debugAssert(plink == &groupSlots(d, pos->table, g)[ipos] &&
                    groupCtrl(d, pos->table, g)[ipos] != CTRL_EMPTY);
        tableRemove(d, pos->table, g, ipos, pos->hash);
    }
    dictFreeKey(d, de);
    dictFreeVal(d, de);
    if (!entryIsKey(de)) zfree(decodeMaskedPtr(de));

    debugAssert(d->pauserehash > 0);
    d->pauserehash--;
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
 * and values: the hash tables, overflow stash nodes and, for dicts with values,
 * one dictEntry per element. no_value dicts store the key directly in the slot. */
size_t dictMemUsage(const dict *d) {
    return dictSize(d) * dictEntryMemUsage(d->type->no_value) +
           dictTableMemUsage(d) + dictStashMemUsage(d);
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
    unsigned long long integers[7], hash = 0;
    int j;

    integers[0] = (long) d->ht_table[0];
    integers[1] = d->ht_size_exp[0];
    integers[2] = d->ht_used[0];
    integers[3] = (long) d->ht_table[1];
    integers[4] = d->ht_size_exp[1];
    integers[5] = d->ht_used[1];
    integers[6] = (long) d->stash;

    /* We hash N integers by summing every successive integer with the integer
     * hashing of the previous sum. Basically:
     *
     * Result = hash(hash(hash(int1)+int2)+int3) ...
     *
     * This way the same set of integers in a different order will (likely) hash
     * to a different number. */
    for (j = 0; j < 7; j++) {
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
    iter->stash_next = NULL;
    iter->safe = 0;
    iter->done = 0;
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

/* Iterates the slots of table 0, then of table 1 if rehashing, then the stash.
 * A safe iterator pauses rehashing, so no entry moves: every entry present for
 * the whole iteration is returned exactly once. Entries added meanwhile may or
 * may not be returned. */
dictEntry *dictNext(dictIterator *iter)
{
    dict *d = iter->d;
    if (iter->done) return NULL;

    if (iter->index == -1 && iter->table == 0) {
        /* First call. A safe iterator pauses rehashing, so that entries don't
         * move. */
        if (iter->safe)
            dictPauseRehashing(d);
        else
            iter->fingerprint = dictFingerprint(d);
        iter->index = 0;
        iter->pos = -1;
        if (d->ht_table[0] == NULL) {
            iter->done = 1; /* no table, nothing to iterate */
            return NULL;
        }
    }

    while (iter->table < 2) {
        if (++iter->pos >= GROUP_SLOTS) {
            iter->pos = 0;
            if ((size_t)++iter->index >= numGroups(d->ht_size_exp[iter->table])) {
                if (dictIsRehashing(d) && iter->table == 0) {
                    iter->table++;
                    iter->index = 0;
                } else {
                    iter->table = 2;
                    iter->stash_next = d->stash;
                    break;
                }
            }
        }
        if (groupCtrl(d, iter->table, iter->index)[iter->pos] != CTRL_EMPTY)
            return groupSlots(d, iter->table, iter->index)[iter->pos];
    }
    while (iter->stash_next) {
        dictStashEntry *se = iter->stash_next;
        iter->stash_next = se->next;
        if (se->slot) return se->slot;
    }
    iter->done = 1;
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
    if (dictSize(d) == 0) return NULL;
    if (dictIsRehashing(d)) _dictRehashStep(d);
    if (unlikely(d->stash != NULL)) {
        size_t live = stashLive(d);
        if (live && randomULong() % dictSize(d) < live) return stashRandom(d, live);
    }
    while (1) {
        int table = 0;
        size_t g;
        if (dictIsRehashing(d)) {
            /* A random group of either table. Groups of the old table before
             * rehashidx are mostly, but not always, empty: displaced entries
             * of later home groups can sit there. */
            size_t s0 = numGroups(d->ht_size_exp[0]);
            g = randomULong() % (s0 + numGroups(d->ht_size_exp[1]));
            if (g >= s0) {
                table = 1;
                g -= s0;
            }
        } else {
            g = randomULong() & expToMask(d->ht_size_exp[0]);
        }
        groupMask full = groupMatchFull(groupCtrl(d, table, g));
        if (!full) continue;

        /* Now we found a non empty group and we need to get a random element
         * from it. */
        unsigned target = random() % __builtin_popcount(full);
        while (target--) full &= full - 1;
        return groupSlots(d, table, g)[__builtin_ctz(full)];
    }
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
    if (count == 0) return 0;
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
    unsigned long emptylen = 0; /* Continuous empty groups so far. */
    while(stored < count && maxsteps--) {
        for (j = 0; j < tables; j++) {
            /* Visit physical groups even before rehashidx: they may hold
             * entries displaced from home groups that have not moved yet. */
            if (i >= numGroups(d->ht_size_exp[j])) continue; /* Out of range for this table. */
            groupMask full = groupMatchFull(groupCtrl(d, j, i));

            /* Count contiguous empty groups, and jump to other
             * locations if they reach 'count' (with a minimum of 5). */
            if (!full) {
                emptylen++;
                if (emptylen >= 5 && emptylen > count) {
                    i = randomULong() & maxsizemask;
                    emptylen = 0;
                }
            } else {
                emptylen = 0;
                dictEntry **slots = groupSlots(d, j, i);
                while (full) {
                    dictEntry *he = slots[__builtin_ctz(full)];
                    full &= full - 1;
                    /* Collect all the elements of the groups found non empty while iterating.
                     * To avoid the issue of being unable to sample the end of a group,
                     * we utilize the Reservoir Sampling algorithm to optimize the sampling process.
                     * This means that even when the maximum number of samples has been reached,
                     * we continue sampling until we reach the end of the group.
                     * See https://en.wikipedia.org/wiki/Reservoir_sampling. */
                    if (stored < count) {
                        des[stored] = he;
                    } else {
                        unsigned long r = randomULong() % (stored + 1);
                        if (r < count) des[r] = he;
                    }
                    stored++;
                }
                if (stored >= count) goto end;
            }
        }
        i = (i+1) & maxsizemask;
    }

end:
    /* Include live stashed entries in the reservoir. Rehashing remains paused
     * while a stash exists, so these slots stay valid throughout sampling. */
    for (dictStashEntry *se = d->stash; se; se = se->next) {
        if (!se->slot) continue;
        if (stored < count) {
            des[stored] = se->slot;
        } else {
            unsigned long r = randomULong() % (stored + 1);
            if (r < count) des[r] = se->slot;
        }
        stored++;
    }
    return stored > count ? count : stored;
}

/* Reallocate the dictEntry, key and value allocations of the entry in a slot
 * using the provided allocation functions in order to defrag them. The tables
 * themselves are defragged by dictDefragTables(). */
static void dictDefragSlot(dict *d, dictEntry **slot, dictDefragFunctions *defragfns) {
    dictDefragAllocFunction *defragalloc = defragfns->defragAlloc;
    dictDefragAllocFunction *defragkey = defragfns->defragKey;
    dictDefragAllocFunction *defragval = defragfns->defragVal;
    dictEntry *de = *slot, *newde = NULL;
    void *newkey = defragkey ? defragkey(entryStoredKey(de)) : NULL;
    if (d->type->no_value) {
        if (newkey) *slot = encodeEntryKey(d, newkey);
    } else {
        void *newval = defragval ? defragval(dictGetVal(de)) : NULL;
        assert(entryIsNormal(de));
        newde = defragalloc ? defragalloc(de) : NULL;
        if (newde) de = newde;
        if (newkey) de->key = newkey;
        if (newval) de->v.val = newval;
        if (newde) *slot = newde;
    }
}

/* This is like dictGetRandomKey() from the POV of the API, but returns every
 * entry with the same probability: it picks slots uniformly at random until one
 * holds an entry. dictGetRandomKey() instead selects a random group and then a
 * random entry of it, which favors entries of groups with few entries.
 *
 * The expected number of picks is slots / entries, which the resize policy
 * keeps below 8 (32 while a fork child runs). If unlucky, fall back to sampling
 * a run of groups. */
#define GETFAIR_MAX_PICKS 64
#define GETFAIR_NUM_ENTRIES 15
dictEntry *dictGetFairRandomKey(dict *d) {
    if (dictSize(d) == 0) return NULL;
    if (dictIsRehashing(d)) _dictRehashStep(d);
    if (unlikely(d->stash != NULL)) {
        size_t live = stashLive(d);
        if (live && randomULong() % dictSize(d) < live) return stashRandom(d, live);
    }
    size_t slots0 = tableCapacity(d->ht_size_exp[0]);
    size_t slots = slots0 + (dictIsRehashing(d) ? tableCapacity(d->ht_size_exp[1]) : 0);
    for (int i = 0; i < GETFAIR_MAX_PICKS; i++) {
        size_t r = randomULong() % slots;
        int table = 0;
        if (r >= slots0) {
            table = 1;
            r -= slots0;
        }
        size_t g = r / GROUP_SLOTS;
        int pos = r % GROUP_SLOTS;
        if (groupCtrl(d, table, g)[pos] != CTRL_EMPTY)
            return groupSlots(d, table, g)[pos];
    }

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
 * Scan callback rules: the callback may delete entries, including the one
 * passed to it, and may insert or replace entries. Entries never move while a
 * scan step runs, so this can't hide an entry from the scan.
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
 * dict.c hash tables are always power of two in size (in groups), and every
 * entry has a home group given by the bitwise AND between the mixed hash of
 * its key and SIZE-1. A cursor step returns exactly the entries whose home
 * group is the cursor: those stored in the group itself, and those displaced
 * from it to later groups of its probe sequence. Entries of other home groups
 * stored in the same groups are left to their own cursor step, so a table
 * that doesn't change in size returns every entry once.
 *
 * For example if the current table has 16 groups, the mask is
 * (in binary) 1111. The home group of a key in the hash table will always be
 * the last four bits of the mixed hash, and so forth.
 *
 * WHAT HAPPENS IF THE TABLE CHANGES IN SIZE?
 *
 * If the hash table grows, elements can go anywhere in one multiple of
 * the old group: for example let's say we already iterated with
 * a 4 bit cursor 1100 (the mask is 1111 because hash table size = 16).
 *
 * If the hash table will be resized to 64 groups, then the new mask will
 * be 111111. The new home groups you obtain by substituting in ??1100
 * with either 0 or 1 can be targeted only by keys we already visited
 * when scanning the home group 1100 in the smaller hash table.
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
 * Rehashing moves all the entries of a home group at once, so they are
 * either still in the old table or all in the new one.
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
 *    return all the keys of a given home group, and all the expansions, so
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

/* Emit the entries of table `table` whose home group is `home`: the native
 * entries of that group, then the entries displaced from it along its probe
 * sequence. Control bytes are re-read for every slot, as callbacks may delete
 * and insert entries. */
static void dictScanHome(dict *d, int table, size_t home, dictScanFunction *fn,
                         dictDefragFunctions *defragfns, void *privdata)
{
    size_t mask = expToMask(d->ht_size_exp[table]);
    size_t g = home, step = 0;
    for (;;) {
        for (int pos = 0; pos < GROUP_SLOTS; pos++) {
            uint8_t c = groupCtrl(d, table, g)[pos];
            if (c == CTRL_EMPTY) continue;
            /* In the home group only native entries have this home; elsewhere
             * only displaced ones may. */
            if ((g == home) == ((c & CTRL_DISPLACED) != 0)) continue;
            dictEntry **slot = &groupSlots(d, table, g)[pos];
            if (g != home && (entryMixedHash(d, *slot) & mask) != home) continue;
            if (defragfns) dictDefragSlot(d, slot, defragfns);
            fn(privdata, *slot, slot);
        }
        if (groupCtrl(d, table, g)[OVF_POS] == 0 || step == mask) break;
        g = probeNext(g, &step, mask);
    }
    /* Stashed entries belong to the new table. Entries stashed by the
     * callbacks are added before the start of the list and not visited. */
    if (table == 1) {
        for (dictStashEntry *se = d->stash; se; se = se->next) {
            if (!se->slot || (se->mixed & mask) != home) continue;
            if (defragfns) dictDefragSlot(d, &se->slot, defragfns);
            fn(privdata, se->slot, &se->slot);
        }
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

    if (!dictIsRehashing(d)) {
        htidx0 = 0;
        m0 = expToMask(d->ht_size_exp[htidx0]);
        dictScanHome(d, htidx0, v & m0, fn, defragfns, privdata);

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
        if (numGroups(d->ht_size_exp[htidx0]) > numGroups(d->ht_size_exp[htidx1])) {
            htidx0 = 1;
            htidx1 = 0;
        }

        m0 = expToMask(d->ht_size_exp[htidx0]);
        m1 = expToMask(d->ht_size_exp[htidx1]);

        dictScanHome(d, htidx0, v & m0, fn, defragfns, privdata);

        /* Iterate over indices in larger table that are the expansion
         * of the index pointed to by the cursor in the smaller table */
        do {
            dictScanHome(d, htidx1, v & m1, fn, defragfns, privdata);

            /* Increment the reverse cursor not covered by the smaller mask.*/
            v |= ~m1;
            v = rev(v);
            v++;
            v = rev(v);

            /* Continue while bits covered by mask difference is non-zero */
        } while (v & (m0 ^ m1));
    }

    dictResumeRehashing(d);

    return v;
}

/* ------------------------- private functions ------------------------------ */

/* Because we may need to allocate huge memory chunk at once when dict
 * resizes, we will check this allocation is allowed or not if the dict
 * type has resizeAllowed member function. */
static int dictTypeResizeAllowed(dict *d, size_t size) {
    if (d->type->resizeAllowed == NULL) return 1;
    return d->type->resizeAllowed(
                    tableBytes(nextGroupExp(size)),
                    (double)d->ht_used[0] / tableCapacity(d->ht_size_exp[0]));
}

/* Returning DICT_OK indicates a successful expand or the dictionary is undergoing rehashing,
 * and there is nothing else we need to do about this dictionary currently. While DICT_ERR indicates
 * that expand has not been triggered (may be try shrinking?)*/
int dictExpandIfNeeded(dict *d) {
    /* Incremental rehashing already in progress. Return. */
    if (dictIsRehashing(d)) return DICT_OK;

    /* If the hash table is empty expand it to the initial size. */
    if (numGroups(d->ht_size_exp[0]) == 0) {
        dictExpand(d, 1);
        return DICT_OK;
    }

    /* If we reached 7/8 of the slots and we are allowed to resize the hash
     * table (global setting), we resize making room for one more element. At
     * the hard fill limit the table grows regardless of the policy and of
     * resizeAllowed(). */
    size_t used = d->ht_used[0];
    int hard = used >= hardFill(d->ht_size_exp[0]);
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if (hard || (can_resize == DICT_RESIZE_ENABLE && used >= maxFill(d->ht_size_exp[0]))) {
        if (hard || dictTypeResizeAllowed(d, used + 1))
            dictExpand(d, used + 1);
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

    /* If the table is a single group, don't shrink it. */
    if (d->ht_size_exp[0] <= DICT_HT_INITIAL_EXP) return DICT_OK;

    /* If we reached below 1:8 elements/slots ratio, and we are allowed to resize
     * the hash table (global setting) or we should avoid it but the ratio is below 1:32,
     * we'll trigger a resize of the hash table. */
    size_t capacity = tableCapacity(d->ht_size_exp[0]);
    dictResizeEnable can_resize;
    atomicGet(dict_can_resize, can_resize);
    if ((can_resize == DICT_RESIZE_ENABLE &&
         d->ht_used[0] * HASHTABLE_MIN_FILL <= capacity) ||
        (can_resize != DICT_RESIZE_FORBID &&
         d->ht_used[0] * HASHTABLE_MIN_FILL * dict_force_resize_ratio <= capacity))
    {
        /* The new table is sized for twice the entries, so inserts during
         * and right after the shrink don't fill it up again. */
        size_t target = d->ht_used[0] * 2;
        if (dictTypeResizeAllowed(d, target))
            dictShrink(d, target);
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
    if (d->pauserehash > 0) {
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

    /* Subtract the size of all tables. */
    tableMemChanged(d, -(long long)dictTableMemUsage(d));
    stashMemChanged(d, -(long long)dictStashMemUsage(d));

    stashClear(d);
    _dictClear(d,0,callback);
    _dictClear(d,1,callback);
    d->rehashidx = -1;
    d->pauserehash = 0;
    d->pauseAutoResize = 0;
    d->pending_shrink = 0;
}

void dictSetResizeEnabled(dictResizeEnable enable) {
    atomicSet(dict_can_resize, enable);
}

/* Provides the old and new ht size, in groups, for a given dictionary
 * during rehashing. This method should only be invoked during
 * initialization/rehashing. */
void dictRehashingInfo(dict *d, unsigned long long *from_size, unsigned long long *to_size) {
    /* Invalid method usage if rehashing isn't ongoing. */
    assert(dictIsRehashing(d));
    *from_size = numGroups(d->ht_size_exp[0]);
    *to_size = numGroups(d->ht_size_exp[1]);
}

/* ------------------------------- Debugging ---------------------------------*/
#define DICT_STATS_VECTLEN 50
void dictFreeStats(dictStats *stats) {
    zfree(stats->clvector);
    zfree(stats);
}

void dictCombineStats(dictStats *from, dictStats *into) {
    into->groups += from->groups;
    into->overflowGroups += from->overflowGroups;
    into->displaced += from->displaced;
    into->maxProbeLen = (from->maxProbeLen > into->maxProbeLen) ? from->maxProbeLen : into->maxProbeLen;
    into->totalProbeLen += from->totalProbeLen;
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
    stats->groups = numGroups(d->ht_size_exp[htidx]);
    stats->htSize = tableCapacity(d->ht_size_exp[htidx]);
    stats->htUsed = d->ht_used[htidx];
    if (!full) return stats;
    /* Compute stats about probe lengths: the number of groups a lookup of each
     * entry visits. */
    size_t mask = expToMask(d->ht_size_exp[htidx]);
    for (size_t g = 0; g < stats->groups; g++) {
        const uint8_t *ctrl = groupCtrl(d, htidx, g);
        if (ctrl[OVF_POS]) stats->overflowGroups++;
        groupMask full_slots = groupMatchFull(ctrl);
        while (full_slots) {
            int pos = __builtin_ctz(full_slots);
            full_slots &= full_slots - 1;
            unsigned long len = 1;
            if (ctrl[pos] & CTRL_DISPLACED) {
                stats->displaced++;
                size_t q = entryMixedHash(d, groupSlots(d, htidx, g)[pos]) & mask, step = 0;
                while (q != g && step < mask) {
                    q = probeNext(q, &step, mask);
                    len++;
                }
            }
            if (len > stats->maxProbeLen) stats->maxProbeLen = len;
            stats->totalProbeLen += len;
            clvector[(len < DICT_STATS_VECTLEN) ? len : (DICT_STATS_VECTLEN-1)]++;
        }
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
                      " groups: %lu\n"
                      " groups with overflow: %lu\n"
                      " displaced entries: %lu\n"
                      " max probe length: %lu\n"
                      " avg probe length: %.02f\n"
                      " Probe length distribution (groups probed per entry):\n",
                      stats->groups, stats->overflowGroups, stats->displaced,
                      stats->maxProbeLen, (double) stats->totalProbeLen / stats->htUsed);

        for (unsigned long i = 1; i < DICT_STATS_VECTLEN; i++) {
            if (stats->clvector[i] == 0) continue;
            if (l >= bufsize) break;
            l += snprintf(buf + l, bufsize - l,
                          "   %ld: %ld (%.02f%%)\n",
                          i, stats->clvector[i], ((float) stats->clvector[i] / stats->htUsed) * 100);
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
    return dictBuckets(d) * GROUP_SLOTS;
}

/* Defrag the dict struct and its hash tables with `defragfn`, which receives an
 * allocation and returns a new one, or NULL if it was not moved.
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

/* Bytes of the table being rehashed away from. Only meaningful while rehashing
 * (and valid inside the rehashingStarted/Completed callbacks). */
size_t dictRehashingMemUsage(const dict *d) {
    return tableBytes(d->ht_size_exp[0]);
}

/* Call fn(ptr, size) for every hash table allocation of the dict. */
void dictDismissTables(dict *d, void (*fn)(void *ptr, size_t size)) {
    if (!d) return;
    fn(d->ht_table[0], tableBytes(d->ht_size_exp[0]));
    fn(d->ht_table[1], tableBytes(d->ht_size_exp[1]));
}

/* ------------------------- Prefetch state machine -------------------------- */

enum { PF_GROUP, PF_MATCH, PF_SLOT, PF_ENTRY, PF_ENTRY_KEY, PF_ENTRY_VALUE, PF_DONE };

void dictPrefetchInit(dictPrefetchState *st, dict *d, const void *key) {
    st->d = d;
    st->key = key;
    st->ht_idx = -1;
    st->slot = NULL;
    st->current_entry = NULL;
    st->group = 0;
    st->step = 0;
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
    st->hash = keyMixedHash(d, key);
    st->stage = PF_GROUP;
    st->done = 0;
}

/* Advance the lookup until a useful address to prefetch is produced.
 * Returns NULL, and sets st->done, when the lookup is complete. The stages
 * mirror a dictFind: control bytes of the group -> tag match (control bytes
 * are now in cache) -> slot of a candidate -> entry -> entry key payload ->
 * compare and entry value payload. */
void *dictPrefetchNext(dictPrefetchState *st) {
    dict *d = st->d;
    for (;;) {
        switch (st->stage) {
        case PF_GROUP: {
            /* Pick the next table that may hold the key; none left means done. */
            int t = st->ht_idx + 1;
            for (; t <= 1; t++) {
                if (t == 1 && !dictIsRehashing(d)) { t = 2; break; }
                if (d->ht_used[t] == 0) continue;
                if (t == 0 && d->rehashidx >= 0 &&
                    (long)(st->hash & expToMask(d->ht_size_exp[0])) < d->rehashidx)
                    continue;
                break;
            }
            if (t > 1) {
                st->stage = PF_DONE;
                st->done = 1;
                return NULL;
            }
            st->ht_idx = t;
            st->group = st->hash & expToMask(d->ht_size_exp[t]);
            st->step = 0;
            st->stage = PF_MATCH;
            return groupCtrl(d, t, st->group);
        }
        case PF_MATCH: {
            const uint8_t *ctrl = groupCtrl(d, st->ht_idx, st->group);
            st->candidates = groupMatchTag(ctrl, hashTag(st->hash));
            st->lone = (st->candidates && (st->candidates & (st->candidates - 1)) == 0 &&
                        ctrl[OVF_POS] == 0 && !dictIsRehashing(d));
            st->stage = PF_SLOT;
            break;
        }
        case PF_SLOT: {
            if (st->candidates) {
                int pos = __builtin_ctz(st->candidates);
                st->candidates &= st->candidates - 1;
                st->slot = &groupSlots(d, st->ht_idx, st->group)[pos];
                st->stage = PF_ENTRY;
                return st->slot;
            }
            /* No (more) candidates in this group: next group of the probe
             * sequence, else the next table. */
            const uint8_t *ctrl = groupCtrl(d, st->ht_idx, st->group);
            size_t mask = expToMask(d->ht_size_exp[st->ht_idx]);
            size_t step = st->step;
            if (ctrl[OVF_POS] == 0 || step == mask) {
                st->stage = PF_GROUP;
                break;
            }
            st->group = probeNext(st->group, &step, mask);
            st->step = step;
            st->stage = PF_MATCH;
            return groupCtrl(d, st->ht_idx, st->group);
        }
        case PF_ENTRY:
            st->current_entry = *st->slot;
            st->stage = PF_ENTRY_KEY;
            return st->current_entry;
        case PF_ENTRY_KEY:
            st->stage = PF_ENTRY_VALUE;
            if (d->type->prefetchEntryKey) {
                void *addr = d->type->prefetchEntryKey(st->current_entry);
                if (addr) return addr;
            }
            break;
        case PF_ENTRY_VALUE: {
            const void *cmp_key = slotLookupKey(d, st->current_entry);
            /* A single tag match in a group nothing probed past, while not
             * rehashing, is assumed to be a hit without comparing the keys.
             * Otherwise compare. */
            if (st->lone || dictCompareKeys(d, st->key, cmp_key)) {
                st->stage = PF_DONE;
                st->done = 1;
                if (d->type->prefetchEntryValue)
                    return d->type->prefetchEntryValue(st->current_entry);
                return NULL;
            }
            st->stage = PF_SLOT;
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
static long long test_stash_bytes = 0;

static void testTableMemChanged(dict *d, long long delta_bytes) {
    UNUSED(d);
    test_table_bytes += delta_bytes;
}

static void testStashMemChanged(dict *d, long long delta_bytes) {
    UNUSED(d);
    test_stash_bytes += delta_bytes;
}

/* Returns a raw hash whose mixed form (see mixHash()) is `mixed`, so tests can
 * choose home groups and tags. */
static uint64_t unmixHash(uint64_t mixed) {
    const uint64_t c = 0x9e3779b97f4a7c15ULL;
    uint64_t inv = c; /* Newton's iteration for the inverse modulo 2^64 */
    for (int i = 0; i < 5; i++) inv *= 2 - c * inv;
    return (mixed ^ (mixed >> 32)) * inv;
}

/* Every key has the same home group and tag, to build long probe sequences. */
static uint64_t constHashCallback(const void *key) {
    UNUSED(key);
    return 0x5a5a5a5a5a5a5a5aULL;
}

/* Every key has one of four home groups (in tables of up to 2^20 groups) and a
 * random tag: lots of displaced entries sharing groups with other homes. */
static uint64_t fewHomesHashCallback(const void *key) {
    uint64_t h = hashCallback(key);
    return unmixHash((h & ~(uint64_t)0xfffff) | (h & 0x3));
}

/* k0 lives in group 1, and the other keys wrap from the last group into
 * group 0 in four- and eight-group tables. */
static uint64_t wrappedHomeHashCallback(const void *key) {
    return unmixHash(strcmp(key, "k0") == 0 ? 1 : 7);
}

static dictType verifyDictType = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
    .keys_are_odd = 0,
    .tableMemChanged = testTableMemChanged,
    .stashMemChanged = testStashMemChanged,
};

static dictType verifyDictTypeVal = {
    .hashFunction = hashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .tableMemChanged = testTableMemChanged,
    .stashMemChanged = testStashMemChanged,
};

static dictType oneHomeDictType = {
    .hashFunction = constHashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .no_value = 1,
    .tableMemChanged = testTableMemChanged,
    .stashMemChanged = testStashMemChanged,
};

static dictType fewHomesDictType = {
    .hashFunction = fewHomesHashCallback,
    .keyCompare = compareCallback,
    .keyDestructor = freeCallback,
    .tableMemChanged = testTableMemChanged,
    .stashMemChanged = testStashMemChanged,
};

/* Checks the structural invariants of the dict: control bytes (tags and the
 * displaced flag), overflow counters, entry counts, the rehashing invariant
 * and memory accounting. */
static void dictVerify(dict *d) {
    for (int t = 0; t <= 1; t++) {
        if (!d->ht_table[t]) {
            assert(d->ht_size_exp[t] == -1);
            assert(d->ht_used[t] == 0);
            continue;
        }
        size_t ng = numGroups(d->ht_size_exp[t]), mask = ng - 1, counted = 0;
        unsigned long *passed = zcalloc(ng * sizeof(*passed));
        for (size_t g = 0; g < ng; g++) {
            const uint8_t *ctrl = groupCtrl(d, t, g);
            for (int pos = 0; pos < GROUP_SLOTS; pos++) {
                uint8_t c = ctrl[pos];
                if (c == CTRL_EMPTY) continue;
                dictEntry *e = groupSlots(d, t, g)[pos];
                if (d->type->no_value) assert(entryIsKey(e));
                else assert(entryIsNormal(e));
                uint64_t mixed = entryMixedHash(d, e);
                assert((c & CTRL_TAG_MASK) == hashTag(mixed));
                size_t home = mixed & mask;
                assert(((c & CTRL_DISPLACED) != 0) == (home != g));
                /* Entries of homes before rehashidx moved to the new table. */
                if (t == 0 && dictIsRehashing(d)) assert((long)home >= d->rehashidx);
                size_t q = home, step = 0;
                while (q != g) {
                    passed[q]++;
                    assert(step < mask);
                    q = probeNext(q, &step, mask);
                }
                counted++;
            }
        }
        for (size_t g = 0; g < ng; g++) {
            unsigned stored = groupCtrl(d, t, g)[OVF_POS];
            /* Exact, unless the counter saturated and stuck. */
            assert(stored == passed[g] || stored == OVF_MAX);
        }
        zfree(passed);
        /* Stashed entries are counted in ht_used[1]. */
        assert(counted + (t == 1 ? stashLive(d) : 0) == d->ht_used[t]);
    }
    unsigned long stash_count = 0;
    for (dictStashEntry *se = d->stash; se; se = se->next) stash_count++;
    assert(stash_count == d->stash_count);
    if (d->stash) assert(dictIsRehashing(d) && d->pauserehash > 0);
    if (dictIsRehashing(d)) {
        assert(d->ht_table[0] && d->ht_table[1]);
        assert(d->rehashidx >= 0 && (size_t)d->rehashidx <= numGroups(d->ht_size_exp[0]));
    } else {
        assert(d->ht_table[1] == NULL && d->ht_used[1] == 0);
    }
    if (d->type->tableMemChanged == testTableMemChanged)
        assert(test_table_bytes == (long long)dictTableMemUsage(d));
    if (d->type->stashMemChanged == testStashMemChanged)
        assert(test_stash_bytes == (long long)dictStashMemUsage(d));
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

/* Keys of the scan mutation test: k0..k<SCAN_KEYS-1> share one home group and
 * spill into the next groups of its probe sequence; more are added by the
 * callback. */
#define SCAN_KEYS (2 * DICT_GROUP_SLOTS + 1)

typedef struct scanMutationCtx {
    dict *d;
    int seen[SCAN_KEYS + 2];
    int deletion; /* dictDelete, dictUnlink, or two-phase unlink */
    int additions;
    int nested;
} scanMutationCtx;

static void scanMutationCb(void *privdata, const dictEntry *de, dictEntryLink plink);

/* A full scan cycle. */
static void scanAll(dict *d, dictScanFunction *fn, void *privdata) {
    unsigned long cursor = 0;
    do {
        cursor = dictScan(d, cursor, fn, privdata);
    } while (cursor);
}

static void scanMutationCb(void *privdata, const dictEntry *de, dictEntryLink plink) {
    scanMutationCtx *ctx = privdata;
    int id = keyId(dictGetKey(de));
    UNUSED(plink);
    assert(id >= 0 && id < SCAN_KEYS + 2);
    ctx->seen[id]++;
    if (ctx->deletion == 3) {
        /* Deleting and adding back the same key reuses a slot that the scan
         * already passed, so the key is not returned again. */
        assert(ctx->seen[id] == 1);
        assert(dictDelete(ctx->d, dictGetKey(de)) == DICT_OK);
        assert(dictAdd(ctx->d, keyForId(id), NULL) == DICT_OK);
        return;
    }
    if (id != 0) return;

    if (ctx->nested == 1) {
        ctx->nested = 2;
        scanAll(ctx->d, scanMutationCb, ctx);
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
        assert(dictAdd(ctx->d, keyForId(SCAN_KEYS + i), NULL) == DICT_OK);
}

static void countScanCb(void *privdata, const dictEntry *de, dictEntryLink plink) {
    modelCtx *m = privdata;
    UNUSED(plink);
    m->seen[keyId(dictGetKey(de))]++;
}

/* Counts scanned entries whose keys are numbers, into a uint32_t array. */
static void countNumericScanCb(void *privdata, const dictEntry *de, dictEntryLink plink) {
    uint32_t *seen = privdata;
    UNUSED(plink);
    seen[atoi(dictGetKey(de))]++;
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
                int p = rand() % 3;
                dictSetResizeEnabled(p == 0 ? DICT_RESIZE_ENABLE :
                                     p == 1 ? DICT_RESIZE_AVOID : DICT_RESIZE_FORBID);
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
    unsigned long current_dict_used, remain_keys, capacity, limit, new_groups;
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

    TEST("Group matching finds tags, empty and displaced slots") {
        uint8_t ctrl[GROUP_WIDTH] = {0};
        for (int i = 0; i < GROUP_SLOTS; i++)
            ctrl[i] = (i % 3 == 0) ? CTRL_EMPTY : (uint8_t)((i % 2 ? CTRL_DISPLACED : 0) | (i % 5 + 1));
        ctrl[OVF_POS] = 0xff; /* the counter must never match */
        for (uint8_t tag = 1; tag <= CTRL_TAG_MASK; tag++) {
            groupMask want = 0;
            for (int i = 0; i < GROUP_SLOTS; i++)
                if (ctrl[i] != CTRL_EMPTY && (ctrl[i] & CTRL_TAG_MASK) == tag) want |= 1u << i;
            assert(groupMatchTag(ctrl, tag) == want);
        }
        groupMask empty = 0, displaced = 0;
        for (int i = 0; i < GROUP_SLOTS; i++) {
            if (ctrl[i] == CTRL_EMPTY) empty |= 1u << i;
            if (ctrl[i] & CTRL_DISPLACED) displaced |= 1u << i;
        }
        assert(groupMatchEmpty(ctrl) == empty);
        assert(groupMatchDisplaced(ctrl) == displaced);
        assert(groupMatchNative(ctrl) == (groupMask)(~empty & ~displaced & GROUP_SLOT_BITS));
        ctrl[OVF_POS] = 0; /* a zero counter is not an empty slot either */
        assert(groupMatchEmpty(ctrl) == empty);
        assert(groupMatchFull(ctrl) == (groupMask)(~empty & GROUP_SLOT_BITS));
        assert(mixHash(unmixHash(0x0123456789abcdefULL)) == 0x0123456789abcdefULL);
    }

    TEST("Initial table is a single group") {
        assert(dictBuckets(d) == 0 && dictSlots(d) == 0);
        retval = dictAdd(d, stringFromLongLong(1000000), (void*)1);
        assert(retval == DICT_OK);
        assert(dictBuckets(d) == 1);
        assert(dictSlots(d) == DICT_GROUP_SLOTS);
        assert(dictTableMemUsage(d) == DICT_GROUP_BYTES);
        assert(dictDelete(d, "1000000") == DICT_OK);
        assert(dictBuckets(d) == 1); /* never shrinks below one group */
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
        /* A power of two number of groups. */
        assert((dictBuckets(d) & (dictBuckets(d) - 1)) == 0);
        dictVerify(d);
    }

    TEST("Use DICT_RESIZE_AVOID to disable the dict resize and fill up to the hard fill limit") {
        dictSetResizeEnabled(DICT_RESIZE_AVOID);
        capacity = dictSlots(d);
        limit = hardFill(d->ht_size_exp[0]);
        assert(limit < capacity);
        for (j = 16; j < (long)limit; j++) {
            retval = dictAdd(d,stringFromLongLong(j),(void*)j);
            assert(retval == DICT_OK);
        }
        current_dict_used = limit;
        assert(dictSize(d) == current_dict_used);
        assert(dictSlots(d) == capacity);
        assert(!dictIsRehashing(d));
        dictVerify(d);
    }

    TEST("Add one more key, the dict grows despite DICT_RESIZE_AVOID") {
        retval = dictAdd(d,stringFromLongLong(current_dict_used),(void*)(current_dict_used));
        assert(retval == DICT_OK);
        current_dict_used++;
        new_groups = numGroups(nextGroupExp(current_dict_used));
        assert(dictSize(d) == current_dict_used);
        assert(dictIsRehashing(d));
        assert(tableCapacity(d->ht_size_exp[0]) == capacity);
        assert(numGroups(d->ht_size_exp[1]) == new_groups);
        dictVerify(d);

        /* Wait for rehashing. */
        dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        drainRehash(d);
        assert(dictSize(d) == current_dict_used);
        assert(numGroups(d->ht_size_exp[0]) == new_groups);
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
        /* The shrunk table leaves room for twice the entries. */
        new_groups = numGroups(nextGroupExp(current_dict_used * 2));
        assert(retval == DICT_OK);
        assert(dictSize(d) == current_dict_used);
        assert(dictIsRehashing(d));
        assert(tableCapacity(d->ht_size_exp[0]) == capacity);
        assert(numGroups(d->ht_size_exp[1]) == new_groups);

        /* Wait for rehashing. */
        drainRehash(d);
        assert(dictSize(d) == current_dict_used);
        assert(numGroups(d->ht_size_exp[0]) == new_groups);
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
        assert(tableCapacity(d->ht_size_exp[0]) == capacity);
        new_groups = numGroups(d->ht_size_exp[1]);

        /* Wait for rehashing. */
        for (int steps = 0; dictIsRehashing(d) && steps < 1000; steps++)
            dictRehash(d, 100);
        assert(!dictIsRehashing(d));
        assert(dictSize(d) == current_dict_used);
        assert(numGroups(d->ht_size_exp[0]) == new_groups);
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
        assert(numGroups(ds->ht_size_exp[1]) == numGroups(nextGroupExp(threshold * 2)));
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

    TEST("DICT_RESIZE_FORBID, paused auto-resize and refused resizes still grow at the hard fill limit") {
        for (int mode = 0; mode < 3; mode++) {
            test_table_bytes = 0;
            dict *df = dictCreate(&verifyDictType);
            assert(dictExpand(df, 100) == DICT_OK);
            size_t exp = df->ht_size_exp[0];
            if (mode == 0) dictSetResizeEnabled(DICT_RESIZE_FORBID);
            if (mode == 1) dictPauseAutoResize(df);
            if (mode == 2) dictSetResizeEnabled(DICT_RESIZE_AVOID);
            size_t n = hardFill(exp);
            for (size_t i = 0; i < n; i++)
                assert(dictAdd(df, stringFromLongLong(i), NULL) == DICT_OK);
            assert(!dictIsRehashing(df) && (size_t)df->ht_size_exp[0] == exp);
            assert(dictAdd(df, stringFromLongLong(n), NULL) == DICT_OK);
            assert(dictIsRehashing(df) && df->ht_size_exp[1] > df->ht_size_exp[0]);
            dictVerify(df);
            /* Inserts keep going to the new table; once it fills up the old
             * one is drained even though the policy says not to rehash. */
            for (size_t i = n + 1; i < 3 * n; i++)
                assert(dictAdd(df, stringFromLongLong(i), NULL) == DICT_OK);
            assert(dictSize(df) == 3 * n);
            dictVerify(df);
            if (mode == 1) dictResumeAutoResize(df);
            dictSetResizeEnabled(DICT_RESIZE_ENABLE);
            drainRehash(df);
            dictVerify(df);
            dictRelease(df);
            assert(test_table_bytes == 0);
        }
    }

    TEST("Forced rehash completes under AVOID and FORBID, including after a pause") {
        for (int policy = DICT_RESIZE_AVOID; policy <= DICT_RESIZE_FORBID; policy++) {
            for (int paused = 0; paused <= 1; paused++) {
                dictSetResizeEnabled(DICT_RESIZE_ENABLE);
                test_table_bytes = 0;
                dictType type = verifyDictType;
                type.force_full_rehash = 1;
                dict *df = dictCreate(&type);
                assert(dictExpand(df, 100) == DICT_OK);
                size_t capacity = dictSlots(df);
                size_t limit = hardFill(df->ht_size_exp[0]);
                size_t n = paused ? 3 * capacity : limit + 1;
                if (paused) dictPauseRehashing(df);
                dictSetResizeEnabled(policy);
                for (size_t i = 0; i < n; i++) {
                    assert(dictAdd(df, stringFromLongLong(i), NULL) == DICT_OK);
                    if (!paused) assert(!dictIsRehashing(df));
                }
                if (paused) {
                    assert(dictIsRehashing(df) && df->ht_used[0] == limit);
                    assert(df->stash != NULL);
                    dictResumeRehashing(df);
                }
                assert(!dictIsRehashing(df) && df->stash == NULL);
                assert(dictSize(df) == n && dictSlots(df) > capacity);
                dictVerify(df);
                dictRelease(df);
                assert(test_table_bytes == 0 && test_stash_bytes == 0);
            }
        }
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

        /* Identical keys and resize policy => identical table geometry, so the
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

    TEST("One home group: long probe sequences, saturated overflow counters, deletes") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&oneHomeDictType);
        const int n = 300; /* all keys share one home group and tag */
        for (int i = 0; i < n; i++) {
            assert(dictAdd(dc, stringFromLongLong(i), NULL) == DICT_OK);
            if (i % 50 == 0) dictVerify(dc);
        }
        drainRehash(dc);
        dictVerify(dc);
        size_t home = mixHash(constHashCallback(NULL)) & expToMask(dc->ht_size_exp[0]);
        /* More than 255 entries probed past the home group. */
        assert(groupCtrl(dc, 0, home)[OVF_POS] == OVF_MAX);
        /* Delete in a scattered order, every deletion must keep the
         * counters right. */
        for (int i = 0; i < n; i += 2) {
            char *k = stringFromLongLong(i);
            assert(dictDelete(dc, k) == DICT_OK);
            zfree(k);
            if (i % 20 == 0) dictVerify(dc);
        }
        drainRehash(dc);
        dictVerify(dc);
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
        dictRelease(dc);
        assert(test_table_bytes == 0);
    }

    TEST("dictGetFairRandomKey is fair when entries share one home group") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&oneHomeDictType);
        /* 30 entries: a full home group and one full group after it, in a
         * table of 16 groups that are otherwise empty. */
        assert(dictExpand(dc, 200) == DICT_OK);
        for (int i = 0; i < 30; i++) assert(dictAdd(dc, stringFromLongLong(i), NULL) == DICT_OK);
        unsigned long hits[30] = {0};
        const int picks = 300000;
        for (int i = 0; i < picks; i++) hits[atoi(dictGetKey(dictGetFairRandomKey(dc)))]++;
        double chi2 = 0, expected = picks / 30.0;
        for (int i = 0; i < 30; i++) chi2 += (hits[i] - expected) * (hits[i] - expected) / expected;
        assert(chi2 < 73); /* df = 29, p = 0.00001 */
        dictRelease(dc);
        assert(test_table_bytes == 0);
    }

    TEST("Sampling finds entries displaced before rehashidx during growth and shrink") {
        for (int shrinking = 0; shrinking <= 1; shrinking++) {
            test_table_bytes = 0;
            dictType type = verifyDictType;
            type.hashFunction = wrappedHomeHashCallback;
            dict *ds = dictCreate(&type);
            assert(dictExpand(ds, maxFill(shrinking ? 3 : 2)) == DICT_OK);
            dictPauseAutoResize(ds);
            char *wrapped = NULL;
            for (int i = 0; i <= GROUP_SLOTS + 1; i++) {
                char *key = keyForId(i);
                assert(dictAdd(ds, key, NULL) == DICT_OK);
                wrapped = key;
            }
            if (shrinking) assert(dictShrink(ds, maxFill(2)) == DICT_OK);
            else assert(dictExpand(ds, maxFill(3)) == DICT_OK);
            assert(dictRehash(ds, 1));
            assert(ds->rehashidx == 2);
            assert(groupMatchDisplaced(groupCtrl(ds, 0, 0)) != 0);
            dictSetResizeEnabled(DICT_RESIZE_FORBID);
            dictEntry *samples[GROUP_SLOTS + 2];
            for (int i = 0; i < 100; i++) {
                unsigned int n = dictGetSomeKeys(ds, samples, GROUP_SLOTS + 2);
                int found = 0;
                for (unsigned int j = 0; j < n; j++)
                    found |= dictGetKey(samples[j]) == wrapped;
                assert(found);
            }
            dictVerify(ds);
            dictResumeAutoResize(ds);
            dictRelease(ds);
            assert(test_table_bytes == 0);
            dictSetResizeEnabled(DICT_RESIZE_ENABLE);
        }
    }

    TEST("Safe iterator deleting the returned entry in a long probe sequence returns each entry once") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&oneHomeDictType);
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
        dict *dc = dictCreate(&oneHomeDictType);
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

    TEST("Scan returns every entry exactly once while the table doesn't change") {
        dictType *types[] = {&verifyDictType, &fewHomesDictType, &oneHomeDictType};
        for (int ti = 0; ti < 3; ti++) {
            test_table_bytes = 0;
            dict *dc = dictCreate(types[ti]);
            modelCtx *m = zcalloc(sizeof(*m));
            m->d = dc;
            int n = types[ti] == &oneHomeDictType ? 300 : MODEL_KEYS;
            for (int i = 0; i < n; i++) modelAdd(m, dc, i);
            drainRehash(dc);
            unsigned long cursor = 0;
            do {
                cursor = dictScan(dc, cursor, countScanCb, m);
            } while (cursor);
            for (int i = 0; i < n; i++) assert(m->seen[i] == 1);
            dictRelease(dc);
            assert(test_table_bytes == 0);
            zfree(m);
        }
    }

    TEST("Scan: callback deletes the entry it is given, displaced entries stay found") {
        test_table_bytes = 0;
        dict *dc = dictCreate(&fewHomesDictType);
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
            for (int deletion = 0; deletion < 4; deletion++) {
                for (int additions = 1; additions <= 2; additions++) {
                    for (int nested = 0; nested <= 1; nested++) {
                        if (deletion == 3 && (additions > 1 || nested)) continue;
                        dictType type = oneHomeDictType;
                        type.no_value = no_value;
                        test_table_bytes = 0;
                        dict *dc = dictCreate(&type);
                        /* Room for everything, so nothing resizes. */
                        assert(dictExpand(dc, 4 * SCAN_KEYS) == DICT_OK);
                        for (int i = 0; i < SCAN_KEYS; i++)
                            assert(dictAdd(dc, keyForId(i), NULL) == DICT_OK);
                        dictVerify(dc);
                        scanMutationCtx ctx = {.d = dc, .deletion = deletion,
                                               .additions = additions, .nested = nested};
                        scanAll(dc, scanMutationCb, &ctx);
                        if (deletion == 3) {
                            for (int i = 0; i < SCAN_KEYS; i++) assert(ctx.seen[i] == 1);
                            assert(dictSize(dc) == SCAN_KEYS);
                        } else {
                            for (int i = 1; i < SCAN_KEYS; i++) assert(ctx.seen[i] >= 1);
                            assert(dictFind(dc, "k0") == NULL);
                            assert(dictSize(dc) == SCAN_KEYS - 1 + (unsigned long)additions);
                        }
                        dictVerify(dc);
                        dictRelease(dc);
                        assert(test_table_bytes == 0);
                    }
                }
            }
        }
    }

    TEST("Randomized operations against a reference model (no_value)") {
        dictRandomizedTest(&verifyDictType, accurate ? 600 : 60, 1500);
    }

    TEST("Randomized operations against a reference model (with values)") {
        dictRandomizedTest(&verifyDictTypeVal, accurate ? 300 : 30, 1500);
    }

    TEST("Randomized operations with heavy displacement") {
        dictRandomizedTest(&fewHomesDictType, accurate ? 200 : 20, 1000);
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

    TEST("Inserts while rehashing is paused fill the new table, then go to the stash") {
        test_table_bytes = 0;
        dict *dp = dictCreate(&verifyDictType);
        assert(dictExpand(dp, 100) == DICT_OK);
        size_t exp0 = dp->ht_size_exp[0];
        dictPauseRehashing(dp);
        /* Growing starts at 7/8 fill, but no entry may move while paused. */
        long i = 0;
        while (!dictIsRehashing(dp)) assert(dictAdd(dp, stringFromLongLong(i++), NULL) == DICT_OK);
        assert((size_t)dp->ht_size_exp[0] == exp0 && dp->ht_used[0] == maxFill(exp0));
        size_t cap1 = tableCapacity(dp->ht_size_exp[1]);
        /* The new table must keep room for the old one's entries. */
        while (dictSize(dp) + 1 < cap1) assert(dictAdd(dp, stringFromLongLong(i++), NULL) == DICT_OK);
        assert(dp->stash == NULL);
        long first_stashed = i;
        while (i < (long)(3 * cap1)) assert(dictAdd(dp, stringFromLongLong(i++), NULL) == DICT_OK);
        assert(dp->stash != NULL && stashLive(dp) == (size_t)(i - first_stashed));
        assert(dictSize(dp) == (unsigned long)i);
        assert(dictRehash(dp, 100) == 0 && dp->ht_used[0] == maxFill(exp0));
        dictVerify(dp);
        /* Lookups find stashed entries; delete some of them, and some table
         * entries, with both delete paths. */
        uint8_t *gone = zcalloc(i);
        for (long k = 0; k < i; k += 7) {
            char *key = stringFromLongLong(k);
            if (k % 2) {
                assert(dictDelete(dp, key) == DICT_OK);
            } else {
                dictPosition pos;
                dictEntryLink link = dictTwoPhaseUnlinkFind(dp, key, &pos);
                assert(link != NULL && dictCompareKeys(dp, dictGetKey(*link), key));
                dictTwoPhaseUnlinkFree(dp, link, &pos); /* the stash stays: still paused */
            }
            gone[k] = 1;
            zfree(key);
        }
        assert(dp->stash != NULL);
        dictVerify(dp);
        for (long k = 0; k < i; k++) {
            char *key = stringFromLongLong(k);
            assert((dictFind(dp, key) == NULL) == gone[k]);
            zfree(key);
        }
        /* A safe iterator and a full scan return every entry, stashed or not. */
        uint32_t *seen = zcalloc(i * sizeof(*seen));
        dictIterator it;
        dictEntry *e;
        dictInitSafeIterator(&it, dp);
        while ((e = dictNext(&it))) seen[atoi(dictGetKey(e))]++;
        dictResetIterator(&it);
        for (long k = 0; k < i; k++) assert(seen[k] == !gone[k]);
        memset(seen, 0, i * sizeof(*seen));
        scanAll(dp, countNumericScanCb, seen);
        for (long k = 0; k < i; k++) assert((seen[k] >= 1) == !gone[k]);
        unsigned long size = dictSize(dp);
        /* Resuming moves the stash into the table, growing it. */
        dictResumeRehashing(dp);
        assert(dp->stash == NULL && dictSize(dp) == size);
        dictVerify(dp);
        drainRehash(dp);
        assert(tableCapacity(dp->ht_size_exp[0]) > 3 * cap1);
        for (long k = 0; k < i; k++) {
            char *key = stringFromLongLong(k);
            assert((dictFind(dp, key) == NULL) == gone[k]);
            zfree(key);
        }
        dictVerify(dp);
        zfree(seen);
        zfree(gone);
        dictRelease(dp);
        assert(test_table_bytes == 0);
    }

    TEST("Sampling includes live stash entries and accounts for retained deleted nodes") {
        for (int no_value = 0; no_value <= 1; no_value++) {
            test_table_bytes = 0;
            dictType type = verifyDictType;
            type.no_value = no_value;
            dict *ds = dictCreate(&type);
            dictPauseRehashing(ds);
            const int n = 100;
            for (int i = 0; i < n; i++)
                assert(dictAdd(ds, keyForId(i), NULL) == DICT_OK);
            assert(ds->stash != NULL);
            size_t bytes = dictStashMemUsage(ds);
            assert(bytes != 0 && test_stash_bytes == (long long)bytes);
            assert(dictMemUsage(ds) == dictTableMemUsage(ds) + bytes +
                   dictSize(ds) * dictEntryMemUsage(no_value));

            /* With both storage kinds populated, each must remain eligible. */
            dictEntry *samples[100];
            int saw_table = 0, saw_stash = 0;
            for (int i = 0; i < 1000 && !(saw_table && saw_stash); i++) {
                unsigned int count = dictGetSomeKeys(ds, samples, 5);
                for (unsigned int j = 0; j < count; j++) {
                    int table;
                    void *key = dictGetKey(samples[j]);
                    assert(findKey(ds, keyMixedHash(ds, key), key, &table, NULL, NULL));
                    saw_stash |= table == 2;
                    saw_table |= table != 2;
                }
            }
            assert(saw_table && saw_stash);

            /* Remove all table entries and some stash entries while paused. */
            for (int t = 0; t < 2; t++) {
                for (size_t g = 0; g < numGroups(ds->ht_size_exp[t]); g++) {
                    for (int pos = 0; pos < GROUP_SLOTS; pos++) {
                        if (groupCtrl(ds, t, g)[pos] == CTRL_EMPTY) continue;
                        void *key = dictGetKey(groupSlots(ds, t, g)[pos]);
                        assert(dictDelete(ds, key) == DICT_OK);
                    }
                }
            }
            for (dictStashEntry *se = ds->stash; se; se = se->next) {
                if (keyId(dictGetKey(se->slot)) % 2 == 0)
                    assert(dictDelete(ds, dictGetKey(se->slot)) == DICT_OK);
            }
            assert(dictSize(ds) != 0 && dictSize(ds) == stashLive(ds));
            assert(dictStashMemUsage(ds) == bytes); /* deleted nodes still exist */
            dictVerify(ds);
            assert(dictGetSomeKeys(ds, NULL, 0) == 0);
            unsigned int count = dictGetSomeKeys(ds, samples, n);
            assert(count == dictSize(ds));
            uint8_t seen[100] = {0};
            for (unsigned int j = 0; j < count; j++) {
                int id = keyId(dictGetKey(samples[j]));
                assert(id >= 0 && id < n && id % 2 != 0 && !seen[id]);
                seen[id] = 1;
            }
            dictResumeRehashing(ds);
            assert(dictStashMemUsage(ds) == 0 && test_stash_bytes == 0);
            dictVerify(ds);
            dictRelease(ds);
            assert(test_table_bytes == 0 && test_stash_bytes == 0);
        }
    }

    TEST("A stashed dict can be emptied and released while paused") {
        test_table_bytes = 0;
        dict *dp = dictCreate(&verifyDictTypeVal);
        dictPauseRehashing(dp);
        for (long i = 0; i < 1000; i++) assert(dictAdd(dp, stringFromLongLong(i), NULL) == DICT_OK);
        assert(dp->stash != NULL);
        dictEmpty(dp, NULL);
        assert(dictSize(dp) == 0 && dp->stash == NULL && dp->pauserehash == 0);
        assert(dictStashMemUsage(dp) == 0 && test_stash_bytes == 0);
        dictPauseRehashing(dp);
        for (long i = 0; i < 1000; i++) assert(dictAdd(dp, stringFromLongLong(i), NULL) == DICT_OK);
        assert(dp->stash != NULL);
        dictRelease(dp);
        assert(test_table_bytes == 0 && test_stash_bytes == 0);
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

    TEST("A link stays valid across inserts and deletes of other keys") {
        test_table_bytes = 0;
        dict *dl = dictCreate(&oneHomeDictType);
        /* Big enough not to grow, small enough not to shrink after the deletes. */
        assert(dictExpand(dl, 300) == DICT_OK);
        for (int i = 0; i < 40; i++) assert(dictAdd(dl, stringFromLongLong(i), NULL) == DICT_OK);
        dictEntryLink link = dictFindLink(dl, "20", NULL);
        assert(link != NULL);
        void *key = dictGetKey(*link);
        for (int i = 40; i < 200; i++) assert(dictAdd(dl, stringFromLongLong(i), NULL) == DICT_OK);
        for (int i = 0; i < 20; i++) {
            char *k = stringFromLongLong(i);
            assert(dictDelete(dl, k) == DICT_OK);
            zfree(k);
        }
        assert(!dictIsRehashing(dl));
        assert(dictGetKey(*link) == key && dictFindLink(dl, "20", NULL) == link);
        dictVerify(dl);
        dictRelease(dl);
        assert(test_table_bytes == 0);
    }

    return 0;
}
#endif
