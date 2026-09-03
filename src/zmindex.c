/* zmindex.c -- compact member index used beside the sorted set B+ tree.
 *
 * A large sorted set has to answer two questions: where does a (score, member)
 * pair sit in the sorted order, and what is the score of a given member. The
 * B+ tree in zbtree.c answers the first. This file answers the second, and
 * replaces the general-purpose dict that used to do it.
 *
 * A dict pays for generality the sorted set does not need: a bucket array of
 * one pointer per slot, a chain pointer for every colliding key, and a
 * separate entry allocation once a bucket is shared. Here the index is open
 * addressed with linear probing, so there are no chains and no per-entry
 * allocations at all, and eight entries share one bucket:
 *
 *   +----------------+----------------+---------------------------+
 *   | tags (8 bytes) | home_tags (8B) | 8 element pointers (64B)  |
 *   +----------------+----------------+---------------------------+
 *
 * 'tags' packs one byte of member hash per slot, so a single 64-bit load and
 * a SWAR byte compare reject all eight candidates at once. Only a matching
 * tag causes an element to be dereferenced and its member compared, which is
 * the one expensive step. A tag byte of zero marks a slot that was never
 * used and is what terminates a probe.
 *
 * 'home_tags' is a small filter over the tags of the members whose probe
 * chain *starts* in this bucket. Three bits summarise each tag, so it answers
 * "perhaps present" or "definitely absent" and never the other way round.
 * A lookup for a missing member normally stops on this word instead of
 * walking a chain that may be long in a nearly full table.
 *
 * Deletion cannot leave a hole, or the chain that runs through it would break.
 * An emptied slot keeps its tag and stores a tombstone sentinel instead of an
 * element, which stops it matching while keeping later slots reachable. Enough
 * tombstones trigger the same copy that a resize uses.
 *
 * RESIZING
 * ========
 *
 * A resize allocates a second table and copies one bucket per operation, so no
 * single command pays for the whole table. Entries are *copied*, not moved:
 * the old table keeps them until the copy finishes. Both slots name the same
 * element, so a lookup that finds either is correct, and a scan sweeping the
 * old table cannot miss a member that has already been copied ahead of it.
 * Insertions during a resize only go to the new table.
 *
 * The old table is released in one step at the end. Because a scan cursor is
 * bound to the table's revision, and installing the new table changes it, any
 * cursor in flight restarts and therefore covers everything. Returning a
 * member twice is allowed by SCAN; missing one is not.
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "server.h"

#include <limits.h>

/* An emptied slot keeps its tag and points here. Element allocations are
 * aligned, so this can never be mistaken for one. */
#define ZMI_TOMBSTONE ((zbtElem *)(uintptr_t)1)

#define ZMI_INITIAL_BUCKETS 4

/* Live entries are kept under 31/32 of the slots so that probes stay short,
 * and total used slots under 63/64 so that a probe always meets an unused one
 * and an insertion always has somewhere to go. Dropping to an eighth starts a
 * copy into a smaller table. */
#define ZMI_MAX_LOAD_NUM 31
#define ZMI_MAX_LOAD_DEN 32
#define ZMI_MIN_LOAD_NUM 1
#define ZMI_MIN_LOAD_DEN 8
#define ZMI_MAX_FILLED_NUM 63
#define ZMI_MAX_FILLED_DEN 64

/* Buckets copied into the new table per indexed operation. One bucket holds
 * up to eight members, so a doubling finishes well before the destination,
 * sized for twice the live members, could run out of room. */
#define ZMI_REHASH_BUCKETS_PER_STEP 1

/* ------------------------------- Slot access ----------------------------- */

/* Fold tag zero into one: zero is reserved to mark an unused slot. */
static inline uint8_t zmiTag(uint64_t hash) {
    uint8_t tag = (uint8_t)(hash >> 56);
    return tag ? tag : 1;
}

static inline uint64_t zmiHash(sds member) {
    return dictGenHashFunction(member, sdslen(member));
}

static inline uint8_t zmiTagAt(const zmiBucket *b, unsigned int pos) {
    return (uint8_t)(b->tags >> (pos * 8));
}

/* Replace one tag byte without disturbing the other seven. */
static inline void zmiSetTag(zmiBucket *b, unsigned int pos, uint8_t tag) {
    uint64_t shift = pos * 8;
    b->tags = (b->tags & ~(UINT64_C(0xff) << shift)) | ((uint64_t)tag << shift);
}

/* Set the high bit of every byte of 'tags' that equals 'tag'. A byte above a
 * real match can also be set, so each candidate is still checked against the
 * stored byte before its element is touched. */
static inline uint64_t zmiTagMask(uint64_t tags, uint8_t tag) {
    uint64_t x = tags ^ (UINT64_C(0x0101010101010101) * tag);
    return (x - UINT64_C(0x0101010101010101)) & ~x &
           UINT64_C(0x8080808080808080);
}

static inline unsigned int zmiFirstTag(uint64_t mask) {
    return (unsigned int)(__builtin_ctzll(mask) >> 3);
}

/* Three bits stand for one tag in a bucket's filter word. Two tags sharing
 * all three only costs a normal probe. */
static inline uint64_t zmiTagBits(uint8_t tag) {
    uint64_t mixed = (uint64_t)tag * UINT64_C(0x9e3779b97f4a7c15);
    return (UINT64_C(1) << (tag & 63)) |
           (UINT64_C(1) << (mixed >> 58)) |
           (UINT64_C(1) << ((mixed >> 36) & 63));
}

/* Record a member in the filter of the bucket its chain starts in. */
static inline void zmiRecordHomeTag(zmiTable *t, uint64_t hash) {
    zmiBucket *home = &t->buckets[hash & (t->size - 1)];
    home->home_tags |= zmiTagBits(zmiTag(hash));
}

/* Zero only when the home bucket proves the member is absent. */
static inline int zmiHomeMayContain(const zmiTable *t, uint64_t hash) {
    const zmiBucket *home = &t->buckets[hash & (t->size - 1)];
    uint64_t bits = zmiTagBits(zmiTag(hash));
    return (home->home_tags & bits) == bits;
}

static inline unsigned long zmiTableSlots(const zmiTable *t) {
    return t->size * ZMI_BUCKET_ITEMS;
}

static inline int zmiOwnsBucket(const zmiTable *t, const zmiBucket *b) {
    return t->buckets != NULL && b >= t->buckets && b < t->buckets + t->size;
}

static inline int zmiIsRehashing(const zmindex *mi) {
    return mi->rehashidx >= 0;
}

/* Where new entries go. */
static inline zmiTable *zmiDestTable(zmindex *mi) {
    return zmiIsRehashing(mi) ? &mi->t[1] : &mi->t[0];
}

/* --------------------------- Table lifecycle ----------------------------- */

/* Scan cursors carry this, so a table that replaces another must not reuse
 * its value. Zero is never handed out: a cursor of zero means "start". */
static uint32_t zmiNextRevision(void) {
    static uint32_t counter = 0;
    if (++counter == 0) counter = 1;
    return counter;
}

static unsigned long zmiNextPower(unsigned long size) {
    unsigned long result = ZMI_INITIAL_BUCKETS;
    while (result < size) {
        if (result > ULONG_MAX / 2) return result;
        result <<= 1;
    }
    return result;
}

/* Buckets needed to hold 'elements' members without exceeding maximum load. */
static unsigned long zmiBucketsForElements(unsigned long elements) {
    if (elements == 0) return ZMI_INITIAL_BUCKETS;
    unsigned long slots =
        (elements * ZMI_MAX_LOAD_DEN + ZMI_MAX_LOAD_NUM - 1) / ZMI_MAX_LOAD_NUM;
    return zmiNextPower((slots + ZMI_BUCKET_ITEMS - 1) / ZMI_BUCKET_ITEMS);
}

/* Allocate a zeroed table. With 'try' set, returns 0 rather than panicking
 * when the allocation fails. */
static int zmiTableInit(zmiTable *t, unsigned long buckets, int try) {
    memset(t, 0, sizeof(*t));
    unsigned long size = zmiNextPower(buckets);
    size_t bytes = (size_t)size * sizeof(zmiBucket);
    zmiBucket *b = try ? ztrycalloc(bytes) : zcalloc(bytes);
    if (b == NULL) return 0;
    t->buckets = b;
    t->size = size;
    t->scan_revision = zmiNextRevision();
    return 1;
}

static void zmiTableRelease(zmiTable *t) {
    if (t->buckets) zfree(t->buckets);
    memset(t, 0, sizeof(*t));
}

zmindex *zmiCreate(void) {
    zmindex *mi = zmalloc(sizeof(*mi));
    memset(mi, 0, sizeof(*mi));
    mi->rehashidx = -1;
    mi->revision = zmiNextRevision();
    return mi;
}

void zmiRelease(zmindex *mi) {
    if (mi == NULL) return;
    zmiTableRelease(&mi->t[0]);
    zmiTableRelease(&mi->t[1]);
    zfree(mi);
}

size_t zmiMemUsage(const zmindex *mi) {
    size_t size = sizeof(*mi);
    for (int i = 0; i < 2; i++)
        size += (size_t)mi->t[i].size * sizeof(zmiBucket);
    return size;
}

void zmiDismissMemory(zmindex *mi) {
    for (int i = 0; i < 2; i++) {
        if (mi->t[i].buckets)
            dismissMemory(mi->t[i].buckets,
                          (size_t)mi->t[i].size * sizeof(zmiBucket));
    }
}

/* ------------------------------ Raw insertion ---------------------------- */

/* Take the first unused or emptied slot on the chain. Reusing a tombstone
 * leaves every existing chain intact and does not consume a further slot.
 * Duplicates are not checked for: callers either know the member is absent or
 * have just established it. */
static void zmiTableInsertRaw(zmiTable *t, uint64_t hash, zbtElem *elem) {
    uint8_t tag = zmiTag(hash);
    unsigned long mask = t->size - 1;
    unsigned long index = hash & mask;

    for (unsigned long probes = 0; probes < t->size; probes++) {
        zmiBucket *b = &t->buckets[index];
        for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++) {
            uint8_t oldtag = zmiTagAt(b, pos);
            if (oldtag == 0 || b->slot[pos] == ZMI_TOMBSTONE) {
                if (oldtag == 0) t->filled++;
                zmiSetTag(b, pos, tag);
                b->slot[pos] = elem;
                t->live++;
                zmiRecordHomeTag(t, hash);
                return;
            }
        }
        index = (index + 1) & mask;
    }
    /* The maximum-filled rule leaves an unused slot in every table. */
    serverPanic("zmindex: no free slot in member index");
}

/* --------------------------------- Lookup -------------------------------- */

/* Search one table.
 *
 * 'ins_bucket' asks for the slot an insertion should use if the member turns
 * out to be absent; the first tombstone on the chain is preferred over the
 * unused slot that ends it, since both are reached by the same probe.
 *
 * Returns the element, or NULL when the member is not in this table. */
static zbtElem *zmiTableFind(zmiTable *t, uint64_t hash,
                             const char *member, size_t mlen,
                             zmiBucket **found_bucket, unsigned int *found_pos,
                             zmiBucket **ins_bucket, unsigned int *ins_pos)
{
    if (t->size == 0) return NULL;

    /* The filter can rule the member out, but an insertion still needs a
     * slot, and finding it costs the same walk. */
    int absent = !zmiHomeMayContain(t, hash);
    if (absent && ins_bucket == NULL) return NULL;

    uint8_t tag = zmiTag(hash);
    unsigned long mask = t->size - 1;
    unsigned long index = hash & mask;
    zmiBucket *tomb_bucket = NULL;
    unsigned int tomb_pos = 0;

    for (unsigned long probes = 0; probes < t->size; probes++) {
        zmiBucket *b = &t->buckets[index];

        if (!absent) {
            uint64_t matches = zmiTagMask(b->tags, tag);
            while (matches) {
                unsigned int pos = zmiFirstTag(matches);
                matches &= matches - 1;
                if (zmiTagAt(b, pos) != tag) continue; /* SWAR false positive */
                zbtElem *elem = b->slot[pos];
                if (elem == ZMI_TOMBSTONE) continue;
                sds cur = zbtGetEle(elem);
                if (sdslen(cur) == mlen && memcmp(cur, member, mlen) == 0) {
                    if (found_bucket) *found_bucket = b;
                    if (found_pos) *found_pos = pos;
                    return elem;
                }
            }
        }

        /* An unused slot ends the chain: nothing this hash could reach lies
         * beyond it. A tombstone does not, so keep walking past one. */
        for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++) {
            if (zmiTagAt(b, pos) == 0) {
                if (ins_bucket) {
                    *ins_bucket = tomb_bucket ? tomb_bucket : b;
                    *ins_pos = tomb_bucket ? tomb_pos : pos;
                }
                return NULL;
            }
            if (tomb_bucket == NULL && b->slot[pos] == ZMI_TOMBSTONE) {
                tomb_bucket = b;
                tomb_pos = pos;
            }
        }
        index = (index + 1) & mask;
    }

    if (ins_bucket && tomb_bucket) {
        *ins_bucket = tomb_bucket;
        *ins_pos = tomb_pos;
    }
    return NULL;
}

/* Locate a known element by pointer. Used where the member has already been
 * resolved, so no string comparison is needed. */
static int zmiTableFindPtr(zmiTable *t, uint64_t hash, zbtElem *target,
                           zmiBucket **found_bucket, unsigned int *found_pos)
{
    if (t->size == 0) return 0;
    if (!zmiHomeMayContain(t, hash)) return 0;

    uint8_t tag = zmiTag(hash);
    unsigned long mask = t->size - 1;
    unsigned long index = hash & mask;

    for (unsigned long probes = 0; probes < t->size; probes++) {
        zmiBucket *b = &t->buckets[index];
        uint64_t matches = zmiTagMask(b->tags, tag);
        while (matches) {
            unsigned int pos = zmiFirstTag(matches);
            matches &= matches - 1;
            if (zmiTagAt(b, pos) == tag && b->slot[pos] == target) {
                *found_bucket = b;
                *found_pos = pos;
                return 1;
            }
        }
        for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++)
            if (zmiTagAt(b, pos) == 0) return 0;
        index = (index + 1) & mask;
    }
    return 0;
}

/* --------------------------------- Resizing ------------------------------- */

static void zmiRehashStep(zmindex *mi, unsigned long buckets);

/* Begin copying into a table of 'buckets' buckets. */
static int zmiStartResize(zmindex *mi, unsigned long buckets, int try) {
    serverAssert(!zmiIsRehashing(mi));
    if (!zmiTableInit(&mi->t[1], buckets, try)) return 0;
    mi->rehashidx = 0;
    mi->revision = zmiNextRevision(); /* Saved insert positions are stale. */
    return 1;
}

/* Copy whatever is left in one go. Used when the destination is about to run
 * out of room, and by callers that need the index at a definite size. */
static void zmiRehashFinish(zmindex *mi) {
    while (zmiIsRehashing(mi))
        zmiRehashStep(mi, 64);
}

/* Copy up to 'buckets' more buckets of the old table into the new one, and
 * install the new table once the last one is done. */
static void zmiRehashStep(zmindex *mi, unsigned long buckets) {
    if (!zmiIsRehashing(mi)) return;
    zmiTable *src = &mi->t[0];
    zmiTable *dst = &mi->t[1];

    while (buckets-- && (unsigned long)mi->rehashidx < src->size) {
        zmiBucket *b = &src->buckets[mi->rehashidx];
        for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++) {
            if (zmiTagAt(b, pos) == 0) continue;
            zbtElem *elem = b->slot[pos];
            if (elem == ZMI_TOMBSTONE) continue;
            /* The member supplies the hash the compact entry does not keep. */
            zmiTableInsertRaw(dst, zmiHash(zbtGetEle(elem)), elem);
        }
        mi->rehashidx++;
    }

    if ((unsigned long)mi->rehashidx >= src->size) {
        serverAssert(dst->live == mi->used);
        zmiTableRelease(src);
        mi->t[0] = mi->t[1];
        memset(&mi->t[1], 0, sizeof(mi->t[1]));
        mi->rehashidx = -1;
        /* t[0] now carries the new table's revision, so cursors bound to the
         * old one restart and cover the whole index again. */
        mi->revision = zmiNextRevision();
    }
}

/* Make room for 'add' more members, growing, or clearing out tombstones. */
static void zmiExpandIfNeeded(zmindex *mi, unsigned long add) {
    if (mi->t[0].size == 0) {
        serverAssert(!zmiIsRehashing(mi));
        zmiTableInit(&mi->t[0], zmiBucketsForElements(add), 0);
        mi->revision = zmiNextRevision();
        return;
    }

    if (zmiIsRehashing(mi)) {
        zmiTable *dst = &mi->t[1];
        unsigned long slots = zmiTableSlots(dst);
        /* Everything still to be copied, plus the new members, plus the
         * tombstones the destination has already collected. */
        unsigned long live = mi->used + add;
        unsigned long filled = live + (dst->filled - dst->live);
        if (live * ZMI_MAX_LOAD_DEN <= slots * ZMI_MAX_LOAD_NUM &&
            filled * ZMI_MAX_FILLED_DEN < slots * ZMI_MAX_FILLED_NUM)
            return;
        /* The destination would overflow before the copy ends. Finish it now
         * so the checks below can start a correctly sized one. */
        zmiRehashFinish(mi);
    }

    zmiTable *t = &mi->t[0];
    unsigned long slots = zmiTableSlots(t);
    unsigned long live = mi->used + add;
    unsigned long filled = t->filled + add;
    if (live * ZMI_MAX_LOAD_DEN > slots * ZMI_MAX_LOAD_NUM) {
        /* Twice the buckets. That halves the load and leaves the copy far
         * more room than the insertions advancing it can take: the copy ends
         * after t[0].size insertions, while the new table holds nearly twice
         * as many members as the old one could. */
        zmiStartResize(mi, t->size * 2, 0);
    } else if (filled * ZMI_MAX_FILLED_DEN >= slots * ZMI_MAX_FILLED_NUM) {
        /* The load is fine and only tombstones are in the way, so a table of
         * the same size is enough to reclaim them. */
        zmiStartResize(mi, t->size, 0);
    }
}

static int zmiExpandGeneric(zmindex *mi, unsigned long n, int try) {
    if (n < mi->used) n = mi->used;
    unsigned long buckets = zmiBucketsForElements(n);

    if (zmiIsRehashing(mi)) {
        if (mi->t[1].size >= buckets) return 1;
        zmiRehashFinish(mi);
    }
    if (mi->t[0].size >= buckets) return 1;

    /* Nothing to preserve: take the new table straight away. */
    if (mi->used == 0 && mi->t[0].filled == 0) {
        zmiTable fresh;
        if (!zmiTableInit(&fresh, buckets, try)) return 0;
        zmiTableRelease(&mi->t[0]);
        mi->t[0] = fresh;
        mi->revision = zmiNextRevision();
        return 1;
    }
    return zmiStartResize(mi, buckets, try);
}

int zmiExpand(zmindex *mi, unsigned long n) {
    return zmiExpandGeneric(mi, n, 0);
}

int zmiTryExpand(zmindex *mi, unsigned long n) {
    return zmiExpandGeneric(mi, n, 1);
}

void zmiShrinkIfNeeded(zmindex *mi) {
    /* The whole index is dead weight once the last member goes. Range
     * deletion empties the set before the command layer can drop the object,
     * so release the tables here rather than shrink them. */
    if (mi->used == 0) {
        zmiTableRelease(&mi->t[0]);
        zmiTableRelease(&mi->t[1]);
        mi->rehashidx = -1;
        mi->revision = zmiNextRevision();
        return;
    }
    if (mi->pause_autoresize || zmiIsRehashing(mi)) return;
    if (mi->t[0].size <= ZMI_INITIAL_BUCKETS) return;
    if (mi->used * ZMI_MIN_LOAD_DEN <= zmiTableSlots(&mi->t[0]) * ZMI_MIN_LOAD_NUM)
        zmiStartResize(mi, zmiBucketsForElements(mi->used), 0);
}

/* ---------------------------- Public lookup API --------------------------- */

zbtElem *zmiFind(zmindex *mi, sds member) {
    if (mi->used == 0) return NULL;
    zmiRehashStep(mi, ZMI_REHASH_BUCKETS_PER_STEP);

    uint64_t hash = zmiHash(member);
    size_t mlen = sdslen(member);
    zbtElem *elem;

    /* Members added during a copy exist only in the new table. Either table
     * may hold a copied member, and both name the same element. */
    if (zmiIsRehashing(mi)) {
        elem = zmiTableFind(&mi->t[1], hash, member, mlen, NULL, NULL, NULL, NULL);
        if (elem) return elem;
    }
    return zmiTableFind(&mi->t[0], hash, member, mlen, NULL, NULL, NULL, NULL);
}

/* The bucket a lookup for 'hash' starts at, for prefetching. Always a
 * readable address, even before any table exists. */
static inline const void *zmiBucketHint(zmindex *mi, uint64_t hash) {
    zmiTable *t = zmiDestTable(mi);
    if (t->size == 0) return t;
    return &t->buckets[hash & (t->size - 1)];
}

/* Look up several members at once. Independent lookups have nothing to wait
 * on between them, so the dependent bucket-load cache miss of each one is
 * hidden behind a lookahead window instead of being paid serially. */
void zmiFindBatch(zmindex *mi, sds *members, zbtElem **results, size_t n) {
    if (n == 0) return;

    enum { PF = 8 };

    size_t primed = n < PF ? n : PF;
    for (size_t j = 0; j < primed; j++)
        redis_prefetch_read(zmiBucketHint(mi, zmiHash(members[j])));

    for (size_t i = 0; i < n; i++) {
        /* Two windows ahead: bring the member's own bytes into cache before
         * it is hashed. */
        if (i + 2 * PF < n) redis_prefetch_read(members[i + 2 * PF]);

        results[i] = zmiFind(mi, members[i]);

        /* One window ahead: hash the member and prefetch its bucket. */
        if (i + PF < n)
            redis_prefetch_read(zmiBucketHint(mi, zmiHash(members[i + PF])));
    }
}

/* Look up 'member' and, when it is absent, record where it would go so the
 * insertion that normally follows does not probe again. */
zbtElem *zmiFindForAdd(zmindex *mi, sds member, zmiPosition *pos) {
    zmiRehashStep(mi, ZMI_REHASH_BUCKETS_PER_STEP);

    uint64_t hash = zmiHash(member);
    size_t mlen = sdslen(member);
    zmiBucket *ins_bucket = NULL;
    unsigned int ins_pos = 0;
    zbtElem *elem = NULL;

    pos->bucket = NULL;
    pos->pos = 0;
    pos->hash = hash;
    pos->revision = mi->revision;

    if (zmiIsRehashing(mi)) {
        /* Only a slot in the destination is of any use to the caller. */
        elem = zmiTableFind(&mi->t[1], hash, member, mlen, NULL, NULL,
                            &ins_bucket, &ins_pos);
        if (elem) return elem;
        elem = zmiTableFind(&mi->t[0], hash, member, mlen, NULL, NULL,
                            NULL, NULL);
        if (elem) return elem;
    } else {
        elem = zmiTableFind(&mi->t[0], hash, member, mlen, NULL, NULL,
                            &ins_bucket, &ins_pos);
        if (elem) return elem;
    }

    pos->bucket = ins_bucket;
    pos->pos = ins_pos;
    return NULL;
}

/* Place 'elem' in the slot a preceding zmiFindForAdd() picked, if the index
 * has not changed shape since. The member must be absent. */
void zmiInsertAt(zmindex *mi, zbtElem *elem, const zmiPosition *pos) {
    zmiExpandIfNeeded(mi, 1);

    zmiTable *t = zmiDestTable(mi);
    uint64_t hash = pos ? pos->hash : zmiHash(zbtGetEle(elem));

    if (pos && pos->bucket && pos->revision == mi->revision &&
        pos->pos < ZMI_BUCKET_ITEMS && zmiOwnsBucket(t, pos->bucket))
    {
        zmiBucket *b = pos->bucket;
        uint8_t oldtag = zmiTagAt(b, pos->pos);
        if (oldtag == 0 || b->slot[pos->pos] == ZMI_TOMBSTONE) {
            if (oldtag == 0) t->filled++;
            zmiSetTag(b, pos->pos, zmiTag(hash));
            b->slot[pos->pos] = elem;
            t->live++;
            zmiRecordHomeTag(t, hash);
            mi->used++;
            zmiRehashStep(mi, ZMI_REHASH_BUCKETS_PER_STEP);
            return;
        }
    }

    zmiTableInsertRaw(t, hash, elem);
    mi->used++;
    zmiRehashStep(mi, ZMI_REHASH_BUCKETS_PER_STEP);
}

void zmiAdd(zmindex *mi, zbtElem *elem) {
    zmiInsertAt(mi, elem, NULL);
}

/* Add 'elem' unless its member is already indexed. Returns 1 when added. */
int zmiAddUnique(zmindex *mi, zbtElem *elem) {
    zmiPosition pos;
    if (zmiFindForAdd(mi, zbtGetEle(elem), &pos) != NULL) return 0;
    zmiInsertAt(mi, elem, &pos);
    return 1;
}

void zmiAddBatch(zmindex *mi, zbtElem **elems, unsigned long n) {
    if (n == 0) return;
    zmiExpand(mi, mi->used + n);
    for (unsigned long i = 0; i < n; i++)
        zmiAdd(mi, elems[i]);
}

/* ---------------------------------- Removal ------------------------------- */

static void zmiTombstoneAt(zmindex *mi, zmiTable *t, zmiBucket *b,
                           unsigned int pos)
{
    UNUSED(mi);
    serverAssert(b->slot[pos] != ZMI_TOMBSTONE && zmiTagAt(b, pos) != 0);
    b->slot[pos] = ZMI_TOMBSTONE;
    t->live--;
}

/* Drop every reference to 'elem', which a copy in progress may have put in
 * both tables. Returns 1 if it was indexed. */
static int zmiRemoveElem(zmindex *mi, zbtElem *elem, uint64_t hash) {
    int removed = 0;
    for (int i = 0; i < 2; i++) {
        zmiBucket *b;
        unsigned int pos;
        if (zmiTableFindPtr(&mi->t[i], hash, elem, &b, &pos)) {
            zmiTombstoneAt(mi, &mi->t[i], b, pos);
            removed = 1;
        }
    }
    if (removed) mi->used--;
    return removed;
}

/* Remove 'member' and return the element it named, or NULL if absent. The
 * element itself is untouched: the caller owns it from here. */
zbtElem *zmiUnlink(zmindex *mi, sds member) {
    if (mi->used == 0) return NULL;
    zmiRehashStep(mi, ZMI_REHASH_BUCKETS_PER_STEP);

    uint64_t hash = zmiHash(member);
    size_t mlen = sdslen(member);
    zbtElem *elem = NULL;

    if (zmiIsRehashing(mi))
        elem = zmiTableFind(&mi->t[1], hash, member, mlen, NULL, NULL, NULL, NULL);
    if (elem == NULL)
        elem = zmiTableFind(&mi->t[0], hash, member, mlen, NULL, NULL, NULL, NULL);
    if (elem == NULL) return NULL;

    zmiRemoveElem(mi, elem, hash);
    return elem;
}

/* Same, for callers that already hold the element. */
int zmiDeleteElem(zmindex *mi, zbtElem *elem) {
    if (mi->used == 0) return 0;
    return zmiRemoveElem(mi, elem, zmiHash(zbtGetEle(elem)));
}

/* Point every slot naming 'old' at 'newelem' instead. Both hold the same
 * member bytes; active defragmentation uses this after moving an element. */
void zmiReplaceElem(zmindex *mi, zbtElem *old, zbtElem *newelem) {
    uint64_t hash = zmiHash(zbtGetEle(newelem));
    for (int i = 0; i < 2; i++) {
        zmiBucket *b;
        unsigned int pos;
        if (zmiTableFindPtr(&mi->t[i], hash, old, &b, &pos))
            b->slot[pos] = newelem;
    }
}

/* -------------------------------- Traversal ------------------------------- */

/* While a copy is in progress the members of buckets below rehashidx are in
 * both tables. Everything is named exactly once by the new table plus the
 * part of the old one that has not been copied yet. */
static inline unsigned long zmiFirstOwnBucket(const zmindex *mi) {
    return zmiIsRehashing(mi) ? (unsigned long)mi->rehashidx : 0;
}

void zmiInitIterator(zmiIterator *it, zmindex *mi) {
    it->mi = mi;
    it->table = 0;
    it->bucket = zmiFirstOwnBucket(mi);
    it->pos = 0;
}

zbtElem *zmiNext(zmiIterator *it) {
    zmindex *mi = it->mi;

    for (;;) {
        zmiTable *t = &mi->t[it->table];
        if (it->bucket >= t->size) {
            if (it->table == 0 && zmiIsRehashing(mi)) {
                it->table = 1;
                it->bucket = 0;
                it->pos = 0;
                continue;
            }
            return NULL;
        }
        zmiBucket *b = &t->buckets[it->bucket];
        while (it->pos < ZMI_BUCKET_ITEMS) {
            unsigned int pos = it->pos++;
            if (zmiTagAt(b, pos) == 0) continue;
            zbtElem *elem = b->slot[pos];
            if (elem == ZMI_TOMBSTONE) continue;
            return elem;
        }
        it->bucket++;
        it->pos = 0;
    }
}

/* Return a uniformly chosen member, or NULL when the index is empty. */
zbtElem *zmiRandomElem(zmindex *mi) {
    if (mi->used == 0) return NULL;

    unsigned long first = zmiFirstOwnBucket(mi);
    unsigned long own0 = mi->t[0].size > first ? mi->t[0].size - first : 0;
    unsigned long own1 = zmiIsRehashing(mi) ? mi->t[1].size : 0;
    unsigned long buckets = own0 + own1;
    if (buckets == 0) return NULL;

    /* Rejection sampling over the slots that name a member exactly once. The
     * minimum-load rule keeps at least an eighth of them live, so a live slot
     * normally turns up in a handful of draws. */
    unsigned long slots = buckets * ZMI_BUCKET_ITEMS;
    for (int tries = 0; tries < 100; tries++) {
        unsigned long r = randomULong() % slots;
        unsigned long bucket = r / ZMI_BUCKET_ITEMS;
        unsigned int pos = (unsigned int)(r % ZMI_BUCKET_ITEMS);
        zmiTable *t;
        if (bucket < own0) {
            t = &mi->t[0];
            bucket += first;
        } else {
            t = &mi->t[1];
            bucket -= own0;
        }
        zmiBucket *b = &t->buckets[bucket];
        if (zmiTagAt(b, pos) == 0) continue;
        zbtElem *elem = b->slot[pos];
        if (elem == ZMI_TOMBSTONE) continue;
        return elem;
    }

    /* A table left unusually sparse: walk to a randomly chosen member. */
    unsigned long target = randomULong() % mi->used;
    zmiIterator it;
    zmiInitIterator(&it, mi);
    zbtElem *elem;
    while ((elem = zmiNext(&it)) != NULL) {
        if (target-- == 0) return elem;
    }
    serverPanic("zmindex: live member count disagrees with the tables");
}

/* Visit one bucket per call and return the cursor of the next.
 *
 * The low half of the cursor is a bucket number, of the old table first and
 * then of the new one; the high half is the table revision the cursor belongs
 * to. A copy in progress only appends to that space and never removes a
 * member from the old table, so a sweep cannot step over one. Installing a new
 * table changes the revision and restarts the sweep, which may repeat members
 * but never skips any. */
uint64_t zmiScan(zmindex *mi, uint64_t cursor, zmiScanFunction *fn,
                 void *privdata)
{
    if (mi->used == 0) return 0;

    uint32_t revision = (uint32_t)(cursor >> 32);
    uint64_t bucket = cursor ? (uint32_t)cursor - 1 : 0;
    if (cursor == 0 || revision != mi->t[0].scan_revision) {
        revision = mi->t[0].scan_revision;
        bucket = 0;
    }

    uint64_t first = mi->t[0].size;
    uint64_t total = first + (zmiIsRehashing(mi) ? mi->t[1].size : 0);
    if (bucket >= total) return 0;

    zmiTable *t;
    uint64_t local;
    if (bucket < first) {
        t = &mi->t[0];
        local = bucket;
    } else {
        t = &mi->t[1];
        local = bucket - first;
    }

    zmiBucket *b = &t->buckets[local];
    for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++) {
        if (zmiTagAt(b, pos) == 0) continue;
        zbtElem *elem = b->slot[pos];
        if (elem == ZMI_TOMBSTONE) continue;
        fn(privdata, elem);
    }

    if (++bucket >= total) return 0;
    serverAssert(bucket < UINT32_MAX);
    return ((uint64_t)revision << 32) | (uint32_t)(bucket + 1);
}

/* ---------------------------- Defragmentation ----------------------------- */

/* Visit one bucket per call, for active defragmentation.
 *
 * Unlike zmiScan(), the cursor here is a plain bucket number. Defragmentation
 * does not need SCAN's completeness guarantee: relocating an element twice is
 * harmless and anything a table swap hides is picked up by the next cycle. In
 * exchange the cursor fits in an unsigned long on every platform, which is
 * what the shared defrag "scan later" cursor is. */
unsigned long zmiScanDefrag(zmindex *mi, unsigned long cursor,
                            zmiScanFunction *fn, void *privdata)
{
    unsigned long first = mi->t[0].size;
    unsigned long total = first + (zmiIsRehashing(mi) ? mi->t[1].size : 0);
    /* The index may have been resized or emptied since the last call. */
    if (cursor >= total) return 0;

    zmiTable *t;
    unsigned long local;
    if (cursor < first) {
        t = &mi->t[0];
        local = cursor;
    } else {
        t = &mi->t[1];
        local = cursor - first;
    }

    zmiBucket *b = &t->buckets[local];
    for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++) {
        if (zmiTagAt(b, pos) == 0) continue;
        zbtElem *elem = b->slot[pos];
        if (elem == ZMI_TOMBSTONE) continue;
        fn(privdata, elem);
    }

    return ++cursor >= total ? 0 : cursor;
}

zmindex *zmiDefragTables(zmindex *mi, zmiDefragAllocFunction *defragfn) {
    zmindex *newmi = defragfn(mi);
    if (newmi) mi = newmi;
    for (int i = 0; i < 2; i++) {
        if (mi->t[i].buckets == NULL) continue;
        zmiBucket *buckets = defragfn(mi->t[i].buckets);
        if (buckets) mi->t[i].buckets = buckets;
    }
    return newmi;
}

/* ------------------------------- Introspection ---------------------------- */

static size_t zmiTableStats(char *buf, size_t bufsize, zmiTable *t,
                            int tableid, int full)
{
    if (t->size == 0)
        return snprintf(buf, bufsize, "No stats available for empty table %d\n",
                        tableid);

    unsigned long slots = zmiTableSlots(t);
    unsigned long tombstones = t->filled - t->live;
    unsigned long maxrun = 0, run = 0, runs = 0;
    for (unsigned long i = 0; i < t->size; i++) {
        int used = 0;
        for (unsigned int pos = 0; pos < ZMI_BUCKET_ITEMS; pos++)
            if (zmiTagAt(&t->buckets[i], pos) != 0) { used = 1; break; }
        if (used) {
            run++;
            if (run > maxrun) maxrun = run;
        } else {
            if (run) runs++;
            run = 0;
        }
    }
    if (run) runs++;

    size_t l = snprintf(buf, bufsize,
        "Member index table %d stats:\n"
        " table size: %lu buckets (%lu slots)\n"
        " number of elements: %lu\n"
        " tombstones: %lu\n"
        " load factor: %.02f%%\n"
        " bucket runs: %lu, longest: %lu\n",
        tableid, t->size, slots, t->live, tombstones,
        slots ? (double)t->live / slots * 100 : 0.0, runs, maxrun);
    UNUSED(full);
    return l > bufsize ? bufsize : l;
}

void zmiGetStats(char *buf, size_t bufsize, zmindex *mi, int full) {
    if (bufsize == 0) return;
    size_t l = zmiTableStats(buf, bufsize, &mi->t[0], 0, full);
    if (zmiIsRehashing(mi) && bufsize > l)
        l += zmiTableStats(buf + l, bufsize - l, &mi->t[1], 1, full);
    /* Make sure there is a NULL term at the end. */
    buf[bufsize - 1] = '\0';
    UNUSED(l);
}

/* --------------------------------- Testing -------------------------------- */

#ifdef REDIS_TEST
#include <assert.h>
#include "testhelp.h"

/* Members are "member:<index>:<tail>" so a test callback can recover the
 * index it was built from. The varying tail keeps lengths uneven. */
static sds zmiTestMember(int i) {
    char buf[64];
    snprintf(buf, sizeof(buf), "member:%d:%s", i, i % 3 ? "x" : "longer-tail");
    return sdsnew(buf);
}

static int zmiTestIndexOf(zbtElem *elem) {
    return atoi(zbtGetEle(elem) + strlen("member:"));
}

typedef struct {
    char *hits;
    int n;
    unsigned long count;
} zmiTestScanCtx;

static void zmiTestScanCollect(void *privdata, zbtElem *elem) {
    zmiTestScanCtx *ctx = privdata;
    int idx = zmiTestIndexOf(elem);
    assert(idx >= 0 && idx < ctx->n);
    ctx->hits[idx] = 1;
    ctx->count++;
}

int zmindexTest(int argc, char **argv, int flags) {
    UNUSED(argc);
    UNUSED(argv);
    UNUSED(flags);

    printf("Testing the sorted set member index\n");

    const int N = 5000;
    zbtElem **elems = zmalloc(sizeof(zbtElem *) * N);
    sds *members = zmalloc(sizeof(sds) * N);
    for (int i = 0; i < N; i++) {
        members[i] = zmiTestMember(i);
        elems[i] = zbtCreateElem((double)i, members[i]);
    }

    /* --- Insert, find, and the absence of anything not inserted --- */
    {
        zmindex *mi = zmiCreate();
        for (int i = 0; i < N; i++) {
            zmiPosition pos;
            assert(zmiFindForAdd(mi, members[i], &pos) == NULL);
            zmiInsertAt(mi, elems[i], &pos);
            assert(zmiSize(mi) == (unsigned long)(i + 1));
        }
        for (int i = 0; i < N; i++)
            assert(zmiFind(mi, members[i]) == elems[i]);

        sds missing = sdsnew("definitely-not-there");
        assert(zmiFind(mi, missing) == NULL);
        sdsfree(missing);

        /* A duplicate must be rejected without changing the size. */
        assert(zmiAddUnique(mi, elems[7]) == 0);
        assert(zmiSize(mi) == (unsigned long)N);
        zmiRelease(mi);
        test_cond("Insert and find every member", 1);
    }

    /* --- Deletion leaves the rest of every probe chain reachable --- */
    {
        zmindex *mi = zmiCreate();
        zmiAddBatch(mi, elems, N);
        assert(zmiSize(mi) == (unsigned long)N);

        for (int i = 0; i < N; i += 2)
            assert(zmiUnlink(mi, members[i]) == elems[i]);
        assert(zmiSize(mi) == (unsigned long)N / 2);

        for (int i = 0; i < N; i++) {
            zbtElem *found = zmiFind(mi, members[i]);
            assert(found == (i % 2 ? elems[i] : NULL));
        }
        /* Re-adding must reuse tombstones rather than grow without bound. */
        for (int i = 0; i < N; i += 2)
            zmiAdd(mi, elems[i]);
        assert(zmiSize(mi) == (unsigned long)N);
        for (int i = 0; i < N; i++)
            assert(zmiFind(mi, members[i]) == elems[i]);

        assert(zmiDeleteElem(mi, elems[3]) == 1);
        assert(zmiDeleteElem(mi, elems[3]) == 0);
        zmiRelease(mi);
        test_cond("Delete, lookup and reinsert", 1);
    }

    /* --- Iteration yields every member exactly once, mid-resize too --- */
    {
        zmindex *mi = zmiCreate();
        for (int i = 0; i < N; i++) {
            zmiAdd(mi, elems[i]);

            /* Whether or not a copy is in progress, an iteration has to see
             * each member once: the invariant is easiest to break exactly
             * here, so check it while the index is small enough to afford. */
            if (i < 200 || i % 512 == 0) {
                char *hits = zcalloc(N);
                zmiIterator it;
                zmiInitIterator(&it, mi);
                zbtElem *elem;
                unsigned long seen = 0;
                while ((elem = zmiNext(&it)) != NULL) {
                    int idx = zmiTestIndexOf(elem);
                    assert(idx >= 0 && idx < N);
                    assert(hits[idx] == 0);
                    hits[idx] = 1;
                    seen++;
                }
                assert(seen == zmiSize(mi));
                zfree(hits);
            }
        }
        zmiRelease(mi);
        test_cond("Iterator visits every member once", 1);
    }

    /* --- A full scan reaches every member --- */
    {
        zmindex *mi = zmiCreate();
        zmiAddBatch(mi, elems, N);

        char *hits = zcalloc(N);
        zmiTestScanCtx ctx = {hits, N, 0};
        uint64_t cursor = 0;
        unsigned long guard = 0;
        do {
            cursor = zmiScan(mi, cursor, zmiTestScanCollect, &ctx);
            assert(++guard < 1000000);
        } while (cursor != 0);

        for (int i = 0; i < N; i++) assert(hits[i] == 1);
        zfree(hits);
        zmiRelease(mi);
        test_cond("Full scan reaches every member", 1);
    }

    /* --- A scan may repeat members while the index is being copied, but a
     * member present throughout must still come back at least once --- */
    {
        const int M = 1500;
        zmindex *mi = zmiCreate();
        zmiAddBatch(mi, elems, M);

        char *hits = zcalloc(N);
        zmiTestScanCtx ctx = {hits, N, 0};
        uint64_t cursor = 0;
        unsigned long guard = 0;
        int extra = M;
        do {
            cursor = zmiScan(mi, cursor, zmiTestScanCollect, &ctx);
            /* Grow the index under the cursor, which both advances a copy and
             * eventually installs a new table. */
            if (extra < N) zmiAdd(mi, elems[extra++]);
            assert(++guard < 10000000);
        } while (cursor != 0);

        for (int i = 0; i < M; i++) assert(hits[i] == 1);
        zfree(hits);
        zmiRelease(mi);
        test_cond("Scan keeps members present throughout a resize", 1);
    }

    /* --- Shrinking after mass deletion, and release when empty --- */
    {
        zmindex *mi = zmiCreate();
        zmiAddBatch(mi, elems, N);
        unsigned long grown = mi->t[0].size;
        assert(grown > ZMI_INITIAL_BUCKETS);

        zmiPauseAutoResize(mi);
        for (int i = 0; i < N - 10; i++)
            assert(zmiUnlink(mi, members[i]) != NULL);
        zmiResumeAutoResize(mi);
        zmiShrinkIfNeeded(mi);
        zmiRehashFinish(mi);
        assert(mi->t[0].size < grown);
        for (int i = N - 10; i < N; i++)
            assert(zmiFind(mi, members[i]) == elems[i]);

        for (int i = N - 10; i < N; i++)
            assert(zmiUnlink(mi, members[i]) != NULL);
        zmiShrinkIfNeeded(mi);
        assert(zmiSize(mi) == 0);
        assert(mi->t[0].size == 0 && mi->t[0].buckets == NULL);
        /* An emptied index has to come back to life on the next insert. */
        zmiAdd(mi, elems[0]);
        assert(zmiFind(mi, members[0]) == elems[0]);
        zmiRelease(mi);
        test_cond("Shrink, release when empty, and revive", 1);
    }

    /* --- Random selection covers the whole index --- */
    {
        zmindex *mi = zmiCreate();
        const int M = 64;
        zmiAddBatch(mi, elems, M);
        char *hits = zcalloc(M);
        for (int i = 0; i < M * 200; i++) {
            zbtElem *elem = zmiRandomElem(mi);
            assert(elem != NULL);
            int idx = zmiTestIndexOf(elem);
            assert(idx >= 0 && idx < M);
            hits[idx] = 1;
        }
        for (int i = 0; i < M; i++) assert(hits[i] == 1);
        zfree(hits);
        zmiRelease(mi);
        test_cond("Random selection reaches every member", 1);
    }

    /* --- Replacing an element updates every table that names it --- */
    {
        zmindex *mi = zmiCreate();
        zmiAddBatch(mi, elems, N);
        zbtElem *copy = zbtCreateElem(elems[11]->score, members[11]);
        zmiReplaceElem(mi, elems[11], copy);
        assert(zmiFind(mi, members[11]) == copy);
        zmiReplaceElem(mi, copy, elems[11]);
        assert(zmiFind(mi, members[11]) == elems[11]);
        zbtFreeElem(copy);
        zmiRelease(mi);
        test_cond("Replace element pointer", 1);
    }

    for (int i = 0; i < N; i++) {
        zbtFreeElem(elems[i]);
        sdsfree(members[i]);
    }
    zfree(elems);
    zfree(members);
    return 0;
}
#endif
