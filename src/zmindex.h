/* zmindex.h -- compact member index for large sorted sets.
 *
 * Maps a sorted-set member to the zbtElem that holds it, replacing the
 * general-purpose dict that used to sit beside the B+ tree. See zmindex.c for
 * the layout and the reasoning behind it.
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#ifndef __ZMINDEX_H
#define __ZMINDEX_H

#include <stddef.h>
#include <stdint.h>

#include "sds.h"

/* Elements are only ever referenced by pointer here, so the definition in
 * server.h is not needed to describe the index. */
struct zbtElem;

/* Eight entries share a bucket so that one cache line serves the tag word,
 * the filter word and the first slots. */
#define ZMI_BUCKET_ITEMS 8

/* One slot is a byte of member hash inside 'tags' plus an element pointer.
 * A zero tag byte marks a slot that was never used, which is what ends a
 * probe; a used slot whose pointer is the tombstone sentinel was emptied and
 * keeps later slots of the same chain reachable.
 *
 * 'home_tags' summarises the tags of every member whose chain *starts* in
 * this bucket. It answers "perhaps present" or "definitely absent", which is
 * enough to abandon most lookups for missing members before any element is
 * dereferenced. */
typedef struct zmiBucket {
    uint64_t tags;
    uint64_t home_tags;
    struct zbtElem *slot[ZMI_BUCKET_ITEMS];
} zmiBucket;

typedef struct zmiTable {
    zmiBucket *buckets;
    unsigned long size;     /* Buckets, always a power of two (0 if unset). */
    unsigned long live;     /* Slots holding an element. */
    unsigned long filled;   /* Slots ever used (live plus tombstones). */
    uint32_t scan_revision; /* Identity a scan cursor is bound to. */
} zmiTable;

/* t[1] exists only while the index is being copied into a new size. */
typedef struct zmindex {
    zmiTable t[2];
    long rehashidx;             /* -1, or the next t[0] bucket to copy. */
    unsigned long used;         /* Live members. Authoritative. */
    unsigned int pause_autoresize;
    uint32_t revision;          /* Invalidates saved insert positions. */
} zmindex;

/* A failed lookup can hand its empty slot to the following insertion. The
 * position is used only while 'revision' still matches the index. */
typedef struct zmiPosition {
    zmiBucket *bucket;
    unsigned int pos;
    uint64_t hash;
    uint32_t revision;
} zmiPosition;

/* Yields every live member exactly once. The index must not be modified while
 * an iterator is open. */
typedef struct zmiIterator {
    zmindex *mi;
    unsigned long bucket;
    unsigned int pos;
    int table;
} zmiIterator;

typedef void zmiScanFunction(void *privdata, struct zbtElem *elem);
typedef void *zmiDefragAllocFunction(void *ptr);

/* Lifecycle and sizing. */
zmindex *zmiCreate(void);
void zmiRelease(zmindex *mi);
int zmiExpand(zmindex *mi, unsigned long n);
int zmiTryExpand(zmindex *mi, unsigned long n);
void zmiShrinkIfNeeded(zmindex *mi);
size_t zmiMemUsage(const zmindex *mi);
void zmiDismissMemory(zmindex *mi);
void zmiGetStats(char *buf, size_t bufsize, zmindex *mi, int full);

#define zmiSize(mi) ((mi)->used)
#define zmiPauseAutoResize(mi) ((mi)->pause_autoresize++)
#define zmiResumeAutoResize(mi) ((mi)->pause_autoresize--)

/* Lookup and modification. */
struct zbtElem *zmiFind(zmindex *mi, sds member);
void zmiFindBatch(zmindex *mi, sds *members, struct zbtElem **results,
                  size_t n);
struct zbtElem *zmiFindForAdd(zmindex *mi, sds member, zmiPosition *pos);
void zmiInsertAt(zmindex *mi, struct zbtElem *elem, const zmiPosition *pos);
void zmiAdd(zmindex *mi, struct zbtElem *elem);
int zmiAddUnique(zmindex *mi, struct zbtElem *elem);
void zmiAddBatch(zmindex *mi, struct zbtElem **elems, unsigned long n);
struct zbtElem *zmiUnlink(zmindex *mi, sds member);
int zmiDeleteElem(zmindex *mi, struct zbtElem *elem);
void zmiReplaceElem(zmindex *mi, struct zbtElem *old, struct zbtElem *newelem);
struct zbtElem *zmiRandomElem(zmindex *mi);

/* Unordered access. */
void zmiInitIterator(zmiIterator *it, zmindex *mi);
struct zbtElem *zmiNext(zmiIterator *it);
uint64_t zmiScan(zmindex *mi, uint64_t cursor, zmiScanFunction *fn,
                 void *privdata);

/* Active defragmentation. zmiScanDefrag() walks the slots a bucket at a time
 * through a plain bucket cursor; zmiDefragTables() relocates the index itself
 * and its bucket arrays, returning the index's new address when it moved. */
unsigned long zmiScanDefrag(zmindex *mi, unsigned long cursor,
                            zmiScanFunction *fn, void *privdata);
zmindex *zmiDefragTables(zmindex *mi, zmiDefragAllocFunction *defragfn);

#endif /* __ZMINDEX_H */
