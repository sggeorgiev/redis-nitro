/* dictbench.c - micro-benchmark gate comparing dict.c (chained) with
 * hashtable.c (Swiss-style 64-byte buckets).
 *
 * Server-only (redis-cli/redis-benchmark link dict.o but not hashtable.o).
 * Run with:   ./redis-server test dictbench [--large-memory]
 * 1M entries by default, plus 10M with --large-memory.
 *
 * Three entry kinds are measured:
 *   kvobj : no_value dict of 16-byte objects embedding a 64-bit key
 *           (keyFromStoredKey / entryGetKey), like the keyspace.
 *   sdsset: no_value dict of sds strings (keys_are_odd), like sets.
 *   sdsval: sds -> ptr dict (heap dictEntry in dict, {key,val} struct in the
 *           hashtable), like the generic value dicts.
 */
#ifdef REDIS_TEST

#include "fmacros.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "server.h"
#include "dict.h"
#include "hashtable.h"
#include "sds.h"
#include "zmalloc.h"
#include "testhelp.h"

typedef struct { uint64_t key; uint64_t pad; } kvItem;
typedef struct { sds key; void *val; } valItem;

typedef enum { K_KVOBJ, K_SDSSET, K_SDSVAL, K_COUNT } kindId;
static const char *kindNames[] = {"kvobj", "sdsset", "sdsval"};

/* ------------------------------ helpers ---------------------------------- */

static uint64_t nowNs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t rngState = 88172645463325252ULL;
static uint64_t rnd(void) {
    rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
    return rngState;
}

/* ---------------------------- kvobj callbacks ----------------------------- */

static const void *kvFromStored(const void *stored) { return &((const kvItem *)stored)->key; }
static uint64_t kvHash(const void *key) { return mix64(*(const uint64_t *)key); }
static int kvDictCmp(dictCmpCache *c, const void *a, const void *b) {
    (void)c; return *(const uint64_t *)a == *(const uint64_t *)b;
}
static int kvHtCmp(const void *a, const void *b) {
    return *(const uint64_t *)a != *(const uint64_t *)b;
}

static dictType kvDictType = {
    .hashFunction = kvHash, .keyCompare = kvDictCmp,
    .no_value = 1, .keys_are_odd = 0, .keyFromStoredKey = kvFromStored,
};
static hashtableType kvHtType = {
    .entryGetKey = kvFromStored, .hashFunction = kvHash, .keyCompare = kvHtCmp,
};

/* ------------------------------ sds callbacks ----------------------------- */

static uint64_t sdsHtHash(const void *key) { return hashtableGenHashFunction(key, sdslen((sds)key)); }
static int sdsHtCmp(const void *a, const void *b) {
    size_t la = sdslen((sds)a), lb = sdslen((sds)b);
    return la != lb || memcmp(a, b, la) != 0;
}
static const void *valItemKey(const void *e) { return ((const valItem *)e)->key; }

static dictType sdsSetDictType = {
    .hashFunction = dictSdsHash, .keyCompare = dictSdsKeyCompare,
    .no_value = 1, .keys_are_odd = 1,
};
static hashtableType sdsSetHtType = {
    .hashFunction = sdsHtHash, .keyCompare = sdsHtCmp,
};
static dictType sdsValDictType = {
    .hashFunction = dictSdsHash, .keyCompare = dictSdsKeyCompare,
};
static hashtableType sdsValHtType = {
    .entryGetKey = valItemKey, .hashFunction = sdsHtHash, .keyCompare = sdsHtCmp,
};

/* ------------------------------ entry model ------------------------------- */

typedef struct benchSet {
    kindId kind;
    size_t n;
    void **ents;  /* n entries inserted by the benchmark */
    void **miss;  /* n entries never inserted */
} benchSet;

static void *makeEnt(kindId k, uint64_t id) {
    char buf[32];
    switch (k) {
    case K_KVOBJ: { kvItem *it = zmalloc(sizeof(*it)); it->key = id; it->pad = 0; return it; }
    case K_SDSSET: { int l = snprintf(buf, sizeof(buf), "key:%llu", (unsigned long long)id); return sdsnewlen(buf, l); }
    default: { valItem *it = zmalloc(sizeof(*it));
               int l = snprintf(buf, sizeof(buf), "key:%llu", (unsigned long long)id);
               it->key = sdsnewlen(buf, l); it->val = it; return it; }
    }
}
static void freeEnt(kindId k, void *e) {
    if (k == K_SDSSET) sdsfree(e);
    else if (k == K_SDSVAL) { sdsfree(((valItem *)e)->key); zfree(e); }
    else zfree(e);
}
/* The key to use for lookups, given an entry. */
static const void *entKey(kindId k, void *e) {
    return k == K_KVOBJ ? (const void *)&((kvItem *)e)->key :
           k == K_SDSSET ? (const void *)e : (const void *)((valItem *)e)->key;
}
/* What dict stores/returns (stored key). For sdsval dict stores the sds. */
static void *dictStored(kindId k, void *e) {
    return k == K_SDSVAL ? (void *)((valItem *)e)->key : e;
}

static benchSet *benchSetNew(kindId k, size_t n) {
    benchSet *s = zmalloc(sizeof(*s));
    s->kind = k; s->n = n;
    s->ents = zmalloc(n * sizeof(void *));
    s->miss = zmalloc(n * sizeof(void *));
    for (size_t i = 0; i < n; i++) {
        s->ents[i] = makeEnt(k, i);
        s->miss[i] = makeEnt(k, n + i);
    }
    /* Shuffle both so the access order is random relative to the allocation. */
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rnd() % (i + 1);
        void *t = s->ents[i]; s->ents[i] = s->ents[j]; s->ents[j] = t;
        j = rnd() % (i + 1);
        t = s->miss[i]; s->miss[i] = s->miss[j]; s->miss[j] = t;
    }
    return s;
}
static void benchSetFree(benchSet *s) {
    for (size_t i = 0; i < s->n; i++) { freeEnt(s->kind, s->ents[i]); freeEnt(s->kind, s->miss[i]); }
    zfree(s->ents); zfree(s->miss); zfree(s);
}

static dictType *dictTypeFor(kindId k) {
    return k == K_KVOBJ ? &kvDictType : k == K_SDSSET ? &sdsSetDictType : &sdsValDictType;
}
static hashtableType *htTypeFor(kindId k) {
    return k == K_KVOBJ ? &kvHtType : k == K_SDSSET ? &sdsSetHtType : &sdsValHtType;
}

/* ------------------------------- results ---------------------------------- */

typedef struct benchRes {
    double insert, drain, hit, miss, mix, iter, scan, sample, twostep;
    double bytesPerEntry;
    size_t sink;
} benchRes;

static volatile size_t gSink;

static void scanCbDict(void *pd, const dictEntry *de, dictEntryLink plink) {
    (void)plink; (*(size_t *)pd) += (size_t)(uintptr_t)dictGetKey(de) & 1;
}
static void scanCbHt(void *pd, void *e) { (*(size_t *)pd) += (size_t)(uintptr_t)e & 1; }

/* ------------------------------- dict side -------------------------------- */

static void benchDict(benchSet *s, benchRes *r) {
    kindId k = s->kind;
    size_t n = s->n, sink = 0;
    uint64_t t;
    size_t mem0 = zmalloc_used_memory();
    dict *d = dictCreate(dictTypeFor(k));

    t = nowNs();
    for (size_t i = 0; i < n; i++) {
        void *e = s->ents[i];
        if (dictAdd(d, dictStored(k, e), k == K_SDSVAL ? ((valItem *)e)->val : NULL) != DICT_OK) abort();
    }
    r->insert = (double)(nowNs() - t) / n;
    t = nowNs();
    while (dictIsRehashing(d)) dictRehashMicroseconds(d, 1000);
    r->drain = (double)(nowNs() - t) / n;
    r->bytesPerEntry = (double)(zmalloc_used_memory() - mem0) / n;

    r->hit = 1e18;
    for (int pass = 0; pass < 3; pass++) {
        t = nowNs();
        for (size_t i = 0; i < n; i++) sink += dictFind(d, entKey(k, s->ents[i])) != NULL;
        double v = (double)(nowNs() - t) / n; if (v < r->hit) r->hit = v;
    }
    r->miss = 1e18;
    for (int pass = 0; pass < 3; pass++) {
        t = nowNs();
        for (size_t i = 0; i < n; i++) sink += dictFind(d, entKey(k, s->miss[i])) != NULL;
        double v = (double)(nowNs() - t) / n; if (v < r->miss) r->miss = v;
    }

    t = nowNs();
    {
        dictIterator it; dictEntry *de;
        dictInitIterator(&it, d);
        while ((de = dictNext(&it))) sink += (size_t)(uintptr_t)dictGetKey(de) & 1;
        dictResetIterator(&it);
    }
    r->iter = (double)(nowNs() - t) / n;

    t = nowNs();
    { unsigned long cur = 0; do { cur = dictScan(d, cur, scanCbDict, &sink); } while (cur); }
    r->scan = (double)(nowNs() - t) / n;

    {
        dictEntry *des[16]; size_t calls = n / 16 < 200000 ? n / 16 : 200000;
        t = nowNs();
        for (size_t i = 0; i < calls; i++) sink += dictGetSomeKeys(d, des, 16);
        r->sample = (double)(nowNs() - t) / calls;
    }

    /* Delete-heavy mix: delete half, re-add half of the misses, delete again. */
    t = nowNs();
    for (size_t i = 0; i < n / 2; i++) sink += dictDelete(d, entKey(k, s->ents[i])) == DICT_OK;
    for (size_t i = 0; i < n / 2; i++) {
        void *e = s->miss[i];
        dictAdd(d, dictStored(k, e), k == K_SDSVAL ? ((valItem *)e)->val : NULL);
    }
    for (size_t i = 0; i < n / 2; i++) sink += dictDelete(d, entKey(k, s->miss[i])) == DICT_OK;
    r->mix = (double)(nowNs() - t) / (n / 2 * 3);
    /* Re-add the deleted half so the two-step test runs on a full table. */
    for (size_t i = 0; i < n / 2; i++) {
        void *e = s->ents[i];
        dictAdd(d, dictStored(k, e), k == K_SDSVAL ? ((valItem *)e)->val : NULL);
    }

    /* find-then-insert on a miss: the db.c lookup + dbAdd pattern. no_value only. */
    r->twostep = -1;
    if (k != K_SDSVAL) {
        t = nowNs();
        for (size_t i = 0; i < n / 2; i++) {
            dictEntryLink bucket = NULL;
            void *e = s->miss[i];
            dictEntryLink link = dictFindLink(d, entKey(k, e), &bucket);
            if (link) abort();
            dictSetKeyAtLink(d, e, &bucket, 1);
        }
        r->twostep = (double)(nowNs() - t) / (n / 2);
    }
    r->sink = sink;
    gSink += sink;
    dictRelease(d);
}

/* ----------------------------- hashtable side ----------------------------- */

static void benchHt(benchSet *s, benchRes *r) {
    kindId k = s->kind;
    size_t n = s->n, sink = 0;
    uint64_t t;
    size_t mem0 = zmalloc_used_memory();
    hashtable *h = hashtableCreate(htTypeFor(k));

    t = nowNs();
    for (size_t i = 0; i < n; i++) if (!hashtableAdd(h, s->ents[i])) abort();
    r->insert = (double)(nowNs() - t) / n;
    t = nowNs();
    while (hashtableIsRehashing(h)) hashtableRehashMicroseconds(h, 1000);
    r->drain = (double)(nowNs() - t) / n;
    r->bytesPerEntry = (double)(zmalloc_used_memory() - mem0) / n;

    r->hit = 1e18;
    for (int pass = 0; pass < 3; pass++) {
        void *f;
        t = nowNs();
        for (size_t i = 0; i < n; i++) sink += hashtableFind(h, entKey(k, s->ents[i]), &f);
        double v = (double)(nowNs() - t) / n; if (v < r->hit) r->hit = v;
    }
    r->miss = 1e18;
    for (int pass = 0; pass < 3; pass++) {
        void *f;
        t = nowNs();
        for (size_t i = 0; i < n; i++) sink += hashtableFind(h, entKey(k, s->miss[i]), &f);
        double v = (double)(nowNs() - t) / n; if (v < r->miss) r->miss = v;
    }

    t = nowNs();
    {
        hashtableIterator it; void *e;
        hashtableInitIterator(&it, h, 0);
        while (hashtableNext(&it, &e)) sink += (size_t)(uintptr_t)e & 1;
        hashtableCleanupIterator(&it);
    }
    r->iter = (double)(nowNs() - t) / n;

    t = nowNs();
    { size_t cur = 0; do { cur = hashtableScan(h, cur, scanCbHt, &sink); } while (cur); }
    r->scan = (double)(nowNs() - t) / n;

    {
        void *dst[16]; size_t calls = n / 16 < 200000 ? n / 16 : 200000;
        t = nowNs();
        for (size_t i = 0; i < calls; i++) sink += hashtableSampleEntries(h, dst, 16);
        r->sample = (double)(nowNs() - t) / calls;
    }

    t = nowNs();
    for (size_t i = 0; i < n / 2; i++) sink += hashtableDelete(h, entKey(k, s->ents[i]));
    for (size_t i = 0; i < n / 2; i++) hashtableAdd(h, s->miss[i]);
    for (size_t i = 0; i < n / 2; i++) sink += hashtableDelete(h, entKey(k, s->miss[i]));
    r->mix = (double)(nowNs() - t) / (n / 2 * 3);
    for (size_t i = 0; i < n / 2; i++) hashtableAdd(h, s->ents[i]);

    r->twostep = -1;
    if (k != K_SDSVAL) {
        t = nowNs();
        for (size_t i = 0; i < n / 2; i++) {
            hashtablePosition pos; void *existing;
            void *e = s->miss[i];
            if (!hashtableFindPositionForInsert(h, (void *)entKey(k, e), &pos, &existing)) abort();
            hashtableInsertAtPosition(h, e, &pos);
        }
        r->twostep = (double)(nowNs() - t) / (n / 2);
    }
    r->sink = sink;
    gSink += sink;
    hashtableRelease(h);
}

/* -------------------------------- reporting ------------------------------- */

static void row(const char *name, double a, double b) {
    if (a < 0 || b < 0) return;
    printf("    %-22s %9.1f %9.1f   %+6.1f%%\n", name, a, b, (b - a) / a * 100.0);
}

static int runSize(size_t n) {
    int fail = 0;
    printf("\n=== %zu entries (ns/op unless noted; delta<0 = hashtable faster) ===\n", n);
    for (int k = 0; k < K_COUNT; k++) {
        benchSet *s = benchSetNew(k, n);
        benchRes dr, hr;
        memset(&dr, 0, sizeof(dr)); memset(&hr, 0, sizeof(hr));
        benchDict(s, &dr);
        benchHt(s, &hr);
        printf("  [%s]\n    %-22s %9s %9s\n", kindNames[k], "", "dict", "hashtable");
        row("insert", dr.insert, hr.insert);
        row("rehash drain", dr.drain, hr.drain);
        row("find hit", dr.hit, hr.hit);
        row("find miss", dr.miss, hr.miss);
        row("del/add/del mix", dr.mix, hr.mix);
        row("iterate", dr.iter, hr.iter);
        row("scan", dr.scan, hr.scan);
        row("sample 16 (per call)", dr.sample, hr.sample);
        row("find+insert 2-step", dr.twostep, hr.twostep);
        row("bytes/entry (table)", dr.bytesPerEntry, hr.bytesPerEntry);
        /* Gate: lookups >=10% faster, nothing else >5% slower, memory not higher. */
        int pass = (hr.hit <= dr.hit * 0.90) && (hr.miss <= dr.miss * 0.90) &&
                   hr.bytesPerEntry <= dr.bytesPerEntry * 1.001 &&
                   hr.insert + hr.drain <= (dr.insert + dr.drain) * 1.05 &&
                   hr.mix <= dr.mix * 1.05 && hr.iter <= dr.iter * 1.05 &&
                   hr.scan <= dr.scan * 1.05 && hr.sample <= dr.sample * 1.05;
        printf("    gate: %s\n", pass ? "PASS" : "FAIL");
        if (!pass) fail++;
        benchSetFree(s);
    }
    return fail;
}

int dictbenchTest(int argc, char **argv, int flags) {
    UNUSED(argc); UNUSED(argv);
    uint8_t seed[16] = {0};
    dictSetHashFunctionSeed(seed);
    hashtableSetHashFunctionSeed(seed);
    dictSetResizeEnabled(DICT_RESIZE_ENABLE);
    hashtableSetResizePolicy(HASHTABLE_RESIZE_ALLOW);

    int fail = runSize(1000000);
    if (flags & REDIS_TEST_LARGE_MEMORY) fail += runSize(10000000);
    printf("\ndictbench: %d kind/size combination(s) failed the gate (informational)\n", fail);
    return 0;
}

#endif /* REDIS_TEST */
