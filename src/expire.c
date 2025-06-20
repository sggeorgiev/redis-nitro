/* Implementation of EXPIRE (keys with fixed time to live).
 *
 * ----------------------------------------------------------------------------
 *
 * Copyright (c) 2009-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 */

#include "server.h"
#include "expire.h"
#include "redisassert.h"

/* Define the EbucketsType for keys */
EbucketsType estoreBucketsType = {
    .onDeleteItem = NULL,
    .getExpireMeta = kvobjGetExpireMeta,   /* get ExpireMeta attached to each hash */
    .itemsAddrAreOdd = 0,                 /* Addresses of kvobj are even */
    .ebp.precision = 10,
    .ebp.keySize = EB_PRECISION2KEYSIZE(10 /*.precision*/),
    .isEbStack = 1,
};

/* Expiration store - wrapper around ebuckets for key expiration
 * Similar to kvstore, but using ebuckets instead of dict
 * In cluster mode, we have one ebuckets per slot
 * In non-cluster mode, we have a single ebuckets for the entire db
 *
 * The structure is defined in expire.h
 */

/* Create a new expiration store
 * type - The EbucketsType to use for the buckets
 * num_buckets_bits - The log2 of the number of buckets (0 for 1 bucket,
 *                    CLUSTER_SLOT_MASK_BITS for CLUSTER_SLOTS buckets)
 * flags - Configuration flags
 */
estore *estoreCreate(EbucketsType *type, int num_buckets_bits) {
    /* We can't support more than 2^16 buckets to be consistent with kvstore */
    assert(num_buckets_bits <= 16);

    estore *es = zmalloc(sizeof(estore));

    /* Store the bucket type */
    es->bucket_type = type;

    /* Calculate number of buckets based on num_buckets_bits */
    es->num_buckets_bits = num_buckets_bits;
    es->num_buckets = 1 << num_buckets_bits;

    /* Allocate the buckets array */
    es->buckets = zmalloc(sizeof(ebuckets) * es->num_buckets);

    /* Initialize all buckets */
    for (int i = 0; i < es->num_buckets; i++) {
        es->buckets[i] = ebCreate();
    }

    es->count = 0;
    return es;
}

/* Empty an expiration store (clear all entries but keep the structure) */
void estoreEmpty(estore *es) {
    if (es == NULL) return;

    for (int i = 0; i < es->num_buckets; i++) {
        ebDestroy(&es->buckets[i], es->bucket_type, NULL);
        es->buckets[i] = ebCreate();
    }

    es->count = 0;
}

/* Release an expiration store (free all memory) */
void estoreRelease(estore *es) {
    if (es == NULL) return;

    for (int i = 0; i < es->num_buckets; i++) {
        ebDestroy(&es->buckets[i], es->bucket_type, NULL);
    }

    zfree(es->buckets);
    zfree(es);
}

/* Remove kv from estore. Assumed kv is not null and exists in 'es'. */
void estoreRemove(estore *es, int slot, kvobj *kv) {
    if (es == NULL) return;

    ebuckets *bucket = estoreGetBucket(es, slot);
    if (ebRemove(bucket, es->bucket_type, kv)) {
        es->count--;
    }
}

/* Add kv to estore with the given expiration time */
void estoreAdd(estore *es, kvobj *kv, int slot, long long when) {
    if (es == NULL || kv == NULL) return;

    ebuckets *bucket = estoreGetBucket(es, slot);
    if (ebAdd(bucket, es->bucket_type, kv, when) == 0)
        es->count++;
}

void estoreIncrementalCascade(estore *es, uint64_t now, uint64_t maxCascade) {
    if (!server.cluster_enabled) {
        ebCascade(es->buckets + 0, es->bucket_type, now, maxCascade);
        return;
    }

    for(int i = 0; i < es->num_buckets && maxCascade > 0; i++) 
        maxCascade -= ebCascade(es->buckets + i, es->bucket_type, now, maxCascade);
}

void estoreCombineStats(ebucketsStats *from, ebucketsStats *into) {
    into->totalItems += from->totalItems;
    into->totalBuckets += from->totalBuckets;
    into->totalSegments += from->totalSegments;

    if (into->totalBuckets == 0) return;

    into->avgItemsPerBucket = into->totalItems / into->totalBuckets;
    into->avgItemsPerSegment = into->totalItems / into->totalSegments;
    into->avgSegPerBucket = into->totalSegments / into->totalBuckets;        
}

void estoreGetStats(estore *es, char *buf, size_t bufsize, int full) {
    if (!server.cluster_enabled) {
        ebucketsStats stats = {0}; /* must be zeroed */
        ebGetStats(es->buckets[0], es->bucket_type, &stats);
        ebGetStatsMsg(buf, bufsize, &stats, full);
    } else {
        ebucketsStats clusterStats = {0}; /* must be zeroed */
        for (int i = 0; i < es->num_buckets; i++) {
            ebucketsStats stats = {0}; /* must be zeroed */
            ebGetStats(es->buckets[i], es->bucket_type, &stats);
            estoreCombineStats(&stats, &clusterStats);
        }
        ebGetStatsMsg(buf, bufsize, &clusterStats, full);
    }
}

/* Get the total number of items in the expiration store */
unsigned long long estoreSize(estore *es) {
    if (es == NULL) return 0;
    return es->count;
}

/* Get the number of items in a specific slot */
unsigned long long estoreSlotSize(estore *es, int slot) {
    if (es == NULL) return 0;

    ebuckets *bucket = estoreGetBucket(es, slot);
    return ebGetTotalItems(*bucket, es->bucket_type);
}

/* Check if a specific slot is empty */
int estoreSlotIsEmpty(estore *es, int slot) {
    if (es == NULL) return 1;

    ebuckets *bucket = estoreGetBucket(es, slot);
    return ebIsEmpty(*bucket);
}

/*-----------------------------------------------------------------------------
 * Incremental collection of expired keys.
 *
 * When keys are accessed they are expired on-access. However we need a
 * mechanism in order to ensure keys are eventually removed when expired even
 * if no access is performed on them.
 *----------------------------------------------------------------------------*/

/* Constants table from pow(0.98, 1) to pow(0.98, 16).
 * Help calculating the db->avg_ttl. */
//static double avg_ttl_factor[16] = {0.98, 0.9604, 0.941192, 0.922368, 0.903921, 0.885842, 0.868126, 0.850763, 0.833748, 0.817073, 0.800731, 0.784717, 0.769022, 0.753642, 0.738569, 0.723798};
// TODO_MOTI: Support avg_ttl calculation

/* Helper function for the activeExpireCycle() function.
 * This function will try to expire the key-value entry that is stored in the 
 * hash table entry 'de' of the 'expires' hash table of a Redis database.
 *
 * If the key is found to be expired, it is removed from the database and
 * 1 is returned. Otherwise no operation is performed and 0 is returned.
 *
 * When a key is expired, server.stat_expiredkeys is incremented.
 *
 * The parameter 'now' is the current time in milliseconds as is passed
 * to the function to avoid too many gettimeofday() syscalls. */
int activeExpireCycleTryExpire(redisDb *db, kvobj *kv, long long now) {
    if (now < kvobjGetExpire(kv))
        return 0;

    enterExecutionUnit(1, 0);
    sds key = kvobjGetKey(kv);
    robj *keyobj = createStringObject(key,sdslen(key));
    deleteExpiredKeyAndPropagate(db,keyobj);
    decrRefCount(keyobj);
    exitExecutionUnit();
    /* Propagate the DEL command */
    postExecutionUnitOperations();
    return 1;
}

/* Try to expire a few timed out keys. The algorithm used is adaptive and
 * will use few CPU cycles if there are few expiring keys, otherwise
 * it will get more aggressive to avoid that too much memory is used by
 * keys that can be removed from the keyspace.
 *
 * Every expire cycle tests multiple databases: the next call will start
 * again from the next db. No more than CRON_DBS_PER_CALL databases are
 * tested at every iteration.
 *
 * The function can perform more or less work, depending on the "type"
 * argument. It can execute a "fast cycle" or a "slow cycle". The slow
 * cycle is the main way we collect expired cycles: this happens with
 * the "server.hz" frequency (usually 10 hertz).
 *
 * However the slow cycle can exit for timeout, since it used too much time.
 * For this reason the function is also invoked to perform a fast cycle
 * at every event loop cycle, in the beforeSleep() function. The fast cycle
 * will try to perform less work, but will do it much more often.
 *
 * The following are the details of the two expire cycles and their stop
 * conditions:
 *
 * If type is ACTIVE_EXPIRE_CYCLE_FAST the function will try to run a
 * "fast" expire cycle that takes no longer than ACTIVE_EXPIRE_CYCLE_FAST_DURATION
 * microseconds, and is not repeated again before the same amount of time.
 * The cycle will also refuse to run at all if the latest slow cycle did not
 * terminate because of a time limit condition.
 *
 * If type is ACTIVE_EXPIRE_CYCLE_SLOW, that normal expire cycle is
 * executed, where the time limit is a percentage of the REDIS_HZ period
 * as specified by the ACTIVE_EXPIRE_CYCLE_SLOW_TIME_PERC define. In the
 * fast cycle, the check of every database is interrupted once the number
 * of already expired keys in the database is estimated to be lower than
 * a given percentage, in order to avoid doing too much work to gain too
 * little memory.
 *
 * The configured expire "effort" will modify the baseline parameters in
 * order to do more work in both the fast and slow expire cycles.
 */

#define ACTIVE_EXPIRE_CYCLE_KEYS_PER_LOOP 20 /* Keys for each DB loop. */
#define ACTIVE_EXPIRE_CYCLE_FAST_DURATION 1000 /* Microseconds. */
#define ACTIVE_EXPIRE_CYCLE_SLOW_TIME_PERC 25 /* Max % of CPU to use. */
#define ACTIVE_EXPIRE_CYCLE_ACCEPTABLE_STALE 10 /* % of stale keys after which
                                                   we do extra efforts. */

#define HFE_DB_BASE_ACTIVE_EXPIRE_FIELDS_PER_SEC 10000

/* Data used by the expire dict scan callback. */
typedef struct {
    redisDb *db;
    long long now;
    unsigned long sampled; /* num keys checked */
    unsigned long expired; /* num keys expired */
    long long ttl_sum; /* sum of ttl for key with ttl not yet expired */
    int ttl_samples; /* num keys with ttl not yet expired */
} expireScanData;

void expireScanCallback(void *privdata, const dictEntry *de, dictEntryLink plink) {
    UNUSED(plink);
    kvobj *kv = dictGetKV(de);
    expireScanData *data = privdata;
    long long ttl  = kvobjGetExpire(kv) - data->now;
    if (activeExpireCycleTryExpire(data->db, kv, data->now)) {
        data->expired++;
    }
    if (ttl > 0) {
        /* We want the average TTL of keys yet not expired. */
        data->ttl_sum += ttl;
        data->ttl_samples++;
    }
    data->sampled++;
}

static inline int isExpiryDictValidForSamplingCb(dict *d) {
    long long numkeys = dictSize(d);
    unsigned long buckets = dictBuckets(d);
    /* When there are less than 1% filled buckets, sampling the key
     * space is expensive, so stop here waiting for better times...
     * The dictionary will be resized asap. */
    if (buckets > DICT_HT_INITIAL_SIZE && (numkeys * 100/buckets < 1)) {
        return C_ERR;
    }
    return C_OK;
}

/* Active expiration Cycle for hash-fields.
 *
 * Note that releasing fields is expected to be more predictable and rewarding
 * than releasing keys because it is stored in `ebuckets` DS which optimized for
 * active expiration and in addition the deletion of fields is simple to handle. */
static inline void activeExpireHashFieldCycle(int type) {
    /* Remember current db across calls */
    static unsigned int currentDb = 0;

    /* Tracks the count of fields actively expired for the current database.
     * This count continues as long as it fails to actively expire all expired
     * fields of currentDb, indicating a possible need to adjust the value of
     * maxToExpire. */
    static uint64_t activeExpirySequence = 0;
    /* Threshold for adjusting maxToExpire */
    const uint32_t EXPIRED_FIELDS_TH = 1000000;

    redisDb *db = server.db + currentDb;

    /* If db is empty, move to next db and return */
    if (ebIsEmpty(db->hexpires)) {
        activeExpirySequence = 0;
        currentDb = (currentDb + 1) % server.dbnum;
        return;
    }

    /* Maximum number of fields to actively expire on a single call */
    uint32_t maxToExpire = HFE_DB_BASE_ACTIVE_EXPIRE_FIELDS_PER_SEC / server.hz;

    /* If running for a while and didn't manage to active-expire all expired fields of
     * currentDb (i.e. activeExpirySequence becomes significant) then adjust maxToExpire */
    if ((activeExpirySequence > EXPIRED_FIELDS_TH) && (type == ACTIVE_EXPIRE_CYCLE_SLOW)) {
        /* maxToExpire is multiplied by a factor between 1 and 32, proportional to
         * the number of times activeExpirySequence exceeded EXPIRED_FIELDS_TH */
        uint64_t factor = activeExpirySequence / EXPIRED_FIELDS_TH;
        maxToExpire *= (factor<32) ? factor : 32;
    }

    if (hashTypeDbActiveExpire(db, maxToExpire) == maxToExpire) {
        /* active-expire reached maxToExpire limit */
        activeExpirySequence += maxToExpire;
    } else {
        /* Managed to active-expire all expired fields of currentDb */
        activeExpirySequence = 0;
        currentDb = (currentDb + 1) % server.dbnum;
    }
}

/* Callback for active expiration of keys */
static ExpireAction keyExpireCallback(eItem item, void *ctx) {
    redisDb *db = (redisDb *)ctx;
    kvobj *kv = (kvobj *)item;

    /* Get key name */
    sds keyname = kvobjGetKey(kv);
    robj *key = createStringObject(keyname, sdslen(keyname));

    /* Delete the key from the database */
    enterExecutionUnit(1, 0);
    deleteExpiredKeyAndPropagate(db, key);
    decrRefCount(key);
    exitExecutionUnit();
    postExecutionUnitOperations();

    server.stat_expiredkeys++;

    return ACT_REMOVE_EXP_ITEM;
}

/* Perform active expiration on keys using ebuckets */
unsigned int estoreActiveExpire(redisDb *db, unsigned int max_keys) {
    if (db == NULL || db->expiresNew == NULL) return 0;
    if (estoreSize(db->expiresNew) == 0) return 0;

    unsigned int keys_expired = 0;
    mstime_t now = mstime();

    /* In cluster mode, we need to process only slots that belong to this node */
    int start_slot = 0;
    int end_slot = db->expiresNew->num_buckets;

    UNUSED(start_slot);
    UNUSED(end_slot);

    if (server.cluster_enabled) {
//        /* Process only slots that belong to this node */
//        for (int i = start_slot; i < end_slot && keys_expired < max_keys; i++) {
//            /* Skip slots that don't belong to this node */
//            if (server.cluster && !clusterNodeCoversSlot(getMyClusterNode(), i)) {
//                continue;
//            }
//
//            /* Skip empty slots */
//            if (estoreSlotIsEmpty(db->expiresNew, i)) continue;
//
//            ebuckets *bucket = estoreGetBucket(db->expiresNew, i);
//
//            ExpireInfo info = {
//                .maxToExpire = max_keys - keys_expired,
//                .onExpireItem = keyExpireCallback,
//                .ctx = db,
//                .now = now,
//                .itemsExpired = 0
//            };
//
//            ebExpire(bucket, db->expiresNew->bucket_type, &info);
//            keys_expired += info.itemsExpired;
//        }
    } else {
        /* In non-cluster mode, we just have a single bucket */
        ebuckets *bucket = estoreGetBucket(db->expiresNew, 0);

        ExpireInfo info = {
            .maxToExpire = max_keys,
            .onExpireItem = keyExpireCallback,
            .ctx = db,
            .now = now,
            .itemsExpired = 0
        };

        ebExpire(bucket, db->expiresNew->bucket_type, &info);
        
        /* Move expired keys from secondary ebuckets */
        
        keys_expired = info.itemsExpired;
    }

    return keys_expired;
}

/* Get the appropriate bucket for a given slot */
ebuckets *estoreGetBucket(estore *es, int slot) {
    if (server.cluster_enabled)
        return &es->buckets[slot];
    else
        return &es->buckets[0];
}

size_t estoreMemUsage(estore *es) {
    if (es == NULL) return 0;

    // TODO_MOTI: Follow kvstoreMemUsage() to imp for estoreMemUsage()
//    size_t mem = sizeof(*es);
//    for (int i = 0; i < es->num_buckets; i++) {
//        mem += ebMemUsage(es->buckets[i], es->bucket_type);
//    }
    return 0;
}


void activeExpireCycle(int type) {
    /* Adjust the running parameters according to the configured expire
     * effort. The default effort is 1, and the maximum configurable effort
     * is 10. */
    unsigned long
    effort = server.active_expire_effort-1, /* Rescale from 0 to 9. */
    config_keys_per_loop = ACTIVE_EXPIRE_CYCLE_KEYS_PER_LOOP +
                           ACTIVE_EXPIRE_CYCLE_KEYS_PER_LOOP/4*effort,
    config_cycle_fast_duration = ACTIVE_EXPIRE_CYCLE_FAST_DURATION +
                                 ACTIVE_EXPIRE_CYCLE_FAST_DURATION/4*effort,
    config_cycle_slow_time_perc = ACTIVE_EXPIRE_CYCLE_SLOW_TIME_PERC +
                                  2*effort,
    config_cycle_acceptable_stale = ACTIVE_EXPIRE_CYCLE_ACCEPTABLE_STALE-
                                    effort;

    /* This function has some global state in order to continue the work
     * incrementally across calls. */
    static unsigned int current_db = 0; /* Next DB to test. */
    static int timelimit_exit = 0;      /* Time limit hit in previous call? */
    static long long last_fast_cycle = 0; /* When last fast cycle ran. */

    int j, iteration = 0;
    int dbs_per_call = CRON_DBS_PER_CALL;
    int dbs_performed = 0;
    long long start = ustime(), timelimit, elapsed;

    /* If 'expire' action is paused, for whatever reason, then don't expire any key.
     * Typically, at the end of the pause we will properly expire the key OR we
     * will have failed over and the new primary will send us the expire. */
    if (isPausedActionsWithUpdate(PAUSE_ACTION_EXPIRE)) return;

    if (type == ACTIVE_EXPIRE_CYCLE_FAST) {
        /* Don't start a fast cycle if the previous cycle did not exit
         * for time limit, unless the percentage of estimated stale keys is
         * too high. Also never repeat a fast cycle for the same period
         * as the fast cycle total duration itself. */
        if (!timelimit_exit &&
            server.stat_expired_stale_perc < config_cycle_acceptable_stale)
            return;

        if (start < last_fast_cycle + (long long)config_cycle_fast_duration*2)
            return;

        last_fast_cycle = start;
    }

    /* We usually should test CRON_DBS_PER_CALL per iteration, with
     * two exceptions:
     *
     * 1) Don't test more DBs than we have.
     * 2) If last time we hit the time limit, we want to scan all DBs
     * in this iteration, as there is work to do in some DB and we don't want
     * expired keys to use memory for too much time. */
    if (dbs_per_call > server.dbnum || timelimit_exit)
        dbs_per_call = server.dbnum;

    /* We can use at max 'config_cycle_slow_time_perc' percentage of CPU
     * time per iteration. Since this function gets called with a frequency of
     * server.hz times per second, the following is the max amount of
     * microseconds we can spend in this function. */
    timelimit = config_cycle_slow_time_perc*1000000/server.hz/100;
    timelimit_exit = 0;
    if (timelimit <= 0) timelimit = 1;

    if (type == ACTIVE_EXPIRE_CYCLE_FAST)
        timelimit = config_cycle_fast_duration; /* in microseconds. */

    /* Accumulate some global stats as we expire keys, to have some idea
     * about the number of keys that are already logically expired, but still
     * existing inside the database. */
    long total_sampled = 0;
    long total_expired = 0;

    /* Try to smoke-out bugs (server.also_propagate should be empty here) */
    serverAssert(server.also_propagate.numops == 0);

    /* Stop iteration when one of the following conditions is met:
     *
     * 1) We have checked a sufficient number of databases with expiration time.
     * 2) The time limit has been exceeded.
     * 3) All databases have been traversed. */
    for (j = 0; dbs_performed < dbs_per_call && timelimit_exit == 0 && j < server.dbnum; j++) {
        /* Scan callback data including expired and checked count per iteration. */
        expireScanData data;
        data.ttl_sum = 0;
        data.ttl_samples = 0;
        UNUSED(data);

        redisDb *db = server.db+(current_db % server.dbnum);
        data.db = db;

        //int db_done = 0; /* The scan of the current DB is done? */
        //int update_avg_ttl_times = 0, repeat = 0;

        /* Increment the DB now so we are sure if we run out of time
         * in the current DB we'll restart from the next. This allows to
         * distribute the time evenly across DBs. */
        current_db++;

        /* Interleaving hash-field expiration with key expiration. Better
         * call it before handling expired keys because HFE DS is optimized for
         * active expiration */
        activeExpireHashFieldCycle(type);

//        /* Check if we have keys to expire in the old dict-based structure */
//        if (kvstoreSize(db->expires)) {
//            dbs_performed++;
//
//            /* Continue to expire if at the end of the cycle there are still
//             * a big percentage of keys to expire, compared to the number of keys
//             * we scanned. The percentage, stored in config_cycle_acceptable_stale
//             * is not fixed, but depends on the Redis configured "expire effort". */
//            do {
//                unsigned long num;
//                iteration++;
//
//                /* If there is nothing to expire try next DB ASAP. */
//                if ((num = kvstoreSize(db->expires)) == 0) {
//                    db->avg_ttl = 0;
//                    break;
//                }
//                data.now = mstime();
//
//                /* The main collection cycle. Scan through keys among keys
//                 * with an expire set, checking for expired ones. */
//                data.sampled = 0;
//                data.expired = 0;
//
//                if (num > config_keys_per_loop)
//                    num = config_keys_per_loop;
//
//                /* Here we access the low level representation of the hash table
//                 * for speed concerns: this makes this code coupled with dict.c,
//                 * but it hardly changed in ten years.
//                 *
//                 * Note that certain places of the hash table may be empty,
//                 * so we want also a stop condition about the number of
//                 * buckets that we scanned. However scanning for free buckets
//                 * is very fast: we are in the cache line scanning a sequential
//                 * array of NULL pointers, so we can scan a lot more buckets
//                 * than keys in the same time. */
//                long max_buckets = num*20;
//                long checked_buckets = 0;
//
//                int origin_ttl_samples = data.ttl_samples;
//
//                while (data.sampled < num && checked_buckets < max_buckets) {
//                    db->expires_cursor = kvstoreScan(db->expires, db->expires_cursor, -1, expireScanCallback, isExpiryDictValidForSamplingCb, &data);
//                    if (db->expires_cursor == 0) {
//                        db_done = 1;
//                        break;
//                    }
//                    checked_buckets++;
//                }
//                total_expired += data.expired;
//                total_sampled += data.sampled;
//
//                /* If find keys with ttl not yet expired, we need to update the average TTL stats once. */
//                if (data.ttl_samples - origin_ttl_samples > 0) update_avg_ttl_times++;
//
//                /* We don't repeat the cycle for the current database if the db is done
//                 * for scanning or an acceptable number of stale keys (logically expired
//                 * but yet not reclaimed). */
//                repeat = db_done ? 0 : (data.sampled == 0 || (data.expired * 100 / data.sampled) > config_cycle_acceptable_stale);
//
//                /* We can't block forever here even if there are many keys to
//                 * expire. So after a given amount of microseconds return to the
//                 * caller waiting for the other active expire cycle. */
//                if ((iteration & 0xf) == 0 || !repeat) { /* Update the average TTL stats every 16 iterations or about to exit. */
//                    /* Update the average TTL stats for this database,
//                     * because this may reach the time limit. */
//                    if (data.ttl_samples) {
//                        long long avg_ttl = data.ttl_sum / data.ttl_samples;
//
//                        /* Do a simple running average with a few samples.
//                         * We just use the current estimate with a weight of 2%
//                         * and the previous estimate with a weight of 98%. */
//                        if (db->avg_ttl == 0) {
//                            db->avg_ttl = avg_ttl;
//                        } else {
//                            /* The origin code is as follow.
//                             * for (int i = 0; i < update_avg_ttl_times; i++) {
//                             *   db->avg_ttl = (db->avg_ttl/50)*49 + (avg_ttl/50);
//                             * }
//                             * We can convert the loop into a sum of a geometric progression.
//                             * db->avg_ttl = db->avg_ttl * pow(0.98, update_avg_ttl_times) +
//                             *                  avg_ttl / 50 * (pow(0.98, update_avg_ttl_times - 1) + ... + 1)
//                             *             = db->avg_ttl * pow(0.98, update_avg_ttl_times) +
//                             *                  avg_ttl * (1 - pow(0.98, update_avg_ttl_times))
//                             *             = avg_ttl +  (db->avg_ttl - avg_ttl) * pow(0.98, update_avg_ttl_times)
//                             * Notice that update_avg_ttl_times is between 1 and 16, we use a constant table
//                             * to accelerate the calculation of pow(0.98, update_avg_ttl_times).*/
//                            db->avg_ttl = avg_ttl + (db->avg_ttl - avg_ttl) * avg_ttl_factor[update_avg_ttl_times - 1] ;
//                        }
//                        update_avg_ttl_times = 0;
//                        data.ttl_sum = 0;
//                        data.ttl_samples = 0;
//                    }
//                    if ((iteration & 0xf) == 0) { /* check time limit every 16 iterations. */
//                        elapsed = ustime()-start;
//                        if (elapsed > timelimit) {
//                            timelimit_exit = 1;
//                            server.stat_expired_time_cap_reached_count++;
//                            break;
//                        }
//                    }
//                }
//            } while (repeat);
//        }

        /* Cascade items from L2 to L1 if needed */
        // TODO_MOTI: Fine tune estoreIncrementalCascade()
        uint64_t maxCascade = 20000 / server.hz; 
        estoreIncrementalCascade(db->expiresNew, commandTimeSnapshot(), maxCascade);

        /* Now check if we have keys to expire in the new ebuckets-based structure */
        if (estoreSize(db->expiresNew)) {
            dbs_performed++;
            unsigned int expired;
            /* Perform active expiration using ebuckets */
            do {
                iteration++;
                expired = estoreActiveExpire(db, config_keys_per_loop);
                total_expired += expired;
                if ((iteration & 0xf) == 0) { /* check time limit every 16 iterations. */
                    elapsed = ustime()-start;
                    if (elapsed > timelimit) {
                        timelimit_exit = 1;
                        server.stat_expired_time_cap_reached_count++;
                        break;
                    }
                }            
            } while (expired == config_keys_per_loop);
            
            

            /* We can't block forever here even if there are many keys to
             * expire. So after a given amount of milliseconds return to the
             * caller waiting for the other active expire cycle. */
            elapsed = ustime()-start;
            if (elapsed > timelimit) {
                timelimit_exit = 1;
                server.stat_expired_time_cap_reached_count++;
            }
        }
    }

    elapsed = ustime()-start;
    server.stat_expire_cycle_time_used += elapsed;
    latencyAddSampleIfNeeded("expire-cycle",elapsed/1000);

    /* Update our estimate of keys existing but yet to be expired.
     * Running average with this sample accounting for 5%. */
    double current_perc;
    if (total_sampled) {
        current_perc = (double)total_expired/total_sampled;
    } else
        current_perc = 0;
    server.stat_expired_stale_perc = (current_perc*0.05)+
                                     (server.stat_expired_stale_perc*0.95);
}

/*-----------------------------------------------------------------------------
 * Expires of keys created in writable slaves
 *
 * Normally slaves do not process expires: they wait the masters to synthesize
 * DEL operations in order to retain consistency. However writable slaves are
 * an exception: if a key is created in the slave and an expire is assigned
 * to it, we need a way to expire such a key, since the master does not know
 * anything about such a key.
 *
 * In order to do so, we track keys created in the slave side with an expire
 * set, and call the expireSlaveKeys() function from time to time in order to
 * reclaim the keys if they already expired.
 *
 * Note that the use case we are trying to cover here, is a popular one where
 * slaves are put in writable mode in order to compute slow operations in
 * the slave side that are mostly useful to actually read data in a more
 * processed way. Think at sets intersections in a tmp key, with an expire so
 * that it is also used as a cache to avoid intersecting every time.
 *
 * This implementation is currently not perfect but a lot better than leaking
 * the keys as implemented in 3.2.
 *----------------------------------------------------------------------------*/

/* The dictionary where we remember key names and database ID of keys we may
 * want to expire from the slave. Since this function is not often used we
 * don't even care to initialize the database at startup. We'll do it once
 * the feature is used the first time, that is, when rememberSlaveKeyWithExpire()
 * is called.
 *
 * The dictionary has an SDS string representing the key as the hash table
 * key, while the value is a 64 bit unsigned integer with the bits corresponding
 * to the DB where the keys may exist set to 1. Currently the keys created
 * with a DB id > 63 are not expired, but a trivial fix is to set the bitmap
 * to the max 64 bit unsigned value when we know there is a key with a DB
 * ID greater than 63, and check all the configured DBs in such a case. */
dict *slaveKeysWithExpire = NULL;

/* Check the set of keys created by the master with an expire set in order to
 * check if they should be evicted. */
void expireSlaveKeys(void) {
    if (slaveKeysWithExpire == NULL ||
        dictSize(slaveKeysWithExpire) == 0) return;

    int cycles = 0, noexpire = 0;
    mstime_t start = mstime();
    while(1) {
        dictEntry *de = dictGetRandomKey(slaveKeysWithExpire);
        sds keyname = dictGetKey(de);
        uint64_t dbids = dictGetUnsignedIntegerVal(de);
        uint64_t new_dbids = 0;

        /* Check the key against every database corresponding to the
         * bits set in the value bitmap. */
        int dbid = 0;
        while(dbids && dbid < server.dbnum) {
            if ((dbids & 1) != 0) {
                redisDb *db = server.db+dbid;
                kvobj *kv = dbFind(db, keyname);
                int expired = kv && activeExpireCycleTryExpire(server.db+dbid, kv, start);

                /* If the key was not expired in this DB, we need to set the
                 * corresponding bit in the new bitmap we set as value.
                 * At the end of the loop if the bitmap is zero, it means we
                 * no longer need to keep track of this key. */
                if (kv && !expired) {
                    noexpire++;
                    new_dbids |= (uint64_t)1 << dbid;
                }
            }
            dbid++;
            dbids >>= 1;
        }

        /* Set the new bitmap as value of the key, in the dictionary
         * of keys with an expire set directly in the writable slave. Otherwise
         * if the bitmap is zero, we no longer need to keep track of it. */
        if (new_dbids)
            dictSetUnsignedIntegerVal(de,new_dbids);
        else
            dictDelete(slaveKeysWithExpire,keyname);

        /* Stop conditions: found 3 keys we can't expire in a row or
         * time limit was reached. */
        cycles++;
        if (noexpire > 3) break;
        if ((cycles % 64) == 0 && mstime()-start > 1) break;
        if (dictSize(slaveKeysWithExpire) == 0) break;
    }
}

/* Track keys that received an EXPIRE or similar command in the context
 * of a writable slave. */
void rememberSlaveKeyWithExpire(redisDb *db, sds key) {
    if (slaveKeysWithExpire == NULL) {
        static dictType dt = {
            dictSdsHash,                /* hash function */
            NULL,                       /* key dup */
            NULL,                       /* val dup */
            dictSdsKeyCompare,          /* key compare */
            dictSdsDestructor,          /* key destructor */
            NULL,                       /* val destructor */
            NULL                        /* allow to expand */
        };
        slaveKeysWithExpire = dictCreate(&dt);
    }
    if (db->id > 63) return;

    dictEntry *de = dictAddOrFind(slaveKeysWithExpire, key);
    /* If the entry was just created, set it to a copy of the SDS string
     * representing the key: we don't want to need to take those keys
     * in sync with the main DB. The keys will be removed by expireSlaveKeys()
     * as it scans to find keys to remove. */
    if (dictGetKey(de) == key) {
        dictSetKey(slaveKeysWithExpire, de, sdsdup(key));
        dictSetUnsignedIntegerVal(de,0);
    }

    uint64_t dbids = dictGetUnsignedIntegerVal(de);
    dbids |= (uint64_t)1 << db->id;
    dictSetUnsignedIntegerVal(de,dbids);
}

/* Return the number of keys we are tracking. */
size_t getSlaveKeyWithExpireCount(void) {
    if (slaveKeysWithExpire == NULL) return 0;
    return dictSize(slaveKeysWithExpire);
}

/* Remove the keys in the hash table. We need to do that when data is
 * flushed from the server. We may receive new keys from the master with
 * the same name/db and it is no longer a good idea to expire them.
 *
 * Note: technically we should handle the case of a single DB being flushed
 * but it is not worth it since anyway race conditions using the same set
 * of key names in a writable slave and in its master will lead to
 * inconsistencies. This is just a best-effort thing we do. */
void flushSlaveKeysWithExpireList(void) {
    if (slaveKeysWithExpire) {
        dictRelease(slaveKeysWithExpire);
        slaveKeysWithExpire = NULL;
    }
}

int checkAlreadyExpired(long long when) {
    /* EXPIRE with negative TTL, or EXPIREAT with a timestamp into the past
     * should never be executed as a DEL when load the AOF or in the context
     * of a slave instance.
     *
     * Instead we add the already expired key to the database with expire time
     * (possibly in the past) and wait for an explicit DEL from the master. */
    return (when <= commandTimeSnapshot() && !server.loading && !server.masterhost);
}

#define EXPIRE_NX (1<<0)
#define EXPIRE_XX (1<<1)
#define EXPIRE_GT (1<<2)
#define EXPIRE_LT (1<<3)

/* Parse additional flags of expire commands
 *
 * Supported flags:
 * - NX: set expiry only when the key has no expiry
 * - XX: set expiry only when the key has an existing expiry
 * - GT: set expiry only when the new expiry is greater than current one
 * - LT: set expiry only when the new expiry is less than current one */
int parseExtendedExpireArgumentsOrReply(client *c, int *flags) {
    int nx = 0, xx = 0, gt = 0, lt = 0;

    int j = 3;
    while (j < c->argc) {
        char *opt = c->argv[j]->ptr;
        if (!strcasecmp(opt,"nx")) {
            *flags |= EXPIRE_NX;
            nx = 1;
        } else if (!strcasecmp(opt,"xx")) {
            *flags |= EXPIRE_XX;
            xx = 1;
        } else if (!strcasecmp(opt,"gt")) {
            *flags |= EXPIRE_GT;
            gt = 1;
        } else if (!strcasecmp(opt,"lt")) {
            *flags |= EXPIRE_LT;
            lt = 1;
        } else {
            addReplyErrorFormat(c, "Unsupported option %s", opt);
            return C_ERR;
        }
        j++;
    }

    if ((nx && xx) || (nx && gt) || (nx && lt)) {
        addReplyError(c, "NX and XX, GT or LT options at the same time are not compatible");
        return C_ERR;
    }

    if (gt && lt) {
        addReplyError(c, "GT and LT options at the same time are not compatible");
        return C_ERR;
    }

    return C_OK;
}

/*-----------------------------------------------------------------------------
 * Expires Commands
 *----------------------------------------------------------------------------*/

/* This is the generic command implementation for EXPIRE, PEXPIRE, EXPIREAT
 * and PEXPIREAT. Because the command second argument may be relative or absolute
 * the "basetime" argument is used to signal what the base time is (either 0
 * for *AT variants of the command, or the current time for relative expires).
 *
 * unit is either UNIT_SECONDS or UNIT_MILLISECONDS, and is only used for
 * the argv[2] parameter. The basetime is always specified in milliseconds.
 *
 * Additional flags are supported and parsed via parseExtendedExpireArguments */
void expireGenericCommand(client *c, long long basetime, int unit) {
    robj *key = c->argv[1], *param = c->argv[2];
    long long when; /* unix time in milliseconds when the key will expire. */
    long long current_expire = -1;
    int flag = 0;

    /* checking optional flags */
    if (parseExtendedExpireArgumentsOrReply(c, &flag) != C_OK) {
        return;
    }

    if (getLongLongFromObjectOrReply(c, param, &when, NULL) != C_OK)
        return;

    /* EXPIRE allows negative numbers, but we can at least detect an
     * overflow by either unit conversion or basetime addition. */
    if (unit == UNIT_SECONDS) {
        if (when > LLONG_MAX / 1000 || when < LLONG_MIN / 1000) {
            addReplyErrorExpireTime(c);
            return;
        }
        when *= 1000;
    }

    if (when > LLONG_MAX - basetime) {
        addReplyErrorExpireTime(c);
        return;
    }
    when += basetime;

    /* No key, return zero. */
    kvobj *kv = lookupKeyWrite(c->db,key); 
    if (kv == NULL) {
        addReply(c,shared.czero);
        return;
    }

    if (flag) {
        current_expire = kvobjGetExpire(kv);

        /* NX option is set, check current expiry */
        if (flag & EXPIRE_NX) {
            if (current_expire != -1) {
                addReply(c,shared.czero);
                return;
            }
        }

        /* XX option is set, check current expiry */
        if (flag & EXPIRE_XX) {
            if (current_expire == -1) {
                /* reply 0 when the key has no expiry */
                addReply(c,shared.czero);
                return;
            }
        }

        /* GT option is set, check current expiry */
        if (flag & EXPIRE_GT) {
            /* When current_expire is -1, we consider it as infinite TTL,
             * so expire command with gt always fail the GT. */
            if (when <= current_expire || current_expire == -1) {
                /* reply 0 when the new expiry is not greater than current */
                addReply(c,shared.czero);
                return;
            }
        }

        /* LT option is set, check current expiry */
        if (flag & EXPIRE_LT) {
            /* When current_expire -1, we consider it as infinite TTL,
             * but 'when' can still be negative at this point, so if there is
             * an expiry on the key and it's not less than current, we fail the LT. */
            if (current_expire != -1 && when >= current_expire) {
                /* reply 0 when the new expiry is not less than current */
                addReply(c,shared.czero);
                return;
            }
        }
    }

    if (checkAlreadyExpired(when)) {
        robj *aux;

        int deleted = dbGenericDelete(c->db,key,server.lazyfree_lazy_expire,DB_FLAG_KEY_EXPIRED);
        serverAssertWithInfo(c,key,deleted);
        server.dirty++;

        /* Replicate/AOF this as an explicit DEL or UNLINK. */
        aux = server.lazyfree_lazy_expire ? shared.unlink : shared.del;
        rewriteClientCommandVector(c,2,aux,key);
        signalModifiedKey(c,c->db,key);
        notifyKeyspaceEvent(NOTIFY_GENERIC,"del",key,c->db->id);
        addReply(c, shared.cone);
        return;
    } else {
        kv = setExpire(c,c->db,key,when); /* might realloc kv */
        addReply(c,shared.cone);
        /* Propagate as PEXPIREAT millisecond-timestamp
         * Only rewrite the command arg if not already PEXPIREAT */
        if (c->cmd->proc != pexpireatCommand) {
            rewriteClientCommandArgument(c,0,shared.pexpireat);
        }

        /* Avoid creating a string object when it's the same as argv[2] parameter  */
        if (basetime != 0 || unit == UNIT_SECONDS) {
            robj *when_obj = createStringObjectFromLongLong(when);
            rewriteClientCommandArgument(c,2,when_obj);
            decrRefCount(when_obj);
        }

        signalModifiedKey(c,c->db,key);
        notifyKeyspaceEvent(NOTIFY_GENERIC,"expire",key,c->db->id);
        server.dirty++;
        return;
    }
}

/* EXPIRE key seconds [ NX | XX | GT | LT] */
void expireCommand(client *c) {
    expireGenericCommand(c,commandTimeSnapshot(),UNIT_SECONDS);
}

/* EXPIREAT key unix-time-seconds [ NX | XX | GT | LT] */
void expireatCommand(client *c) {
    expireGenericCommand(c,0,UNIT_SECONDS);
}

/* PEXPIRE key milliseconds [ NX | XX | GT | LT] */
void pexpireCommand(client *c) {
    expireGenericCommand(c,commandTimeSnapshot(),UNIT_MILLISECONDS);
}

/* PEXPIREAT key unix-time-milliseconds [ NX | XX | GT | LT] */
void pexpireatCommand(client *c) {
    expireGenericCommand(c,0,UNIT_MILLISECONDS);
}

/* Implements TTL, PTTL, EXPIRETIME and PEXPIRETIME */
void ttlGenericCommand(client *c, int output_ms, int output_abs) {
    long long expire, ttl = -1;

    /* If the key does not exist at all, return -2 */
    kvobj *kv = lookupKeyReadWithFlags(c->db,c->argv[1],LOOKUP_NOTOUCH);
    if (kv == NULL) {
        addReplyLongLong(c,-2);
        return;
    }

    /* The key exists. Return -1 if it has no expire, or the actual
     * TTL value otherwise. */
    expire = kvobjGetExpire(kv);
    if (expire != -1) {
        ttl = output_abs ? expire : expire-commandTimeSnapshot();
        if (ttl < 0) ttl = 0;
    }
    if (ttl == -1) {
        addReplyLongLong(c,-1);
    } else {
        addReplyLongLong(c,output_ms ? ttl : ((ttl+500)/1000));
    }
}

/* TTL key */
void ttlCommand(client *c) {
    ttlGenericCommand(c, 0, 0);
}

/* PTTL key */
void pttlCommand(client *c) {
    ttlGenericCommand(c, 1, 0);
}

/* EXPIRETIME key */
void expiretimeCommand(client *c) {
    ttlGenericCommand(c, 0, 1);
}

/* PEXPIRETIME key */
void pexpiretimeCommand(client *c) {
    ttlGenericCommand(c, 1, 1);
}

/* PERSIST key */
void persistCommand(client *c) {
    kvobj *kv;
    if ((kv = lookupKeyWrite(c->db,c->argv[1])) != NULL) {
        if (removeExpire(c->db,c->argv[1], kv)) {
            signalModifiedKey(c,c->db,c->argv[1]);
            notifyKeyspaceEvent(NOTIFY_GENERIC,"persist",c->argv[1],c->db->id);
            addReply(c,shared.cone);
            server.dirty++;
        } else {
            addReply(c,shared.czero);
        }
    } else {
        addReply(c,shared.czero);
    }
}

/* TOUCH key1 [key2 key3 ... keyN] */
void touchCommand(client *c) {
    int touched = 0;
    for (int j = 1; j < c->argc; j++)
        if (lookupKeyRead(c->db,c->argv[j]) != NULL) touched++;
    addReplyLongLong(c,touched);
}

#ifdef REDIS_TEST
#include <stdio.h>
#include "testhelp.h"

#define TEST(name) printf("test — %s\n", name);
typedef struct TestKVObj {
    int index;
    ExpireMeta mexpire;
} TestKVObj;

ExpireMeta *getTestKVObjExpireMeta(const void *item) {
    return &((TestKVObj *)item)->mexpire;
}

void deleteTestKVObjCallback(eItem item, void *ctx) {
    UNUSED(ctx);
    UNUSED(item);
}

EbucketsType testEbType = {
    .getExpireMeta = getTestKVObjExpireMeta,
    .onDeleteItem = deleteTestKVObjCallback,
    .itemsAddrAreOdd = 0,
    .ebp.precision = 0,
    .ebp.keySize = EB_PRECISION2KEYSIZE(0 /*.precision*/),
};

/* ./redis-server test expire */
int expireTest(int argc, char **argv, int flags) {
    UNUSED(argc);
    UNUSED(argv);
    UNUSED(flags);

    TEST("Create estore with pre-allocated buckets") {
        int num_bits = 3; // 2^3 = 8 buckets
        int num_buckets = 1 << num_bits;

        estore *es = estoreCreate(&testEbType, num_bits);

        for (int i = 0; i < num_buckets; i++) {
            ebuckets b = es->buckets[i];
            assert(b == NULL); // ebCreate returns NULL
        }

        assert(es->num_buckets == num_buckets);
        assert(es->num_buckets_bits == num_bits);
        assert(es->bucket_type == &testEbType);
        assert(es->count == 0);

        estoreRelease(es); // Clean up
    }

    TEST("estoreGetBucket with cluster mode OFF") {
        server.cluster_enabled = 0; // Non-cluster mode

        estore *es = estoreCreate(&testEbType, 2); // 2^2 = 4 buckets
        ebuckets *expected = &es->buckets[0];

        for (int i = 0; i < 4; i++) {
            ebuckets *b = estoreGetBucket(es, i);
            assert(b == expected); // All slots should return the first bucket in non-cluster mode
        }

        estoreRelease(es);
    }

    TEST("estoreGetBucket with cluster mode ON") {
        server.cluster_enabled = 1; // Cluster mode enabled

        estore *es = estoreCreate(&testEbType, 2); // 4 buckets

        for (int i = 0; i < 4; i++) {
            ebuckets *b = estoreGetBucket(es, i);
            assert(b == &es->buckets[i]); // Each slot should map to its corresponding bucket
        }

        estoreRelease(es);
    }

    TEST("estoreAdd inserts kvobj at correct address in expected bucket") {
        server.cluster_enabled = 0;

        estore *es = estoreCreate(&testEbType, 2); // 4 buckets (slots 0-3)
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));
        long long expire_time = 1;
        int slot = 1;

        assert(estoreSize(es) == 0);

        estoreAdd(es, (kvobj*)kv, slot, expire_time);
        assert(estoreSize(es) == 1);

        // Get the expected bucket
        ebuckets *bucket = estoreGetBucket(es, slot);
        EbucketsIterator iter;
        int found = 0;

        ebStart(&iter, *bucket, es->bucket_type);
        while (iter.currItem) {
            if (iter.currItem == kv) {
                found = 1;
                break;
            }
            ebNext(&iter);
        }
        ebStop(&iter);

        assert(found == 1); // kvobj was found by pointer

        zfree(kv);
        estoreRelease(es);
    }

    TEST("estoreAdd handles NULL estore or kvobj gracefully") {
        estore *es = estoreCreate(&testEbType, 2); // 4 buckets (slots 0-3)
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));

        long long expire_time = 1;
        unsigned long long original_count = estoreSize(es);

        estoreAdd(NULL, (kvobj*)kv, 0, expire_time);
        estoreAdd(es, NULL, 0, expire_time);

        assert(estoreSize(es) == original_count);

        zfree(kv);
        estoreRelease(es);
    }

    TEST("estoreEmpty clears all buckets and resets count") {
        server.cluster_enabled = 1;

        // Create estore with 4 buckets
        estore *es = estoreCreate(&testEbType, 2); // 2^2 = 4 buckets

        TestKVObj *kv[4];
        // Add one TestKVObj to each bucket
        for (int slot = 0; slot < 4; slot++) {
            kv[slot] = zmalloc(sizeof(TestKVObj));
            estoreAdd(es, (kvobj*)kv[slot], slot, 1);
        }

        // Ensure total count is 4
        assert(estoreSize(es) == 4);

        // Confirm all buckets are not empty
        for (int i = 0; i < es->num_buckets; i++) {
            ebuckets *bucket = &es->buckets[i];
            assert(!ebIsEmpty(*bucket));
        }

        // Call estoreEmpty
        estoreEmpty(es);

        // Confirm count is zero
        assert(estoreSize(es) == 0);

        // Confirm all buckets are empty
        for (int i = 0; i < es->num_buckets; i++) {
            ebuckets *bucket = &es->buckets[i];
            assert(ebIsEmpty(*bucket));
        }

        estoreRelease(es);
    }

    TEST("estoreEmpty with NULL estore is a no-op") {
        // This should not crash or produce side effects
        estoreEmpty(NULL);
        // No assertion needed — if it doesn't crash, it passes
    }

    TEST("estoreRelease with NULL does nothing") {
        // Should not crash or do anything
        estoreRelease(NULL);
    }

    TEST("estoreRelease frees all resources for valid estore") {
        // Setup
        estore *es = estoreCreate(&testEbType, 2); // 4 buckets

        // Add an item to ensure some real content exists
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv, 1, 1);

        // Just ensure it was added (not strictly necessary, but good check)
        assert(estoreSize(es) == 1);

        // Release the estore
        estoreRelease(es);

        // We can't assert anything post-release (pointer is now invalid),
        // but Valgrind or AddressSanitizer will catch use-after-free if any.
        zfree(kv); // Caller owns kvobj memory separately
    }

    TEST("estoreRemove removes kvobj and decrements count") {
        server.cluster_enabled = 0;

        // Create estore with 1 bucket
        estore *es = estoreCreate(&testEbType, 1);

        // Allocate and add a test kvobj
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));
        long long expire_time = 1;

        estoreAdd(es, (kvobj*)kv, 0, expire_time);
        assert(estoreSize(es) == 1);

        // Remove the kvobj
        estoreRemove(es, 0, (kvobj*)kv);
        assert(estoreSize(es) == 0);

        // Cleanup
        zfree(kv);
        estoreRelease(es);
    }

    TEST("estoreRemove with NULL estore does nothing") {
        // Should not crash or do anything
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));
        estoreRemove(NULL, 0, (kvobj*)kv);
        zfree(kv);
    }

    TEST("estoreSlotSize reports correct number of items in a slot") {
        server.cluster_enabled = 1;
        estore *es = estoreCreate(&testEbType, 2); // 4 slots

        // Slot 1 should initially be 0
        assert(estoreSlotSize(es, 1) == 0);

        // Add two kvobj to slot 1
        TestKVObj *kv1 = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv1, 1, 1);
        
        TestKVObj *kv2 = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv2, 1, 2);

        // Add one kvobj to slot 2
        TestKVObj *kv3 = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv3, 2, 1);

        // Now slot 1 should report size 2
        assert(estoreSlotSize(es, 1) == 2);

        // Slot 2 should report size 1
        assert(estoreSlotSize(es, 2) == 1);

        // Other slots still 0
        assert(estoreSlotSize(es, 0) == 0);
        assert(estoreSlotSize(es, 3) == 0);

        zfree(kv1);
        zfree(kv2);
        zfree(kv3);
        estoreRelease(es);
    }

    TEST("estoreSlotSize returns 0 when estore is NULL") {
        assert(estoreSlotSize(NULL, 0) == 0);
        assert(estoreSlotSize(NULL, 1) == 0);
    }

    TEST("estoreSlotIsEmpty reflects whether a slot has items") {
        server.cluster_enabled = 1;
        estore *es = estoreCreate(&testEbType, 2); // 4 slots

        // Slot 2 should be empty initially
        assert(estoreSlotIsEmpty(es, 2) == 1);

        // Add one kvobj to slot 2
        TestKVObj *kv = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv, 2, 1234567890);

        // Now slot 2 should not be empty
        assert(estoreSlotIsEmpty(es, 2) == 0);

        zfree(kv);
        estoreRelease(es);
    }

    TEST("estoreSlotIsEmpty returns 1 when estore is NULL") {
        assert(estoreSlotIsEmpty(NULL, 0) == 1);
        assert(estoreSlotIsEmpty(NULL, 3) == 1);
    }

    TEST("estoreCombineStats correctly combines statistics") {
        ebucketsStats from = {
            .totalItems = 20,
            .totalBuckets = 4,
            .totalSegments = 2,
            .avgItemsPerBucket = 0, // will be recalculated
            .avgItemsPerSegment = 0,
            .avgSegPerBucket = 0
        };

        ebucketsStats into = {
            .totalItems = 10,
            .totalBuckets = 2,
            .totalSegments = 1,
            .avgItemsPerBucket = 0,
            .avgItemsPerSegment = 0,
            .avgSegPerBucket = 0
        };

        estoreCombineStats(&from, &into);

        // Validate combined totals
        assert(into.totalItems == 30);
        assert(into.totalBuckets == 6);
        assert(into.totalSegments == 3);

        // Validate recomputed averages
        assert(into.avgItemsPerBucket == 5);   // 30 / 6
        assert(into.avgItemsPerSegment == 10); // 30 / 3
        assert(into.avgSegPerBucket == 0);     // 3 / 6 = 0 (integer division)

        // Edge case: totalBuckets = 0
        ebucketsStats zeroBucketsFrom = {
            .totalItems = 5,
            .totalBuckets = 0,
            .totalSegments = 2
        };

        ebucketsStats zeroBucketsInto = {
            .totalItems = 0,
            .totalBuckets = 0,
            .totalSegments = 0
        };

        estoreCombineStats(&zeroBucketsFrom, &zeroBucketsInto);

        // Should add values, but skip average calculations
        assert(zeroBucketsInto.totalItems == 5);
        assert(zeroBucketsInto.totalBuckets == 0);
        assert(zeroBucketsInto.totalSegments == 2);
        assert(zeroBucketsInto.avgItemsPerBucket == 0);
        assert(zeroBucketsInto.avgItemsPerSegment == 0);
        assert(zeroBucketsInto.avgSegPerBucket == 0);
    }

    TEST("estoreGetStats returns correct stats in non-clustered mode") {
        server.cluster_enabled = 0;

        estore *es = estoreCreate(&testEbType, 1); // only one bucket

        TestKVObj *kv1 = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv1, 0, 1);

        TestKVObj *kv2 = zmalloc(sizeof(TestKVObj));
        estoreAdd(es, (kvobj*)kv2, 0, 1);

        char buf[1024];
        memset(buf, 0, sizeof(buf));

        estoreGetStats(es, buf, sizeof(buf), 1);

        // Validate that output includes expected stats
        assert(strstr(buf, " total items: 2") != NULL);
        assert(strstr(buf, " total buckets: 1") != NULL);
        assert(strstr(buf, " total segments: 1") != NULL);

        assert(strstr(buf, " avg items per bucket: 2") != NULL);
        assert(strstr(buf, " avg items per segment: 2") != NULL);
        assert(strstr(buf, " avg segments per bucket: 1") != NULL);

        zfree(kv1);
        zfree(kv2);
        estoreRelease(es);
    }

    TEST("estoreGetStats returns combined stats in clustered mode") {
        server.cluster_enabled = 1;

        estore *es = estoreCreate(&testEbType, 2); // two buckets

        // Add objects to different buckets
        TestKVObj *kv1 = zmalloc(sizeof(TestKVObj));
        TestKVObj *kv2 = zmalloc(sizeof(TestKVObj));
        TestKVObj *kv3 = zmalloc(sizeof(TestKVObj));
        TestKVObj *kv4 = zmalloc(sizeof(TestKVObj));

        estoreAdd(es, (kvobj*)kv1, 0, 1); // bucket 0
        estoreAdd(es, (kvobj*)kv2, 1, 1); // bucket 1
        estoreAdd(es, (kvobj*)kv3, 1, 2); // bucket 1
        estoreAdd(es, (kvobj*)kv4, 0, 2); // bucket 0

        char buf[1024];
        memset(buf, 0, sizeof(buf));

        estoreGetStats(es, buf, sizeof(buf), 1); // full = 1 for verbose output

        // Validate that combined values are present
        assert(strstr(buf, " total items: 4") != NULL);
        assert(strstr(buf, " total buckets: 2") != NULL);
        assert(strstr(buf, " total segments: 2") != NULL);

        assert(strstr(buf, " avg items per bucket: 2") != NULL);
        assert(strstr(buf, " avg items per segment: 2") != NULL);
        assert(strstr(buf, " avg segments per bucket: 1") != NULL);

        zfree(kv1);
        zfree(kv2);
        zfree(kv3);
        zfree(kv4);
        estoreRelease(es);
    }
    
    return 0;
}
#endif

