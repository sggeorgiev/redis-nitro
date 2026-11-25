/*
 * This file utilizes prefetching keys and data for multiple commands in a batch,
 * to improve performance by amortizing memory access costs across multiple operations.
 *
 * Copyright (c) 2025-Present, Redis Ltd. and contributors.
 * All rights reserved.
 *
 * Copyright (c) 2024-present, Valkey contributors.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * Portions of this file are available under BSD3 terms; see REDISCONTRIBUTIONS for more information.
 */

#include "server.h"
#include "memory_prefetch.h"
#include "dict.h"

/* --------------------------------------------------------------------------
 * Dict prefetching
 * -------------------------------------------------------------------------- */

/* dictPrefetcher drives a batch of dict lookups, each one a dictPrefetchState
 * advanced by dictPrefetchNext() (see dict.c), yielding to the next in-flight
 * lookup each time a prefetch is issued — so one lookup's memory stall
 * overlaps another's work. The lookups are fully dict-pure: any value
 * payload prefetching is delegated to the dictType->prefetchEntryValue
 * callback of each key's dict. The same prefetcher is
 * used by both the cross-command batch path and the intra-command
 * dictPrefetchKeys() API. */
typedef struct dictPrefetcher {
    size_t pos;                  /* Cursor into active[]; advances on each prefetch issue */
    size_t nactive;              /* Number of in-flight key lookups (entries in active[]) */
    dictPrefetchState *lookups;  /* Per-key lookup state, capacity == max_keys */
    uint32_t *active;            /* Indexes of the in-flight lookups, capacity == max_keys */
    size_t max_keys;             /* Capacity of lookups[] and active[] */
} dictPrefetcher;

/* Issue a software prefetch for `addr`, then yield to the next in-flight
 * lookup by advancing the cursor. */
static inline void dictPrefetchAdvance(dictPrefetcher *p, void *addr) {
    redis_prefetch_read(addr);
    if (++p->pos >= p->nactive) p->pos = 0;
}

/* Retire the current lookup. It is swap-removed from active[], so the cursor
 * now names the next in-flight lookup: finished lookups are never rescanned. */
static inline void dictPrefetchMarkDone(dictPrefetcher *p) {
    p->active[p->pos] = p->active[--p->nactive];
    if (p->pos >= p->nactive) p->pos = 0;
    server.stat_total_prefetch_entries++;
}

/* Allocate the per-key lookup array. The prefetcher can then be reused across
 * many batches by repeated dictPrefetcherReset / dictPrefetcherRun calls. */
static void dictPrefetcherInit(dictPrefetcher *p, size_t max_keys) {
    p->lookups = zcalloc(max_keys * sizeof(dictPrefetchState));
    p->active = zcalloc(max_keys * sizeof(uint32_t));
    p->max_keys = max_keys;
}

static void dictPrefetcherFree(dictPrefetcher *p) {
    zfree(p->lookups);
    zfree(p->active);
    p->lookups = NULL;
    p->active = NULL;
    p->max_keys = 0;
}

/* Configure the prefetcher for a single batch and seed every lookup's
 * starting state. The dicts and keys must remain valid until
 * dictPrefetcherRun returns; only the pointers are stored. If `find` is set,
 * the lookups also find their keys, see dictPrefetchFound(). */
static void dictPrefetcherReset(dictPrefetcher *p, dict **dicts, void **keys, size_t nkeys, int find) {
    serverAssert(nkeys <= p->max_keys);
    p->pos = 0;

    size_t nactive = 0;
    for (size_t i = 0; i < nkeys; i++) {
        dictPrefetchInit(&p->lookups[i], dicts[i], keys[i], find);
        if (!p->lookups[i].done) p->active[nactive++] = (uint32_t)i;
    }
    p->nactive = nactive;
}

/* Drive the prefetch state machine across all dict lookups until every lookup
 * is done.
 *
 * Conceptually each dict lookup is a dictFind broken into stages (see
 * dictPrefetchNext()):
 *   bucket → tag match → entry → entry value payload
 * If the key is not found in a bucket, the lookup goes on with its child
 * bucket, and then retries against ht[1] if the dict is mid-rehash.
 *
 * Instead of waiting for each stage's memory access to complete, the FSM
 * issues a prefetch and yields to another in-flight lookup, hiding the
 * memory access latency.
 *
 * Any prefetching of the entry's value payload (e.g. kv->ptr for a RAW
 * string) is delegated to dictType->prefetchEntryValue. */
static void dictPrefetcherRun(dictPrefetcher *p) {
    while (p->nactive) {
        dictPrefetchState *lk = &p->lookups[p->active[p->pos]];
        void *addr = dictPrefetchNext(lk);
        if (lk->done) {
            /* The value payload: the cursor already names the next lookup. */
            if (addr) redis_prefetch_read(addr);
            dictPrefetchMarkDone(p);
        } else if (addr) {
            dictPrefetchAdvance(p, addr);
        }
    }
}

/* --------------------------------------------------------------------------
 * Intra-command prefetch API
 * --------------------------------------------------------------------------
 * dictPrefetchKeys() allows a single multi-key command (e.g. MGET) to
 * prefetch dict data for a batch of its own keys, reusing the same state
 * machine that the cross-command path uses.
 *
 * Typical usage from a command implementation:
 *
 *   #define BATCH 16
 *   void myMultiKeyCommand(client *c) {
 *       dict *d = kvstoreGetDict(c->db->keys, slot);
 *       for (int j = 0; j < numkeys; j += BATCH) {
 *           int n = MIN(BATCH, numkeys - j);
 *           void *keys[BATCH]; dict *dicts[BATCH];
 *           for (int k = 0; k < n; k++) {
 *               keys[k] = c->argv[j+k+1]->ptr;
 *               dicts[k] = d;
 *           }
 *           dictPrefetchKeys(dicts, keys, n, NULL, NULL);
 *           // Now process these n keys — dict bucket / entry (the kvobj)
 *           // (and value payload, if dictType->prefetchEntryValue is set)
 *           // are warm in cache.
 *       }
 *   }
 * ----------------------------------------------------------------------- */
int dictPrefetchKeys(dict **dicts, void **keys, size_t nkeys, uint64_t *hashes, void **found) {
    /* Single-key prefetch has no benefit — nothing to interleave with.
     * Callers passing nkeys==1 (e.g. tail of a multi-key batch) should
     * fall through to a direct lookup. */
    if (nkeys <= 1) {
        if (hashes && nkeys == 1 && dicts[0]) hashes[0] = dictGetHash(dicts[0], keys[0]);
        return 0;
    }

    /* Guard the fixed-size stack array below; callers must batch larger
     * inputs into chunks of DICT_PREFETCH_MAX_SIZE or smaller. */
    serverAssert(nkeys <= DICT_PREFETCH_MAX_SIZE);
    server.stat_total_prefetch_batches++;

    dictPrefetchState lookups[DICT_PREFETCH_MAX_SIZE];
    uint32_t active[DICT_PREFETCH_MAX_SIZE];
    dictPrefetcher p = { .lookups = lookups, .active = active, .max_keys = nkeys };
    dictPrefetcherReset(&p, dicts, keys, nkeys, found != NULL);
    if (hashes) {
        /* Hand the hashes to the caller so its lookups can skip hashFunction(). */
        for (size_t i = 0; i < nkeys; i++) {
            if (!lookups[i].done) hashes[i] = dictPrefetchHash(&lookups[i]);
            else if (dicts[i]) hashes[i] = dictGetHash(dicts[i], keys[i]);
        }
    }
    dictPrefetcherRun(&p);
    if (found) {
        for (size_t i = 0; i < nkeys; i++) {
            dictEntry *de = dictPrefetchFound(&lookups[i]);
            found[i] = de ? dictGetKey(de) : NULL;
        }
    }
    return found != NULL;
}

/* --------------------------------------------------------------------------
 * Cross-command batch prefetching
 * -------------------------------------------------------------------------- */

/* PrefetchCommandsBatch structure holds the state of the current batch of client commands being processed. */
typedef struct PrefetchCommandsBatch {
    size_t key_count;               /* Number of keys in the current batch */
    size_t client_count;            /* Number of clients in the current batch */
    size_t pcmd_count;              /* Number of pending commands in the current batch */
    size_t max_prefetch_size;       /* Maximum number of keys to prefetch in a batch */
    void **keys;                    /* Array of keys to prefetch in the current batch */
    client **clients;               /* Array of clients in the current batch */
    pendingCommand **pending_cmds;  /* Array of pending commands in the current batch */
    dict **keys_dicts;              /* Main dict for each key */
    dictPrefetcher prefetcher;      /* Initialized once; reset and reused per batch. */
} PrefetchCommandsBatch;

static PrefetchCommandsBatch *batch = NULL;

void freePrefetchCommandsBatch(void) {
    if (batch == NULL) {
        return;
    }

    zfree(batch->clients);
    zfree(batch->pending_cmds);
    zfree(batch->keys);
    zfree(batch->keys_dicts);
    dictPrefetcherFree(&batch->prefetcher);
    zfree(batch);
    batch = NULL;
}

void prefetchCommandsBatchInit(void) {
    serverAssert(!batch);

    /* To avoid prefetching small batches, we set the max size to twice
     * the configured size, so if not exceeding twice the limit, we can
     * prefetch all of it. See also `determinePrefetchCount` */
    size_t max_prefetch_size = server.prefetch_batch_max_size * 2;

    if (max_prefetch_size == 0) {
        return;
    }

    batch = zcalloc(sizeof(PrefetchCommandsBatch));
    batch->max_prefetch_size = max_prefetch_size;
    batch->clients = zcalloc(max_prefetch_size * sizeof(client *));
    batch->pending_cmds = zcalloc(max_prefetch_size * sizeof(pendingCommand *));
    batch->keys = zcalloc(max_prefetch_size * sizeof(void *));
    batch->keys_dicts = zcalloc(max_prefetch_size * sizeof(dict *));
    dictPrefetcherInit(&batch->prefetcher, max_prefetch_size);
}

void onMaxBatchSizeChange(void) {
    if (batch && batch->client_count > 0) {
        /* We need to process the current batch before updating the size */
        return;
    }

    freePrefetchCommandsBatch();
    prefetchCommandsBatchInit();
}

void resetCommandsBatch(void) {
    if (batch == NULL) {
        /* Handle the case where prefetching becomes enabled from disabled. */
        if (server.prefetch_batch_max_size) prefetchCommandsBatchInit();
        return;
    }

    batch->key_count = 0;
    batch->client_count = 0;
    batch->pcmd_count = 0;

    /* Handle the case where the max prefetch size has been changed. */
    if (batch->max_prefetch_size != (size_t)server.prefetch_batch_max_size * 2) {
        onMaxBatchSizeChange();
    }
}

/* Prefetching in very small batches tends to be ineffective because the technique
 * relies on a small gap—typically a few CPU cycles—between issuing the prefetch
 * and performing the actual memory access. If the batch is too small, this delay
 * cannot be effectively inserted, and the prefetching yields little to no benefit.
 *
 * To avoid wasting effort, when the remaining data is small (less than twice the
 * maximum batch size), we simply prefetch all of it at once. Otherwise, we only
 * prefetch a limited portion, capped at the configured maximum. */
int determinePrefetchCount(int len) {
    if (!batch) return 0;

    /* The batch max size is double of the configured size. */
    int config_size = batch->max_prefetch_size / 2;
    return len < (int)batch->max_prefetch_size ? len : config_size;
}

/* Prefetch command-related data:
 * 1. Prefetch the command arguments allocated by the I/O thread to bring them
 *    closer to the L1 cache.
 * 2. Prefetch the io_deferred_objects for all clients.
 * 3. Prefetch the keys and values for all commands in the current batch from
 *    the main dictionaries. */
void prefetchCommands(void) {
    if (!batch || server.loading) return;

    /* Prefetch argv's for all pending commands */
    for (size_t i = 0; i < batch->pcmd_count; i++) {
        pendingCommand *pcmd = batch->pending_cmds[i];
        if (unlikely(pcmd->argc <= 0)) continue;
        for (int j = 0; j < pcmd->argc; j++) {
            redis_prefetch_read(pcmd->argv[j]);
        }
    }

    /* Prefetch the argv->ptr if required */
    for (size_t i = 0; i < batch->pcmd_count; i++) {
        pendingCommand *pcmd = batch->pending_cmds[i];
        if (unlikely(pcmd->argc <= 1)) continue;
        /* Skip the first argument (command name), as it's typically short */
        for (int j = 1; j < pcmd->argc; j++) {
            if (pcmd->argv[j]->encoding == OBJ_ENCODING_RAW) {
                redis_prefetch_read(pcmd->argv[j]->ptr);
            }
        }
    }

    /* Prefetch io_deferred_objects for all clients */
    for (size_t i = 0; i < batch->client_count; i++) {
        client *c = batch->clients[i];
        if (!c->io_deferred_objects || c->io_deferred_objects_num == 0) continue;
        for (int j = 0; j < c->io_deferred_objects_num; j++)
            redis_prefetch_read(c->io_deferred_objects[j]);
    }

    /* Get the keys ptrs - we do it here after the key obj was prefetched. */
    for (size_t i = 0; i < batch->key_count; i++) {
        batch->keys[i] = ((robj *)batch->keys[i])->ptr;
    }

    /* Prefetch dict keys for all commands.
     * Prefetching is beneficial only if there are more than one key. */
    if (batch->key_count > 1) {
        server.stat_total_prefetch_batches++;
        /* Prefetch keys from the main dict — value-side prefetch (if any)
         * is driven by dbDictType->prefetchEntryValue. */
        dictPrefetcherReset(&batch->prefetcher, batch->keys_dicts, batch->keys, batch->key_count, 0);
        dictPrefetcherRun(&batch->prefetcher);
    }
}

/* Adds the client's command to the current batch.
 *
 * Returns C_OK if the command was added successfully, C_ERR otherwise. */
int addCommandToBatch(client *c) {
    if (unlikely(!batch)) return C_ERR;

    /* If the batch is full, process it.
     * We also check the client count to handle cases where
     * no keys exist for the clients' commands. */
    if (batch->client_count == batch->max_prefetch_size ||
        batch->key_count == batch->max_prefetch_size ||
        batch->pcmd_count == batch->max_prefetch_size)
    {
        return C_ERR;
    }

    /* Avoid partial prefetching: if the batch already has commands and adding this
     * client's ready commands would likely exceed the batch size limit, reject
     * the entire client. This is a conservative estimate using command count as
     * a proxy for key count to ensure all keys from a client are either fully
     * prefetched together or not prefetched at all. */
    if (batch->pcmd_count > 0 &&
        (c->pending_cmds.ready_len + batch->key_count > batch->max_prefetch_size ||
         c->pending_cmds.ready_len + batch->pcmd_count > batch->max_prefetch_size))
    {
        return C_ERR;
    }

    batch->clients[batch->client_count++] = c;

    pendingCommand *pcmd = c->pending_cmds.head;
    while (pcmd != NULL && batch->pcmd_count < batch->max_prefetch_size) {
        if (pcmd->next) redis_prefetch_read(pcmd->next);

        /* Skip commands that have not been preprocessed, or have errors. */
        if ((pcmd->flags & PENDING_CMD_FLAG_INCOMPLETE) || !pcmd->cmd || pcmd->read_error) break;

        batch->pending_cmds[batch->pcmd_count++] = pcmd;

        serverAssert(pcmd->flags & PENDING_CMD_KEYS_RESULT_VALID);
        dict *cmd_dict = kvstoreGetDict(c->db->keys, pcmd->slot > 0 ? pcmd->slot : 0);
        int i;
        for (i = 0; i < pcmd->keys_result.numkeys && batch->key_count < batch->max_prefetch_size; i++) {
            batch->keys[batch->key_count] = pcmd->argv[pcmd->keys_result.keys[i].pos];
            batch->keys_dicts[batch->key_count] = cmd_dict;
            batch->key_count++;
        }
        /* Record how many of the command's keys the batch covers, so that a
         * multi-key command (MGET, MSET) prefetches only the rest itself: each
         * key is then prefetched once, and none is left out when the batch
         * fills up mid-command. Keep the larger count if batched again. */
        if (i > pcmd->prefetched_keys) pcmd->prefetched_keys = i;
        pcmd = pcmd->next;
    }

    return C_OK;
}
