/* hope.c - HOPE (High-speed Order-Preserving Encoder), 3-gram scheme.
 *
 * C port of the 3-gram encoder from the HOPE project
 * (https://github.com/efficient/HOPE, SIGMOD 2020), plus a decoder that the
 * reference 3-gram encoder lacks. See hope.h for the public API.
 *
 * Build pipeline:
 *   1. Symbol selection: count 3-gram frequencies over the sample, keep the
 *      most frequent ones, and fill the gaps between them with intervals so
 *      that every possible byte string maps to exactly one interval (in
 *      ascending order). This mirrors HOPE's NGramSS.
 *   2. Code assignment: give every interval an optimal order-preserving
 *      (Hu-Tucker) variable-length code.
 *   3. Dictionaries: a sorted interval array for encoding (longest-prefix
 *      match via binary search) and a succinct binary trie for decoding.
 *
 * Losslessness: every string covered by an interval shares the interval's
 * prefix, so a code always maps back to exactly the bytes it consumed.
 *
 * Copyright 2020, Carnegie Mellon University (original C++ implementation).
 * Licensed under the Apache License 2.0.
 */

#include "hope.h"

#include <stdlib.h>
#include <string.h>

#include "zmalloc.h"

/* n for the n-gram scheme. */
#define HOPE_NGRAM 3
/* Symbols whose optimal code needs more than this many bits are rejected so
 * that a code always fits in an int32 and packs cleanly. */
#define HOPE_MAX_CODE_LEN 32
/* Default dictionary-size budget passed to symbol selection. Selection caps
 * this to the number of distinct 3-grams actually seen, so for small inputs
 * essentially all frequent 3-grams are used. */
#define HOPE_DEFAULT_DICT_LIMIT 65536

/* A variable-length code: the low `len` bits of `code` are significant. */
typedef struct {
    int32_t code;
    int8_t len;
} hopeCode;

/* An encode-dictionary entry: an interval identified by its start key (a
 * 1..3 byte boundary, zero-padded to 3 bytes), the shared-prefix length of the
 * interval (how many bytes encoding consumes / decoding emits), and the code. */
typedef struct {
    uint8_t start_key[3];
    uint8_t cpl; /* common prefix length, 1..3 */
    hopeCode code;
} hopeInterval;

/* Succinct binary trie used for decoding. Level-order (BFS) bit encoding of a
 * binary trie built over the code set: for the node at slot `pos`, bit `pos`
 * marks a left child and bit `pos+1` marks a right child. rank() navigates to
 * a node's children in O(1). Leaves map to interval indices. */
typedef struct {
    int num_bits;
    int num_nodes;
    uint64_t *bits;
    int *rank_lut;
    int *leaf_orders;
} hopeSBT;

struct hopeEncoder {
    hopeInterval *dict; /* intervals sorted ascending by start_key */
    int dict_size;
    hopeSBT sbt;        /* decode trie; leaf_orders[k] -> interval index */
    int built;
};

/* A short (<= 3 byte) string used for interval boundaries and prefixes. */
typedef struct {
    uint8_t b[3];
    uint8_t len;
} hopeStr;

/* ------------------------------------------------------------------------- */
/* Small helpers.                                                            */
/* ------------------------------------------------------------------------- */

static hopeStr hopeMk1(int c)
{
    hopeStr s;
    s.b[0] = (uint8_t)c;
    s.b[1] = 0;
    s.b[2] = 0;
    s.len = 1;
    return s;
}

static hopeStr hopeMk3(uint32_t g)
{
    hopeStr s;
    s.b[0] = (uint8_t)((g >> 16) & 0xFF);
    s.b[1] = (uint8_t)((g >> 8) & 0xFF);
    s.b[2] = (uint8_t)(g & 0xFF);
    s.len = 3;
    return s;
}

/* Lexicographic comparison of two short strings (shorter is smaller when one
 * is a prefix of the other). */
static int hopeStrCmp(const hopeStr *a, const hopeStr *b)
{
    int n = a->len < b->len ? a->len : b->len;
    for (int i = 0; i < n; i++) {
        if (a->b[i] != b->b[i]) return a->b[i] < b->b[i] ? -1 : 1;
    }
    if (a->len != b->len) return a->len < b->len ? -1 : 1;
    return 0;
}

/* Growable vector of hopeStr. */
typedef struct {
    hopeStr *a;
    int n;
    int cap;
} hopeStrVec;

static int hopeStrVecPush(hopeStrVec *v, hopeStr s)
{
    if (v->n == v->cap) {
        int nc = v->cap ? v->cap * 2 : 32;
        hopeStr *na = zrealloc(v->a, sizeof(hopeStr) * nc);
        if (!na) return -1;
        v->a = na;
        v->cap = nc;
    }
    v->a[v->n++] = s;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 1: 3-gram symbol selection.                                         */
/* ------------------------------------------------------------------------- */

static uint32_t hopeGramAt(const unsigned char *p)
{
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
}

static int hopeCmpU32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

typedef struct {
    uint32_t g;
    int64_t f;
} hopeGramFreq;

/* Sort by frequency descending, then by gram descending (matches the
 * reference tie-break in NGramSS::pickMostFreqSymbols). */
static int hopeCmpGramFreq(const void *a, const void *b)
{
    const hopeGramFreq *x = a, *y = b;
    if (x->f != y->f) return (x->f < y->f) - (x->f > y->f);
    return (x->g < y->g) - (x->g > y->g);
}

/* Common prefix of two 3-grams that share their first byte (>= 1 byte). */
static hopeStr hopeCommonPrefix(const hopeStr *s1, const hopeStr *s2)
{
    hopeStr out;
    out.b[0] = out.b[1] = out.b[2] = 0;
    for (int i = 1; i < HOPE_NGRAM; i++) {
        if (i >= s1->len || i >= s2->len || s1->b[i] != s2->b[i]) {
            out.len = (uint8_t)i;
            memcpy(out.b, s1->b, i);
            return out;
        }
    }
    out.len = HOPE_NGRAM;
    memcpy(out.b, s1->b, HOPE_NGRAM);
    return out;
}

/* Append single-character intervals for byte values [first, last] to both the
 * boundary and prefix vectors. */
static int hopeFillSingleChar(int first, int last, hopeStrVec *bnd, hopeStrVec *pre)
{
    for (int c = first; c <= last; c++) {
        hopeStr s = hopeMk1(c);
        if (hopeStrVecPush(bnd, s) != 0) return -1;
        if (hopeStrVecPush(pre, s) != 0) return -1;
    }
    return 0;
}

/* Build interval boundaries and prefixes from the selected (ascending)
 * 3-grams. Port of NGramSS::fillInGap. When no 3-gram was selected, fall back
 * to 256 single-character intervals. Returns 0 on success, -1 on failure. */
static int hopeFillInGap(const uint32_t *mf, int num, hopeStrVec *bnd, hopeStrVec *pre)
{
    if (num == 0) return hopeFillSingleChar(0, 255, bnd, pre);

    hopeStr first = hopeMk3(mf[0]);
    if (hopeFillSingleChar(0, first.b[0], bnd, pre) != 0) return -1;

    for (int i = 0; i < num - 1; i++) {
        hopeStr s1 = hopeMk3(mf[i]);
        hopeStr s2 = hopeMk3(mf[i + 1]);
        if (hopeStrVecPush(pre, s1) != 0) return -1;
        if (hopeStrVecPush(bnd, s1) != 0) return -1;

        hopeStr rb = s1;
        for (int j = HOPE_NGRAM - 1; j >= 0; j--) {
            if (rb.b[j] < 255) {
                rb.b[j] = (uint8_t)(rb.b[j] + 1);
                rb.len = (uint8_t)(j + 1);
                break;
            }
        }

        if (hopeStrCmp(&rb, &s2) != 0) {
            if (hopeStrVecPush(bnd, rb) != 0) return -1;
            if (rb.b[0] != s2.b[0]) {
                if (hopeStrVecPush(pre, hopeMk1(s1.b[0])) != 0) return -1;
                if (hopeFillSingleChar(s1.b[0] + 1, s2.b[0], bnd, pre) != 0) return -1;
            } else {
                hopeStr common;
                if (s1.b[0] != rb.b[0])
                    common = rb;
                else
                    common = hopeCommonPrefix(&s1, &s2);
                if (hopeStrVecPush(pre, common) != 0) return -1;
            }
        }
    }

    hopeStr last = hopeMk3(mf[num - 1]);
    if (hopeStrVecPush(pre, last) != 0) return -1;
    if (hopeStrVecPush(bnd, last) != 0) return -1;

    hopeStr lrb = last;
    for (int j = HOPE_NGRAM - 1; j >= 0; j--) {
        if (lrb.b[j] < 255) {
            lrb.b[j] = (uint8_t)(lrb.b[j] + 1);
            lrb.len = (uint8_t)(j + 1);
            break;
        }
    }
    if (hopeStrVecPush(bnd, lrb) != 0) return -1;
    if (hopeStrVecPush(pre, hopeMk1(last.b[0])) != 0) return -1;

    if (last.b[0] < 255) {
        if (hopeFillSingleChar(last.b[0] + 1, 255, bnd, pre) != 0) return -1;
    }
    return 0;
}

/* Compare raw query bytes q[0..qlen) against a boundary (std::string-style:
 * shorter is smaller when a prefix). Used only for interval-frequency
 * estimation during training. */
static int hopeCmpBytesBoundary(const unsigned char *q, int qlen, const hopeStr *b)
{
    int n = qlen < b->len ? qlen : b->len;
    for (int i = 0; i < n; i++) {
        if (q[i] != b->b[i]) return q[i] < b->b[i] ? -1 : 1;
    }
    if (qlen != b->len) return qlen < b->len ? -1 : 1;
    return 0;
}

static int hopeBoundarySearch(const hopeStr *bnd, int n, const unsigned char *q, int qlen)
{
    int l = 0, r = n;
    while (r - l > 1) {
        int m = (l + r) >> 1;
        int cmp = hopeCmpBytesBoundary(q, qlen, &bnd[m]);
        if (cmp < 0) r = m;
        else if (cmp == 0) return m;
        else l = m;
    }
    return l;
}

/* Estimate how often each interval is used by test-encoding the sample keys.
 * Frequencies start at 1. Port of NGramSS::countIntervalFreq. */
static void hopeCountIntervalFreq(const char **keys, const size_t *key_lens,
                                  size_t num_keys, const hopeStrVec *bnd,
                                  const hopeStrVec *pre, int64_t *freq)
{
    for (int i = 0; i < bnd->n; i++) freq[i] = 1;
    for (size_t k = 0; k < num_keys; k++) {
        const unsigned char *key = (const unsigned char *)keys[k];
        int len = (int)key_lens[k];
        int pos = 0;
        while (pos < len) {
            int qlen = len - pos;
            if (qlen > HOPE_NGRAM + 1) qlen = HOPE_NGRAM + 1;
            int idx = hopeBoundarySearch(bnd->a, bnd->n, key + pos, qlen);
            freq[idx]++;
            pos += pre->a[idx].len;
        }
    }
}

/* Run the full symbol-selection stage, producing the interval boundary vector
 * and per-interval frequencies. Returns 0 on success, -1 on failure. */
static int hopeSelectSymbols(const char **keys, const size_t *key_lens,
                             size_t num_keys, hopeStrVec *bnd_out,
                             int64_t **freq_out)
{
    int rc = -1;
    uint32_t *grams = NULL;
    hopeGramFreq *gf = NULL;
    uint32_t *most_freq = NULL;
    hopeStrVec bnd = {0}, pre = {0};
    int64_t *freq = NULL;

    /* Collect all 3-grams. */
    long total = 0;
    for (size_t k = 0; k < num_keys; k++) {
        if (key_lens[k] >= HOPE_NGRAM) total += (long)key_lens[k] - (HOPE_NGRAM - 1);
    }

    int distinct = 0;
    if (total > 0) {
        grams = zmalloc(sizeof(uint32_t) * total);
        if (!grams) goto done;
        long t = 0;
        for (size_t k = 0; k < num_keys; k++) {
            const unsigned char *key = (const unsigned char *)keys[k];
            int len = (int)key_lens[k];
            for (int j = 0; j + HOPE_NGRAM <= len; j++)
                grams[t++] = hopeGramAt(key + j);
        }
        qsort(grams, total, sizeof(uint32_t), hopeCmpU32);

        /* Run-length compress into distinct (gram, freq) pairs. */
        gf = zmalloc(sizeof(hopeGramFreq) * total);
        if (!gf) goto done;
        for (long i = 0; i < total;) {
            long j = i + 1;
            while (j < total && grams[j] == grams[i]) j++;
            gf[distinct].g = grams[i];
            gf[distinct].f = j - i;
            distinct++;
            i = j;
        }
    }

    /* Pick the most frequent 3-grams (half of the adjusted budget). */
    int num_mf = 0;
    if (distinct > 0) {
        int64_t adjust = HOPE_DEFAULT_DICT_LIMIT;
        if (adjust > (int64_t)distinct * 2) adjust = (int64_t)distinct * 2 - 1;
        int k_pick = (int)(adjust / 2);
        if (k_pick > distinct) k_pick = distinct;
        if (k_pick > 0) {
            qsort(gf, distinct, sizeof(hopeGramFreq), hopeCmpGramFreq);
            most_freq = zmalloc(sizeof(uint32_t) * k_pick);
            if (!most_freq) goto done;
            for (int i = 0; i < k_pick; i++) most_freq[i] = gf[i].g;
            qsort(most_freq, k_pick, sizeof(uint32_t), hopeCmpU32);
            num_mf = k_pick;
        }
    }

    if (hopeFillInGap(most_freq, num_mf, &bnd, &pre) != 0) goto done;
    if (bnd.n != pre.n || bnd.n < 2) goto done;

    freq = zmalloc(sizeof(int64_t) * bnd.n);
    if (!freq) goto done;
    hopeCountIntervalFreq(keys, key_lens, num_keys, &bnd, &pre, freq);

    *bnd_out = bnd;
    bnd.a = NULL; /* ownership transferred */
    *freq_out = freq;
    freq = NULL;
    rc = 0;

done:
    zfree(grams);
    zfree(gf);
    zfree(most_freq);
    zfree(pre.a);
    zfree(bnd.a);
    zfree(freq);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Stage 2: Hu-Tucker optimal order-preserving code assignment.             */
/* ------------------------------------------------------------------------- */

/* Compute the optimal Hu-Tucker code length for each symbol (in symbol order)
 * given its frequency. Direct port of HuTuckerCA::genOptimalCodeLen. Requires
 * n >= 2. Returns 0 on success, -1 on allocation failure. */
static int hopeGenOptimalCodeLen(const int64_t *freq, int n, int *code_len)
{
    int *L = zmalloc(sizeof(int) * n);
    int64_t *P = zmalloc(sizeof(int64_t) * n);
    int *s = zmalloc(sizeof(int) * (n - 1));
    int *d = zmalloc(sizeof(int) * (n - 1));
    if (!L || !P || !s || !d) {
        zfree(L); zfree(P); zfree(s); zfree(d);
        return -1;
    }

    int64_t maxp = 1;
    for (int k = 0; k < n; k++) {
        L[k] = 0;
        P[k] = freq[k];
        maxp += freq[k];
    }

    for (int m = 0; m < n - 1; m++) {
        int i = 0, i1 = 0, i2 = 0;
        int64_t pmin = maxp;
        int sumL = -1;
        while (i < n - 1) {
            if (P[i] == 0) {
                i++;
                continue;
            }
            int j1 = i, j2 = -1;
            int64_t min1 = P[i], min2 = maxp;
            int minL1 = L[i], minL2 = -1;

            int j = 0;
            for (j = i + 1; j < n; j++) {
                if (P[j] == 0) continue;
                if (P[j] < min1 || (P[j] == min1 && L[j] < minL1)) {
                    min2 = min1;
                    j2 = j1;
                    minL2 = minL1;
                    min1 = P[j];
                    j1 = j;
                    minL1 = L[j];
                } else if (P[j] < min2 || (P[j] == min2 && L[j] < minL2)) {
                    min2 = P[j];
                    j2 = j;
                    minL2 = L[j];
                }
                if (L[j] == 0) break;
            }

            int64_t pt = P[j1] + P[j2];
            int sumLt = L[j1] + L[j2];
            if (pt < pmin || (pt == pmin && sumLt < sumL)) {
                pmin = pt;
                sumL = sumLt;
                i1 = j1;
                i2 = j2;
            }
            i = j;
        }

        if (i1 > i2) {
            int tmp = i1;
            i1 = i2;
            i2 = tmp;
        }
        s[m] = i1;
        d[m] = i2;
        P[i1] = pmin;
        P[i2] = 0;
        L[i1] = sumL + 1;
    }

    L[s[n - 2]] = 0;
    for (int m = n - 2; m >= 0; m--) {
        L[s[m]] += 1;
        L[d[m]] = L[s[m]];
    }

    for (int k = 0; k < n; k++) code_len[k] = L[k];

    zfree(L);
    zfree(P);
    zfree(s);
    zfree(d);
    return 0;
}

/* Node of the temporary Hu-Tucker binary tree used to read out codes. */
typedef struct hopeHtNode {
    int left_idx;
    int right_idx;
    struct hopeHtNode *left_child;
    struct hopeHtNode *right_child;
} hopeHtNode;

static hopeHtNode *hopeHtNewLeaf(int idx)
{
    hopeHtNode *n = zmalloc(sizeof(*n));
    if (!n) return NULL;
    n->left_idx = idx;
    n->right_idx = idx;
    n->left_child = NULL;
    n->right_child = NULL;
    return n;
}

static hopeCode hopeHtLookup(hopeHtNode *root, int idx)
{
    hopeCode code = {0, 0};
    hopeHtNode *n = root;
    while (n->left_child != NULL) {
        code.code <<= 1;
        if (idx > n->left_child->right_idx) {
            code.code += 1;
            n = n->right_child;
        } else {
            n = n->left_child;
        }
        code.len++;
    }
    return code;
}

static void hopeHtDestroy(hopeHtNode *root, int cap)
{
    if (!root) return;
    hopeHtNode **q = zmalloc(sizeof(hopeHtNode *) * cap);
    if (!q) return;
    int head = 0, tail = 0;
    q[tail++] = root;
    while (head < tail) {
        hopeHtNode *n = q[head++];
        if (n->left_child) q[tail++] = n->left_child;
        if (n->right_child) q[tail++] = n->right_child;
        zfree(n);
    }
    zfree(q);
}

/* Build the Hu-Tucker tree from the per-symbol code lengths and read out a
 * code for every symbol into `codes`. Returns 0 on success, -1 on failure. */
static int hopeAssignCodes(const int *code_len, int n, hopeCode *codes)
{
    int rc = -1;
    hopeHtNode **node_list = zmalloc(sizeof(hopeHtNode *) * n);
    int *tmp_len = zmalloc(sizeof(int) * n);
    int *idx_list = zmalloc(sizeof(int) * n);
    if (!node_list || !tmp_len || !idx_list) goto cleanup;

    for (int i = 0; i < n; i++) node_list[i] = NULL;

    int max_code_len = 0;
    for (int i = 0; i < n; i++) {
        node_list[i] = hopeHtNewLeaf(i);
        if (!node_list[i]) goto cleanup;
        tmp_len[i] = code_len[i];
        if (code_len[i] > max_code_len) max_code_len = code_len[i];
    }

    for (int len = max_code_len; len > 0; len--) {
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (tmp_len[i] == len) idx_list[cnt++] = i;
        }
        for (int i = 0; i + 1 < cnt; i += 2) {
            int idx1 = idx_list[i];
            int idx2 = idx_list[i + 1];
            hopeHtNode *left = node_list[idx1];
            hopeHtNode *right = node_list[idx2];
            hopeHtNode *parent = zmalloc(sizeof(*parent));
            if (!parent) goto cleanup;
            parent->left_idx = left->left_idx;
            parent->right_idx = right->right_idx;
            parent->left_child = left;
            parent->right_child = right;
            node_list[idx1] = parent;
            node_list[idx2] = NULL;
            tmp_len[idx1] = len - 1;
            tmp_len[idx2] = 0;
        }
    }

    hopeHtNode *root = node_list[0];
    for (int i = 0; i < n; i++) codes[i] = hopeHtLookup(root, i);

    hopeHtDestroy(root, 2 * n);
    rc = 0;

cleanup:
    if (rc != 0 && node_list) {
        for (int i = 0; i < n; i++) {
            if (node_list[i]) hopeHtDestroy(node_list[i], 2 * n);
        }
    }
    zfree(node_list);
    zfree(tmp_len);
    zfree(idx_list);
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Stage 3a: succinct binary trie for decoding.                              */
/* ------------------------------------------------------------------------- */

static uint64_t hopePopcountLinear(const uint64_t *bits, uint64_t x, uint64_t nbits)
{
    if (nbits == 0) return 0;
    uint64_t lastword = (nbits - 1) / 64;
    uint64_t p = 0;
    for (uint64_t i = 0; i < lastword; i++)
        p += (uint64_t)__builtin_popcountll(bits[x + i]);
    uint64_t lastshifted = bits[x + lastword] >> (63 - ((nbits - 1) & 63));
    p += (uint64_t)__builtin_popcountll(lastshifted);
    return p;
}

static inline int hopeRbNumWords(int num_bits)
{
    return (num_bits % 64 == 0) ? (num_bits / 64) : (num_bits / 64 + 1);
}

static inline int hopeRbReadBit(const hopeSBT *t, int pos)
{
    return (t->bits[pos >> 6] & (0x8000000000000000ULL >> (pos & 63))) != 0;
}

static inline void hopeRbSetBit(hopeSBT *t, int pos)
{
    t->bits[pos >> 6] |= (0x8000000000000000ULL >> (pos & 63));
}

static void hopeRbInitRankLut(hopeSBT *t)
{
    int num_blocks = t->num_bits / 512 + 1;
    int cumu_rank = 0;
    for (int i = 0; i < num_blocks - 1; i++) {
        t->rank_lut[i] = cumu_rank;
        cumu_rank += (int)hopePopcountLinear(t->bits, (uint64_t)i * 8, 512);
    }
    t->rank_lut[num_blocks - 1] = cumu_rank;
}

static inline int hopeRbRank(const hopeSBT *t, int pos)
{
    int block_id = pos / 512;
    int offset = pos & 511;
    return t->rank_lut[block_id] +
           (int)hopePopcountLinear(t->bits, (uint64_t)block_id * 8, offset + 1);
}

typedef struct hopeBtNode {
    struct hopeBtNode *left;
    struct hopeBtNode *right;
    int leaf_order;
} hopeBtNode;

static hopeBtNode *hopeBtNew(void)
{
    hopeBtNode *n = zmalloc(sizeof(*n));
    if (!n) return NULL;
    n->left = NULL;
    n->right = NULL;
    n->leaf_order = -1;
    return n;
}

static hopeBtNode *hopeBtBuild(const hopeCode *codes, int n, int *num_nodes_out)
{
    int num_nodes = 0;
    hopeBtNode *root = hopeBtNew();
    if (!root) return NULL;
    num_nodes++;

    for (int i = 0; i < n; i++) {
        uint32_t code = (uint32_t)codes[i].code;
        int code_len = codes[i].len;
        code <<= (32 - code_len);
        hopeBtNode *n2 = root;
        for (int j = 0; j < code_len; j++) {
            if (code & 0x80000000u) {
                if (n2->right) {
                    n2 = n2->right;
                } else {
                    n2->right = hopeBtNew();
                    if (!n2->right) goto oom;
                    num_nodes++;
                    n2 = n2->right;
                }
            } else {
                if (n2->left) {
                    n2 = n2->left;
                } else {
                    n2->left = hopeBtNew();
                    if (!n2->left) goto oom;
                    num_nodes++;
                    n2 = n2->left;
                }
            }
            code <<= 1;
        }
        n2->leaf_order = i;
    }

    *num_nodes_out = num_nodes;
    return root;

oom:
    {
        hopeBtNode **stack = zmalloc(sizeof(hopeBtNode *) * (num_nodes + 1));
        if (stack) {
            int top = 0;
            stack[top++] = root;
            while (top > 0) {
                hopeBtNode *n2 = stack[--top];
                if (n2->left) stack[top++] = n2->left;
                if (n2->right) stack[top++] = n2->right;
                zfree(n2);
            }
            zfree(stack);
        }
    }
    return NULL;
}

static void hopeBtDestroy(hopeBtNode *root, int num_nodes)
{
    if (!root) return;
    hopeBtNode **q = zmalloc(sizeof(hopeBtNode *) * num_nodes);
    if (!q) return;
    int head = 0, tail = 0;
    q[tail++] = root;
    while (head < tail) {
        hopeBtNode *n = q[head++];
        if (n->left) q[tail++] = n->left;
        if (n->right) q[tail++] = n->right;
        zfree(n);
    }
    zfree(q);
}

static void hopeSBTFree(hopeSBT *t)
{
    zfree(t->bits);
    zfree(t->rank_lut);
    zfree(t->leaf_orders);
    t->bits = NULL;
    t->rank_lut = NULL;
    t->leaf_orders = NULL;
    t->num_bits = 0;
    t->num_nodes = 0;
}

static int hopeSBTBuild(hopeSBT *t, const hopeCode *codes, int n)
{
    memset(t, 0, sizeof(*t));

    int num_nodes = 0;
    hopeBtNode *root = hopeBtBuild(codes, n, &num_nodes);
    if (!root) return -1;

    t->num_nodes = num_nodes;
    t->num_bits = 2 * num_nodes + 1;

    int num_words = hopeRbNumWords(t->num_bits);
    int num_blocks = t->num_bits / 512 + 1;
    t->bits = zcalloc((size_t)num_words * 8);
    t->rank_lut = zmalloc((size_t)num_blocks * sizeof(int));
    t->leaf_orders = zmalloc((size_t)num_nodes * sizeof(int));
    hopeBtNode **q = zmalloc((size_t)num_nodes * sizeof(hopeBtNode *));
    if (!t->bits || !t->rank_lut || !t->leaf_orders || !q) {
        zfree(q);
        hopeBtDestroy(root, num_nodes);
        hopeSBTFree(t);
        return -1;
    }

    int pos = 0;
    int head = 0, tail = 0;
    q[tail++] = root;
    while (head < tail) {
        hopeBtNode *n = q[head++];
        if (n->left) {
            hopeRbSetBit(t, pos);
            q[tail++] = n->left;
        }
        if (n->right) {
            hopeRbSetBit(t, pos + 1);
            q[tail++] = n->right;
        }
        t->leaf_orders[pos / 2] = n->leaf_order;
        pos += 2;
    }

    hopeRbInitRankLut(t);

    zfree(q);
    hopeBtDestroy(root, num_nodes);
    return 0;
}

static inline int hopeSBTIsLeaf(const hopeSBT *t, int pos)
{
    return !hopeRbReadBit(t, pos) && !hopeRbReadBit(t, pos + 1);
}

static int hopeSBTLookup(const hopeSBT *t, const uint8_t *in, size_t in_bytes,
                         int *key_bit_pos, int *out_idx)
{
    int bit_pos = *key_bit_pos;
    unsigned char cur_char = in[bit_pos >> 3];
    int offset = bit_pos & 7;
    int pos = 0;
    while (bit_pos < (int)in_bytes * 8) {
        if (cur_char & (1 << (7 - offset))) pos += 1;
        pos = hopeRbRank(t, pos) << 1;
        if (pos >= t->num_bits) {
            *key_bit_pos = bit_pos;
            return 0;
        }
        if (hopeSBTIsLeaf(t, pos)) {
            bit_pos++;
            *out_idx = t->leaf_orders[pos / 2];
            *key_bit_pos = bit_pos;
            return 1;
        }
        bit_pos++;
        offset++;
        if (offset == 8) {
            cur_char = in[bit_pos >> 3];
            offset = 0;
        }
    }
    *key_bit_pos = bit_pos;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Stage 3b: interval encode dictionary.                                     */
/* ------------------------------------------------------------------------- */

/* Build the encode dictionary from the interval boundaries and codes. Port of
 * Array3GramDict::build. The common prefix length is the shared prefix of a
 * boundary and its (decremented) successor, i.e. the bytes every string in the
 * interval shares. */
static void hopeBuildIntervalDict(hopeInterval *dict, const hopeStr *bnd, int n,
                                  const hopeCode *codes)
{
    for (int i = 0; i < n; i++) {
        int slen = bnd[i].len;
        for (int j = 0; j < 3; j++)
            dict[i].start_key[j] = (j < slen) ? bnd[i].b[j] : 0;

        uint8_t cpl;
        if (i < n - 1) {
            hopeStr next = bnd[i + 1];
            next.b[next.len - 1] = (uint8_t)(next.b[next.len - 1] - 1);
            cpl = 0;
            int j = 0;
            while (j < slen && j < next.len && bnd[i].b[j] == next.b[j]) {
                cpl++;
                j++;
            }
        } else {
            cpl = (uint8_t)slen;
        }
        if (cpl == 0) cpl = 1; /* defensive: always consume at least one byte */
        dict[i].cpl = cpl;
        dict[i].code = codes[i];
    }
}

static int hopeDictCmp(const unsigned char *q, int qlen, const hopeInterval *iv)
{
    for (int i = 0; i < 3; i++) {
        if (i >= qlen) return (iv->start_key[i] == 0) ? 0 : -1;
        if (q[i] < iv->start_key[i]) return -1;
        if (q[i] > iv->start_key[i]) return 1;
    }
    return qlen > 3 ? 1 : 0;
}

static int hopeDictSearch(const hopeInterval *dict, int n,
                          const unsigned char *q, int qlen)
{
    int l = 0, r = n;
    while (r - l > 1) {
        int m = (l + r) >> 1;
        int cmp = hopeDictCmp(q, qlen, &dict[m]);
        if (cmp < 0) r = m;
        else if (cmp == 0) return m;
        else l = m;
    }
    return l;
}

/* ------------------------------------------------------------------------- */
/* Public API.                                                               */
/* ------------------------------------------------------------------------- */

hopeEncoder *hopeCreate(void)
{
    return zcalloc(sizeof(struct hopeEncoder));
}

void hopeFree(hopeEncoder *e)
{
    if (!e) return;
    hopeSBTFree(&e->sbt);
    zfree(e->dict);
    zfree(e);
}

hopeEncoder *hopeDup(const hopeEncoder *e)
{
    if (!e) return NULL;

    hopeEncoder *copy = zcalloc(sizeof(*copy));
    if (!copy) return NULL;

    copy->built = e->built;
    copy->dict_size = e->dict_size;
    if (e->dict && e->dict_size > 0) {
        copy->dict = zmalloc(sizeof(hopeInterval) * e->dict_size);
        if (!copy->dict) {
            zfree(copy);
            return NULL;
        }
        memcpy(copy->dict, e->dict, sizeof(hopeInterval) * e->dict_size);
    }

    const hopeSBT *src = &e->sbt;
    hopeSBT *dst = &copy->sbt;
    memset(dst, 0, sizeof(*dst));
    dst->num_bits = src->num_bits;
    dst->num_nodes = src->num_nodes;
    if (src->bits) {
        int num_words = hopeRbNumWords(src->num_bits);
        int num_blocks = src->num_bits / 512 + 1;
        dst->bits = zmalloc((size_t)num_words * 8);
        dst->rank_lut = zmalloc((size_t)num_blocks * sizeof(int));
        dst->leaf_orders = zmalloc((size_t)src->num_nodes * sizeof(int));
        if (!dst->bits || !dst->rank_lut || !dst->leaf_orders) {
            hopeFree(copy);
            return NULL;
        }
        memcpy(dst->bits, src->bits, (size_t)num_words * 8);
        memcpy(dst->rank_lut, src->rank_lut, (size_t)num_blocks * sizeof(int));
        memcpy(dst->leaf_orders, src->leaf_orders,
               (size_t)src->num_nodes * sizeof(int));
    }

    return copy;
}

int hopeBuild(hopeEncoder *e, const char **keys, const size_t *key_lens,
              size_t num_keys)
{
    if (!e || !keys || !key_lens || num_keys == 0) return -1;

    int rc = -1;
    hopeStrVec bnd = {0};
    int64_t *freq = NULL;
    int *code_len = NULL;
    hopeCode *codes = NULL;
    hopeInterval *dict = NULL;

    if (hopeSelectSymbols(keys, key_lens, num_keys, &bnd, &freq) != 0) goto cleanup;

    int n = bnd.n;
    code_len = zmalloc(sizeof(int) * n);
    codes = zmalloc(sizeof(hopeCode) * n);
    if (!code_len || !codes) goto cleanup;

    if (hopeGenOptimalCodeLen(freq, n, code_len) != 0) goto cleanup;

    /* Reject inputs whose optimal code would overflow an int32 code slot. */
    for (int i = 0; i < n; i++) {
        if (code_len[i] > HOPE_MAX_CODE_LEN) goto cleanup;
    }

    if (hopeAssignCodes(code_len, n, codes) != 0) goto cleanup;

    dict = zmalloc(sizeof(hopeInterval) * n);
    if (!dict) goto cleanup;
    hopeBuildIntervalDict(dict, bnd.a, n, codes);

    hopeSBT sbt;
    if (hopeSBTBuild(&sbt, codes, n) != 0) goto cleanup;

    /* Commit the new state, replacing anything from a previous build. */
    hopeSBTFree(&e->sbt);
    zfree(e->dict);
    e->dict = dict;
    dict = NULL;
    e->dict_size = n;
    e->sbt = sbt;
    e->built = 1;
    rc = 0;

cleanup:
    zfree(bnd.a);
    zfree(freq);
    zfree(code_len);
    zfree(codes);
    zfree(dict);
    return rc;
}

int hopeEncode(const hopeEncoder *e, const char *key, size_t key_len,
               uint8_t *buffer)
{
    const unsigned char *k = (const unsigned char *)key;
    int64_t *int_buf = (int64_t *)buffer;
    int idx = 0;
    int_buf[0] = 0;
    int int_buf_len = 0;
    int klen = (int)key_len;
    int pos = 0;

    while (pos < klen) {
        int di = hopeDictSearch(e->dict, e->dict_size, k + pos, klen - pos);
        int64_t s_buf = (int64_t)(uint32_t)e->dict[di].code.code;
        int s_len = e->dict[di].code.len;
        int prefix_len = e->dict[di].cpl;

        if (int_buf_len + s_len > 63) {
            int num_bits_left = 64 - int_buf_len;
            int_buf_len = s_len - num_bits_left;
            int_buf[idx] <<= num_bits_left;
            int_buf[idx] |= (s_buf >> int_buf_len);
            int_buf[idx] = __builtin_bswap64(int_buf[idx]);
            int_buf[idx + 1] = s_buf;
            idx++;
        } else {
            int_buf[idx] <<= s_len;
            int_buf[idx] |= s_buf;
            int_buf_len += s_len;
        }
        pos += prefix_len;
    }
    int_buf[idx] <<= (64 - int_buf_len);
    int_buf[idx] = __builtin_bswap64(int_buf[idx]);
    return (idx << 6) + int_buf_len;
}

int hopeDecode(const hopeEncoder *e, const uint8_t *enc, size_t enc_bytes,
               int bit_len, uint8_t *out)
{
    int buf_pos = 0;
    int key_bit_pos = 0;
    int idx = 0;
    while (key_bit_pos < bit_len) {
        if (!hopeSBTLookup(&e->sbt, enc, enc_bytes, &key_bit_pos, &idx)) {
            if (key_bit_pos < bit_len) return 0;
            return buf_pos;
        }
        const hopeInterval *iv = &e->dict[idx];
        for (int j = 0; j < iv->cpl; j++)
            out[buf_pos++] = iv->start_key[j];
    }
    return buf_pos;
}
