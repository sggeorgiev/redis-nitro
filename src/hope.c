/* hope.c - HOPE (High-speed Order-Preserving Encoder), Double-char scheme.
 *
 * C port of the Double-char encoder/decoder from the HOPE project
 * (https://github.com/efficient/HOPE, SIGMOD 2020). See hope.h for the API
 * and a description of the scheme.
 *
 * The build pipeline mirrors the reference C++ implementation:
 *   1. Count the frequency of every two-byte symbol over the sample keys.
 *   2. Assign optimal order-preserving (Hu-Tucker) variable-length codes.
 *   3. Fill the 65536-entry encode dictionary and build the decode trie.
 *
 * Copyright 2020, Carnegie Mellon University (original C++ implementation).
 * Licensed under the Apache License 2.0.
 */

#include "hope.h"

#include <stdlib.h>
#include <string.h>

#include "zmalloc.h"

#define HOPE_N HOPE_NUM_DOUBLE_CHAR

/* A variable-length code: the low `len` bits of `code` are significant. */
typedef struct {
    int32_t code;
    int8_t len;
} hopeCode;

/* Succinct binary trie used for decoding. It is a level-order (BFS) bit
 * encoding of a binary trie built over the code set: for node at slot `pos`,
 * bit `pos` marks a left child and bit `pos+1` marks a right child. rank()
 * over the bitvector navigates to a node's children in O(1). */
typedef struct {
    int num_bits;      /* total bits in the trie bitvector */
    int num_nodes;     /* number of trie nodes */
    uint64_t *bits;    /* level-order child bitmap (MSB-first within a word) */
    int *rank_lut;     /* cumulative popcount per 512-bit block */
    int *leaf_orders;  /* leaf_orders[pos/2] -> symbol index for leaf nodes */
} hopeSBT;

struct hopeEncoder {
    hopeCode dict[HOPE_N]; /* encode dictionary indexed by two-byte symbol */
    hopeSBT sbt;           /* decode trie */
    int built;             /* nonzero once hopeBuild() has succeeded */
};

/* ------------------------------------------------------------------------- */
/* Stage 1: two-byte symbol frequency counting.                              */
/* ------------------------------------------------------------------------- */

/* Count every overlapping two-byte pair (sliding window, stride 1) across all
 * sample keys. Frequencies start at 1 so that every one of the 65536 symbols
 * is representable even if it never appears in the sample. */
static void hopeCountFreq(const char **keys, const size_t *key_lens,
                          size_t num_keys, int64_t *freq)
{
    for (int i = 0; i < HOPE_N; i++) freq[i] = 1;

    for (size_t k = 0; k < num_keys; k++) {
        const unsigned char *key = (const unsigned char *)keys[k];
        size_t len = key_lens[k];
        for (size_t j = 0; j < len; j++) {
            unsigned idx = 256u * key[j];
            if (j + 1 < len) idx += key[j + 1];
            freq[idx]++;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Stage 2: Hu-Tucker optimal order-preserving code assignment.             */
/* ------------------------------------------------------------------------- */

/* Compute the optimal Hu-Tucker code length for each symbol (in symbol order)
 * given its frequency. This is a direct port of HuTuckerCA::genOptimalCodeLen.
 * `code_len` must have room for `n` entries. Returns 0 on success, -1 on
 * allocation failure. */
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

/* Read the order-preserving code for symbol `idx` by walking from the tree
 * root: go left/right depending on which subtree contains `idx`, appending a
 * bit per step. Port of HuTuckerCA::lookup. */
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

/* Free every node reachable from `root` using an explicit queue (the tree can
 * be deep, so recursion is avoided). */
static void hopeHtDestroy(hopeHtNode *root, int cap)
{
    if (!root) return;
    hopeHtNode **q = zmalloc(sizeof(hopeHtNode *) * cap);
    if (!q) return; /* best-effort: nothing else we can do on OOM here */
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
 * code for every symbol into `codes`. Port of HuTuckerCA::buildBinaryTree +
 * the lookup loop in assignCodes. Returns 0 on success, -1 on failure. */
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

    /* Merge nodes bottom-up: at each length, adjacent nodes of that length are
     * paired into a parent whose length is one less. */
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

    /* All nodes are reachable from the root; free the whole tree. A binary
     * tree with n leaves has at most 2n-1 nodes. */
    hopeHtDestroy(root, 2 * n);
    rc = 0;

cleanup:
    if (rc != 0 && node_list) {
        /* On failure, free any leaf/internal nodes still parked in node_list
         * (the tree was not fully assembled, so free per-slot). */
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
/* Stage 3: succinct binary trie for decoding.                               */
/* ------------------------------------------------------------------------- */

/* popcount over a bit range, port of surf::popcountLinear. `bits` is a word
 * array, `x` is the starting word index, `nbits` is how many bits to count. */
static uint64_t hopePopcountLinear(const uint64_t *bits, uint64_t x, uint64_t nbits)
{
    if (nbits == 0) return 0;
    uint64_t lastword = (nbits - 1) / 64;
    uint64_t p = 0;
    for (uint64_t i = 0; i < lastword; i++) {
        p += (uint64_t)__builtin_popcountll(bits[x + i]);
    }
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
    int word_id = pos >> 6;
    int offset = pos & 63;
    return (t->bits[word_id] & (0x8000000000000000ULL >> offset)) != 0;
}

static inline void hopeRbSetBit(hopeSBT *t, int pos)
{
    int word_id = pos >> 6;
    int offset = pos & 63;
    t->bits[word_id] |= (0x8000000000000000ULL >> offset);
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

/* Node of the temporary (pointer-based) binary trie built from the codes. */
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

/* Build a binary trie from the code set. Each code contributes a root->leaf
 * path (MSB-first over its `len` significant bits). Sets *num_nodes_out.
 * Returns the root, or NULL on allocation failure. Port of buildBinaryTrie. */
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
    /* Free the partial trie iteratively. */
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

/* Build the succinct decode trie from the code set. Port of SBT::SBT.
 * Returns 0 on success, -1 on failure. */
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

    /* Level-order (BFS) traversal writing the child bitmap. */
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

static inline void hopeSBTMoveToLeftChild(const hopeSBT *t, int *pos)
{
    *pos = hopeRbRank(t, *pos);
    *pos <<= 1;
}

/* Consume bits from `in` starting at *key_bit_pos, walking the trie until a
 * leaf is reached. On success sets *out_idx to the symbol index, advances
 * *key_bit_pos past the consumed bits, and returns 1. Returns 0 if the walk
 * runs out of input without hitting a leaf. Port of SBT::lookup. */
static int hopeSBTLookup(const hopeSBT *t, const uint8_t *in, size_t in_bytes,
                         int *key_bit_pos, int *out_idx)
{
    int bit_pos = *key_bit_pos;
    unsigned char cur_char = in[bit_pos >> 3];
    int offset = bit_pos & 7;
    int pos = 0;
    while (bit_pos < (int)in_bytes * 8) {
        if (cur_char & (1 << (7 - offset))) pos += 1;
        hopeSBTMoveToLeftChild(t, &pos);
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
/* Public API.                                                               */
/* ------------------------------------------------------------------------- */

hopeEncoder *hopeCreate(void)
{
    hopeEncoder *e = zcalloc(sizeof(*e));
    return e;
}

hopeEncoder *hopeDup(const hopeEncoder *e)
{
    if (!e) return NULL;

    hopeEncoder *copy = zcalloc(sizeof(*copy));
    if (!copy) return NULL;

    /* The encode dictionary is a flat array; a plain copy suffices. */
    memcpy(copy->dict, e->dict, sizeof(e->dict));
    copy->built = e->built;

    /* Deep-copy the succinct decode trie's backing arrays. */
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
            hopeSBTFree(dst);
            zfree(copy);
            return NULL;
        }
        memcpy(dst->bits, src->bits, (size_t)num_words * 8);
        memcpy(dst->rank_lut, src->rank_lut, (size_t)num_blocks * sizeof(int));
        memcpy(dst->leaf_orders, src->leaf_orders,
               (size_t)src->num_nodes * sizeof(int));
    }

    return copy;
}

void hopeFree(hopeEncoder *e)
{
    if (!e) return;
    hopeSBTFree(&e->sbt);
    zfree(e);
}

int hopeBuild(hopeEncoder *e, const char **keys, const size_t *key_lens,
              size_t num_keys)
{
    if (!e || !keys || !key_lens || num_keys == 0) return -1;

    int rc = -1;
    int64_t *freq = zmalloc(sizeof(int64_t) * HOPE_N);
    int *code_len = zmalloc(sizeof(int) * HOPE_N);
    hopeCode *codes = zmalloc(sizeof(hopeCode) * HOPE_N);
    if (!freq || !code_len || !codes) goto cleanup;

    hopeCountFreq(keys, key_lens, num_keys, freq);

    if (hopeGenOptimalCodeLen(freq, HOPE_N, code_len) != 0) goto cleanup;
    if (hopeAssignCodes(code_len, HOPE_N, codes) != 0) goto cleanup;

    /* Fill the encode dictionary. Symbols are in ascending order, so codes[i]
     * is the code for the two-byte symbol i. */
    for (int i = 0; i < HOPE_N; i++) e->dict[i] = codes[i];

    /* (Re)build the decode trie. */
    hopeSBTFree(&e->sbt);
    if (hopeSBTBuild(&e->sbt, codes, HOPE_N) != 0) goto cleanup;

    e->built = 1;
    rc = 0;

cleanup:
    zfree(freq);
    zfree(code_len);
    zfree(codes);
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

    for (int i = 0; i < klen; i += 2) {
        unsigned s_idx = 256u * k[i];
        if (i + 1 < klen) s_idx += k[i + 1];
        int64_t s_buf = e->dict[s_idx].code;
        int s_len = e->dict[s_idx].len;
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
            if (key_bit_pos < bit_len) {
                return 0;
            } else {
                if (buf_pos > 0 && out[buf_pos - 1] == 0) buf_pos--;
                return buf_pos;
            }
        }
        out[buf_pos] = (uint8_t)((idx >> 8) & 0xFF);
        out[buf_pos + 1] = (uint8_t)(idx & 0xFF);
        buf_pos += 2;
    }
    return buf_pos;
}
