/* hope.h - HOPE (High-speed Order-Preserving Encoder), Double-char scheme.
 *
 * This is a C port of the Double-char encoder from the HOPE project
 * (https://github.com/efficient/HOPE, SIGMOD 2020). It is a dictionary-based
 * compressor that encodes arbitrary byte strings into shorter byte strings
 * while preserving their lexicographic order: for any two keys a and b,
 *   memcmp(a, b) < 0  <=>  memcmp(encode(a), encode(b)) < 0.
 *
 * The Double-char scheme uses all 65536 two-byte symbols. It counts the
 * frequency of each two-byte pair over a set of sample keys, assigns an
 * optimal order-preserving (Hu-Tucker) variable-length code to every symbol,
 * and stores the result in a fixed-size dictionary used for fast encoding.
 * Decoding uses a succinct binary trie built over the same code set.
 *
 * Copyright 2020, Carnegie Mellon University (original C++ implementation).
 * Licensed under the Apache License 2.0.
 */

#ifndef HOPE_H
#define HOPE_H

#include <stddef.h>
#include <stdint.h>

/* Number of two-byte symbols handled by the Double-char scheme. */
#define HOPE_NUM_DOUBLE_CHAR 65536

typedef struct hopeEncoder hopeEncoder;

/* Allocate an empty (untrained) encoder. Returns NULL on allocation failure.
 * The encoder must be trained with hopeBuild() before it can encode/decode. */
hopeEncoder *hopeCreate(void);

/* Train the encoder from a set of sample keys.
 *
 * keys      - array of num_keys byte pointers (keys need not be sorted).
 * key_lens  - array of num_keys lengths, one per key.
 * num_keys  - number of sample keys (must be > 0).
 *
 * Returns 0 on success, -1 on error (empty input or allocation failure).
 * hopeBuild() may be called again to retrain an existing encoder. */
int hopeBuild(hopeEncoder *e, const char **keys, const size_t *key_lens, size_t num_keys);

/* Encode key[0..key_len) into buffer, returning the encoded length in BITS.
 * The encoded byte length is (return_value + 7) / 8.
 *
 * The caller owns buffer and must ensure it is large enough. A safe upper
 * bound is 8 + key_len * ((HOPE longest code bits + 7) / 8); in practice the
 * output is smaller than the input. To be fully safe for pathological inputs,
 * size the buffer generously (e.g. a few KB, matching the reference harness).
 *
 * The encoder must have been trained with hopeBuild() first. */
int hopeEncode(const hopeEncoder *e, const char *key, size_t key_len, uint8_t *buffer);

/* Decode an encoded key back into its original bytes.
 *
 * enc       - the encoded bytes (as produced by hopeEncode()).
 * enc_bytes - number of valid bytes in enc, i.e. (bit_len + 7) / 8.
 * bit_len   - the encoded bit length returned by hopeEncode().
 * out       - output buffer for the decoded bytes (owned by the caller).
 *
 * Returns the number of decoded bytes, or 0 if the input could not be
 * decoded. The encoder must have been trained with hopeBuild() first. */
int hopeDecode(const hopeEncoder *e, const uint8_t *enc, size_t enc_bytes,
               int bit_len, uint8_t *out);

/* Deep-copy a trained encoder. Returns a newly allocated encoder that is
 * independent of the source (its own dictionary and decode trie), or NULL on
 * allocation failure or if e is NULL. The copy encodes/decodes identically to
 * the source, so bytes produced by the source remain valid under the copy. */
hopeEncoder *hopeDup(const hopeEncoder *e);

/* Free an encoder created by hopeCreate(). Safe to call with NULL. */
void hopeFree(hopeEncoder *e);

#endif /* HOPE_H */
