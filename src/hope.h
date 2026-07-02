/* hope.h - HOPE (High-speed Order-Preserving Encoder), 3-gram scheme.
 *
 * C port of the 3-gram encoder from the HOPE project
 * (https://github.com/efficient/HOPE, SIGMOD 2020), extended with a decoder
 * (the reference 3-gram encoder is encode-only). It is a dictionary-based
 * compressor that encodes arbitrary byte strings into shorter byte strings
 * while preserving their lexicographic order: for any two keys a and b,
 *   memcmp(a, b) < 0  <=>  memcmp(encode(a), encode(b)) < 0.
 *
 * The 3-gram scheme selects a set of variable-length symbols: the most
 * frequent 3-grams in the sample plus the "gap" intervals between them, so
 * that every possible byte string maps to exactly one interval. Each interval
 * gets an optimal order-preserving (Hu-Tucker) variable-length code. Encoding
 * repeatedly finds the interval covering the current position (longest-prefix
 * match), emits its code, and advances by the interval's shared-prefix length.
 * Because every string in an interval shares that prefix, decoding is lossless:
 * each code maps back to exactly the prefix bytes it consumed.
 *
 * Copyright 2020, Carnegie Mellon University (original C++ implementation).
 * Licensed under the Apache License 2.0.
 */

#ifndef HOPE_H
#define HOPE_H

#include <stddef.h>
#include <stdint.h>

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
 * Returns 0 on success, -1 on error (empty input, allocation failure, or the
 * trained code would need more than 32 bits per symbol). On failure the caller
 * should fall back to storing keys uncompressed. hopeBuild() may be called
 * again to retrain an existing encoder. */
int hopeBuild(hopeEncoder *e, const char **keys, const size_t *key_lens, size_t num_keys);

/* Encode key[0..key_len) into buffer, returning the encoded length in BITS.
 * The encoded byte length is (return_value + 7) / 8.
 *
 * The caller owns buffer and must ensure it is large enough. Because every
 * symbol code is at most 32 bits and consumes at least one input byte, a
 * buffer of key_len * 4 + 64 bytes is always sufficient.
 *
 * The encoder must have been trained with hopeBuild() first. */
int hopeEncode(const hopeEncoder *e, const char *key, size_t key_len, uint8_t *buffer);

/* Decode an encoded key back into its original bytes.
 *
 * enc       - the encoded bytes (as produced by hopeEncode()).
 * enc_bytes - number of valid bytes in enc, i.e. (bit_len + 7) / 8.
 * bit_len   - the encoded bit length returned by hopeEncode().
 * out       - output buffer for the decoded bytes (owned by the caller). Each
 *             code emits at most 3 bytes, so enc_bytes * 24 + 16 bytes is
 *             always sufficient.
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
