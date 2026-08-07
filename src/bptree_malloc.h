/* Bptree -- allocator selection.
 *
 * Copyright (c) 2011-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of (a) the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
 *
 * This file is used in order to change the Bptree allocator at compile time.
 * Just define the following defines to what you want to use. Also add
 * the include of your alternate allocator if needed (not needed in order
 * to use the default libc allocator). */

#ifndef BPTREE_ALLOC_H
#define BPTREE_ALLOC_H
#include "zmalloc.h"
#define bpt_malloc zmalloc
#define bpt_realloc zrealloc
#define bpt_free zfree
#define bpt_malloc_usable zmalloc_usable
#define bpt_malloc_usable_size zmalloc_usable_size
#endif
