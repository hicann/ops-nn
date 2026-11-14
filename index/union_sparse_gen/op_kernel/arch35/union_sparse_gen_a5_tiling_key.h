/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_ARCH35_TILING_KEY_H
#define UNION_SPARSE_GEN_ARCH35_TILING_KEY_H

// A5 algorithm and cooperation scope, shared by host and device. A subgroup has
// 8/16/32 threads; the number does not denote the input capacity K.
// Current public W8 uses SUBGROUP_8 or PACKED_TABLE only. Wider subgroups and
// RADIX_SORT remain reserved implementations, not additional public capability.
// Keep numeric literals for CANN's preprocessed kernel-key extraction.
#define UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_8 0
#define UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_16 16
#define UNION_SPARSE_GEN_A5_KEY_ORDERED_SUBGROUP_32 32
#define UNION_SPARSE_GEN_A5_KEY_PACKED_TABLE 1
#define UNION_SPARSE_GEN_A5_KEY_RADIX_SORT 2

enum class UnionSparseGenA5Algorithm { WarpMerge, PackedMembership, RadixMembership };
#endif
