/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_ARCH22_TILING_KEY_H
#define UNION_SPARSE_GEN_ARCH22_TILING_KEY_H

// A2 execution families and compiled resource layouts, shared by host and device.
// Keep literal values: CANN extracts these constants from the preprocessed entry.
// ORDERED_GENERAL permits single-row, tree, table and resource-dependent vector
// processing. ORDERED_BATCH_* enables eight-row batched vector preparation;
// 256/512 is the compiled vector-window limit, not K or a promise of vector work.
#define UNION_SPARSE_GEN_A2_KEY_ORDERED_GENERAL 0
#define UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_256 8
#define UNION_SPARSE_GEN_A2_KEY_ORDERED_BATCH_512 108
// Membership keys and BATCH_512 are retained but unreachable through the current
// public Y128/W8/sorted/explicit-list contract.
#define UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_TABLE 1
#define UNION_SPARSE_GEN_A2_KEY_MEMBERSHIP_ADAPTIVE 2

#endif
