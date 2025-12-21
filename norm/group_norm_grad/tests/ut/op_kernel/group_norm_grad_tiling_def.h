/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _DIAG_V2_TILING_H_
#define _DIAG_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct GroupNormGradTilingData {
    uint32_t Tiling_key = 0;               // 0
    uint32_t N = 0;                        // 1
    uint32_t C = 0;                        // 2
    uint32_t HXW = 0;                      // 3
    uint32_t G = 0;                        // 4
    uint32_t NXG = 0;                      // 5
    uint32_t C_G = 0;                      // 6
    uint32_t task_num_per_core = 0;        // 7
    uint32_t task_num_per_tail_core = 0;   // 8
    uint32_t tail_core = 0;                // 9
    uint32_t mode1_ub_cap_C_num = 0;       // 10
    uint32_t mode1_ub_iter_C_num = 0;      // 11
    uint32_t mode1_ub_tail_C_num = 0;      // 12
    uint32_t mode2_ub_capacity_ele = 0;    // 13
    uint32_t mode2_ub_iteration_num = 0;   // 14
    uint32_t mode2_ub_tail_num = 0;        // 15
    uint32_t workSpaceSize = 0;            // 16
    uint32_t stage2CoreUsed = 0;           // 17
    uint32_t castEleNum = 0;               // 18
    uint32_t tailCastNum = 0;              // 19
    uint32_t coreBatchParts = 0;           // 20
    uint32_t coreBatchPartsTailRepeat = 0; // 21
    uint32_t repeatTime4Stage2 = 0;        // 22
    bool dx_is_require = 0;                // 23
    bool dgamma_is_require = 0;            // 24
    bool dbeta_is_require = 0;             // 25
};

#define DTYPE_X int64_t

#pragma pack(1)

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    GroupNormGradTilingData tilingData;                                                  \
    INIT_TILING_DATA(GroupNormGradTilingData, tilingDataPointer, tilingPointer);         \
    (tilingData).Tiling_key = tilingDataPointer->Tiling_key;                             \
    (tilingData).N = tilingDataPointer->N;                                               \
    (tilingData).C = tilingDataPointer->C;                                               \
    (tilingData).HXW = tilingDataPointer->HXW;                                           \
    (tilingData).G = tilingDataPointer->G;                                               \
    (tilingData).NXG = tilingDataPointer->NXG;                                           \
    (tilingData).C_G = tilingDataPointer->C_G;                                           \
    (tilingData).task_num_per_core = tilingDataPointer->task_num_per_core;               \
    (tilingData).task_num_per_tail_core = tilingDataPointer->task_num_per_tail_core;     \
    (tilingData).tail_core = tilingDataPointer->tail_core;                               \
    (tilingData).mode1_ub_cap_C_num = tilingDataPointer->mode1_ub_cap_C_num;             \
    (tilingData).mode1_ub_iter_C_num = tilingDataPointer->mode1_ub_iter_C_num;           \
    (tilingData).mode1_ub_tail_C_num = tilingDataPointer->mode1_ub_tail_C_num;           \
    (tilingData).mode2_ub_capacity_ele = tilingDataPointer->mode2_ub_capacity_ele;       \
    (tilingData).mode2_ub_iteration_num = tilingDataPointer->mode2_ub_iteration_num;     \
    (tilingData).mode2_ub_tail_num = tilingDataPointer->mode2_ub_tail_num;               \
    (tilingData).workSpaceSize = tilingDataPointer->workSpaceSize;                       \
    (tilingData).stage2CoreUsed = tilingDataPointer->stage2CoreUsed;                     \
    (tilingData).castEleNum = tilingDataPointer->castEleNum;                             \
    (tilingData).tailCastNum = tilingDataPointer->tailCastNum;                           \
    (tilingData).coreBatchParts = tilingDataPointer->coreBatchParts;                     \
    (tilingData).coreBatchPartsTailRepeat = tilingDataPointer->coreBatchPartsTailRepeat; \
    (tilingData).repeatTime4Stage2 = tilingDataPointer->repeatTime4Stage2;               \
    (tilingData).dx_is_require = tilingDataPointer->dx_is_require;                       \
    (tilingData).dgamma_is_require = tilingDataPointer->dgamma_is_require;               \
    (tilingData).dbeta_is_require = tilingDataPointer->dbeta_is_require;
#endif