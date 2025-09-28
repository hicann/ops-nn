/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _GE_GROUP_NORM_SWISH_GRAD_TILING_H_
#define _GE_GROUP_NORM_SWISH_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define DTYPE_X half
#define DTYPE_GAMMA float
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#pragma pack(1)

struct GroupNormSwishGradTilingData {
    uint32_t Tiling_key;
    uint32_t N;
    uint32_t C;
    uint32_t HXW;
    uint32_t G;
    uint32_t NXG;
    uint32_t C_G;
    uint32_t task_num_per_core;
    uint32_t task_num_per_tail_core;
    uint32_t tail_core;
    uint32_t mode1_ub_cap_C_num;
    uint32_t mode1_ub_iter_C_num;
    uint32_t mode1_ub_tail_C_num;
    uint32_t mode2_ub_capacity_ele;
    uint32_t mode2_ub_iteration_num;
    uint32_t mode2_ub_tail_num;
    uint32_t workSpaceSize;
    uint32_t stage2CoreUsed;
    uint32_t castEleNum;
    uint32_t tailCastNum;
    uint32_t coreBatchParts;
    uint32_t coreBatchPartsTailRepeat;
    uint32_t repeatTime4Stage2;
    uint32_t dgamma_is_require;
    uint32_t dbeta_is_require;
    float swish_scale;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                       \
    GroupNormSwishGradTilingData tilingData;                                             \
    INIT_TILING_DATA(GroupNormSwishGradTilingData, tilingDataPointer, tilingPointer);    \
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
    (tilingData).dgamma_is_require = tilingDataPointer->dgamma_is_require;               \
    (tilingData).dbeta_is_require = tilingDataPointer->dbeta_is_require;                 \
    (tilingData).swish_scale = tilingDataPointer->swish_scale;
#endif
