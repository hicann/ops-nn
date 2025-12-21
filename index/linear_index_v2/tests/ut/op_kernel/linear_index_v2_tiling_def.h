/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_linear_index_v2_tiling.h
 * \brief
 */
#ifndef _FAST_OP_TEST_LINEAR_INDEX_V2_TILING_H_
#define _FAST_OP_TEST_LINEAR_INDEX_V2_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct LinearIndexV2TilingParam {
    uint64_t usedCoreNum = 0;
    uint64_t tensorId = 0;
    uint64_t formerCoreNum = 0;
    uint64_t formerCoreDataNum = 0;
    uint64_t formerCoreFormerDataNum = 0;
    uint64_t formerCoreTailDataNum = 0;
    uint64_t formerCoreFormerTime = 0;
    uint64_t formerCoreTailTime = 0;
    ;
    uint64_t tailCoreNum = 0;
    uint64_t tailCoreDataNum = 0;
    uint64_t tailCoreFormerDataNum = 0;
    uint64_t tailCoreTailDataNum = 0;
    uint64_t tailCoreFormerTime = 0;
    uint64_t tailCoreTailTime = 0;
    uint64_t indicesMask[8];
};

struct LinearIndexV2TilingData {
    LinearIndexV2TilingParam params;
};

inline void InitLinearIndexV2TilingData(uint8_t* tiling, LinearIndexV2TilingData* data)
{
    memcpy(data, tiling, sizeof(LinearIndexV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    LinearIndexV2TilingData tiling_data;         \
    InitLinearIndexV2TilingData(tiling_arg, &tiling_data)
#endif // _FAST_OP_TEST_LINEAR_INDEX_V2_TILING_H_