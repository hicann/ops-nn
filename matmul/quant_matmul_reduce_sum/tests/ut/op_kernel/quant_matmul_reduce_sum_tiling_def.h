/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_sum_tiling_def.h
 * \brief
 */
#ifndef __QUANT_MATMUL_REDUCE_SUM_TILING_DEF_H__
#define __QUANT_MATMUL_REDUCE_SUM_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define __aicore__

struct QuantMatmulReduceSumParams {
    uint32_t batchNum = 0;
    uint32_t coreNum = 0;
    uint32_t ubBaseK = 0;
    uint32_t ubBaseN = 0;
    uint32_t ubRestBytes = 0;
    uint32_t ubCalSize = 0;
    uint32_t isPertoken = 0;
    uint32_t isDetermine = 0;
    uint64_t workspaceSize = 0;
};

struct QuantMatmulReduceSumTilingData {
    QuantMatmulReduceSumParams qbmmReduceSumParams;
    TCubeTiling matmulTiling;
};

#pragma pack()

#define DTYPE_X int8_t
#define DTYPE_W int8_t
#define DTYPE_Y bfloat16_t
#define DTYPE_SCALE bfloat16_t

#if defined(__CCE_KT_TEST__)
template <class T>
inline __aicore__ void InitTilingData(const uint8_t* p_tilingdata, T* tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(
    const __gm__ uint8_t* p_tilingdata, T* tilingdata)
#endif
{
    memcpy(tilingdata, p_tilingdata, sizeof(QuantMatmulReduceSumTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    QuantMatmulReduceSumTilingData tiling_data;  \
    InitTilingData(tiling_arg, &tiling_data)
#endif
