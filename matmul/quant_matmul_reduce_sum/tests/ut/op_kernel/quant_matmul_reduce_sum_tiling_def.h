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
 * \file quant_matmul_reduce_sum_tiling_def.h
 * \brief
 */
#ifndef __QUANT_MATMUL_REDUCE_SUM_TILING_DEF_H__
#define __QUANT_MATMUL_REDUCE_SUM_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"
#include "quant_matmul_reduce_sum_tiling_data.h"

#define __aicore__

#pragma pack()

#define DTYPE_X int8_t
#define DTYPE_W int8_t
#define DTYPE_Y bfloat16_t
#define DTYPE_SCALE bfloat16_t

using namespace QUANT_MATMUL_REDUCE_SUM;

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
