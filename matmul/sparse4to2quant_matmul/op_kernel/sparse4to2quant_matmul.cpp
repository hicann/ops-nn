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
 * \file sparse4to2quant_matmul.cpp
 * \brief
 */

#include "sparse4to2quant_matmul.h"
#include "sparse4to2quant_matmul_tiling_data.h"
#include "kernel_operator.h"

// if run with ttk without bias, can't get DTYPE_BIAS macro
#undef DTYPE_BIAS
#define DTYPE_BIAS bfloat16_t

using namespace AscendC;
using namespace matmul;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X) && FORMAT_X == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

constexpr CubeFormat format_x2 = CubeFormat::NZ;

constexpr CubeFormat format_y = CubeFormat::ND;

extern "C" __global__ __aicore__ void sparse4to2quant_matmul(
    GM_ADDR x, GM_ADDR sparseWeight, GM_ADDR index, GM_ADDR xScale, GM_ADDR sparseWeightScale, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    // Tiling注册入口
    REGISTER_TILING_DEFAULT(SparseQmm::Sparse4to2QuantMatmulTilingData);
    // 宏方式获取TilingData
    GET_TILING_DATA_WITH_STRUCT(SparseQmm::Sparse4to2QuantMatmulTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Sparse4to2QuantMatmul<int8_t, int8_t, DTYPE_Y, format_x1, format_x2> op;
    op.Init(x, sparseWeight, index, bias, xScale, sparseWeightScale, y, user1, &tilingData, &tPipe);
    op.Process();
    tPipe.Destroy();
}