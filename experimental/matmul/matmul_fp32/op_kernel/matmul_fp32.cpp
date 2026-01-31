/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_fp32.cpp
 * \brief
 */

#include "matmul_fp32_common.h"
#include "matmul_fp32_base_kernel.h"
#include "matmul_fp32_bl1_fullload_kernel.h"
#include "matmul_fp32_tiling_key.h"

using namespace AscendC;
using namespace matmul;

#define DTYPE_X1 float
#define DTYPE_X2 float
#define DTYPE_Y float
#define DTYPE_BIAS float

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;


#define MM_FP32_IMPL_CLASS(templateClass, aFormat, ...)                                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulFp32RunInfo.transA == 0 && tilingData.matmulFp32RunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                           \
        } else if (tilingData.matmulFp32RunInfo.transA == 1 && tilingData.matmulFp32RunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                            \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else if (tilingData.matmulFp32RunInfo.transA == 0 && tilingData.matmulFp32RunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, false>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_X1, true>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, user, &tilingData, &pipe);                                     \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

template <int LOADMODE, int SPLITCOREMODE, int FIXOPTI, int MIXND2NZ>
__global__ __aicore__ void matmul_fp32(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulFp32TilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulFp32TilingData, tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
    if constexpr (
        LOADMODE == MAT_MUL_FP32_NO_FULLLOAD && SPLITCOREMODE == MAT_MUL_FP32_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_FP32_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_FP32_MIXND2NZ_TRUE) {
        MM_FP32_IMPL_CLASS(MatmulFp32BaseKernel, format_x1, MatmulFp32BaseBlock, MM_CFG_NO_PRELOAD);
    } else if constexpr (
        LOADMODE == MAT_MUL_FP32_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_FP32_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_FP32_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_FP32_MIXND2NZ_TRUE) {
        MM_FP32_IMPL_CLASS(MatmulFp32BL1FullLoadKernel, format_x1, MatmulFp32BaseBlock, MM_CFG_NO_PRELOAD);
    }
}
