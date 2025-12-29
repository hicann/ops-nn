/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_v3_apt.cpp
 * \brief
 */
#if defined(__DAV_C310__)
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "arch35/gemm_v3_tiling_key.h"
#include "../mat_mul_v3/arch35/mat_mul_asw_kernel.h"

#else
#include "../mat_mul_v3/mat_mul_base_kernel.h"
#endif

using namespace AscendC;
using namespace matmul;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif
#ifndef DTYPE_A
#define DTYPE_A half
#endif
#ifndef DTYPE_B
#define DTYPE_B half
#endif
#ifndef DTYPE_C
#define DTYPE_C half
#endif
#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#define MMV3_IMPL_CLASS_TRNAS(transA, transB, templateClass, ...)                          \
    do {                                                                                   \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_C>;         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;   \
        TPipe pipe;                                                                        \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_A, transA>; \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_B, transB>; \
                                                                                           \
        /* 创建模板类实例并初始化 */                                                         \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                      \
        op.Init(aGM, bGM, yGM, biasGM, nullptr, user, &tilingData, &pipe);                 \
                                                                                           \
        /* 执行处理逻辑 */                                                                  \
        op.Process(1);                                                                     \
    } while (0)

template <
    int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, int8_t MODEL, int8_t FULL_LOAD,
    int8_t L0C2OUT_MODEL>
__global__ __aicore__ void gemm_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);
    // 累加场景，c和y同维度，不支持bias
    GM_ADDR biasGM = nullptr;
    constexpr bool aTran = (A_TRANS == 1);
    constexpr bool bTran = (B_TRANS == 1);
#if defined(__DAV_C310__)
    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // GemmV3复用matmulV3 kernel和tilingKey，暂只支持aswt模板
    if constexpr (
        API_LEVEL == MAT_MUL_HIGH_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        MMV3_IMPL_CLASS_TRNAS(
            aTran, bTran, MatmulV3Advanced::MatmulAswKernel, MatmulV3Advanced::MatmulAswBlock, MM_CFG_NO_PRELOAD);
    }
#endif
}