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
 * \file transpose_batch_mat_mul.cpp
 * \brief
 */
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "arch35/transpose_batch_mat_mul_tiling_key.h"
#include "arch35/transpose_batch_mat_mul_asw_kernel_advanced.h"
#include "arch35/transpose_batch_mat_mul_tiling_key_public.h"
#include "arch35/transpose_batch_mat_mul_asw_block_advanced.h"
using namespace TransposeBatchMatMulAdvanced;

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;

#define TBMM_IMPL_CLASS_COMMON_TRNAS(transA, transB, Mode, templateClass, ...)                                         \
    do {                                                                                                               \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                           \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                               \
        TPipe pipe;                                                                                                    \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;                                 \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;                                 \
        templateClass<aType, bType, cType, biasType, Mode, __VA_ARGS__> op;                                            \
        op.Init(aGM, bGM, cGM, biasGM, scalesGM, user, &tilingData, &pipe);                                            \
        op.Process();                                                                                                  \
    } while (0)

template <int8_t PERM_X1, int8_t PERM_X2, int8_t BATCH_SPLIT>
__global__ __aicore__ void transpose_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM,
                                                   GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);

    constexpr bool aTran = false;
    constexpr bool bTran = (PERM_X2 == 1);

    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if constexpr (BATCH_SPLIT == TRANSPOSE_BATCH_MAT_MUL_BATCH_SPLIT_FALSE &&
                  PERM_X1 == TRANSPOSE_BATCH_MAT_MUL_PERM_X1_0_1_2) {
        TBMM_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, TBMM_MODE::BMM_TRANS,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswKernel,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if constexpr (BATCH_SPLIT == TRANSPOSE_BATCH_MAT_MUL_BATCH_SPLIT_FALSE &&
                         PERM_X1 == TRANSPOSE_BATCH_MAT_MUL_PERM_X1_1_0_2) {
        TBMM_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, TBMM_MODE::TRANS_BMM_TRANS,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswKernel,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if constexpr (BATCH_SPLIT == TRANSPOSE_BATCH_MAT_MUL_BATCH_SPLIT_TRUE &&
                         PERM_X1 == TRANSPOSE_BATCH_MAT_MUL_PERM_X1_0_1_2) {
        TBMM_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, TBMM_MODE::BMM_TRANS_TRANS,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswKernel,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    } else if constexpr (BATCH_SPLIT == TRANSPOSE_BATCH_MAT_MUL_BATCH_SPLIT_TRUE &&
                         PERM_X1 == TRANSPOSE_BATCH_MAT_MUL_PERM_X1_1_0_2) {
        TBMM_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, TBMM_MODE::TRANS_BMM_TRANS_TRANS,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswKernel,
                                     TransposeBatchMatMulAdvanced::TransposeBatchMatMulAswBlock, MM_CFG_NO_PRELOAD);
    }
}