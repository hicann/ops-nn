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
 * \file transpose_quant_batch_mat_mul.cpp
 * \brief
 */
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "arch35/transpose_quant_batch_mat_mul_tiling_key.h"
#include "arch35/transpose_quant_batch_mat_mul_asw_kernel_advanced.h"
#include "arch35/transpose_quant_batch_mat_mul_tiling_key_public.h"
#include "arch35/transpose_quant_batch_mat_mul_asw_block_advanced.h"

using namespace TransposeQuantBatchMatMulAdvanced;

using namespace AscendC;
using namespace matmul;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#define DTYPE_LOC_LOCAL float

#undef DTYPE_PERTOKEN_SCALE
#define DTYPE_PERTOKEN_SCALE float

#undef DTYPE_SCALE
#define DTYPE_SCALE float

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;

constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, false, false, false);

#define TQBMM_IMPL_CLASS_COMMON_TRNAS(transposeX1, transposeX2, templateClass, ...)                                    \
    do {                                                                                                               \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_PERTOKEN_SCALE, DTYPE_Y, transposeX1,         \
                      transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                                       \
            op;                                                                                                        \
        op.Init(aGM, bGM, x2_scaleGM, x1_scaleGM, cGM, user, &tilingData, &pipe);                                      \
        op.Process();                                                                                                  \
    } while (0)

template <int8_t PERM_X1, int8_t PERM_X2, int8_t BATCH_SPLIT>
__global__ __aicore__ void transpose_quant_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1_scaleGM,
                                                         GM_ADDR x2_scaleGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                         GM_ADDR tilingGM)
{
    constexpr bool aTran = false;
    constexpr bool bTran = false;
    TPipe pipe;
    GM_ADDR user = GetUserWorkspace(workspaceGM);
    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TQBMM_IMPL_CLASS_COMMON_TRNAS(aTran, bTran, TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswKernel,
                                  TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswBlock,
                                  MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
}