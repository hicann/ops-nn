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
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "transpose_batch_mat_mul_tiling_key.h"
#include "transpose_batch_mat_mul_tiling_key_public.h"
#include "transpose_batch_mat_mul_asw_kernel.h"
#include "transpose_batch_mat_mul_asw_split_k_kernel.h"

using namespace AscendC;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X1
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr bool X2_IS_FRACTAL_NZ = true;
#else
constexpr bool X2_IS_FRACTAL_NZ = false;
#endif

template <int8_t PERM_X1, int8_t PERM_X2, int8_t BATCH_SPLIT, int8_t API_LEVEL, int8_t TBMODEL>
__global__ __aicore__ void transpose_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM,
                                                   GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    AscendC::InitSocState();

    constexpr bool aTran = false;
    constexpr bool bTran = (PERM_X2 == 1);
    constexpr bool IS_INT_8_OUTPUT = std::is_same_v<DTYPE_Y, signed char>;

    // Fixed-point (supportMmadS8S4 / DAV_RESV) configs use the legacy kernel/tiling.
#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
    using layoutA = AscendC::Std::conditional_t<aTran, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using layoutB = AscendC::Std::conditional_t<
        bTran, AscendC::Std::conditional_t<X2_IS_FRACTAL_NZ, asc::te::zn_layout_ptn, asc::te::dn_ext_layout_ptn>,
        AscendC::Std::conditional_t<X2_IS_FRACTAL_NZ, asc::te::nz_layout_ptn, asc::te::nd_ext_layout_ptn>>;
    // C GM layout
    using layoutC = asc::te::nd_ext_layout_ptn;
#endif

    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
    if constexpr (API_LEVEL == TRANSPOSE_BATCH_MAT_MUL_TENSOR_LEVEL && TBMODEL == TRANSPOSE_BATCH_MAT_MUL_BASIC) {
        TransposeBatchMatMulAdvanced::TBMMTensorKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                       layoutC, PERM_X1>(aGM, bGM, biasGM, scalesGM, cGM, nullptr,
                                                                         tilingData);
    } else if constexpr (API_LEVEL == TRANSPOSE_BATCH_MAT_MUL_TENSOR_LEVEL &&
                         TBMODEL == TRANSPOSE_BATCH_MAT_MUL_BASIC_SPLIT_K && !IS_INT_8_OUTPUT) {
        TransposeBatchMatMulAdvanced::TBMMSplitKTensorKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                             layoutC, PERM_X1>(aGM, bGM, biasGM, scalesGM, cGM, nullptr,
                                                                               tilingData);
    }
#endif
}
