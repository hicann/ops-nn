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
#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "transpose_quant_batch_mat_mul_tiling_key.h"
#include "transpose_quant_batch_mat_mul_asw_kernel_advanced.h"
#include "transpose_quant_batch_mat_mul_tiling_key_public.h"
#include "transpose_quant_batch_mat_mul_asw_block_advanced.h"
#include "transpose_quant_batch_mat_mul_mx.h"

using namespace TransposeQuantBatchMatMulAdvanced;

using namespace AscendC;
using namespace matmul;

#define DTYPE_LOC_LOCAL float

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef DTYPE_X1_SCALE
#define DTYPE_X1_SCALE DTYPE_X2_SCALE
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;
#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true, true,
                                                                       false, false, false);

#define TQBMM_IMPL_CLASS_COMMON_TRANS(transposeX1, transposeX2, precisionMode, templateClass, ...)            \
    do {                                                                                                      \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_Y, precisionMode, \
                      transposeX1, transposeX2, format_x2, DTYPE_LOC_LOCAL, __VA_ARGS__>                      \
            op;                                                                                               \
        op.Init(aGM, bGM, x2_scaleGM, x1_scaleGM, cGM, user, &tilingData, &pipe);                             \
        op.Process();                                                                                         \
    } while (0)

#define TQBMM_MX_BLAZE_IMPL_CLASS(aLayout, bLayout, cLayout)                                                           \
    do {                                                                                                               \
        constexpr uint64_t NON_CONTIGUOUS_TYPE = (PERM_X1 == 1) ?                                                      \
                                                     static_cast<uint64_t>(                                            \
                                                         Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1) : \
                                                     0UL;                                                              \
        if ASCEND_IS_AIC {                                                                                             \
            TqbmmMxTensorApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, cLayout, \
                                   0, PERM_X1, NON_CONTIGUOUS_TYPE>(aGM, bGM, x2_scaleGM, biasGM, x1_scaleGM, cGM,     \
                                                                    tilingData);                                       \
        }                                                                                                              \
    } while (0)

template <int8_t PERM_X1, int8_t PERM_X2, int8_t BATCH_SPLIT, int8_t PRECISION_MODE, int8_t API_LEVEL>
__global__ __aicore__ void transpose_quant_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1_scaleGM,
                                                         GM_ADDR x2_scaleGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                                         GM_ADDR tilingGM)
{
    constexpr bool isMxfp8 = std::is_same_v<DTYPE_X1, __fp8e4m3> && std::is_same_v<DTYPE_X2_SCALE, __fp8e8m0>;
    constexpr bool isMxfp4 = std::is_same_v<DTYPE_X1, __fp4e2m1x2> && std::is_same_v<DTYPE_X2_SCALE, __fp8e8m0>;
    constexpr bool aTran = false;
    constexpr bool bTran = (PERM_X2 == 1);
    TPipe pipe;
    GM_ADDR user = GetUserWorkspace(workspaceGM);
    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    GET_TILING_DATA(tilingData, tilingGM);

    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<
        bTran,
        AscendC::Std::conditional_t<(format_x2 == CubeFormat::NZ), AscendC::Te::ZNLayoutPtn,
                                    AscendC::Te::DNExtLayoutPtn>,
        AscendC::Std::conditional_t<(format_x2 == CubeFormat::NZ), AscendC::Te::NZLayoutPtn,
                                    AscendC::Te::NDExtLayoutPtn>>;
    // C GM layout
    using layoutC = AscendC::Te::NDExtLayoutPtn;

    if constexpr (API_LEVEL == static_cast<int8_t>(TQBMMApiLevel::TENSOR_LEVEL) && (isMxfp8 || isMxfp4)) {
        TQBMM_MX_BLAZE_IMPL_CLASS(layoutA, layoutB, layoutC);
    }
    if constexpr (API_LEVEL == static_cast<int8_t>(TQBMMApiLevel::HIGH_LEVEL)) {
        if constexpr (sizeof(DTYPE_X2_SCALE) == sizeof(uint64_t)) {
            TQBMM_IMPL_CLASS_COMMON_TRANS(aTran, bTran, static_cast<int8_t>(TQBMMPrecisionMode::PRECISION_MODE_HIFP8),
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswKernel,
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswBlock,
                                          MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        } else if constexpr (sizeof(DTYPE_X2_SCALE) == sizeof(float)) {
            TQBMM_IMPL_CLASS_COMMON_TRANS(aTran, bTran, static_cast<int8_t>(TQBMMPrecisionMode::PRECISION_MODE_FP8),
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswKernel,
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswBlock,
                                          MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        } else if constexpr (sizeof(DTYPE_X2_SCALE) == sizeof(uint8_t)) {
            TQBMM_IMPL_CLASS_COMMON_TRANS(aTran, bTran, static_cast<int8_t>(TQBMMPrecisionMode::PRECISION_MODE_MXFP8),
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswKernel,
                                          TransposeQuantBatchMatMulAdvanced::TransposeQuantBatchMatMulAswBlock,
                                          MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        }
    }
}
