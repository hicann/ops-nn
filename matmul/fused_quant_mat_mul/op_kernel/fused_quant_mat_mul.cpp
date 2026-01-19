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
 * \file fused_quant_mat_mul.cpp
 * \brief
 */

#include "../quant_batch_matmul_v3/quant_batch_matmul_v3.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_init_output.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_cube_basic.h"
#if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_X2_SCALE == DT_FLOAT)
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_bf16_basic.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_bf16.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_bf16_opt.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken_basic.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_pertoken_opt.h"
#endif
#endif
#include "kernel_operator.h"
#include "fused_quant_mat_mul_kernel_tilingkey.h"
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_kernel_tiling_data.h"

// if run with ttk without bias, can't get DTYPE_BIAS macro
#undef DTYPE_BIAS
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
    // s8->s32
    #define DTYPE_BIAS int32_t
#else
    // fp8/hif8->fp32
    #define DTYPE_BIAS float
#endif

#define SUPPORT_PERBLOCK false

#undef ORIG_DTYPE_X1_SCALE
#undef DTYPE_X1_SCALE
#define ORIG_DTYPE_X1_SCALE DT_FLOAT
#define DTYPE_X1_SCALE float

using namespace AscendC;
using namespace matmul;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
    #define DTYPE_LOC_LOCAL int32_t
#else
    #define DTYPE_LOC_LOCAL float
#endif

#define INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(transposeX1, transposeX2, epilogueClass)                      \
    do {                                                                                                     \
        const QuantBatchMatmulV3TilingData *qBmmV3TilingData = &tilingData;                                  \
        BmmDequantPertokenBasic<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
                                transposeX2, QuantBatchMatmulV3Update, epilogueClass> op;                                   \
        op.Init(x1, x2, x2_scale, bias, x1_scale, y, user1, qBmmV3TilingData, &tPipe);                     \
        op.Process();                                                                                        \
        tPipe.Destroy();                                                                                     \
    } while (0)

template <int TRANS, int KERNEL_TEMPLATE_TYPE, int PERTOKEN, int OPTIONATTR, int FUSED_OPTYPE>
__global__ __aicore__ void fused_quant_mat_mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, 
                                               GM_ADDR y_scale, GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y_offset,
                                               GM_ADDR x2_table, GM_ADDR x3, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(QuantBatchMatmulV3TilingData);
    GET_TILING_DATA(tilingData, tiling);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
static constexpr FusedOpType fusedOpType = static_cast<FusedOpType>(FUSED_OPTYPE);
using epilogueClass = typename EpilogueTypeTraits<fusedOpType>::Type;

// 6bit from hight to low: needClean, pertoken, opt, basic, transX1, transX2
#if (ORIG_DTYPE_Y == DT_FLOAT16)  // ORIG_DTYPE_Y = fp16

#if (ORIG_DTYPE_X2_SCALE == DT_FLOAT)
    if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false false
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // false true
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true false
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {  // true true
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true, epilogueClass);
    }
#endif // ORIG_DTYPE_X2_SCALE

#else  // ORIG_DTYPE_Y = bf16
    if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_NOT_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_B_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_A_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false, epilogueClass);
    } else if constexpr (
        TRANS == QUANT_BATCH_MATMUL_V3_ALL_TRANS && KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == QUANT_BATCH_MATMUL_V3_IS_PERTOKEN && OPTIONATTR == QUANT_BATCH_MATMUL_V3_OPTION_ATTR_NONE) {
        INVOKE_FUSED_QUANT_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true, epilogueClass);
    }
#endif // ORIG_DTYPE_Y
#endif
}