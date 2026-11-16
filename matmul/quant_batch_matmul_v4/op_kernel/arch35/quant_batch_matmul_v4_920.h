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
 * \file quant_batch_matmul_v4_920.h
 * \brief
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "quant_batch_matmul_v4_tiling_key.h"
#include "quant_batch_matmul_v4_tiling_data_apt.h"
#include "quant_batch_matmul_v4_general.h"

#if (defined(IS_MX_TYPE))
#include "quant_batch_matmul_v4_weight_quant_mx_cube_blaze.h"
#else
#include "quant_batch_matmul_v4_pergroup_fixpipe.h"
#include "quant_batch_matmul_v4_pergroup_vector.h"
#endif

using namespace QuantBatchMatmulV4;

// if run with ttk without bias, can't get DTYPE_BIAS macro
#ifndef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_Y
#endif

#if defined(ORIG_DTYPE_X2) && defined(DT_FLOAT) && ORIG_DTYPE_X2 == DT_FLOAT
#undef DTYPE_X2
#define DTYPE_X2 fp4x2_e2m1_t
#endif

using namespace AscendC;

namespace QuantBatchMatmulV4 {
namespace Arch35 {
template <int TRANS, int QUANT_TYPE, int KERNEL_TEMPLATE_TYPE>
__aicore__ inline void InvokeWeightQuantMatmulOpImpl920(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                        GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
                                                        GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    constexpr bool TRANS_B = TRANS == QBMMV4_B_TRANS || TRANS == QBMMV4_ALL_TRANS;
    REGISTER_TILING_DEFAULT(qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams);

#if (defined(IS_MX_TYPE) || defined(IS_FP8FP4_TYPE))
#if (defined(IS_MX_TYPE))
    if constexpr (QUANT_TYPE == QBMMV4_MX) {
        constexpr uint64_t FULL_LOAD_MODE = KERNEL_TEMPLATE_TYPE == QBMMV4_MX_BLAZE_A_FULL_LOAD ?
                                                Blaze::Gemm::A_FULL_LOAD_MODE :
                                                Blaze::Gemm::NONE_FULL_LOAD_MODE;
        InvokeWeightQuantMxCubeBlaze<FULL_LOAD_MODE>(x1, x2, bias, x1_scale, x2_scale, y, tiling);
    }
#endif
    if constexpr (QUANT_TYPE == QBMMV4_TCG) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams, tilingDataIn, tiling);
        QuantBatchMatmulV4GeneralKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, TRANS_B, DTYPE_Y> op;
        op.Init(x1, x2, x2_scale, y_scale, y, userWS, &tilingDataIn);
        op.Process();
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams, tilingDataIn, tiling);
    if constexpr (KERNEL_TEMPLATE_TYPE == QBMMV4_ANTIQUANT_VECTOR) {
        QuantBatchMatmulV4PerGroupVectorKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, TRANS_B, float, float> op;
        op.Init(x1, x2, x1_scale, x2_scale, y_scale, y, userWS, &tilingDataIn);
        op.Process();
    } else {
        AscendC::TPipe tPipe;
        QuantBatchMatmulV4PerGroupFixpipeKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, TRANS_B, float, uint64_t> op;
        op.Init(x1, x2, x1_scale, x2_scale, y_scale, y, userWS, &tilingDataIn, &tPipe);
        op.Process();
    }
#endif
}
} // namespace Arch35
} // namespace QuantBatchMatmulV4
