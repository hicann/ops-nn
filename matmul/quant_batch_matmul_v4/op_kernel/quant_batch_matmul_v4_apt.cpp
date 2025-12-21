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
 * \file quant_batch_matmul_v4_apt.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
// if run with ttk without bias, can't get DTYPE_BIAS macro
#ifndef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_Y
#endif

// "kernel_operator.h" should before defined(DT_FLOAT)
#if defined(ORIG_DTYPE_X2) && defined(DT_FLOAT) && ORIG_DTYPE_X2 == DT_FLOAT
#undef DTYPE_X2
#define DTYPE_X2 fp4x2_e2m1_t
#endif

#include "arch35/quant_batch_matmul_v4_tiling_key.h"
#include "arch35/quant_batch_matmul_v4_tiling_data.h"
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
// define DTYPE_X2 should before catlass
#include "arch35/catlass_convertor.h"
#include "arch35/quant_batch_matmul_v4_constant.h"
#include "arch35/quant_batch_matmul_v4_perchannel.h"
#else
#include "../quant_batch_matmul_v3/quant_batch_matmul_v3_base.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_al1_full_load.h"
using namespace AscendC;
#endif

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
using namespace QuantBatchMatmulV4;
namespace QuantBatchMatmulV4 {
namespace Arch35 {
template <class TemplateClass>
__aicore__ inline void InvokeWeightQuantBmmOpImpl(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                  GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
                                                  GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams, tilingDataIn, tiling);
    TemplateClass op;
    op.Init(x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, userWS, &tilingDataIn,
            &tPipe);
    op.Process();
}
}  // namespace Arch35
}  // namespace QuantBatchMatmulV4
#endif

template <int TRANS, int QUANT_TYPE, int OPTION_ATTRS, int WEIGHTNZ, int KERNEL_TEMPLATE_TYPE>
__global__ __aicore__ void quant_batch_matmul_v4(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR y_scale, GM_ADDR x1_offset,
    GM_ADDR x2_offset, GM_ADDR y_offset, GM_ADDR x2_table, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
    REGISTER_TILING_DEFAULT(qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams);
    if (QUANT_TYPE == QBMMV4_PER_GROUP) {
        constexpr bool isTransA = TRANS == QBMMV4_A_TRANS || TRANS == QBMMV4_ALL_TRANS;
        constexpr bool isTransB = TRANS == QBMMV4_B_TRANS || TRANS == QBMMV4_ALL_TRANS;
        QuantBatchMatmulV4::Arch35::InvokeWeightQuantBmmOpImpl<QuantBatchMatmulV4PerChannelKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, isTransA, isTransB, false, QuantType::PER_GROUP, bfloat16_t,
            WEIGHTNZ>>(
            x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, y, workspace, tiling);
    } else if (QUANT_TYPE == QBMMV4_MX) {
        QuantBatchMatmulV4::Arch35::Catlass::InvokeActKernel<WEIGHTNZ>(KERNEL_PARAMS);
    }
#else
    GM_ADDR userWS = AscendC::GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;
    // 复用DequantBmm::QuantBatchMatmulV3TilingDataParams
    REGISTER_TILING_DEFAULT(DequantBmm::QuantBatchMatmulV3TilingDataParams);
    GET_TILING_DATA(tilingData, tiling);
    if (KERNEL_TEMPLATE_TYPE == QBMMV4_LUT_ASW) {
        MatMulASWKernel<DTYPE_X1, DTYPE_X2, uint64_t, int32_t, DTYPE_Y, CubeFormat::ND, CubeFormat::NZ, CubeFormat::ND,
                        false, false, true, FusedOpType::NONE>
            op;
        op.Init(x1, x2, bias, x2_scale, x1_scale, x2_table, y, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (KERNEL_TEMPLATE_TYPE == QBMMV4_LUT_AL1FULL) {
        QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, uint64_t, int32_t, DTYPE_Y, CubeFormat::ND,
                                                       CubeFormat::NZ, CubeFormat::ND, false, false, true,
                                                       FusedOpType::NONE>
            op;
        op.Init(x1, x2, bias, x2_scale, x1_scale, x2_table, y, userWS, &tilingData, &tPipe);
        op.Process();
    }
#endif
}
