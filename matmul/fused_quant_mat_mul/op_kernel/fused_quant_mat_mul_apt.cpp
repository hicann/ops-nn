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
 * \file fused_quant_mat_mul_apt.cpp
 * \brief
 */

#include "fused_quant_mat_mul_swiglu.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_abl1_full_load.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_al1_full_load.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_bl1_full_load.h"
#include "../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_iterbatch.h"
#include "arch35/fused_quant_mat_mul_tilingkey.h"

// if run with ttk without bias, can't get DTYPE_BIAS macro
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
// s8->s32
#define DTYPE_BIAS int32_t
#else
// fp8/hif8->fp32
#define DTYPE_BIAS float
#endif

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

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_BIASMODE, int TPL_KERNELTYPE, int TPL_OPTYPE>
__global__ __aicore__ void fused_quant_mat_mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2Scale,
                                               GM_ADDR yScale, GM_ADDR x1Offset, GM_ADDR x2Offset, GM_ADDR yOffset,
                                               GM_ADDR x2Table, GM_ADDR x3, GM_ADDR y, GM_ADDR workSpace,
                                               GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user = GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }

    if (TPL_OPTYPE == F_OPTYPE_RELU) {
        // 复用DequantBmm::QuantBatchMatmulV3TilingDataParams
        REGISTER_TILING_DEFAULT(DequantBmm::QuantBatchMatmulV3TilingDataParams);
        GET_TILING_DATA(tilingData, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 0;
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                            static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), false, FusedOpType::RELU>
                op;
            op.Init(x1, x2, bias, x2Scale, yScale, y, user, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI) {
            QuantBatchMatmulV3::MatmulAswKernelABL1FullLoad<
                DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), false, FusedOpType::RELU>
                op;
            op.Init(x1, x2, bias, x2Scale, yScale, y, user, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
            QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<
            DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
            static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), false, FusedOpType::RELU>
                op;
            op.Init(x1, x2, bias, x2Scale, yScale, y, user, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI) {
            QuantBatchMatmulV3::MatmulAswKernelBL1FullLoad<
                DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), false, FusedOpType::RELU>
                op;
            op.Init(x1, x2, bias, x2Scale, yScale, y, user, &tilingData, &tPipe);
            op.Process();
        }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    } else if (TPL_OPTYPE == F_OPTYPE_SWIGLU) {
        GET_TILING_DATA_WITH_STRUCT(FusedQuantMatmulSwigluTilingData, tilingData, tiling);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        FusedQuantMatmulSwiglu<DTYPE_X1, DTYPE_X2, DTYPE_X2_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                static_cast<bool>(TPL_ATRANS),static_cast<bool>(TPL_BTRANS)> op;
        op.Init(x1, x2, bias, x2Scale, x3, y, user, &tilingData, &tPipe);
        op.Process();
#endif
    }
}
