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
 * \file quant_batch_matmul_v3_apt.cpp
 * \brief
 */
#include "arch35/quant_batch_matmul_v3_tiling_data.h"
#include "arch35/qbmm_cube_on_the_fly.h"
#include "arch35/qbmm_cube_on_the_fly_al1_full_load.h"
#include "arch35/quant_batch_matmul_v3_apt_tiling_key.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#include "arch35/qbmm_cube_on_the_fly_bl1_full_load.h"
#include "arch35/qbmm_cube_on_the_fly_abl1_full_load.h"
#include "arch35/qbmm_cube_on_the_fly_iterbatch.h"
#endif
#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
#include "arch35/qbmm_mix_online_dynamic.h"
#include "arch35/qbmm_mix_online_dynamic_al1_full_load.h"
#include "arch35/qbmm_mix_perblock_act.h"
#endif
#include "kernel_operator.h"

// if run with ttk without bias, can't get DTYPE_BIAS macro
#undef DTYPE_BIAS
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
// s8->s32
#define DTYPE_BIAS int32_t
#else
// fp8/hif8->fp32
#define DTYPE_BIAS float
#endif

// supportMmad平台ORIG_DTYPE_X1可以int8/int4, DTYPE_BIAS都是int32_t
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    #undef DTYPE_BIAS
    #define DTYPE_BIAS int32_t
#endif

#if (defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_SCALE))
#define SUPPORT_PERBLOCK             \
    (ORIG_DTYPE_SCALE == DT_FLOAT && \
     (ORIG_DTYPE_X1 == DT_HIFLOAT8 || ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2))
#else
#define SUPPORT_PERBLOCK false
#endif

#undef ORIG_DTYPE_PERTOKEN_SCALE
#undef DTYPE_PERTOKEN_SCALE
#define ORIG_DTYPE_PERTOKEN_SCALE DT_FLOAT
#define DTYPE_PERTOKEN_SCALE float

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

// DTYPE_LOC_LOCAL的使用场景是: scale 是 float/bfloat16，mix场景
// supportMmadS8S4平台还没有mix场景，这里可以不做修改。将来有mix场景的话，注意此处适配
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
#define DTYPE_LOC_LOCAL int32_t
#else
#define DTYPE_LOC_LOCAL float
#endif

#if defined(ORIG_DTYPE_X2) && defined(DT_INT32) && ORIG_DTYPE_X2 == DT_INT32
    #undef DTYPE_X2
    #define DTYPE_X2 AscendC::int4b_t
#endif

#define QUANT_BMMV3_IMPL_CLASS(formatX1, formatX2, formatY, transposeX1, transposeX2, templateClass, ...)            \
    do {                                                                                                             \
        templateClass<                                                                                               \
            DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_PERTOKEN_SCALE, DTYPE_Y, formatX1, formatX2, formatY, \
            transposeX1, transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                                  \
            op;                                                                                                      \
        op.Init(x1, x2, scale, offset, bias, pertokenScale, y, user1, &tilingData, &tPipe);                          \
        op.Process();                                                                                                \
    } while (0)

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_BIASMODE, int TPL_KERNELTYPE>
__global__ __aicore__ void quant_batch_matmul_v3(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
    GM_ADDR workSpace, GM_ADDR tiling)
{
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(DequantBmm::QuantBatchMatmulV3TilingDataParams);
    GET_TILING_DATA(tilingData, tiling);

#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) {         // Bias Mode = 0;
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 2;
            QUANT_BMMV3_IMPL_CLASS(
                format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS),
                QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, QuantBatchMatmulV3::QuantBmmAswBlock,
                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);

        }
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) { // Kernel Type = 3;
            QUANT_BMMV3_IMPL_CLASS(
                format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS),
                QuantBatchMatmulV3::QuantBmmPertokenAL1FullLoad, QuantBatchMatmulV3::QuantBmmAswBlock,
                MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        }
    }
#endif
    if constexpr (DequantBmm::IsMxType<DTYPE_SCALE>()) {
        if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) {            // Bias Mode = 0;
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 0;
                MatMulASWKernel<
                    DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                    static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
            if constexpr (
                TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE && TPL_ATRANS == 0 &&
                TPL_BTRANS == 1) { // Kernel Type = 1, ATrans = 0, BTrans = 1;
                QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<
                    DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                    static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
        }
    } else {
        if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) {            // Bias Mode = 0
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 0;
                MatMulASWKernel<
                    DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                    static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) { // Kernel Type = 1;
                QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<
                    DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                    static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
        }
    }
#define QBMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout)                                                             \
    do {                                                                                                               \
        QbmmActPerTileKernel<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                            DTYPE_LOC_LOCAL>(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);  \
    } while (0)

#if SUPPORT_PERBLOCK
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE && TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_CUSTOM_MM ) {             // Bias Mode = 0; Kernel Type = 4;
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) { 
            QBMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajor, Act::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::RowMajor, Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajorAlign);
        }
        if constexpr(TPL_ATRANS == 1 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::ColumnMajor, Act::Gemm::layout::RowMajorAlign);
        }
    }
#endif

// supportMmadS8S4平台独有模板
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) {
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI) {
            QuantBatchMatmulV3::MatmulAswKernelABL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                        format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
            op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_BMMAPI) {
             QuantBatchMatmulV3::QbmmIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), MM_CFG_MULTI_BATCH>
            op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT) {
            QuantBatchMatmulV3::QbmmIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS), MM_CFG_MULTI_BATCH_NO_BATCH_OUT>
            op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr(TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI) {
            QuantBatchMatmulV3::MatmulAswKernelBL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                       format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
            op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
    }
    
#endif
}