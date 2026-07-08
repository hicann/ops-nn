/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file quant_batch_matmul_v3.cpp
 * \brief
 */

#include "../../inc/macro.h"

#include "quant_batch_matmul_v3_tiling_data.h"
#include "qbmm_cube_on_the_fly.h"
#include "qbmm_cube_on_the_fly_al1_full_load.h"
#include "quant_batch_matmul_v3_apt_tiling_key.h"

#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
#include "qbmm_cube_on_the_fly_bl1_full_load.h"
#include "qbmm_cube_on_the_fly_abl1_full_load.h"
#include "qbmm_cube_on_the_fly_iterbatch.h"
#endif
#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
#include "qbmm_mix_online_dynamic.h"
#include "qbmm_mix_online_dynamic_al1_full_load.h"
#include "qbmm_mix_pertile_cmct.h"
#endif
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if IS_BLAZE == false
#if (ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0)
#include "qbmm_mx_basic_api_cmct.h"
#else
#include "qbmm_cube_basic_api_cmct.h"
#endif
#endif

#if IS_BLAZE
#include "tensor_api/tensor.h"
#include "qbmm_cube_tensor_api_blaze.h"
#include "qbmm_mix_tensor_api_blaze.h"
#include "qbmm_mix_without_batch_tensor_api_blaze.h"
#if (ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0)
#include "qbmm_mx_tensor_api_blaze.h"
#include "qbmm_mx_without_batch_tensor_api_blaze.h"
#include "qbmm_mx_l0c_pingpong.h"
#include "qbmm_mx_streamk_tensor_api_blaze.h"
#endif
#endif

#ifdef IS_A4W4I
#include "qbmm_int4_to_int8_preprocess.h"
#endif

#ifdef __CCE_KT_TEST__
#define UT_STATIC static
#else
#define UT_STATIC
#endif

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
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
#undef DTYPE_BIAS
#define DTYPE_BIAS int32_t
#endif

#if (defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_SCALE))
#define SUPPORT_PERTILE              \
    (ORIG_DTYPE_SCALE == DT_FLOAT && \
     (ORIG_DTYPE_X1 == DT_HIFLOAT8 || ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2))
#else
#define SUPPORT_PERTILE false
#endif

#undef ORIG_DTYPE_PERTOKEN_SCALE
#undef DTYPE_PERTOKEN_SCALE
#define ORIG_DTYPE_PERTOKEN_SCALE DT_FLOAT
#define DTYPE_PERTOKEN_SCALE float

using namespace AscendC;
using namespace matmul;

#if (defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_SCALE) && defined(FORMAT_X2))
#define IS_MX                                                                                                     \
    ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2 || ORIG_DTYPE_X1 == DT_FLOAT4_E2M1) && \
     (ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X2 == DT_FLOAT8_E5M2 || ORIG_DTYPE_X2 == DT_FLOAT4_E2M1) && \
     ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0)
#else
#define IS_MX false
#endif

#if defined(ORIG_DTYPE_SCALE)
#define CUBE_TEMPLATE_ND                                                                              \
    (ORIG_DTYPE_SCALE == DT_UINT64 || ORIG_DTYPE_SCALE == DT_INT64 || ORIG_DTYPE_SCALE == DT_FLOAT || \
     ORIG_DTYPE_SCALE == DT_BF16) &&                                                                  \
        FORMAT_X2 == FORMAT_ND
#else
#define CUBE_TEMPLATE_ND false
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

#define QUANT_BMMV3_IMPL_CLASS(formatX1, formatX2, formatY, transposeX1, transposeX2, templateClass, ...)             \
    do {                                                                                                              \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_PERTOKEN_SCALE, DTYPE_Y, formatX1, formatX2, \
                      formatY, transposeX1, transposeX2, DTYPE_LOC_LOCAL, __VA_ARGS__>                                \
            op;                                                                                                       \
        op.Init(x1, x2, scale, offset, bias, pertokenScale, y, user1, &tilingData, &tPipe);                           \
        op.Process();                                                                                                 \
    } while (0)

#if defined(ORIG_DTYPE_SCALE) && ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0 && IS_BLAZE
#define QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                               \
    do {                                                                                                       \
        if ASCEND_IS_AIC {                                                                                     \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling); \
            QbmmMxTensorApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>(       \
                x1, x2, scale, bias, pertokenScale, y, &tilingData);                                           \
        }                                                                                                      \
    } while (0)

#define QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                       \
    do {                                                                                                             \
        if ASCEND_IS_AIC {                                                                                           \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData, tilingData,   \
                                        tiling);                                                                     \
            QbmmMxWithoutBatchTensorApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>( \
                x1, x2, scale, bias, pertokenScale, y, &tilingData);                                                 \
        }                                                                                                            \
    } while (0)

#define QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)       \
    do {                                                                                               \
        if ASCEND_IS_AIC {                                                                             \
            GET_TILING_DATA_WITH_STRUCT(                                                               \
                DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData, tilingData, tiling);    \
            QbmmMxL0CPingpongWithoutBatchTensorApiKernel<                                                   \
                DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>(                 \
                x1, x2, scale, bias, pertokenScale, y, &tilingData);                                   \
        }                                                                                              \
    } while (0)

#define QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                             \
    do {                                                                                                       \
        if ASCEND_IS_AIC {                                                                                     \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling); \
            QbmmMxL0CPingpongTensorApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>( \
                x1, x2, scale, bias, pertokenScale, y, &tilingData);                                           \
        }                                                                                                      \
    } while (0)

#define QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                          \
    do {                                                                                                          \
        GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData, tilingData, tiling); \
        QbmmMxStreamKBasicApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>(        \
            x1, x2, scale, bias, pertokenScale, y, user1, &tilingData);                                           \
    } while (0)
#elif defined(ORIG_DTYPE_SCALE) && ORIG_DTYPE_SCALE == DT_FLOAT8_E8M0
#define QUANT_BMMV3_MX_CMCT_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                                \
    do {                                                                                                       \
        if ASCEND_IS_AIC {                                                                                     \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling); \
            QbmmMxBasicApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, cLayout, fullLoadMode>(        \
                x1, x2, scale, bias, pertokenScale, y, &tilingData);                                           \
        }                                                                                                      \
    } while (0)
#endif

#if IS_BLAZE
#define QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                              \
    do {                                                                                                             \
        if ASCEND_IS_AIC {                                                                                           \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling);       \
            QbmmCubeTensorApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, cLayout, \
                                    fullLoadMode>(x1, x2, scale, bias, pertokenScale, y, &tilingData);               \
        }                                                                                                            \
    } while (0)
#define QUANT_BMMV3_MIX_TENSOR_API_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                          \
    do {                                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling);      \
        QbmmMixTensorApiKernel<                                                                                 \
            DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, cLayout, fullLoadMode>(     \
            x1, x2, scale, bias, pertokenScale, y, &tilingData);                                                \
    } while (0)
#define QUANT_BMMV3_MIX_WITHOUT_BATCH_TENSOR_API_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                          \
    do {                                                                                                        \
        GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData, tilingData, tiling);      \
        QbmmMixWithoutBatchTensorApiKernel<                                                                     \
            DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, cLayout, fullLoadMode>(     \
            x1, x2, scale, bias, pertokenScale, y, &tilingData);                                                \
    } while (0)
#endif

// ASCEND_IS_NOT_AIV 等价于 (分离架构ASCEND_IS_AIC OR 耦合架构)
#if CUBE_TEMPLATE_ND
#define QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(aLayout, bLayout, cLayout, fullLoadMode)                                   \
    do {                                                                                                            \
        if ASCEND_IS_NOT_AIV {                                                                                      \
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling);      \
            QbmmCubeBasicApiKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, cLayout, \
                                   fullLoadMode>(x1, x2, scale, bias, pertokenScale, y, &tilingData);               \
        }                                                                                                           \
    } while (0)
#endif

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_BIASMODE, int TPL_KERNELTYPE, int TPL_APILEVEL>
UT_STATIC __global__ __aicore__ void quant_batch_matmul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
                                                           GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
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

    REGISTER_NONE_TILING;

#ifdef IS_A4W4I
#if defined(ORIG_DTYPE_SCALE) && (ORIG_DTYPE_SCALE == DT_UINT64 || ORIG_DTYPE_SCALE == DT_INT64)
    GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling);
    uint64_t m = tilingData.matmulTiling.m;
    uint64_t n = tilingData.matmulTiling.n;
    uint64_t k = tilingData.matmulTiling.k;
    uint64_t batchC = tilingData.params.batchC;
#else
    GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
    uint64_t m = tilingData.matmulTiling.M;
    uint64_t n = tilingData.matmulTiling.N;
    uint64_t k = tilingData.matmulTiling.Ka;
    uint64_t batchC = tilingData.params.batchC;
#endif
    if ASCEND_IS_AIV {
        QbmmInt4ToInt8Preprocess preprocessOp;
        preprocessOp.Init(x1, x2, user1, tPipe, m, n, k, batchC);
        preprocessOp.Process();
        tPipe.Reset();
    }
    x1 = user1;
    x2 = user1 + DequantBmm::Align(batchC * m * k * sizeof(int8_t), ALIGN_SIZE_128);
    SyncAll<false>();
#endif

#if (ORIG_DTYPE_SCALE == DT_FLOAT || ORIG_DTYPE_SCALE == DT_BF16)
#if SUPPORT_MIX_WITHOUT_BATCH_TILING_KEY
    // Bias Mode = 0;
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE && TPL_APILEVEL == TPL_API_LEVEL_BLAZE && TPL_ATRANS == 0) {
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 2;
            if constexpr (TPL_BTRANS == 0) {
                QUANT_BMMV3_MIX_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_BTRANS == 1) {
                QUANT_BMMV3_MIX_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        }
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH) { // Kernel Type = 11;
            if constexpr (TPL_BTRANS == 0) {
                QUANT_BMMV3_MIX_WITHOUT_BATCH_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_BTRANS == 1) {
                QUANT_BMMV3_MIX_WITHOUT_BATCH_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        }
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH) { // Kernel Type = 12;
            if constexpr (TPL_BTRANS == 0) {
                QUANT_BMMV3_MIX_WITHOUT_BATCH_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_BTRANS == 1) {
                QUANT_BMMV3_MIX_WITHOUT_BATCH_TENSOR_API_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        }
    } else {
        if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE && TPL_APILEVEL == TPL_API_LEVEL_HIGH) {// Bias Mode = 0;
            if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 2;
                GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
                QUANT_BMMV3_IMPL_CLASS(
                    format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS),
                    QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel, QuantBatchMatmulV3::QuantBmmAswBlock,
                    MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
            }
            if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) { // Kernel Type = 3;
                GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
                QUANT_BMMV3_IMPL_CLASS(
                    format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS),
                    QuantBatchMatmulV3::QuantBmmPertokenAL1FullLoad, QuantBatchMatmulV3::QuantBmmAswBlock,
                    MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
            }
        }
    }
#else
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE && TPL_APILEVEL == TPL_API_LEVEL_HIGH) {   // Bias Mode = 0;
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_MMAPI) { // Kernel Type = 2;
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS),
                                   static_cast<bool>(TPL_BTRANS), QuantBatchMatmulV3::QuantBmmPertokenRegbaseKernel,
                                   QuantBatchMatmulV3::QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        }
        if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) { // Kernel Type = 3;
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QUANT_BMMV3_IMPL_CLASS(format_x1, format_x2, format_y, static_cast<bool>(TPL_ATRANS),
                                   static_cast<bool>(TPL_BTRANS), QuantBatchMatmulV3::QuantBmmPertokenAL1FullLoad,
                                   QuantBatchMatmulV3::QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG);
        }
    }
#endif
#endif

    if constexpr (DequantBmm::IsMxType<DTYPE_SCALE>()) {
#if IS_MX && IS_BLAZE
#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH &&
                      TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (
            TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH &&
            TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_STREAMK_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        }
#else
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH &&
                      TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_WITHOUT_BATCH_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                              AscendC::Te::NDExtLayoutPtn,
                                                              Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (
            TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG_WITHOUT_BATCH &&
            TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_WITHOUT_BATCH_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_VEC_EPILOGUE_STREAMK_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_STREAMK_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                        AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_MX_L0C_PINGPONG &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_L0C_PINGPONG_IMPL_CLASS(
                    AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                    Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_BLAZE_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
            }
        }
#endif
#elif IS_MX
#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI && TPL_APILEVEL == TPL_API_LEVEL_BASIC) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Nz,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Zn,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Nz,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Zn,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BASIC) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Nz,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Zn,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Nz,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Zn,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            }
        }
#else
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI && TPL_APILEVEL == TPL_API_LEVEL_BASIC) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, 0);
            }
        } else if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                             TPL_APILEVEL == TPL_API_LEVEL_BASIC) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_MX_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                               Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            }
        }
#endif
#endif
    } else {
        if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) { // Bias Mode = 0
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#if CUBE_TEMPLATE_ND
#if IS_BLAZE
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_BLAZE) { // Kernel Type = 0;
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                }
            }
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_BLAZE) { // Kernel Type = 1;
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                }
            }
#else
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_BASIC) { // Kernel Type = 0;
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, 0);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, 0);
                }
            }
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_BASIC) { // Kernel Type = 1;
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                                     Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
                }
            }
#endif
#elif defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
#if IS_BLAZE
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI && TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, 0);
                }
            }
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_BLAZE) {
                if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::NDExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                    QUANT_BMMV3_CUBE_TENSOR_API_IMPL_CLASS(AscendC::Te::DNExtLayoutPtn, AscendC::Te::ZNLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn, Blaze::Gemm::A_FULL_LOAD_MODE);
                }
            }
#endif
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI && TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
                GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
                MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                                static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
            if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                          TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
                GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
                QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<
                    DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                    static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                    op;
                op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
                op.Process();
            }
#endif
#endif
        }
    }
#define QBMM_QUANT_GB_IMPL_CLASS(xLayout, wLayout, yLayout)                                                           \
    do {                                                                                                              \
        GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3BasicAPITilingData, tilingData, tiling);            \
        QbmmCmctPertileKernel<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_SCALE, float, DTYPE_Y, xLayout, wLayout, yLayout, \
                              DTYPE_LOC_LOCAL>(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);    \
    } while (0)

#if SUPPORT_PERTILE
    if constexpr (
        TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE &&
        TPL_KERNELTYPE == TPL_VEC_EPILOGUE_WITH_CUSTOM_MM &&
        TPL_APILEVEL == TPL_API_LEVEL_BASIC) { // Bias Mode = 0; Kernel Type = 4;
#if defined(FORMAT_X2) && (FORMAT_X2 == FORMAT_FRACTAL_NZ)
        // WeightNz: B in Fractal NZ format. B layout is Nz(non-transB)/Zn(transB).
        // Note: only transA=False is supported (guarded by host tiling IsCapable !transA check);
        // all 4 trans instantiations are kept for TilingKey compile-time enum completeness (ATRANS=1 is placeholder now).
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
            QBMM_QUANT_GB_IMPL_CLASS(
                Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Nz, Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMM_QUANT_GB_IMPL_CLASS(
                Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Nz, Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(
                Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::Zn, Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(
                Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::Zn, Cmct::Gemm::layout::RowMajorAlign);
        }
#else
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
            QBMM_QUANT_GB_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                     Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMM_QUANT_GB_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                     Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                     Cmct::Gemm::layout::RowMajorAlign);
        }
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
            QBMM_QUANT_GB_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                     Cmct::Gemm::layout::RowMajorAlign);
        }
#endif
    }
#endif

/*
 * 纯cube场景
 * IS_BLAZE && WEIGHT_ND  -> 3510(blaze asw/al1)       FIXED_POINT_ONLY_CUBE_TO_L0C(none)
 * IS_BLAZE && WEIGHT_NZ  -> 3510(blaze asw/al1)       FIXED_POINT_ONLY_CUBE_TO_L0C(none)
 * !IS_BLAZE && WEIGHT_ND -> 3510(cmct asw/al1)        FIXED_POINT_ONLY_CUBE_TO_L0C(cmct asw/al1/bl1; highapi iterbatch)
 * !IS_BLAZE && WEIGHT_NZ -> 3510(highapi asw/al1)     FIXED_POINT_ONLY_CUBE_TO_L0C(highapi asw/al1/bl1/abl1/iterbatch)
 */
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    if constexpr (TPL_BIASMODE == TPL_EXCLUDE_FROM_TEMPLATE) {
        // high api iterbatch
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_BMMAPI && TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QuantBatchMatmulV3::QbmmIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                    format_x2, format_y, static_cast<bool>(TPL_ATRANS),
                                                    static_cast<bool>(TPL_BTRANS), MM_CFG_MULTI_BATCH>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_BMMAPI_NO_BATCH_OUT &&
                      TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QuantBatchMatmulV3::QbmmIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1,
                                                    format_x2, format_y, static_cast<bool>(TPL_ATRANS),
                                                    static_cast<bool>(TPL_BTRANS), MM_CFG_MULTI_BATCH_NO_BATCH_OUT>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }

#if CUBE_TEMPLATE_ND && defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(DT_INT8) && \
    ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_BASIC) { // Kernel Type = 0;
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, 0);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, 0);
            }
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_BASIC) { // Kernel Type = 1;
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::A_FULL_LOAD_MODE);
            }
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_BASIC) {
            if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::B_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::B_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::B_FULL_LOAD_MODE);
            } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
                QUANT_BMMV3_CUBE_CMCT_IMPL_CLASS(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                                                 Cmct::Gemm::layout::RowMajorAlign, Cmct::Gemm::B_FULL_LOAD_MODE);
            }
        }
#else
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_HIGH) { // Kernel Type = 0;
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            MatMulASWKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                            static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_HIGH) { // Kernel Type = 1;
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QuantBatchMatmulV3::MatmulAswKernelAL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                           format_x1, format_x2, format_y,
                                                           static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOABL1_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QuantBatchMatmulV3::MatmulAswKernelABL1FullLoad<
                DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y, format_x1, format_x2, format_y,
                static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TPL_KERNELTYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOBL1_WITH_MMAPI &&
                      TPL_APILEVEL == TPL_API_LEVEL_HIGH) {
            GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TilingDataParams, tilingData, tiling);
            QuantBatchMatmulV3::MatmulAswKernelBL1FullLoad<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_BIAS, DTYPE_Y,
                                                           format_x1, format_x2, format_y,
                                                           static_cast<bool>(TPL_ATRANS), static_cast<bool>(TPL_BTRANS)>
                op;
            op.Init(x1, x2, bias, scale, pertokenScale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
#endif
    }
#endif
}
