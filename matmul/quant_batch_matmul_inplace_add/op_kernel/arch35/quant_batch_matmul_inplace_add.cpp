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
 * \file quant_batch_matmul_inplace_add.cpp
 * \brief
 */
#if defined(ASC_DEVKIT_MAJOR) && defined(ASC_DEVKIT_MINOR) && ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
#define IS_BLAZE true
#else
#define IS_BLAZE false
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_batch_matmul_inplace_add_tiling_data.h"
#include "quant_batch_matmul_inplace_add_tiling_key.h"

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_X2_SCALE) && \
    defined(DT_FLOAT8_E4M3FN) && defined(DT_FLOAT8_E5M2) && defined(DT_FLOAT8_E8M0)
#define QBMMIA_IS_MX                                                                                 \
    ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2) &&                       \
     (ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X2 == DT_FLOAT8_E5M2) &&                       \
     ORIG_DTYPE_X2_SCALE == DT_FLOAT8_E8M0)
#else
#define QBMMIA_IS_MX false
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(DT_HIFLOAT8)
#define QBMMIA_IS_HIF8 (ORIG_DTYPE_X1 == DT_HIFLOAT8 && ORIG_DTYPE_X2 == DT_HIFLOAT8)
#else
#define QBMMIA_IS_HIF8 false
#endif

#include "lib/matmul_intf.h"
#if QBMMIA_IS_HIF8
#include "qbmmia_cube_basic_api_cmct.h"
#endif
#if QBMMIA_IS_MX
#if IS_BLAZE
#include "qbmmia_mx_tensor_api_blaze.h"
#include "qbmmia_mx_without_batch_tensor_api_blaze.h"
#else
#include "qbmmia_mx_basic_api_cmct.h"
#endif
#endif

using namespace AscendC;
using namespace matmul;

#if IS_BLAZE && QBMMIA_IS_MX
#define QBMMIA_MX_WITHOUT_BATCH_DISPATCH(aLayout, bLayout, fullLoadMode)                           \
    QbmmiaMxWithoutBatchTensorApiKernel<                                                           \
        DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, AscendC::Te::NDExtLayoutPtn, fullLoadMode>( \
        x1, x2, x2_scale, x1_scale, y, &tilingData)

#define QBMMIA_MX_WITHOUT_BATCH_LAYOUT_DISPATCH(fullLoadMode)                                                         \
    do {                                                                                                              \
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {                                                           \
            QBMMIA_MX_WITHOUT_BATCH_DISPATCH(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {                                                    \
            QBMMIA_MX_WITHOUT_BATCH_DISPATCH(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {                                                    \
            QBMMIA_MX_WITHOUT_BATCH_DISPATCH(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {                                                    \
            QBMMIA_MX_WITHOUT_BATCH_DISPATCH(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, fullLoadMode); \
        }                                                                                                             \
    } while (0)
#endif

#if IS_BLAZE && QBMMIA_IS_MX
#define QBMMIA_MX_BLAZE_DISPATCH(aLayout, bLayout, fullLoadMode)                                  \
    QbmmiaMxTensorApiKernel<                                                                       \
        DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, AscendC::Te::NDExtLayoutPtn, fullLoadMode>( \
        x1, x2, x2_scale, x1_scale, y, &tilingData)

#define QBMMIA_MX_BLAZE_LAYOUT_DISPATCH(fullLoadMode)                                                         \
    do {                                                                                                      \
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {                                                   \
            QBMMIA_MX_BLAZE_DISPATCH(AscendC::Te::NDExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {                                            \
            QBMMIA_MX_BLAZE_DISPATCH(AscendC::Te::NDExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {                                            \
            QBMMIA_MX_BLAZE_DISPATCH(AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn, fullLoadMode); \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {                                            \
            QBMMIA_MX_BLAZE_DISPATCH(AscendC::Te::DNExtLayoutPtn, AscendC::Te::DNExtLayoutPtn, fullLoadMode); \
        }                                                                                                     \
    } while (0)
#endif

#if QBMMIA_IS_MX && !IS_BLAZE
#define QBMMIA_MX_DISPATCH(aLayout, bLayout, fullLoadMode)                                               \
    QbmmiaMxBasicApiKernel<                                                                              \
        DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, Cmct::Gemm::layout::RowMajorAlign, fullLoadMode>( \
        x1, x2, x2_scale, x1_scale, y, &tilingData)

#define QBMMIA_MX_CMCT_LAYOUT_DISPATCH(fullLoadMode)                                                            \
    do {                                                                                                        \
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {                                                     \
            QBMMIA_MX_DISPATCH(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, fullLoadMode);       \
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {                                              \
            QBMMIA_MX_DISPATCH(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, fullLoadMode);    \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {                                              \
            QBMMIA_MX_DISPATCH(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, fullLoadMode);    \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {                                              \
            QBMMIA_MX_DISPATCH(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor, fullLoadMode); \
        }                                                                                                       \
    } while (0)
#endif

#if QBMMIA_IS_HIF8
#define QBMMIA_CUBE_DISPATCH(aLayout, bLayout, fullLoadMode)                                                           \
    QbmmiaCubeBasicApiKernel<                                                                                          \
        DTYPE_X1, DTYPE_X2, float, DTYPE_Y, float, aLayout, bLayout, Cmct::Gemm::layout::RowMajorAlign, fullLoadMode>( \
        x1, x2, x2_scale, x1_scale, y, &tilingData)
#endif

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_KERNEL_TYPE>
__global__ __aicore__ void quant_batch_matmul_inplace_add(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR yIn, GM_ADDR x1_scale, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe tPipe;
    REGISTER_NONE_TILING;
#if QBMMIA_IS_MX && IS_BLAZE
    if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI_WITHOUT_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(
            QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData, tilingData, tiling);
        QBMMIA_MX_WITHOUT_BATCH_LAYOUT_DISPATCH(0);
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI_WITHOUT_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(
            QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData, tilingData, tiling);
        QBMMIA_MX_WITHOUT_BATCH_LAYOUT_DISPATCH(Blaze::Gemm::A_FULL_LOAD_MODE);
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        QBMMIA_MX_BLAZE_LAYOUT_DISPATCH(0);
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        QBMMIA_MX_BLAZE_LAYOUT_DISPATCH(Blaze::Gemm::A_FULL_LOAD_MODE);
    }
#elif QBMMIA_IS_MX
    if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        QBMMIA_MX_CMCT_LAYOUT_DISPATCH(0);
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        QBMMIA_MX_CMCT_LAYOUT_DISPATCH(Cmct::Gemm::A_FULL_LOAD_MODE);
    }
#endif
#if QBMMIA_IS_HIF8
    if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMMIA_CUBE_DISPATCH(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, 0);
        }
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
        GET_TILING_DATA_WITH_STRUCT(QMMIA::QuantBatchMatmulInplaceAddTilingData, tilingData, tiling);
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMMIA_CUBE_DISPATCH(
                Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::A_FULL_LOAD_MODE);
        }
    }
#endif
}
