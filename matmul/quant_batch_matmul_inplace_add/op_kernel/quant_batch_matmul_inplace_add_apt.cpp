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
 * \file quant_batch_matmul_inplace_add_apt.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "arch35/qbmmia_mx_basic_api_cmct.h"
#include "arch35/qbmmia_cube_basic_api_cmct.h"

using namespace AscendC;
using namespace matmul;

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2)
#define QBMMIA_IS_MX \
    ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2) && \
     (ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))
#define QBMMIA_IS_HIF8 (ORIG_DTYPE_X1 == DT_HIFLOAT8 && ORIG_DTYPE_X2 == DT_HIFLOAT8)
#else
#define QBMMIA_IS_MX false
#define QBMMIA_IS_HIF8 false
#endif

#define QBMMIA_MX_DISPATCH(aLayout, bLayout, fullLoadMode)                                                           \
    QbmmiaMxBasicApiKernel<                                                                                          \
        DTYPE_X1, DTYPE_X2, DTYPE_Y, aLayout, bLayout, Cmct::Gemm::layout::RowMajorAlign, fullLoadMode>(            \
        x1, x2, x2_scale, x1_scale, y, &tilingData)

#define QBMMIA_CUBE_DISPATCH(aLayout, bLayout, fullLoadMode)                                                         \
    QbmmiaCubeBasicApiKernel<                                                                                        \
        DTYPE_X1, DTYPE_X2, float, DTYPE_Y, float, aLayout, bLayout, Cmct::Gemm::layout::RowMajorAlign,             \
        fullLoadMode>(x1, x2, x2_scale, x1_scale, y, &tilingData)

#define QBMMIA_LAYOUT_DISPATCH(dispatchMacro, fullLoadMode)                                                          \
    do {                                                                                                             \
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {                                                         \
            dispatchMacro(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor, fullLoadMode);                \
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {                                                  \
            dispatchMacro(Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor, fullLoadMode);             \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {                                                  \
            dispatchMacro(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, fullLoadMode);             \
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {                                                  \
            dispatchMacro(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor, fullLoadMode);          \
        }                                                                                                            \
    } while (0)

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_KERNEL_TYPE>
__global__ __aicore__ void quant_batch_matmul_inplace_add(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR yIn, GM_ADDR x1_scale, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe tPipe;
    REGISTER_TILING_DEFAULT(QMMIA::QuantBatchMatmulInplaceAddTilingData);
    GET_TILING_DATA(tilingData, tiling);
#if QBMMIA_IS_MX
    if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) {
        QBMMIA_LAYOUT_DISPATCH(QBMMIA_MX_DISPATCH, 0);
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
        QBMMIA_LAYOUT_DISPATCH(QBMMIA_MX_DISPATCH, 1);
    }
#endif
#if QBMMIA_IS_HIF8
    if constexpr (TPL_KERNEL_TYPE == TPL_CUBE_FIXPIPE_DEFAULT_LOAD_WITH_MMAPI) {
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMMIA_CUBE_DISPATCH(Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, 0);
        }
    } else if constexpr (TPL_KERNEL_TYPE == TPL_CUBE_FIXPIPE_A_FULL_LOAD_WITH_MMAPI) {
        if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QBMMIA_CUBE_DISPATCH(
                Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::A_FULL_LOAD_MODE);
        }
    }
#endif
}
