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
#include "kernel_basic_intf.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_data.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "arch35/qbmmia_mx_basic_api_cmct.h"

using namespace AscendC;
using namespace matmul;

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_KERNEL_TYPE>
__global__ __aicore__ void quant_batch_matmul_inplace_add(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR x2_scale, GM_ADDR yIn, GM_ADDR x1_scale, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(QMMIA::QuantBatchMatmulInplaceAddTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_WITH_MMAPI) {
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                Cmct::Gemm::layout::RowMajorAlign, 0>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                Cmct::Gemm::layout::RowMajorAlign, 0>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                Cmct::Gemm::layout::RowMajorAlign, 0>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                Cmct::Gemm::layout::RowMajorAlign, 0>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        }
    } else if constexpr (TPL_KERNEL_TYPE == TPL_NO_VEC_EPILOGUE_CUSTOM_GMTOAL1_WITH_MMAPI) {
        if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 0) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::RowMajor,
                Cmct::Gemm::layout::RowMajorAlign, 1>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 0 && TPL_BTRANS == 1) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::RowMajor, Cmct::Gemm::layout::ColumnMajor,
                Cmct::Gemm::layout::RowMajorAlign, 1>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 0) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::RowMajor,
                Cmct::Gemm::layout::RowMajorAlign, 1>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        } else if constexpr (TPL_ATRANS == 1 && TPL_BTRANS == 1) {
            QbmmiaMxBasicApiKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, Cmct::Gemm::layout::ColumnMajor, Cmct::Gemm::layout::ColumnMajor,
                Cmct::Gemm::layout::RowMajorAlign, 1>(x1, x2, x2_scale, x1_scale, y, &tilingData);
        }
    }
}