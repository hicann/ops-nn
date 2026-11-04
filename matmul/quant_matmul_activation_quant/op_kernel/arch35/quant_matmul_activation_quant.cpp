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
 * \file quant_matmul_activation_quant.cpp
 * \brief DAV_3510 entry dispatch for GELU, SwiGLU and SiLU MX quantization.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "quant_matmul_activation_quant_tiling_data.h"
#include "quant_matmul_activation_quant_tiling_key.h"
#include "quant_matmul_activation_mx_quant.h"
#include "quant_matmul_activation_mx_quant_without_batch.h"
#include "tensor_api/tensor.h"

namespace {
template <int TPL_TRANS>
struct QbmmAQLayoutA;

template <>
struct QbmmAQLayoutA<0> {
    using Type = asc::te::nd_ext_layout_ptn;
};

template <>
struct QbmmAQLayoutA<1> {
    using Type = asc::te::dn_ext_layout_ptn;
};

template <int TPL_TRANS>
struct QbmmAQLayoutBNd;

template <>
struct QbmmAQLayoutBNd<0> {
    using Type = asc::te::nd_ext_layout_ptn;
};

template <>
struct QbmmAQLayoutBNd<1> {
    using Type = asc::te::dn_ext_layout_ptn;
};

template <int TPL_TRANS>
struct QbmmAQLayoutBNz;

template <>
struct QbmmAQLayoutBNz<0> {
    using Type = asc::te::nz_layout_ptn;
};

template <>
struct QbmmAQLayoutBNz<1> {
    using Type = asc::te::zn_layout_ptn;
};

} // namespace

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_BATCHMODE, int TPL_FULLLOADMODE, int TPL_EPILOGUETYPE>
__global__ __aicore__ void quant_matmul_activation_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                         GM_ADDR x2_scale, GM_ADDR y, GM_ADDR y_scale,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_NONE_TILING;

    using LayoutA = typename QbmmAQLayoutA<TPL_ATRANS>::Type;
#if defined(FORMAT_X2) && defined(FORMAT_FRACTAL_NZ) && FORMAT_X2 == FORMAT_FRACTAL_NZ
    using LayoutB = typename QbmmAQLayoutBNz<TPL_BTRANS>::Type;
#else
    using LayoutB = typename QbmmAQLayoutBNd<TPL_BTRANS>::Type;
#endif

    if constexpr (TPL_BATCHMODE == TPL_WITH_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(QMMAQ::QMMAQTilingData, tilingData, tiling);
        if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_GELU) {
            QuantMatmulGeluMxQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB, asc::te::nd_ext_layout_ptn,
                                         TPL_FULLLOADMODE>(x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace,
                                                           &tilingData);
        } else if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_SWIGLU) {
            QuantMatmulSwiGluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB, asc::te::nd_ext_layout_ptn,
                                           TPL_FULLLOADMODE>(x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace,
                                                             &tilingData);
        } else if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_SILU) {
            QuantMatmulSiluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB, asc::te::nd_ext_layout_ptn,
                                         TPL_FULLLOADMODE>(x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace,
                                                           &tilingData);
        }
    } else if constexpr (TPL_BATCHMODE == TPL_WITHOUT_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(QMMAQ::QMMAQWithoutBatchTilingData, tilingData, tiling);
        if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_GELU) {
            QuantMatmulGeluMxQuantWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB,
                                                     asc::te::nd_ext_layout_ptn, TPL_FULLLOADMODE>(
                x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
        } else if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_SWIGLU) {
            QuantMatmulSwiGluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB,
                                                       asc::te::nd_ext_layout_ptn, TPL_FULLLOADMODE>(
                x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
        } else if constexpr (TPL_EPILOGUETYPE == TPL_EPILOGUE_SILU) {
            QuantMatmulSiluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutB,
                                                     asc::te::nd_ext_layout_ptn, TPL_FULLLOADMODE>(
                x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
