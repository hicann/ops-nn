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
 * \brief Ascend950 entry dispatch for GELU and SwiGLU MX quantization.
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

template <int TPL_ATRANS, int TPL_BTRANS, int TPL_BATCHMODE, int TPL_KERNELTYPE>
__global__ __aicore__ void quant_matmul_activation_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale,
                                                         GM_ADDR x2_scale, GM_ADDR y, GM_ADDR y_scale,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    REGISTER_NONE_TILING;

    using LayoutA = typename QbmmAQLayoutA<TPL_ATRANS>::Type;
    using LayoutBNd = typename QbmmAQLayoutBNd<TPL_BTRANS>::Type;
    using LayoutBNz = typename QbmmAQLayoutBNz<TPL_BTRANS>::Type;

    {
        if constexpr (TPL_BATCHMODE == TPL_WITH_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(QMMAQ::QMMAQTilingData, tilingData, tiling);
#if defined(FORMAT_X2) && defined(FORMAT_FRACTAL_NZ) && FORMAT_X2 == FORMAT_FRACTAL_NZ
            if constexpr (TPL_KERNELTYPE == TPL_GELU_NO_FULLLOAD) {
                QuantMatmulGeluMxQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                             asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_GELU_FULLLOAD) {
                QuantMatmulGeluMxQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                             asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_NO_FULLLOAD) {
                QuantMatmulSwiGluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                               asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_FULLLOAD) {
                QuantMatmulSwiGluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                               asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            }
#else
            if constexpr (TPL_KERNELTYPE == TPL_GELU_NO_FULLLOAD) {
                QuantMatmulGeluMxQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                             asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_GELU_FULLLOAD) {
                QuantMatmulGeluMxQuantKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                             asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_NO_FULLLOAD) {
                QuantMatmulSwiGluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                               asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_FULLLOAD) {
                QuantMatmulSwiGluQuantMxKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                               asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            }
#endif
        } else if constexpr (TPL_BATCHMODE == TPL_WITHOUT_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(QMMAQ::QMMAQWithoutBatchTilingData, tilingData, tiling);
#if defined(FORMAT_X2) && defined(FORMAT_FRACTAL_NZ) && FORMAT_X2 == FORMAT_FRACTAL_NZ
            if constexpr (TPL_KERNELTYPE == TPL_GELU_NO_FULLLOAD) {
                QuantMatmulGeluMxQuantWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                                         asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_GELU_FULLLOAD) {
                QuantMatmulGeluMxQuantWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                                         asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_NO_FULLLOAD) {
                QuantMatmulSwiGluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                                           asc::te::nd_ext_layout_ptn,
                                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_FULLLOAD) {
                QuantMatmulSwiGluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNz,
                                                           asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            }
#else
            if constexpr (TPL_KERNELTYPE == TPL_GELU_NO_FULLLOAD) {
                QuantMatmulGeluMxQuantWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                                         asc::te::nd_ext_layout_ptn, Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_GELU_FULLLOAD) {
                QuantMatmulGeluMxQuantWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                                         asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_NO_FULLLOAD) {
                QuantMatmulSwiGluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                                           asc::te::nd_ext_layout_ptn,
                                                           Blaze::Gemm::NONE_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            } else if constexpr (TPL_KERNELTYPE == TPL_SWIGLU_FULLLOAD) {
                QuantMatmulSwiGluQuantMxWithoutBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, LayoutA, LayoutBNd,
                                                           asc::te::nd_ext_layout_ptn, Blaze::Gemm::A_FULL_LOAD_MODE>(
                    x1, x2, bias, x1_scale, x2_scale, y, y_scale, workspace, &tilingData);
            }
#endif
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
