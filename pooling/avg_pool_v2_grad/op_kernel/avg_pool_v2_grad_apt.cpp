/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_v2_grad.cpp
 * \brief
 */
#include "arch35/avg_pool_v2_grad_simt.h"
#include "arch35/avg_pool_v2_grad_nhwc_kernel.h"
#include "arch35/avg_pool_v2_grad_nchw_kernel.h"
#include "arch35/avg_pool_v2_grad_tiling_data.h"
#include "arch35/avg_pool_v2_grad_tiling_key.h"
using namespace AscendC;
using namespace AvgPoolV2Grad;
using namespace AvgPoolV2GradNHWCNameSpace;
using namespace AvgPoolV2GradNCHWNameSpace;
template <
    uint32_t schMode, uint32_t format, uint32_t isInt32Meet, uint32_t isPad, uint32_t isCheckRange,
    uint32_t countIncludePad, uint32_t hasDivsor>
__global__ __aicore__ void avg_pool_v2_grad(
    GM_ADDR orig_input_shape, GM_ADDR input_grad, GM_ADDR out_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AvgPoolV2GradTilingData);
    if constexpr (schMode == TPL_SIMT_KERNEL) {
        REGISTER_TILING_FOR_TILINGKEY("schMode == TPL_SIMT_KERNEL", AvgPoolV2GradSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AvgPoolV2GradSimtTilingData, tilingData, tiling);
        if constexpr (isInt32Meet == TPL_INT32) {
            AvgPoolV2GradSimtNamespace::AvgPoolV2GradSimt<DTYPE_INPUT_GRAD, int32_t, format, countIncludePad, hasDivsor> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        } else {
            AvgPoolV2GradSimtNamespace::AvgPoolV2GradSimt<DTYPE_INPUT_GRAD, int64_t, format, countIncludePad, hasDivsor> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        }
    } else if constexpr (schMode == TPL_NCHW_KERNEL) {
        REGISTER_TILING_FOR_TILINGKEY("schMode == TPL_NCHW_KERNEL", AvgPoolV2GradNCHWTilingData);
        if constexpr (isInt32Meet == 1) {
            GET_TILING_DATA_WITH_STRUCT(AvgPoolV2GradNCHWTilingData, tilingData, tiling);
            AvgPoolV2GradNCHWKernel<DTYPE_INPUT_GRAD, int32_t, hasDivsor, isCheckRange, countIncludePad> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(AvgPoolV2GradNCHWTilingData, tilingData, tiling);
            AvgPoolV2GradNCHWKernel<DTYPE_INPUT_GRAD, int64_t, hasDivsor, isCheckRange, countIncludePad> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        }
    } else if constexpr (schMode == TPL_NHWC_KERNEL) {    //NHWC
        REGISTER_TILING_FOR_TILINGKEY("schMode == TPL_NHWC_KERNEL", AvgPoolV2GradNHWCTilingData);
        if constexpr (isInt32Meet == TPL_INT32){
            GET_TILING_DATA_WITH_STRUCT(AvgPoolV2GradNHWCTilingData, tilingData, tiling);
            AvgPoolV2GradNHWCNameSpace::AvgPoolV2GradKernelNHWC<DTYPE_INPUT_GRAD, int32_t, hasDivsor, isCheckRange, countIncludePad> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(AvgPoolV2GradNHWCTilingData, tilingData, tiling);
            AvgPoolV2GradNHWCNameSpace::AvgPoolV2GradKernelNHWC<DTYPE_INPUT_GRAD, int64_t, hasDivsor, isCheckRange, countIncludePad> op(&pipe, &tilingData);
            op.Init(input_grad, out_grad);
            op.Process();
        }
    }
}