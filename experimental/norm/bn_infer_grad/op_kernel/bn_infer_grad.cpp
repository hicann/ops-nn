/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file bn_infer_grad_arch32.cpp
 * \brief BnInferGrad Kernel 入口（arch32 架构）
 *
 * 模板参数说明（与 bn_infer_grad_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T_X: 数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义
 *   - SCH_MODE: 调度模式（0=CONTIGUOUS, 1=NC1HWC0）
 */

#include "common/bn_infer_grad.h"

template <typename D_T_X, int SCH_MODE>
__global__ __aicore__ void bn_infer_grad(GM_ADDR grads, GM_ADDR scale,
    GM_ADDR batch_variance, GM_ADDR x_backprop, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BnInferGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(BnInferGradTilingData, tilingData, tiling);

    if constexpr (SCH_MODE == 0) {
        // CONTIGUOUS 分支（NCHW/NHWC）
        NsBnInferGrad::BnInferGradContiguous<D_T_X> op;
        op.Init(grads, scale, batch_variance, x_backprop, workspace, &tilingData);
        op.Process();
    } else if constexpr (SCH_MODE == 1) {
        // NC1HWC0 分支
        NsBnInferGrad::BnInferGradNc1hwc0<D_T_X> op;
        op.Init(grads, scale, batch_variance, x_backprop, workspace, &tilingData);
        op.Process();
    }
}
