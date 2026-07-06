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
 * \file apply_adam_d.cpp
 * \brief ApplyAdamD kernel entry (ascend910b / DAV_2201)
 *
 * Template parameters (matching apply_adam_d_tiling_key.h):
 *   - D_T:          var data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - USE_NESTEROV: nesterov flag {0,1}, from ASCENDC_TPL_UINT_DECL
 *
 * GM_ADDR layout (15 = 10 inputs + 3 outputs + workspace + tiling), 与现有 arch35
 * 入口 / proto(10in/3out) 一致：
 *   var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad,
 *   varOut, mOut, vOut, workspace, tiling
 * 6 个标量(beta1_power..epsilon)为 GM [1] 张量，由 kernel 在 Init 中 GM 读取。
 */
#include "apply_adam_d_kernel.h"
#include "apply_adam_d_tiling_key.h"

template <typename D_T, int USE_NESTEROV>
__global__ __aicore__ void apply_adam_d(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1_power, GM_ADDR beta2_power,
                                        GM_ADDR lr, GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad,
                                        GM_ADDR varOut, GM_ADDR mOut, GM_ADDR vOut, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyAdamDTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamDTilingData, tilingData, tiling);
    NsApplyAdamD::ApplyAdamD<D_T, USE_NESTEROV> op;
    op.Init(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, varOut, mOut, vOut, &tilingData);
    op.Process();
}
