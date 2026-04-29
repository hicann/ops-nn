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
 * \file apply_adamax.cpp
 * \brief ApplyAdamax kernel entry (arch35 / Ascend950)
 *
 * Template parameters (matching apply_adamax_tiling_key.h):
 *   - D_T:     Data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - K_ALIGN: 32B alignment flag (0=unaligned, 1=aligned), from ASCENDC_TPL_UINT_DECL
 *
 * GM_ADDR layout (9 = 4 input tensors + 3 output tensors + workspace + tiling).
 * The 5 scalar parameters (beta1Power, lr, beta1, beta2, epsilon) are passed via
 * TilingData (set from Attr by the Tiling function).
 */

#include "apply_adamax.h"

template <typename D_T, int K_ALIGN>
__global__ __aicore__ void apply_adamax(
    GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR mOut, GM_ADDR vOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyAdamaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdamaxTilingData, tilingData, tiling);
    NsApplyAdamax::ApplyAdamax<D_T, K_ALIGN> op;
    op.Init(var, m, v, grad, varOut, mOut, vOut, &tilingData);
    op.Process();
}
