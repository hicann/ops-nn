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
 * \file apply_proximal_adagrad.cpp
 * \brief ApplyProximalAdagrad kernel entry (arch35 / Ascend950).
 *
 * Signature follows registry-invoke convention:
 *   all inputs -> all outputs -> workspace -> tiling
 *
 * Template parameters are driven by ASCENDC_TPL_SEL_PARAM on the host:
 *   - D_T_VAR  : input dtype (currently C_DT_FLOAT only).
 *   - PAD_TAIL : whether the per-core tail is non-32B-aligned (0/1).
 *                In iteration-2 the kernel always uses DataCopyPad on the
 *                trailing tile, so the parameter is reserved for future
 *                aligned-only fast-path optimisations and serves as the
 *                TilingKey 10001 vs 10002 discriminator.
 *   - HAS_L1   : compile-time hint that l1 may be non-zero.  When 0, the
 *                kernel statically drops the sign + soft-threshold branch
 *                (TilingKey 10003 fast path).
 */

#include "apply_proximal_adagrad.h"

template <typename D_T_VAR, uint32_t PAD_TAIL, uint32_t HAS_L1>
__global__ __aicore__ void apply_proximal_adagrad(
    GM_ADDR var, GM_ADDR accum,
    GM_ADDR lr, GM_ADDR l1, GM_ADDR l2,
    GM_ADDR grad,
    GM_ADDR var_out, GM_ADDR accum_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyProximalAdagradTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyProximalAdagradTilingData, tilingData, tiling);
    NsApplyProximalAdagrad::ApplyProximalAdagrad<D_T_VAR, (PAD_TAIL != 0U), (HAS_L1 != 0U)> op;
    op.Init(var, accum, lr, l1, l2, grad, var_out, accum_out, &tilingData);
    op.Process();
}
