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

/**
 * \file apply_centered_rms_prop_arch35.cpp
 * \brief ApplyCenteredRMSProp kernel entry (arch35 / Ascend950).
 *
 * Signature follows registry-invoke convention:
 *   all inputs (4 Ref + 4 scalar + 1 grad) ->
 *   all outputs (var_out / mg_out / ms_out / mom_out) ->
 *   workspace -> tiling
 *
 * Template parameter:
 *   - D_T_VAR : input dtype (C_DT_FLOAT16 -> TilingKey 2 / fp16 path,
 *                            C_DT_FLOAT   -> TilingKey 1 / fp32 path).
 *
 * Iteration-1 skeleton implements the fp16 path. The fp32 path is registered
 * in the template-arg declaration so the autogen pipeline produces both
 * binary slots, but the kernel body for fp32 will land in iteration-2.
 */

#include "apply_centered_rms_prop.h"

template <typename D_T_VAR>
__global__ __aicore__ void apply_centered_rms_prop(
    GM_ADDR var, GM_ADDR mg, GM_ADDR ms, GM_ADDR mom,
    GM_ADDR lr, GM_ADDR rho, GM_ADDR momentum, GM_ADDR epsilon,
    GM_ADDR grad,
    GM_ADDR var_out, GM_ADDR mg_out, GM_ADDR ms_out, GM_ADDR mom_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyCenteredRMSPropTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyCenteredRMSPropTilingData, tilingData, tiling);
    NsApplyCenteredRMSProp::ApplyCenteredRMSProp<D_T_VAR> op;
    op.Init(var, mg, ms, mom, lr, rho, momentum, epsilon, grad,
            var_out, mg_out, ms_out, mom_out, &tilingData);
    op.Process();
}
