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
 * \file apply_ftrl.cpp
 * \brief ApplyFtrl kernel entry (ascend910b / DAV_2201).
 *
 * Signature follows the registry-invoke convention:
 *   all inputs (8) -> all declared outputs (1: var_out) -> workspace -> tiling.
 * accum / linear are updated in place through their own input GM pointers
 * (Ref Tensor), so they do not appear as additional kernel output ports
 * (DESIGN s3.4).
 *
 * Template parameters are driven by ASCENDC_TPL_SEL_PARAM on the host:
 *   - D_T_VAR  : input dtype (C_DT_FLOAT16 / C_DT_BF16 / C_DT_FLOAT).
 *   - PAD_TAIL : whether the per-core tail is non-32B-aligned (0/1).
 *   - HAS_L1   : compile-time hint that l1 may be non-zero (0/1).
 */

#include "apply_ftrl_kernel.h"

template <typename D_T_VAR, uint32_t PAD_TAIL, uint32_t HAS_L1>
__global__ __aicore__ void apply_ftrl(GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad, GM_ADDR lr, GM_ADDR l1,
                                      GM_ADDR l2, GM_ADDR lr_power, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyFtrlTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyFtrlTilingData, tilingData, tiling);
    NsApplyFtrl::ApplyFtrl<D_T_VAR, (PAD_TAIL != 0U), (HAS_L1 != 0U)> op;
    // accum / linear are both read and written back through their input GM (in-place).
    op.Init(var, accum, linear, grad, lr, l1, l2, lr_power, var_out, &tilingData);
    op.Process();
}
