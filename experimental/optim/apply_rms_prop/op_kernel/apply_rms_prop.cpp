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
 * \file apply_rms_prop.cpp
 * \brief ApplyRmsProp 算子 Kernel 入口（AscendC 标准编程）
 *
 * Kernel 签名（保留 9 参，与 op_def 输入顺序一致）：
 *   输入（4 tensor）: var, ms, mom, grad
 *   输出（3 tensor）: var_out, ms_out, mom_out
 *   + workspace + tiling
 *
 * 4 个 scalar (lr / rho / momentum / epsilon) 通过 Attr 下发，
 * 经 Host Tiling 写入 TilingData，由 Kernel 直接读取。
 *
 * 多 dtype 分发（编译期）：
 *   - C_DT_FLOAT   → ApplyRmsPropFp32<float>
 *   - C_DT_FLOAT16 → ApplyRmsPropCast<half>
 *   - C_DT_BF16    → ApplyRmsPropCast<bfloat16_t>
 */

#include "kernel_operator.h"
#include "apply_rms_prop.h"

template <typename D_T_VAR>
__global__ __aicore__ void apply_rms_prop(
    GM_ADDR var, GM_ADDR ms, GM_ADDR mom, GM_ADDR grad,
    GM_ADDR var_out, GM_ADDR ms_out, GM_ADDR mom_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    REGISTER_TILING_DEFAULT(ApplyRmsPropTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyRmsPropTilingData, tilingData, tiling);

    if constexpr (std::is_same_v<D_T_VAR, float>) {
        NsApplyRmsProp::ApplyRmsPropFp32<float> op;
        op.Init(var, ms, mom, grad, var_out, ms_out, mom_out, &tilingData);
        op.Process();
    } else {
        // half / bfloat16_t —— Cast 到 fp32 计算后 Cast 回 T
        NsApplyRmsProp::ApplyRmsPropCast<D_T_VAR> op;
        op.Init(var, ms, mom, grad, var_out, ms_out, mom_out, &tilingData);
        op.Process();
    }
}
