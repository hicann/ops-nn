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
 * \file cla_gate_backward_apt.cpp
 * \brief ClaGateBackward arch35 kernel entry.
 */

#include "arch35/cla_gate_backward.h"
#include "arch35/cla_gate_backward_tiling_key.h"

using namespace ClaGateBackwardOps;

template <typename T>
__aicore__ inline void RunClaGateBackward(GM_ADDR grad_merged, GM_ADDR global_attn, GM_ADDR local_attn,
                                          GM_ADDR global_gate_logits, GM_ADDR local_gate_logits,
                                          GM_ADDR grad_global_attn_out, GM_ADDR grad_local_attn_out,
                                          GM_ADDR grad_global_gate_logits_out, GM_ADDR grad_local_gate_logits_out,
                                          const ClaGateBackwardTilingData* tilingData, TPipe* pipe)
{
    ClaGateBackwardKernel<T> op;
    op.Init(grad_merged, global_attn, local_attn, global_gate_logits, local_gate_logits, grad_global_attn_out,
            grad_local_attn_out, grad_global_gate_logits_out, grad_local_gate_logits_out, tilingData, pipe);
    op.Process();
}

template <uint64_t KERNEL_MODE>
__global__ __aicore__ void cla_gate_backward(GM_ADDR grad_merged, GM_ADDR global_attn, GM_ADDR local_attn,
                                             GM_ADDR global_gate_logits, GM_ADDR local_gate_logits,
                                             GM_ADDR grad_global_attn_out, GM_ADDR grad_local_attn_out,
                                             GM_ADDR grad_global_gate_logits_out, GM_ADDR grad_local_gate_logits_out,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC || workspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(ClaGateBackwardTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClaGateBackwardTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (KERNEL_MODE == 0) {
        // dtype 分支：5 路输入同 dtype，由 DTYPE_GRAD_MERGED 决定模板实例
        if constexpr (std::is_same<DTYPE_GRAD_MERGED, bfloat16_t>::value) {
            RunClaGateBackward<bfloat16_t>(grad_merged, global_attn, local_attn, global_gate_logits, local_gate_logits,
                                           grad_global_attn_out, grad_local_attn_out, grad_global_gate_logits_out,
                                           grad_local_gate_logits_out, &tilingData, &pipe);
        } else {
            RunClaGateBackward<half>(grad_merged, global_attn, local_attn, global_gate_logits, local_gate_logits,
                                     grad_global_attn_out, grad_local_attn_out, grad_global_gate_logits_out,
                                     grad_local_gate_logits_out, &tilingData, &pipe);
        }
    }
    pipe.Destroy();
}
