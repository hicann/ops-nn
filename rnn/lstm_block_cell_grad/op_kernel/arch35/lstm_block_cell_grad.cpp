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
 * Ascend C kernel entry for LSTMBlockCellGrad on arch35 (Ascend 950).
 * TilingKey-templated entry <int32_t DTYPE, bool USE_PEEPHOLE>: the two TPL
 * parameters map one-to-one (and in order) onto the tilingKey bit segments
 * declared in lstm_block_cell_grad_struct.h (DTYPE bit[0..1] / USE_PEEPHOLE
 * bit[2]), instantiated for all 4 combinations {0, 1, 4, 5}.  The DTYPE
 * segment resolves to the C++ dtype T at compile time (float32 -> float /
 * float16 -> half), so the kernel body has zero runtime dtype/attr branching.
 *
 * 16 inputs + 5 outputs (order = OpDef order), workspace, tiling.
 */

#include "kernel_operator.h"
#include "lstm_block_cell_grad_struct.h"      // TPL DECL/SEL — tilingKey {0,1,4,5} instances
#include "lstm_block_cell_grad.h"             // unified kernel class template
#include "lstm_block_cell_grad_tiling_data.h" // 13-field TilingData summary struct

/**
 * LSTMBlockCellGrad: tilingKey-templated kernel entry.
 *
 * Template parameters (one-to-one with the tilingKey bit segments, in order —
 * lstm_block_cell_grad_struct.h):
 *   DTYPE        — 0 = float32, 1 = float16
 *   USE_PEEPHOLE — the use_peephole attr (bit[2])
 */
template <int32_t DTYPE, bool USE_PEEPHOLE>
__global__ __aicore__ void LSTMBlockCellGrad(GM_ADDR x, GM_ADDR cs_prev, GM_ADDR h_prev, GM_ADDR w, GM_ADDR wci,
                                             GM_ADDR wcf, GM_ADDR wco, GM_ADDR b, GM_ADDR i, GM_ADDR cs, GM_ADDR f,
                                             GM_ADDR o, GM_ADDR ci, GM_ADDR co, GM_ADDR cs_grad, GM_ADDR h_grad,
                                             GM_ADDR cs_prev_grad, GM_ADDR dicfo, GM_ADDR wci_grad, GM_ADDR wcf_grad,
                                             GM_ADDR wco_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING must precede GET_TILING_DATA_WITH_STRUCT (the macro
    // resolves the struct type name).
    REGISTER_NONE_TILING;

    // AIV-only task type — mandatory for TPL-templated kernels.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;

    // Deserialise the TilingData summary struct (shared with the host TilingFunc).
    GET_TILING_DATA_WITH_STRUCT(LSTMBlockCellGradTilingData, tilingData, tiling);

    // TPL template parameters -> kernel instance mapping: the DTYPE segment
    // resolves to the C++ dtype at compile time, USE_PEEPHOLE to bool; no
    // runtime dtype/attr branching inside the kernel body.
    if constexpr (DTYPE == LSTM_BLOCK_CELL_GRAD_DTYPE_FP32) { // float32 -> key 0/4
        NsLSTMBlockCellGrad::LSTMBlockCellGradKernel<float, USE_PEEPHOLE> op;
        op.Init(cs_prev, wci, wcf, wco, i, cs, f, o, ci, co, cs_grad, h_grad, cs_prev_grad, dicfo, wci_grad, wcf_grad,
                wco_grad, workspace, &tilingData, &pipe);
        op.Process();
    } else if constexpr (DTYPE == LSTM_BLOCK_CELL_GRAD_DTYPE_FP16) { // float16 -> key 1/5
        NsLSTMBlockCellGrad::LSTMBlockCellGradKernel<half, USE_PEEPHOLE> op;
        op.Init(cs_prev, wci, wcf, wco, i, cs, f, o, ci, co, cs_grad, h_grad, cs_prev_grad, dicfo, wci_grad, wcf_grad,
                wco_grad, workspace, &tilingData, &pipe);
        op.Process();
    }
    // bfloat16 / fp64 / integer dtypes are rejected by the host-side dtype
    // validation — unreachable here.
    // x / h_prev / w / b do not enter the gradient formulas (their gradients
    // are rebuilt by the caller via one dicfo matmul outside the operator);
    // they only participate in the host-side shape contract checks.
}
