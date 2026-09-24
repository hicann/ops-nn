/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "gru_block_cell_tiling_struct.h"
#include "gru_block_cell_struct.h"
#include "gru_block_cell_kernel.h"

// Kernel 入口（MIX 1 AIC : 2 AIV）。签名对齐 proto：6 入 4 出 + workspace + tiling。
template <uint32_t dtype>
__global__ __aicore__ void gru_block_cell(GM_ADDR x, GM_ADDR hPrev, GM_ADDR wRu, GM_ADDR wC, GM_ADDR bRu, GM_ADDR bC,
                                          GM_ADDR r, GM_ADDR u, GM_ADDR c, GM_ADDR h, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GruBlockCellTilingData);
    // ⚠ MIX 任务类型只能用本宏：__mix__(1,2) 属性形式在框架对入口改名后被拒，
    // 不写则 arch35 不自动推导任务类型。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static_assert(dtype == GRU_BLOCK_CELL_TPL_FP32, "GruBlockCell is fp32-only");
    AscendC::InitSocState(); // MIX 跨核通信前置

    GET_TILING_DATA_WITH_STRUCT(GruBlockCellTilingData, td, tiling);

    // host gate 已拒非法域，此处为防御性复检：下界挡空 tensor，上界挡
    // int64→uint32 窄化回绕（见 tiling_struct.h GRU_BLOCK_CELL_MAX_DIM）。
    // 本算子无 GM workspace（中间量全走片上：L1 常驻 + UB drain + UB→L1 回灌），
    // workspace 形参不消费。
    if (td.batchSize <= 0 || td.inputSize <= 0 || td.hiddenSize <= 0 || td.coreNumUsed <= 0 ||
        td.batchSize > GRU_BLOCK_CELL_MAX_DIM || td.inputSize > GRU_BLOCK_CELL_MAX_DIM ||
        td.hiddenSize > GRU_BLOCK_CELL_MAX_DIM) {
        return;
    }

    GruBlockCellKernel op;
    op.Init(x, hPrev, wRu, wC, bRu, bC, r, u, c, h, &td);
    op.Process();
}
