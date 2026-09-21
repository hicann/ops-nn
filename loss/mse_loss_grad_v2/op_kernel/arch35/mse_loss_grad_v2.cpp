/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2.cpp
 * \brief arch35 (Ascend950) kernel entry: RANK is dispatched by the TilingKey template
 *        (RANK=4 -> tilingKey 0, RANK=8 -> tilingKey 1); dtype by the framework-injected
 *        DTYPE_DOUT macro (predict/label/dout/y share one dtype).
 */

#include "kernel_operator.h"              // Ascend C kernel framework (AscendC:: namespace)
#include "mse_loss_grad_v2_tiling_key.h"  // ASCENDC_TPL_ARGS_DECL (RANK) — template compile path
#include "mse_loss_grad_v2_tiling_data.h" // MseLossGradV2TilingData<RANK> struct
#include "mse_loss_grad_v2_kernel.h"      // MseLossGradV2Kernel<T, RANK> implementation

// GET_TILING_DATA_WITH_STRUCT needs two distinct type names because the RANK=4/8
// structs differ in size. RANK_4/RANK_8 are named constants from the tiling_key
// header — never bare 4/8.
using TilingData4 = MseLossGradV2TilingData<MSE_LOSS_GRAD_V2_RANK_4>;
using TilingData8 = MseLossGradV2TilingData<MSE_LOSS_GRAD_V2_RANK_8>;

template <int RANK>
__global__ __aicore__ void mse_loss_grad_v2(GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace,
                                            GM_ADDR tiling)
{
    (void)workspace; // Broadcast needs no workspace (design §8: workspace = 0)
    GM_ADDR ins[MAX_INPUT_SLOTS] = {predict, label, dout};
    GM_ADDR outs[MAX_OUTPUT_SLOTS] = {y};

    // REGISTER_NONE_TILING must precede GET_TILING_DATA_WITH_STRUCT (two
    // TilingData sizes exist, no single default struct).
    REGISTER_NONE_TILING;
    // AIV-only task type — mandatory for templated operators.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == MSE_LOSS_GRAD_V2_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        MseLossGradV2Kernel<DTYPE_DOUT, MSE_LOSS_GRAD_V2_RANK_4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        MseLossGradV2Kernel<DTYPE_DOUT, MSE_LOSS_GRAD_V2_RANK_8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }

    // Drain all pipes before kernel exit (broadcast-standard-kernel-template
    // §7.1: the last tile's CopyOut is not covered by any MTE3_MTE2 pair).
    AscendC::PipeBarrier<PIPE_ALL>();
}
