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
 * \file bn3_d_training_update.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/bn3_d_training_update_kernel.h"        // BN3DTrainingUpdateKernel<T, RANK>
#include "arch35/bn3_d_training_update_tiling_struct.h" // BN3DTrainingUpdateTilingData<RANK>
#include "arch35/bn3_d_training_update_struct.h"        // ASCENDC_TPL_ARGS_DECL/SEL (RANK)

// Kernel entry must be a function template taking `uint32_t RANK` to match the
// ASCENDC_TPL_UINT calling convention. `extern "C"` is intentionally absent:
// a function template cannot have C linkage, and the Ascend C framework
// resolves the symbol via the auto-generated wrapper.
template <uint32_t RANK>
__global__ __aicore__ void bn3_d_training_update(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum, GM_ADDR scale,
                                                 GM_ADDR offset, GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                                 GM_ADDR mean_out, GM_ADDR variance_out, GM_ADDR batch_mean,
                                                 GM_ADDR batch_variance, GM_ADDR workspace, GM_ADDR tiling)
{
    // DTYPE_X is defined by the build chain from spec.yaml's first input name
    // ("x"); it picks the user dtype (float / half / bfloat16_t) for this
    // kernel binary variant.
    GM_ADDR ins[kMaxInputSlots] = {x, sum, square_sum, scale, offset, mean, variance};
    GM_ADDR outs[kMaxOutputSlots] = {y, mean_out, variance_out, batch_mean, batch_variance};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == 4) {
        // Branch-0 (rank(x)==4 → tilingKey=0): full Init + Process.
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingUpdateTilingDataRank4, td, tiling);
        BN3DTrainingUpdateKernel<DTYPE_X, 4> kernel;
        kernel.Init(ins, outs, workspace, &td);
        kernel.Process();
    } else {
        // Branch-1 (rank(x)==5 → tilingKey=1): 5D NDDMA (nddma_dims_=5) single
        // DataCopy covering all dims; same Init/Process/CopyInBrc/CopyOut/Sync
        // structure as Branch-0, instantiated with the RANK=5 TilingData
        // (DESIGN-BRANCH-1.md §3 / §4 / §5).
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingUpdateTilingDataRank5, td, tiling);
        BN3DTrainingUpdateKernel<DTYPE_X, 5> kernel;
        kernel.Init(ins, outs, workspace, &td);
        kernel.Process();
    }
}
