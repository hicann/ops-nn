/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_kernel/bn3d_training_update_grad_apt.cpp
// =============================================================================
//
// Ascend C kernel entry for BN3DTrainingUpdateGrad (arch35 / Ascend950).
//
//   Key 传递: ASCENDC_TPL_ARGS_DECL(isGroup, isEmptyTensor)
//             -> template <bool isGroup, bool isEmptyTensor>
//   2 份 TilingData（base/group 共用 BN3DTrainingUpdateGradTilingData；empty 独立
//     BN3DTrainingUpdateGradEmptyTilingData）=> REGISTER_NONE_TILING
//     + 各分支内 GET_TILING_DATA_WITH_STRUCT
//   入参顺序 = 算子原型（grads, x, batch_mean, batch_variance -> diff_scale, diff_offset）
//     + 末尾 workspace + tiling
//   DTYPE_GRADS：框架按 REG_OP 首输入名 grads 自动生成（dtype 不进 key，编译期实例化）
// =============================================================================

#include "kernel_operator.h"
#include "arch35/bn3d_training_update_grad_tiling_struct.h"
#include "arch35/bn3d_training_update_grad_struct.h"
#include "arch35/bn3d_training_update_grad_base_kernel.h"
#include "arch35/bn3d_training_update_grad_empty_kernel.h"
#include "arch35/bn3d_training_update_grad_group_kernel.h"

template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void bn3d_training_update_grad(GM_ADDR grads, GM_ADDR x, GM_ADDR batch_mean,
                                                     GM_ADDR batch_variance, GM_ADDR diff_scale, GM_ADDR diff_offset,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) { // combine (0,1)：空 tensor
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingUpdateGradEmptyTilingData, tilingData, tiling);
        BN3DTrainingUpdateGradEmptyKernel<DTYPE_GRADS> op;
        op.Init(grads, x, batch_mean, batch_variance, diff_scale, diff_offset, &tilingData, &pipe);
        op.Process();
    } else if constexpr (isGroup) { // combine (1,0)：A 小借 R
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingUpdateGradTilingData, tilingData, tiling);
        BN3DTrainingUpdateGradGroupKernel<DTYPE_GRADS> op;
        op.InitGroup(grads, x, batch_mean, batch_variance, diff_scale, diff_offset, workspace, &tilingData, &pipe);
        op.ProcessGroup();
    } else { // combine (0,0)：else 兜底 = base
        GET_TILING_DATA_WITH_STRUCT(BN3DTrainingUpdateGradTilingData, tilingData, tiling);
        BN3DTrainingUpdateGradBaseKernel<DTYPE_GRADS> op;
        op.Init(grads, x, batch_mean, batch_variance, diff_scale, diff_offset, &tilingData, &pipe);
        op.Process();
    }
}
