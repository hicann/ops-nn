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
 * \file bn3_d_training_reduce_grad.cpp
 * \brief BN3DTrainingReduceGrad 的 Ascend C kernel 入口
 */

#include "kernel_operator.h"                              // Ascend C kernel framework
#include "arch35/bn3d_training_reduce_grad_kernel.h"      // 内含 struct.h（ASCENDC_TPL_ARGS_DECL）
#include "arch35/bn3d_training_reduce_grad_tiling_data.h" // BN3DTrainingReduceGradTilingData<RANK>

using TilingData4 = BN3DTrainingReduceGradTilingData<4>;
using TilingData8 = BN3DTrainingReduceGradTilingData<8>;

// ===========================================================================
// template<int RANK> __global__ __aicore__ void bn3_d_training_reduce_grad(...)
//
// 主 NPU kernel 函数，每个 AIV core 执行一次。
//
// 参数（GM_ADDR 顺序与 proto.md / OpDef 声明序一致）:
//   grads / x / diff_scale / diff_offset / scale / batch_mean / batch_variance
//     — 7 输入（前 2 个 dtype=DTYPE_GRADS, 后 5 个恒 f32）
//   y          — 输出（shape = grads.shape, dtype = grads.dtype）
//   workspace  — 本算子无 workspace
//   tiling     — TilingData 缓冲（BN3DTrainingReduceGradTilingData<RANK>）
// ===========================================================================
template <int RANK>
__global__ __aicore__ void bn3_d_training_reduce_grad(GM_ADDR grads, GM_ADDR x, GM_ADDR diff_scale, GM_ADDR diff_offset,
                                                      GM_ADDR scale, GM_ADDR batch_mean, GM_ADDR batch_variance,
                                                      GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[7] = {grads, x, diff_scale, diff_offset, scale, batch_mean, batch_variance};
    GM_ADDR outs[1] = {y};
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        BN3DTrainingReduceGradKernel<DTYPE_GRADS, 4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        BN3DTrainingReduceGradKernel<DTYPE_GRADS, 8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
