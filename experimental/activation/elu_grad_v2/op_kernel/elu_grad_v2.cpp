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
 * @file elu_grad_v2.cpp
 */
#include "elu_grad_v2.h"

using NsEluGradV2::KernelEluGradV2;
using NsEluGradV2::KernelEluGradV2ResultMixedOutput;
using NsEluGradV2::KernelEluGradV2SingleTile;
using NsEluGradV2::KernelEluGradV2Float16AlignedFast;
using NsEluGradV2::KernelEluGradV2MixedAlignedFast;

// Dispatch specialized kernels by tiling mode selected on the host side.
 
template <uint32_t schMode>
__global__ __aicore__ void elu_grad_v2(
     GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
     (void)workspace;
    REGISTER_TILING_DEFAULT(EluGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(EluGradV2TilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    constexpr bool kMixedResultOutput =
        !NsEluGradV2::kIsSameType<DTYPE_X, DTYPE_Y> && NsEluGradV2::kIsSameType<DTYPE_Y, float>;
    if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT) {
        if constexpr (kMixedResultOutput) {
            KernelEluGradV2ResultMixedOutput<DTYPE_X, DTYPE_Y> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else {
        KernelEluGradV2<DTYPE_X, true> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
        }
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP) {
        KernelEluGradV2SingleTile<DTYPE_X, false> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT) {
        if constexpr (kMixedResultOutput) {
            KernelEluGradV2ResultMixedOutput<DTYPE_X, DTYPE_Y> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else {
        KernelEluGradV2SingleTile<DTYPE_X, true> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
        }
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_MIXED_EXP_FAST) {
        KernelEluGradV2MixedAlignedFast<DTYPE_X, false> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_MIXED_RESULT_FAST) {
        if constexpr (kMixedResultOutput) {
            KernelEluGradV2ResultMixedOutput<DTYPE_X, DTYPE_Y> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else {
        KernelEluGradV2MixedAlignedFast<DTYPE_X, true> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
        }
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_EXP_FAST) {
        if constexpr (!NsEluGradV2::kIsSameType<DTYPE_X, float> &&
                      !NsEluGradV2::kIsSameType<DTYPE_X, bfloat16_t>) {
            KernelEluGradV2Float16AlignedFast<DTYPE_X, false> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else {
            KernelEluGradV2<DTYPE_X, false> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        }
    } else if constexpr (schMode == ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_RESULT_FAST) {
        if constexpr (kMixedResultOutput) {
            KernelEluGradV2ResultMixedOutput<DTYPE_X, DTYPE_Y> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else if constexpr (!NsEluGradV2::kIsSameType<DTYPE_X, float> &&
                             !NsEluGradV2::kIsSameType<DTYPE_X, bfloat16_t>) {
            KernelEluGradV2Float16AlignedFast<DTYPE_X, true> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        } else {
            KernelEluGradV2<DTYPE_X, true> op;
            op.Init(grads, activations, y, &tilingData, &pipe);
            op.Process();
        }
    } else {
        KernelEluGradV2<DTYPE_X, false> op;
        op.Init(grads, activations, y, &tilingData, &pipe);
        op.Process();
    }
}
 
