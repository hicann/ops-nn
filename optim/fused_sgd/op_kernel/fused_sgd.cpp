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
 * \file fused_sgd.cpp
 * \brief
 */
#include "kernel_operator_list_tensor_intf.h"
#include "fused_sgd_f32.h"
#include "fused_sgd_f16_bf16.h"

using namespace AscendC;
using namespace FusedSgd;

#ifdef __CCE_UT_TEST__
extern "C" __global__ __aicore__ void fused_sgd(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list,
                                                GM_ADDR grad_scale, GM_ADDR params_ref, GM_ADDR grads_ref,
                                                GM_ADDR momentum_buffer_list_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(FusedSgdTilingData, tilingData, tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint64_t tensorStart = static_cast<uint64_t>(blockIdx) * tilingData.tensorsPerCore;
    uint64_t tensorEnd = tensorStart + tilingData.tensorsPerCore;
    if (tensorEnd > tilingData.tensorNum) {
        tensorEnd = tilingData.tensorNum;
    }

    FusedSgdF32<DTYPE_X> op(&pipe);
    op.Init(params, grads, momentum_buffer_list, grad_scale, params_ref, grads_ref, momentum_buffer_list_out,
            tilingData, tensorStart, tensorEnd);
    op.Process();
}
#else
extern "C" __global__ __aicore__ void fused_sgd(GM_ADDR params, GM_ADDR grads, GM_ADDR momentum_buffer_list,
                                                GM_ADDR grad_scale, GM_ADDR params_ref, GM_ADDR grads_ref,
                                                GM_ADDR momentum_buffer_list_out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(FusedSgdTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedSgdTilingData, tilingData, tiling);

    // 计算当前核处理的tensor范围 [tensorStart, tensorEnd)
    uint32_t blockIdx = GetBlockIdx();
    uint64_t tensorStart = static_cast<uint64_t>(blockIdx) * tilingData.tensorsPerCore;
    uint64_t tensorEnd = tensorStart + tilingData.tensorsPerCore;
    if (tensorEnd > tilingData.tensorNum) {
        tensorEnd = tilingData.tensorNum;
    }

#if (ORIG_DTYPE_PARAMS == DT_BF16)
    FusedSgdF16Bf16<bfloat16_t> op(&pipe);
    op.Init(params, grads, momentum_buffer_list, grad_scale, params_ref, grads_ref, momentum_buffer_list_out,
            tilingData, tensorStart, tensorEnd);
    op.Process();
#elif (ORIG_DTYPE_PARAMS == DT_FLOAT16)
    FusedSgdF16Bf16<half> op(&pipe);
    op.Init(params, grads, momentum_buffer_list, grad_scale, params_ref, grads_ref, momentum_buffer_list_out,
            tilingData, tensorStart, tensorEnd);
    op.Process();
#elif (ORIG_DTYPE_PARAMS == DT_FLOAT32)
    FusedSgdF32<float> op(&pipe);
    op.Init(params, grads, momentum_buffer_list, grad_scale, params_ref, grads_ref, momentum_buffer_list_out,
            tilingData, tensorStart, tensorEnd);
    op.Process();
#endif
}
#endif
