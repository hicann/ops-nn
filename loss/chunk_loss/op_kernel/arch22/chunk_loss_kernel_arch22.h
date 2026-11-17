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
 * ChunkLoss forward:
 *   Z = X W^T
 *   max_i / sum_i = online logsumexp state over V
 *   loss = alpha * sum_i (max_i + log(sum_i) - Z_{i,y_i})   (ignore -> 0)
 *   dZ = alpha * (softmax(Z) - one_hot(y))
 *   dH = dZ @ W
 *
 * arch22 split: common + CubeOp (hand PreloadZ/FullLoadA, CV NZ workspace) + VecOp.
 */

#ifndef CHUNK_LOSS_KERNEL_ARCH22_H_
#define CHUNK_LOSS_KERNEL_ARCH22_H_

#include "chunk_loss_common_arch22.h"
#include "chunk_loss_cube_op.h"
#include "chunk_loss_vec_op.h"

namespace chunkloss {

template <class Element, uint32_t TileM = kTileMDefault, bool kPreloadZ = true, bool kFullLoadH = true,
          bool kCallbackZ = false, bool kBatchZ = false, bool kWsNz = kWsNzDefault>
class Kernel {
public:
    CHUNK_LOSS_DEVICE
    Kernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CHUNK_LOSS_DEVICE void operator()(Params const& params);

    template <>
    CHUNK_LOSS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        CubeOp<Element, TileM, kPreloadZ, kFullLoadH, kCallbackZ, kBatchZ, kWsNz> cube;
        cube.Process(params);
    }

    template <>
    CHUNK_LOSS_DEVICE void operator()<AscendC::AIV>(Params const& params)
    {
        VecOp<Element, TileM, kPreloadZ, kFullLoadH, kCallbackZ, kBatchZ, kWsNz> vec;
        vec.Process(params);
    }
};

template <class Element, uint32_t TileM = kTileMDefault, bool kPreloadZ = true, bool kFullLoadH = true,
          bool kCallbackZ = false, bool kBatchZ = false, bool kWsNz = kWsNzDefault>
CHUNK_LOSS_GLOBAL void Launch(uint64_t hardwareSyncAddr, Params params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSyncBaseAddr(hardwareSyncAddr);
    Kernel<Element, TileM, kPreloadZ, kFullLoadH, kCallbackZ, kBatchZ, kWsNz> op;
    op(params);
}

} // namespace chunkloss

#endif
