/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GATHER_V2_KERNEL_H
#define GATHER_V2_KERNEL_H

#include "gather_v2_tiling_data.h"
#include "kernel_operator.h"

namespace GatherV2 {
using namespace AscendC;

template <typename IndexT>
class GatherV2Kernel {
public:
    __aicore__ inline GatherV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace,
                                optiling::GatherV2TilingData* tiling)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ IndexT*>(indices));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = tiling;
        blockIdx_ = GetBlockIdx();
        blockCount_ = GetBlockNum();
        pipe_.InitBuffer(indicesQueue_, 1, tiling_->get_tileCount() * sizeof(IndexT));
        pipe_.InitBuffer(outputQueue_, 1, tiling_->get_tileCount() * tiling_->get_xTypeBytes());
    }

    __aicore__ inline void Process()
    {
        // Gather calculation batches indices, normalizes negative indices in UB,
        // and uses DataCopyPad for GM/UB movement.
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> indicesQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<IndexT> indicesGm_;
    GlobalTensor<uint8_t> yGm_;
    optiling::GatherV2TilingData* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t blockCount_ = 1;
};
} // namespace GatherV2

#endif // GATHER_V2_KERNEL_H
