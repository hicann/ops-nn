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
 * \file group_norm_silu_regbase_empty.h
 * \brief
 */

#ifndef GROUP_NORM_SILU_REGBASE_EMPTY_
#define GROUP_NORM_SILU_REGBASE_EMPTY_

#include <cmath>
#include "group_norm_silu_regbase_base.h"
#include "../inc/kernel_utils.h"

namespace GroupNormSilu {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class GroupNormSiluEmpty
{
public:
    __aicore__ inline GroupNormSiluEmpty(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR silu, GM_ADDR mean, GM_ADDR rstd, GM_ADDR workspace,
        const GroupNormSiluRegbaseTilingData* tilingData)
    {
        tiling_ = tilingData;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        meanGm_.SetGlobalBuffer((__gm__ T*)mean);
        rstdGm_.SetGlobalBuffer((__gm__ T*)rstd);
        ParseTilingData();
        pipe_.InitBuffer(outQue_, BUFFER_NUM, ops::CeilAlign(ubNumPerLoop_, onceBlockAlignNum_) * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (shapeN_ == 0 || blockIdx_ >= blockNum_) {
            return;
        }

        uint32_t numPerCoreLoop = CeilDiv(numPerCore_, innerNumPerCore_);
        uint32_t numPerCoreTail =
            numPerCore_ % innerNumPerCore_ == 0 ? innerNumPerCore_ : numPerCore_ % innerNumPerCore_;
        uint32_t numPerCoreOneLoop = innerNumPerCore_;
        for (uint64_t i = 0; i < numPerCoreLoop; i++) {
            if (i == numPerCoreLoop - 1) {
                numPerCoreOneLoop = numPerCoreTail;
            }

            DataCopyExtParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = numPerCoreOneLoop * sizeof(T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;

            uint64_t gmOffset = blockIdx_ * tiling_->numPerCore + i * innerNumPerCore_;
            ProcessMeanAndRstd<T>(outQue_, meanGm_[gmOffset], dataCopyParams, numPerCoreOneLoop);
            ProcessMeanAndRstd<T>(outQue_, rstdGm_[gmOffset], dataCopyParams, numPerCoreOneLoop);
        }
    }

private:
    __aicore__ inline void ParseTilingData()
    {
        if (blockIdx_ == blockNum_ - 1) {
            numPerCore_ = tiling_->numLastCore;
        } else {
            numPerCore_ = tiling_->numPerCore;
        }

        ubSize_ = tiling_->ubSize;
        shapeN_ = tiling_->shapeN;
        innerNumPerCore_ = ubSize_ / BUFFER_NUM / static_cast<int64_t>(sizeof(T));
        ubNumPerLoop_ = numPerCore_ < innerNumPerCore_ ? numPerCore_ : innerNumPerCore_;
        onceBlockAlignNum_ = static_cast<int64_t>(BLOCK_SIZE / sizeof(T));
    }

private:
    const GroupNormSiluRegbaseTilingData* tiling_;
    TPipe pipe_;

    // output GM tensors
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;

    // tiling_ parameters
    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t numPerCore_;
    int64_t ubSize_;
    int64_t shapeN_;
    int64_t innerNumPerCore_;
    int64_t ubNumPerLoop_;
    int64_t onceBlockAlignNum_;

    // out que
    TQue<TPosition::VECOUT, 1> outQue_;
};

} // namespace GroupNormSilu

#endif
