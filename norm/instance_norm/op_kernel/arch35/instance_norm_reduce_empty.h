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
 * \file instance_norm_reduce_empty.h
 * \brief
 */

#ifndef INSTANCE_NORM_REDUCE_EMPTY_H_
#define INSTANCE_NORM_REDUCE_EMPTY_H_

#include "instance_norm_common.h"

namespace InstanceNormOps {
using namespace AscendC;

template <typename T_MEAN>
class InstanceNormReduceEmpty {
public:
    __aicore__ inline InstanceNormReduceEmpty(const InstanceNormReduceEmptyTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR mean, GM_ADDR variance)
    {
        // Init
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = GetBlockNum();
        ASSERT(usedCoreNum_ != 0 && "block dim can not be zero!"); 
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        
        perCoreElements_ = tilingData_->perCoreElements;
        if (blockIdx_ < usedCoreNum_ - 1) {
            curCoreElements_ = tilingData_->perCoreElements;
            coreLoopsNum_ = tilingData_->perCoreLoops;
            perCorePerLoopElements_ = tilingData_->perCorePerLoopElements;
            perCoreLastLoopElements_ = tilingData_->perCoreLastLoopElements;
        } else if (blockIdx_ == usedCoreNum_ - 1) {
            curCoreElements_ = tilingData_->lastCoreElements;
            coreLoopsNum_ = tilingData_->lastCoreLoops;
            perCorePerLoopElements_ = tilingData_->lastCorePerLoopElements;
            perCoreLastLoopElements_ = tilingData_->lastCoreLastLoopElements;
        }

        meanGm_.SetGlobalBuffer((__gm__ T_MEAN*)mean + this->perCoreElements_ * blockIdx_, this->curCoreElements_);
        varianceGm_.SetGlobalBuffer((__gm__ T_MEAN*)variance + this->perCoreElements_ * blockIdx_, this->curCoreElements_);

        // init local memory
        pipe_.InitBuffer(outNanQueue_, BUFFER_NUM, perCorePerLoopElements_ * sizeof(T_MEAN));
    }

    __aicore__ inline void Process()
    {
        // Process
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        LocalTensor<T_MEAN> meanTypeNanLocal = outNanQueue_.AllocTensor<T_MEAN>();

        T_MEAN t_nan = AscendC::NumericLimits<T_MEAN>::QuietNaN();
        Duplicate(meanTypeNanLocal, t_nan, perCorePerLoopElements_);

        outNanQueue_.EnQue(meanTypeNanLocal);
        meanTypeNanLocal = outNanQueue_.DeQue<T_MEAN>();

        for (uint64_t i = 0; i < coreLoopsNum_; i++) {
            uint64_t perLoopElements =
                (i == (coreLoopsNum_ - 1)) ? perCoreLastLoopElements_ : perCorePerLoopElements_;
            CopyOut(i, perLoopElements, meanTypeNanLocal);
        }
        outNanQueue_.FreeTensor(meanTypeNanLocal);
    }

private:
    __aicore__ inline void CopyOut(
        uint64_t process, uint64_t curLoopElements, LocalTensor<T_MEAN>& runningMeanTypeNanLocal)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = curLoopElements * sizeof(T_MEAN);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(meanGm_[process * this->perCorePerLoopElements_], runningMeanTypeNanLocal, copyInParams);
        DataCopyPad(varianceGm_[process * this->perCorePerLoopElements_], runningMeanTypeNanLocal, copyInParams);
    }

    // Constants
    constexpr static int64_t BUFFER_NUM = 1;

    TPipe pipe_;
    const InstanceNormReduceEmptyTilingData* tilingData_;

    // Global memory address 
    GlobalTensor<T_MEAN> meanGm_;
    GlobalTensor<T_MEAN> varianceGm_;

    // TQue
    TQue<QuePosition::VECOUT, 1> outNanQueue_;

    // Variable
    uint64_t usedCoreNum_{0};
    uint64_t blockIdx_{0};
    uint64_t perCoreElements_{0};
    uint64_t curCoreElements_{0};
    uint64_t coreLoopsNum_{0};
    uint64_t tilesPerCore_{0};
    uint64_t perCorePerLoopElements_{0};
    uint64_t perCoreLastLoopElements_{0};
};
} // namespace InstanceNormOps
#endif // INSTANCE_NORM_REDUCE_EMPTY_H_
