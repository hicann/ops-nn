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
 * \file add_rms_norm_dynamic_quant_empty.h
 * \brief
 */
#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_EMPTY_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_EMPTY_H_

#include "add_rms_norm_dynamic_quant_regbase_common.h"
#include "kernel_operator.h"

namespace AddRmsNormDynamicQuant {
using namespace AscendC;
template <int32_t BUFFER_NUM = 2>
class KernelAddRmsNormDynamicQuantEmpty {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantEmpty(TPipe* pipe, const AddRmsNormDynamicQuantEmptyTilingData *tilingData) : pipe_(pipe), tiling_(tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR scale1, GM_ADDR scale2)
    {
        blockIdx_ = GetBlockIdx();
        hasScale1Out_ = tiling_->hasSmoothScale1;
        hasScale2Out_ = tiling_->hasSmoothScale2;
        usedCoreNum_ = tiling_->usedCoreNum;
        mPerCore_ = tiling_->mPerCore;
        mLastCore_ = tiling_->mLastCore;

        mPerUB_ = tiling_->mPerUB;
        mTailUb_ = tiling_->mTailUb;
        mlastCoreTailUb_ = tiling_->mlastCoreTailUb;
        ubLoopCount_ = tiling_->coreUbBlockCount;
        lastUbLoopCount_ = tiling_->lastCoreBlockCount;

        scalesGmOffset = blockIdx_ * mPerCore_;
        loopCount_ = (blockIdx_ == usedCoreNum_ - 1) ? lastUbLoopCount_ : ubLoopCount_;
        tailUb_ = (blockIdx_ == usedCoreNum_ - 1) ? mlastCoreTailUb_ : mTailUb_;

        scale1Gm_.SetGlobalBuffer((__gm__ float*)scale1);
        pipe_->InitBuffer(outQueueScale1_, BUFFER_NUM, (mPerUB_ * sizeof(float)));

        if (hasScale2Out_) {
            scale2Gm_.SetGlobalBuffer((__gm__ float*)scale2);
            pipe_->InitBuffer(outQueueScale2_, BUFFER_NUM, (mPerUB_ * sizeof(float)));
        }
    }

    __aicore__ inline void CalcScale(uint32_t gmOffset_, uint32_t realM)
    {
        LocalTensor<float> scale1Local = outQueueScale1_.AllocTensor<float>();
        Duplicate(scale1Local, -std::numeric_limits<float>::infinity(), realM);
        outQueueScale1_.EnQue<float>(scale1Local);
        CopyOutScale(scale1Gm_, outQueueScale1_, gmOffset_, realM);

        if (hasScale2Out_) {
            outQueueScale2_.EnQue<float>(scale1Local);
            CopyOutScale(scale2Gm_, outQueueScale2_, gmOffset_, realM);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        int64_t outputOffset = 0;
        for (uint32_t curLoop = 0; curLoop < loopCount_; curLoop++) {
            outputOffset = curLoop * mPerUB_ + scalesGmOffset;
            CalcScale(outputOffset, mPerUB_);
        }
        outputOffset = loopCount_ * mPerUB_ + scalesGmOffset;
        CalcScale(outputOffset, tailUb_);
    }

private:
    TPipe* pipe_ = nullptr;

    GlobalTensor<float> scale1Gm_;
    GlobalTensor<float> scale2Gm_;

    TQue<QuePosition::VECOUT, 1> outQueueScale1_;
    TQue<QuePosition::VECOUT, 1> outQueueScale2_;

    uint64_t numM_{0};
    bool hasScale1Out_{0};
    bool hasScale2Out_{0};
    uint64_t blockIdx_{0};
    uint64_t scalesGmOffset{0};
    uint64_t usedCoreNum_{0};
    uint64_t mPerUB_{0};
    uint64_t mPerCore_{0};
    uint64_t mLastCore_{0};
    uint64_t loopCount_{0};
    uint64_t ubLoopCount_{0};
    uint64_t lastUbLoopCount_{0};
    uint64_t tailUb_{0};
    uint64_t mTailUb_{0};
    uint64_t mlastCoreTailUb_{0};

    const AddRmsNormDynamicQuantEmptyTilingData *tiling_;
};
} // namespace AddRmsNormDynamicQuant
#endif // _ADD_RMS_NORM_DYNAMIC_QUANT_EMPTY_H_
