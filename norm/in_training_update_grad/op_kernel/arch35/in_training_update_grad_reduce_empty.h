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
 * \file in_training_update_grad_reduce_empty.h
 * \brief TilingKey 50000: R == 0 (empty spatial). Both outputs are the sum over an empty set, i.e. 0.0
 *        (note: this differs from instance_norm's ReduceEmpty which writes NaN for mean/variance).
 */
#ifndef IN_TRAINING_UPDATE_GRAD_REDUCE_EMPTY_H_
#define IN_TRAINING_UPDATE_GRAD_REDUCE_EMPTY_H_

#include "in_training_update_grad_common.h"

namespace InTrainingUpdateGrad {
using namespace AscendC;

class InTrainingUpdateGradReduceEmpty {
public:
    __aicore__ inline InTrainingUpdateGradReduceEmpty(const InTrainingUpdateGradReduceEmptyTilingData* tilingData)
    {
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR resGamma, GM_ADDR resBeta)
    {
        blockIdx_ = GetBlockIdx();
        usedCoreNum_ = GetBlockNum();
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }

        perCoreElements_ = tilingData_->perCoreElements;
        if (blockIdx_ < usedCoreNum_ - 1) {
            curCoreElements_ = tilingData_->perCoreElements;
            coreLoopsNum_ = tilingData_->perCoreLoops;
            perLoopElements_ = tilingData_->perCorePerLoopElements;
            lastLoopElements_ = tilingData_->perCoreLastLoopElements;
        } else {
            curCoreElements_ = tilingData_->lastCoreElements;
            coreLoopsNum_ = tilingData_->lastCoreLoops;
            perLoopElements_ = tilingData_->lastCorePerLoopElements;
            lastLoopElements_ = tilingData_->lastCoreLastLoopElements;
        }

        resGammaGm_.SetGlobalBuffer((__gm__ float*)resGamma + perCoreElements_ * blockIdx_, curCoreElements_);
        resBetaGm_.SetGlobalBuffer((__gm__ float*)resBeta + perCoreElements_ * blockIdx_, curCoreElements_);
        pipe_.InitBuffer(outQueue_, 1, perLoopElements_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        LocalTensor<float> zeroLocal = outQueue_.AllocTensor<float>();
        Duplicate(zeroLocal, 0.0f, perLoopElements_);
        outQueue_.EnQue(zeroLocal);
        zeroLocal = outQueue_.DeQue<float>();

        for (uint32_t i = 0; i < coreLoopsNum_; i++) {
            uint32_t curElements = (i == coreLoopsNum_ - 1) ? lastLoopElements_ : perLoopElements_;
            CopyOut(i, curElements, zeroLocal);
        }
        outQueue_.FreeTensor(zeroLocal);
    }

private:
    __aicore__ inline void CopyOut(uint32_t loop, uint32_t curElements, LocalTensor<float>& zeroLocal)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curElements * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(resGammaGm_[loop * perLoopElements_], zeroLocal, copyParams);
        DataCopyPad(resBetaGm_[loop * perLoopElements_], zeroLocal, copyParams);
    }

    TPipe pipe_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    const InTrainingUpdateGradReduceEmptyTilingData* tilingData_;

    GlobalTensor<float> resGammaGm_;
    GlobalTensor<float> resBetaGm_;

    uint32_t usedCoreNum_{0};
    uint32_t blockIdx_{0};
    uint32_t perCoreElements_{0};
    uint32_t curCoreElements_{0};
    uint32_t coreLoopsNum_{0};
    uint32_t perLoopElements_{0};
    uint32_t lastLoopElements_{0};
};
} // namespace InTrainingUpdateGrad
#endif // IN_TRAINING_UPDATE_GRAD_REDUCE_EMPTY_H_
