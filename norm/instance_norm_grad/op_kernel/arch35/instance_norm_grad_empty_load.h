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
 * \file instance_norm_grad_empty_load.h
 * \brief Empty-tensor kernel (tilingKey 500): only pd_gamma/pd_beta are duplicated to 0 along C.
 *        Mirrors group_norm_grad's EmptyDgamma.
 */
#ifndef INSTANCE_NORM_GRAD_EMPTY_LOAD_H
#define INSTANCE_NORM_GRAD_EMPTY_LOAD_H
#pragma once

#include "kernel_operator.h"

namespace InstanceNormGrad {
using namespace AscendC;
constexpr int EMPTY_OUTPUT_COUNT = 2;

template <typename T, int32_t BUFFER_NUM = 2>
class EmptyDgamma {
public:
    __aicore__ inline EmptyDgamma(TPipe* pipe, const InstanceNormGradEmptyTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* pd_gamma, __gm__ uint8_t* pd_beta)
    {
        coreIdx_ = AscendC::GetBlockIdx();
        usedCoreNumDG_ = tiling_->usedCoreNumDG;
        if (coreIdx_ >= usedCoreNumDG_) {
            return;
        }
        colsPerCore_ = tiling_->colsPerCoreDG;
        tailUbCols_ = tiling_->tailUbCols;
        colsPerUB_ = tiling_->colsPerUBDG;
        lastCoreTailUbCols_ = tiling_->lastCoreTailUbCols;
        lastUbLoopCount_ = tiling_->lastCoreBlockCount;
        ubLoopCount_ = tiling_->coreUbBlockCount;
        gmOffset_ = colsPerCore_ * coreIdx_;
        colsLastCoreDG_ = tiling_->colsLastCoreDG;
        pdGammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pd_gamma));
        pdBetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(pd_beta));
        Ppipe_->InitBuffer(outQueue_, BUFFER_NUM, (colsPerUB_ * sizeof(T)));
    }

    __aicore__ inline void CopyToGm(uint32_t gmOffset, LocalTensor<T> outLocal, int32_t curCols)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = curCols * sizeof(T);
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(pdGammaGm_[gmOffset], outLocal, params);
        DataCopyPad(pdBetaGm_[gmOffset], outLocal, params);
    }

    __aicore__ inline void CalcZero(uint32_t gmOffset, uint32_t currentCols)
    {
        LocalTensor<T> outLocal = outQueue_.template AllocTensor<T>();
        Duplicate<T>(outLocal, (T)0, currentCols);
        outQueue_.EnQue(outLocal);
        outLocal = outQueue_.template DeQue<T>();
        CopyToGm(gmOffset, outLocal, currentCols);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void Process()
    {
        if (coreIdx_ >= usedCoreNumDG_) {
            return;
        }
        bool isLastCore = (coreIdx_ == usedCoreNumDG_ - 1);
        uint64_t loopCount = isLastCore ? lastUbLoopCount_ : ubLoopCount_;
        uint64_t tailCols = isLastCore ? lastCoreTailUbCols_ : tailUbCols_;
        int64_t outputOffset = 0;
        for (uint32_t curLoop = 0; curLoop < loopCount; curLoop++) {
            outputOffset = curLoop * colsPerUB_ + gmOffset_;
            CalcZero(outputOffset, colsPerUB_);
        }
        outputOffset = loopCount * colsPerUB_ + gmOffset_;
        CalcZero(outputOffset, tailCols);
    }

private:
    TQue<QuePosition::VECOUT, EMPTY_OUTPUT_COUNT> outQueue_;
    GlobalTensor<T> pdGammaGm_;
    GlobalTensor<T> pdBetaGm_;
    uint64_t colsLastCoreDG_ = 0;
    uint32_t coreIdx_ = 0;
    uint64_t colsPerCore_ = 0;
    uint64_t gmOffset_ = 0;
    uint64_t usedCoreNumDG_ = 0;
    uint64_t colsPerUB_ = 0;
    uint64_t tailUbCols_ = 0;
    uint64_t ubLoopCount_ = 0;
    uint64_t lastUbLoopCount_ = 0;
    uint64_t lastCoreTailUbCols_ = 0;
    TPipe* Ppipe_ = nullptr;
    const InstanceNormGradEmptyTilingData* tiling_;
};
} // namespace InstanceNormGrad
#endif // INSTANCE_NORM_GRAD_EMPTY_LOAD_H
