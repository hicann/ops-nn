/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file group_norm_grad_empty_load.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_EMPTY_LOAD_H
#define GROUP_NORM_GRAD_EMPTY_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace GroupNormGrad {
using namespace AscendC;
constexpr int OUTPUT_COUNT = 2;

template <int32_t BUFFER_NUM = 2>
class EmptyDgamma {
public:
    __aicore__ inline EmptyDgamma(TPipe *pipe, const GroupNormGradEmptyTilingData *tilingData) : Ppipe_(pipe), tiling_(tilingData)
    {
    }
    __aicore__ inline void Init(__gm__ uint8_t *dgamma, __gm__ uint8_t *dbeta)
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
        dgammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dgamma));
        dbetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dbeta));
        Ppipe_->InitBuffer(dgammaQueue_, BUFFER_NUM, (colsPerUB_ * sizeof(float)));
    }

    __aicore__ inline void CopyDgammaAndDbetaToGm(uint32_t dgammaGmOffset, LocalTensor<float> outLocal, int32_t curCols,
        int32_t dgammaUBOffset)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = curCols * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(dgammaGm_[dgammaGmOffset], outLocal[dgammaUBOffset], dataCopyParams);
        DataCopyPad(dbetaGm_[dgammaGmOffset], outLocal[dgammaUBOffset], dataCopyParams);
    }

    __aicore__ inline void VFDuplicateRows(LocalTensor<float> &dstAddr, uint32_t currentCols)
    {
        AscendC::NumericLimits<float>::QuietNaN(dstAddr, currentCols);
    }

    __aicore__ inline void CalcDgamma(uint32_t gmOffset_, uint32_t currentCols)
    {
        LocalTensor<float> dgammaOutLocal = dgammaQueue_.template AllocTensor<float>();

        VFDuplicateRows(dgammaOutLocal, currentCols);
        dgammaQueue_.EnQue(dgammaOutLocal);
        dgammaOutLocal = dgammaQueue_.template DeQue<float>();
        CopyDgammaAndDbetaToGm(gmOffset_, dgammaOutLocal, currentCols, 0);

        dgammaQueue_.FreeTensor(dgammaOutLocal);
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
        CalcDgamma(outputOffset, colsPerUB_);
    }
    outputOffset = loopCount * colsPerUB_ + gmOffset_;
    CalcDgamma(outputOffset, tailCols);
}

private:
TQue<QuePosition::VECOUT, OUTPUT_COUNT> dgammaQueue_;
GlobalTensor<float> dgammaGm_;
GlobalTensor<float> dbetaGm_;
uint64_t colsLastCoreDG_;
uint32_t coreIdx_;
uint64_t colsPerCore_;
uint64_t gmOffset_;
uint64_t usedCoreNumDG_;
uint64_t colsPerUB_;
uint64_t tailUbCols_;
uint64_t ubLoopCount_;
uint64_t lastUbLoopCount_;
uint64_t lastCoreTailUbCols_;
TPipe *Ppipe_ = nullptr;

const GroupNormGradEmptyTilingData *tiling_;
};
} // namespace GroupNormGrad
#endif
