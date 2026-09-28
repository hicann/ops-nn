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
 * \file in_training_update_v2_nhwc.h
 * \brief NHWC [N,R,C] channel-tile RegBase implementation.
 */

#ifndef IN_TRAINING_UPDATE_V2_NHWC_H
#define IN_TRAINING_UPDATE_V2_NHWC_H

#include "in_training_update_v2_common.h"

namespace INTrainingUpdateV2Ops {

template <typename T, bool HAS_AFFINE, bool HAS_RUNNING>
class INTrainingUpdateV2Nhwc : public INTrainingUpdateV2Base<T, HAS_AFFINE, HAS_RUNNING> {
    using Base = INTrainingUpdateV2Base<T, HAS_AFFINE, HAS_RUNNING>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean,
                                GM_ADDR variance, GM_ADDR y, GM_ADDR batchMean, GM_ADDR batchVariance,
                                const INTrainingUpdateV2TilingData* tilingData, TPipe* pipe)
    {
        Base::InitCommon(x, sum, squareSum, gamma, beta, mean, variance, y, batchMean, batchVariance, tilingData, pipe);
        constexpr int64_t elementsPerGmBlock = DMA_BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
        this->GetOwnedYRange(elementsPerGmBlock, yStart_, yCount_);
    }

    __aicore__ inline void Process()
    {
        this->ProcessOwnedStats();
        int64_t position = yStart_;
        const int64_t end = yStart_ + yCount_;
        const int64_t sampleElements = this->tiling_->r * this->tiling_->c;
        while (position < end) {
            const int64_t n = position / sampleElements;
            const int64_t sampleEnd = (n + 1) * sampleElements;
            const int64_t ownedSampleEnd = (end < sampleEnd) ? end : sampleEnd;
            const int64_t cStart = position % this->tiling_->c;
            if (cStart == 0) {
                const int64_t fullRows = (ownedSampleEnd - position) / this->tiling_->c;
                if (fullRows > 0) {
                    const int64_t rowStart = (position % sampleElements) / this->tiling_->c;
                    ProcessFullRows(n, rowStart, fullRows);
                    position += fullRows * this->tiling_->c;
                    continue;
                }
            }
            int64_t currentC = this->tiling_->c - cStart;
            if (currentC > CHANNEL_TILE) {
                currentC = CHANNEL_TILE;
            }
            if (currentC > end - position) {
                currentC = end - position;
            }
            const int64_t statOffset = n * this->tiling_->c + cStart;
            const int64_t gammaOffset = n * this->tiling_->gammaBatchStride + cStart;
            const int64_t betaOffset = n * this->tiling_->betaBatchStride + cStart;
            this->PrepareStats(statOffset, gammaOffset, betaOffset, currentC);
            ProcessChannelSpan(position, currentC);
            position += currentC;
        }
    }

private:
    __aicore__ inline void ProcessFullRows(int64_t n, int64_t rowStart, int64_t rowCount)
    {
        const int64_t channelUnits = this->tiling_->c / CHANNEL_TILE + ((this->tiling_->c % CHANNEL_TILE != 0) ? 1 : 0);
        const int64_t baseC = this->tiling_->c / channelUnits;
        const int64_t extraC = this->tiling_->c % channelUnits;
        for (int64_t channelIndex = 0; channelIndex < channelUnits; ++channelIndex) {
            const int64_t cStart = channelIndex * baseC + ((channelIndex < extraC) ? channelIndex : extraC);
            const int64_t currentC = baseC + ((channelIndex < extraC) ? 1 : 0);
            const int64_t statOffset = n * this->tiling_->c + cStart;
            const int64_t gammaOffset = n * this->tiling_->gammaBatchStride + cStart;
            const int64_t betaOffset = n * this->tiling_->betaBatchStride + cStart;
            this->PrepareStats(statOffset, gammaOffset, betaOffset, currentC);
            for (int64_t rowOffset = 0; rowOffset < rowCount;) {
                const int64_t rows = (rowCount - rowOffset < this->tiling_->rTile) ? rowCount - rowOffset :
                                                                                     this->tiling_->rTile;
                const int64_t gmOffset = (n * this->tiling_->r + rowStart + rowOffset) * this->tiling_->c + cStart;
                ProcessChannelRows(gmOffset, rows, currentC);
                rowOffset += rows;
            }
        }
    }

    __aicore__ inline void ProcessChannelSpan(int64_t gmOffset, int64_t currentC)
    {
        ProcessChannelRows(gmOffset, 1, currentC);
    }

    __aicore__ inline void ProcessChannelRows(int64_t gmOffset, int64_t rows, int64_t currentC)
    {
        const int64_t rowBytes = currentC * static_cast<int64_t>(sizeof(T));
        const int64_t rowStrideBytes = ((rowBytes + DMA_BLOCK_BYTES - 1) / DMA_BLOCK_BYTES) * DMA_BLOCK_BYTES;
        const int64_t rowStrideElems = rowStrideBytes / static_cast<int64_t>(sizeof(T));
        LocalTensor<T> xLocal = this->xQue_.template AllocTensor<T>();
        const int64_t gmStrideBytes = (rows == 1) ? 0 : (this->tiling_->c - currentC) * static_cast<int64_t>(sizeof(T));
        DataCopyExtParams copyIn{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowBytes), gmStrideBytes, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad<T, PaddingMode::Normal>(xLocal, this->xGm_[gmOffset], copyIn, pad);
        this->xQue_.EnQue(xLocal);
        xLocal = this->xQue_.template DeQue<T>();

        LocalTensor<T> yLocal = this->yQue_.template AllocTensor<T>();
        ComputeRows((__ubuf__ T*)xLocal.GetPhyAddr(), (__ubuf__ T*)yLocal.GetPhyAddr(), rows, currentC, rowStrideElems);
        this->yQue_.EnQue(yLocal);
        yLocal = this->yQue_.template DeQue<T>();
        DataCopyExtParams copyOut{static_cast<uint16_t>(rows), static_cast<uint32_t>(rowBytes), 0, gmStrideBytes, 0};
        DataCopyPad<T, PaddingMode::Normal>(this->yGm_[gmOffset], yLocal, copyOut);
        this->yQue_.FreeTensor(yLocal);
        this->xQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeRows(__ubuf__ T* x, __ubuf__ T* y, int64_t rows, int64_t currentC,
                                       int64_t rowStrideElems)
    {
        if (this->tiling_->epsilon == 0.0f) {
            ComputeRowsImpl<true>(x, y, rows, currentC, rowStrideElems);
        } else {
            ComputeRowsImpl<false>(x, y, rows, currentC, rowStrideElems);
        }
    }

    template <bool ZERO_EPSILON>
    __aicore__ inline void ComputeRowsImpl(__ubuf__ T* x, __ubuf__ T* y, int64_t rows, int64_t currentC,
                                           int64_t rowStrideElems)
    {
        __ubuf__ float* sum = this->BiasAddr();
        __ubuf__ float* mean = this->MeanAddr();
        __ubuf__ float* scale = this->ScaleAddr();
        __ubuf__ float* beta = this->BiasAddr();
        __ubuf__ float* restore = this->VarAddr();
        const float negativeInvR = -this->tiling_->invR;
        const float negativeInvRCorrection = -this->tiling_->invRCorrection;
        const uint16_t rowLoops = static_cast<uint16_t>(rows);
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> meanReg;
            RegTensor<float> scaleReg;
            RegTensor<float> betaReg;
            RegTensor<float> restoreReg;
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            uint32_t validCount = static_cast<uint32_t>(currentC);
            MaskReg validMask = UpdateMask<float>(validCount);
            Reg::LoadAlign<float, LoadDist::DIST_NORM>(scaleReg, scale);
            if constexpr (HAS_AFFINE) {
                Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanReg, mean);
                Reg::LoadAlign<float, LoadDist::DIST_NORM>(betaReg, beta);
                Reg::LoadAlign<float, LoadDist::DIST_NORM>(restoreReg, restore);
            } else {
                Reg::LoadAlign<float, LoadDist::DIST_NORM>(sumReg, sum);
                if constexpr (ZERO_EPSILON) {
                    Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanReg, mean);
                }
            }
            for (uint16_t row = 0; row < rowLoops; ++row) {
                const uint32_t offset = static_cast<uint32_t>(row * rowStrideElems);
                LoadXToFp32(x, xReg, validMask, offset);
                ComputeNormalizedY<HAS_AFFINE, ZERO_EPSILON>(yReg, xReg, sumReg, meanReg, scaleReg, betaReg, restoreReg,
                                                             negativeInvR, negativeInvRCorrection, validMask);
                StoreFp32ToY(y, yReg, validMask, offset);
            }
        }
    }

    int64_t yStart_ = 0;
    int64_t yCount_ = 0;
};

} // namespace INTrainingUpdateV2Ops

#endif // IN_TRAINING_UPDATE_V2_NHWC_H
