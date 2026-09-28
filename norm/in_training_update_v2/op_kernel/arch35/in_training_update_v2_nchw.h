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
 * \file in_training_update_v2_nchw.h
 * \brief NCHW [N*C,R] RegBase implementation.
 */

#ifndef IN_TRAINING_UPDATE_V2_NCHW_H
#define IN_TRAINING_UPDATE_V2_NCHW_H

#include "in_training_update_v2_common.h"

namespace INTrainingUpdateV2Ops {

template <typename T, bool HAS_AFFINE, bool HAS_RUNNING>
class INTrainingUpdateV2Nchw : public INTrainingUpdateV2Base<T, HAS_AFFINE, HAS_RUNNING> {
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
        if (this->tiling_->r == 1) {
            ProcessSingleSpatialElement();
            return;
        }
        this->ProcessOwnedStats();
        int64_t position = yStart_;
        const int64_t end = yStart_ + yCount_;
        while (position < end) {
            const int64_t unit = position / this->tiling_->r;
            const int64_t rStart = position % this->tiling_->r;
            if (rStart == 0 && end - position >= this->tiling_->r) {
                int64_t planeCount = (end - position) / this->tiling_->r;
                if (planeCount > STAT_CHUNK) {
                    planeCount = STAT_CHUNK;
                }
                const int64_t n = unit / this->tiling_->c;
                const int64_t cStart = unit % this->tiling_->c;
                if constexpr (HAS_AFFINE) {
                    if ((this->tiling_->gammaBatchStride == 0 || this->tiling_->betaBatchStride == 0) &&
                        planeCount > this->tiling_->c - cStart) {
                        planeCount = this->tiling_->c - cStart;
                    }
                }
                const int64_t gammaOffset = n * this->tiling_->gammaBatchStride + cStart;
                const int64_t betaOffset = n * this->tiling_->betaBatchStride + cStart;
                this->PrepareStats(unit, gammaOffset, betaOffset, planeCount);
                for (int64_t plane = 0; plane < planeCount; ++plane) {
                    ProcessPlane(unit + plane, static_cast<uint32_t>(plane), 0, this->tiling_->r);
                }
                position += planeCount * this->tiling_->r;
                continue;
            }
            int64_t extent = this->tiling_->r - rStart;
            if (extent > end - position) {
                extent = end - position;
            }
            const int64_t n = unit / this->tiling_->c;
            const int64_t c = unit % this->tiling_->c;
            const int64_t gammaOffset = n * this->tiling_->gammaBatchStride + c;
            const int64_t betaOffset = n * this->tiling_->betaBatchStride + c;
            this->PrepareStats(unit, gammaOffset, betaOffset, 1);
            ProcessPlane(unit, 0, rStart, extent);
            position += extent;
        }
    }

private:
    __aicore__ inline void ProcessSingleSpatialElement()
    {
        int64_t position = yStart_;
        const int64_t end = yStart_ + yCount_;
        while (position < end) {
            const int64_t n = position / this->tiling_->c;
            const int64_t cStart = position % this->tiling_->c;
            int64_t count = end - position;
            if (count > STAT_CHUNK) {
                count = STAT_CHUNK;
            }
            if constexpr (HAS_AFFINE) {
                if ((this->tiling_->gammaBatchStride == 0 || this->tiling_->betaBatchStride == 0) &&
                    count > this->tiling_->c - cStart) {
                    count = this->tiling_->c - cStart;
                }
            }

            const int64_t gammaOffset = n * this->tiling_->gammaBatchStride + cStart;
            const int64_t betaOffset = n * this->tiling_->betaBatchStride + cStart;
            this->PrepareStats(position, gammaOffset, betaOffset, count, true);
            ProcessSingleSpatialSpan(position, count);
            position += count;
        }
    }

    __aicore__ inline void ProcessSingleSpatialSpan(int64_t gmOffset, int64_t count)
    {
        LocalTensor<T> xLocal = this->xQue_.template AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        DataCopyPad(xLocal, this->xGm_[gmOffset], copy, pad);
        this->xQue_.EnQue(xLocal);
        xLocal = this->xQue_.template DeQue<T>();

        LocalTensor<T> yLocal = this->yQue_.template AllocTensor<T>();
        ComputeSingleSpatialSpan((__ubuf__ T*)xLocal.GetPhyAddr(), (__ubuf__ T*)yLocal.GetPhyAddr(), count);
        this->yQue_.EnQue(yLocal);
        yLocal = this->yQue_.template DeQue<T>();
        DataCopyPad(this->yGm_[gmOffset], yLocal, copy);
        this->yQue_.FreeTensor(yLocal);
        this->xQue_.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeSingleSpatialSpan(__ubuf__ T* x, __ubuf__ T* y, int64_t count)
    {
        if (this->tiling_->epsilon == 0.0f) {
            ComputeSingleSpatialSpanImpl<true>(x, y, count);
        } else {
            ComputeSingleSpatialSpanImpl<false>(x, y, count);
        }
    }

    template <bool ZERO_EPSILON>
    __aicore__ inline void ComputeSingleSpatialSpanImpl(__ubuf__ T* x, __ubuf__ T* y, int64_t count)
    {
        __ubuf__ float* sum = this->BiasAddr();
        __ubuf__ float* mean = this->MeanAddr();
        __ubuf__ float* scale = this->ScaleAddr();
        __ubuf__ float* beta = this->BiasAddr();
        __ubuf__ float* restore = this->VarAddr();
        const float negativeInvR = -this->tiling_->invR;
        const float negativeInvRCorrection = -this->tiling_->invRCorrection;
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> meanReg;
            RegTensor<float> scaleReg;
            RegTensor<float> betaReg;
            RegTensor<float> restoreReg;
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            uint32_t validCount = static_cast<uint32_t>(count);
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
            LoadXToFp32(x, xReg, validMask, 0);
            ComputeNormalizedY<HAS_AFFINE, ZERO_EPSILON>(yReg, xReg, sumReg, meanReg, scaleReg, betaReg, restoreReg,
                                                         negativeInvR, negativeInvRCorrection, validMask);
            StoreFp32ToY(y, yReg, validMask, 0);
        }
    }

    __aicore__ inline void ProcessPlane(int64_t unit, uint32_t coefficientIndex, int64_t rStart, int64_t rCount)
    {
        const int64_t gmBase = unit * this->tiling_->r + rStart;
        for (int64_t offset = 0; offset < rCount;) {
            const int64_t extent = (rCount - offset < this->tiling_->tileElems) ? rCount - offset :
                                                                                  this->tiling_->tileElems;
            LocalTensor<T> xLocal = this->xQue_.template AllocTensor<T>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(extent * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            DataCopyPad(xLocal, this->xGm_[gmBase + offset], copy, pad);
            this->xQue_.EnQue(xLocal);
            xLocal = this->xQue_.template DeQue<T>();

            LocalTensor<T> yLocal = this->yQue_.template AllocTensor<T>();
            ComputeTile((__ubuf__ T*)xLocal.GetPhyAddr(), (__ubuf__ T*)yLocal.GetPhyAddr(), coefficientIndex, extent);
            this->yQue_.EnQue(yLocal);
            yLocal = this->yQue_.template DeQue<T>();
            DataCopyPad(this->yGm_[gmBase + offset], yLocal, copy);
            this->yQue_.FreeTensor(yLocal);
            this->xQue_.FreeTensor(xLocal);
            offset += extent;
        }
    }

    __aicore__ inline void ComputeTile(__ubuf__ T* x, __ubuf__ T* y, uint32_t coefficientIndex, int64_t extent)
    {
        if (this->tiling_->epsilon == 0.0f) {
            ComputeTileImpl<true>(x, y, coefficientIndex, extent);
        } else {
            ComputeTileImpl<false>(x, y, coefficientIndex, extent);
        }
    }

    template <bool ZERO_EPSILON>
    __aicore__ inline void ComputeTileImpl(__ubuf__ T* x, __ubuf__ T* y, uint32_t coefficientIndex, int64_t extent)
    {
        __ubuf__ float* sum = this->BiasAddr();
        __ubuf__ float* mean = this->MeanAddr();
        __ubuf__ float* scale = this->ScaleAddr();
        __ubuf__ float* beta = this->BiasAddr();
        __ubuf__ float* restore = this->VarAddr();
        const float negativeInvR = -this->tiling_->invR;
        const float negativeInvRCorrection = -this->tiling_->invRCorrection;
        const uint16_t fullLoops = static_cast<uint16_t>(extent / VL_FP32);
        const uint16_t totalLoops = static_cast<uint16_t>((extent + VL_FP32 - 1) / VL_FP32);
        const uint32_t tailCount = static_cast<uint32_t>(extent) - fullLoops * VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> sumReg;
            RegTensor<float> meanReg;
            RegTensor<float> scaleReg;
            RegTensor<float> betaReg;
            RegTensor<float> restoreReg;
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, scale + coefficientIndex);
            if constexpr (HAS_AFFINE) {
                Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(meanReg, mean + coefficientIndex);
                Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, beta + coefficientIndex);
                Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(restoreReg, restore + coefficientIndex);
            } else {
                Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(sumReg, sum + coefficientIndex);
                if constexpr (ZERO_EPSILON) {
                    Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(meanReg, mean + coefficientIndex);
                }
            }
            for (uint16_t loop = 0; loop < fullLoops; ++loop) {
                const uint32_t offset = loop * VL_FP32;
                LoadXToFp32(x, xReg, fullMask, offset);
                ComputeNormalizedY<HAS_AFFINE, ZERO_EPSILON>(yReg, xReg, sumReg, meanReg, scaleReg, betaReg, restoreReg,
                                                             negativeInvR, negativeInvRCorrection, fullMask);
                StoreFp32ToY(y, yReg, fullMask, offset);
            }
            for (uint16_t loop = fullLoops; loop < totalLoops; ++loop) {
                uint32_t tail = tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                const uint32_t offset = loop * VL_FP32;
                LoadXToFp32(x, xReg, tailMask, offset);
                ComputeNormalizedY<HAS_AFFINE, ZERO_EPSILON>(yReg, xReg, sumReg, meanReg, scaleReg, betaReg, restoreReg,
                                                             negativeInvR, negativeInvRCorrection, tailMask);
                StoreFp32ToY(y, yReg, tailMask, offset);
            }
        }
    }

    int64_t yStart_ = 0;
    int64_t yCount_ = 0;
};

} // namespace INTrainingUpdateV2Ops

#endif // IN_TRAINING_UPDATE_V2_NCHW_H
