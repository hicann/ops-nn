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
 * \file layer_norm_v4_split_r.h
 * \brief
 */
#ifndef LAYER_NORM_V4_SPLIT_R_H
#define LAYER_NORM_V4_SPLIT_R_H

#include "layer_norm_v4_welford_base.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace LayerNormV4 {
// Build weighted contributions before invoking the common length-aware reduction.
class LayerNormV4SplitRCommon {
public:
    __aicore__ inline static void FillCountBlock(__ubuf__ float* dst, AscendC::Reg::RegTensor<float>& value,
                                                 AscendC::Reg::MaskReg& full, AscendC::Reg::MaskReg& active,
                                                 float count, uint32_t num, uint16_t loops, uint32_t vl)
    {
        AscendC::Reg::Duplicate(value, count, full);
        for (uint16_t i = 0; i < loops; ++i) {
            active = AscendC::Reg::UpdateMask<float>(num);
            AscendC::Reg::StoreAlign(dst + i * vl, value, active);
        }
    }

    __aicore__ inline static void Merge(LocalTensor<float>& meanOut, LocalTensor<float>& varianceOut,
                                        LocalTensor<float>& means, LocalTensor<float>& variances,
                                        LocalTensor<float>& counts, LocalTensor<float>& contributions,
                                        LocalTensor<float>& reduceTmp, uint32_t splits, float inverseCount)
    {
        auto* meanAddr = (__ubuf__ float*)means.GetPhyAddr();
        auto* varAddr = (__ubuf__ float*)variances.GetPhyAddr();
        auto* countAddr = (__ubuf__ float*)counts.GetPhyAddr();
        auto* contributionAddr = (__ubuf__ float*)contributions.GetPhyAddr();
        auto* meanOutAddr = (__ubuf__ float*)meanOut.GetPhyAddr();
        constexpr uint32_t vl = NormCommon::V_LENGTH;
        uint32_t foldPoint = vl;
        while (foldPoint <= splits / 2) {
            foldPoint *= 2;
        }
        // RegBase vector loops require a uint16_t induction variable and a
        // bound divisible by the step; the mask covers the aligned tail.
        uint32_t splitsAligned = (splits + vl - 1) / vl * vl;
        __VEC_SCOPE__
        {
            using namespace AscendC::Reg;
            RegTensor<float> mean, weight, reference, delta;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(reference, meanAddr);
            uint32_t remaining = splits;
            for (uint16_t offset = 0; offset < splitsAligned; offset += vl) {
                MaskReg active = UpdateMask<float>(remaining);
                LoadAlign(mean, meanAddr + offset);
                LoadAlign(weight, countAddr + offset);
                Muls(weight, weight, inverseCount, active);
                // Retain the centered-mean fix for large-offset inputs.
                Sub(delta, mean, reference, active);
                Mul(delta, delta, weight, active);
                StoreAlign(contributionAddr + offset, delta, active);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        NormCommon::NormCommonRegbase::CalculateReduceSum(contributions, meanOut, reduceTmp, splits, foldPoint);
        __VEC_SCOPE__
        {
            using namespace AscendC::Reg;
            RegTensor<float> mean, variance, weight, reference, delta, mergedMean;
            MaskReg first = CreateMask<float, MaskPattern::VL1>();
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(reference, meanAddr);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mergedMean, meanOutAddr);
            Add(mergedMean, mergedMean, reference, first);
            StoreAlign(meanOutAddr, mergedMean, first);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign<float, LoadDist::DIST_BRC_B32>(mergedMean, meanOutAddr);
            uint32_t remaining = splits;
            for (uint16_t offset = 0; offset < splitsAligned; offset += vl) {
                MaskReg active = UpdateMask<float>(remaining);
                LoadAlign(mean, meanAddr + offset);
                LoadAlign(variance, varAddr + offset);
                LoadAlign(weight, countAddr + offset);
                // Weight before multiplying variance to avoid count-amplified overflow.
                Muls(weight, weight, inverseCount, active);
                Sub(delta, mean, mergedMean, active);
                Mul(delta, delta, delta, active);
                Mul(delta, delta, weight, active);
                Mul(variance, variance, weight, active);
                Add(variance, variance, delta, active);
                StoreAlign(contributionAddr + offset, variance, active);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        NormCommon::NormCommonRegbase::CalculateReduceSum(contributions, varianceOut, reduceTmp, splits, foldPoint);
    }
};

template <typename T, typename U, typename M>
class LayerNormV4RegbaseSplitR : public LayerNormV4WelfordBase<T, U, M, LayerNormV4TilingDataSplitR> {
    using Base = LayerNormV4WelfordBase<T, U, M, LayerNormV4TilingDataSplitR>;

public:
    __aicore__ inline LayerNormV4RegbaseSplitR(const LayerNormV4TilingDataSplitR* tilingData, TPipe* pipeIn)
        : Base(tilingData, pipeIn)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                GM_ADDR workspace)
    {
        row_ = GetBlockIdx() / this->td_->splitR;
        split_ = GetBlockIdx() % this->td_->splitR;
        segmentStart_ = SegmentStart(split_);
        segmentCount_ = SegmentCount(split_);
        this->xGm.SetGlobalBuffer((__gm__ T*)x + row_ * this->td_->N + segmentStart_, segmentCount_);
        this->yGm.SetGlobalBuffer((__gm__ T*)y + row_ * this->td_->N + segmentStart_, segmentCount_);
        this->meanGm.SetGlobalBuffer((__gm__ M*)mean + row_, 1);
        this->rstdGm.SetGlobalBuffer((__gm__ M*)rstd + row_, 1);
        partialGm_.SetGlobalBuffer((__gm__ float*)workspace);
        // Each GM record occupies a UB block; compact copies pack lane 0 from
        // every record into vector-aligned mean/variance planes.
        partialHalf_ = (this->td_->splitR + VECTOR_FLOATS - 1) / VECTOR_FLOATS * VECTOR_FLOATS;
        this->pipe_->InitBuffer(partialIn_, 1, 2 * partialHalf_ * sizeof(float));
        this->InitCommon(gamma, beta);
    }

    __aicore__ inline void Process()
    {
        this->ResetCache();
        this->mean_ = this->welfordTempBuffer.template Get<float>();
        this->variance_ = this->mean_[this->td_->tileLength];
        this->varianceTensor = this->variance_[this->td_->tileLength];
        this->shared_ = this->apiTempBuffer.template Get<uint8_t>();
        this->WelfordInitialize(this->mean_, this->variance_, this->td_->tileLength);
        int64_t loops = segmentCount_ / this->td_->tileLength;
        int64_t tail = segmentCount_ % this->td_->tileLength;
        for (int64_t i = 0; i < loops; ++i) {
            this->ProcessWelfordUpdate(i * this->td_->tileLength, this->td_->tileLength);
        }
        if (tail > 0) {
            this->ProcessWelfordUpdate(loops * this->td_->tileLength, tail);
        }
        WelfordFinalizePara para;
        para.rnLength = this->welfordCount;
        para.abLength = this->td_->tileLength;
        para.headCount = this->welfordCount;
        para.headCountLength = tail;
        para.tailCount = this->welfordCount - (tail > 0);
        para.tailCountLength = this->td_->tileLength - tail;
        para.abRec = 1.0f / static_cast<float>(this->td_->tileLength);
        para.rRec = 1.0f / static_cast<float>(segmentCount_);
        WelfordFinalize<true>(this->meanTensor, this->varianceTensor, this->mean_, this->variance_, this->shared_,
                              para);

        // Publish mean and population variance in separate GM planes.
        // SyncAll includes the preceding MTE3 writes before peer reads.
        auto vToMte3 = this->pipe_->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        int64_t planeLength = this->td_->M * this->td_->splitR * PARTIAL_FLOATS;
        DataCopyExtParams partialCopyParams{1, PARTIAL_FLOATS * sizeof(float), 0, 0, 0};
        DataCopyPad(partialGm_[GetBlockIdx() * PARTIAL_FLOATS], this->meanTensor, partialCopyParams);
        DataCopyPad(partialGm_[planeLength + GetBlockIdx() * PARTIAL_FLOATS], this->varianceTensor, partialCopyParams);
        SyncAll();

        auto all = partialIn_.template AllocTensor<float>();
        auto allMean = all;
        auto allVar = all[partialHalf_];
        int64_t rowOffset = row_ * this->td_->splitR * PARTIAL_FLOATS;
        DataCopyExtParams compactCopy{static_cast<uint16_t>(this->td_->splitR), sizeof(float),
                                      static_cast<uint32_t>((PARTIAL_FLOATS - 1) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> noPadding{false, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(allMean, partialGm_[rowOffset], compactCopy, noPadding);
        DataCopyPad<float, PaddingMode::Compact>(allVar, partialGm_[planeLength + rowOffset], compactCopy, noPadding);
        partialIn_.EnQue(all);
        all = partialIn_.template DeQue<float>();
        allMean = all;
        allVar = all[partialHalf_];
        // The local Welford state is dead; reuse mean_ for compact segment counts.
        FillMergeCounts();
        // With splitR <= 16, the dead tile-sized M2 and 256-float rstd buffers
        // cover the contributions and common reduction scratch, respectively.
        LayerNormV4SplitRCommon::Merge(this->meanTensor, this->varianceTensor, allMean, allVar, this->mean_,
                                       this->variance_, this->rstdTensor, this->td_->splitR,
                                       1.0f / static_cast<float>(this->td_->N));
        partialIn_.FreeTensor(all);
        for (int64_t offset = 0; offset < segmentCount_; offset += this->td_->tileLength) {
            int64_t length = segmentCount_ - offset;
            length = length < this->td_->tileLength ? length : this->td_->tileLength;
            this->ProcessNormalize(offset, segmentStart_ + offset, length);
        }
        if (split_ == 0) {
            this->cacheCount = 1;
            this->paramAddr = 0;
            this->RefreshCache();
        } else {
            this->outQueueMean.FreeTensor(this->meanTensor);
            this->outQueueRstd.FreeTensor(this->rstdTensor);
        }
    }

private:
    __aicore__ inline void FillMergeCounts()
    {
        int64_t units = (this->td_->N + SPLIT_ALIGNMENT - 1) / SPLIT_ALIGNMENT;
        float baseCount = static_cast<float>(units / this->td_->splitR * SPLIT_ALIGNMENT);
        float lastCount = static_cast<float>(SegmentCount(this->td_->splitR - 1));
        uint32_t longCount = units % this->td_->splitR;
        uint32_t splitCount = this->td_->splitR;
        uint16_t firstLoopCount = (splitCount + VECTOR_FLOATS - 1) / VECTOR_FLOATS;
        uint16_t secondLoopCount = (splitCount - 1 + VECTOR_FLOATS - 1) / VECTOR_FLOATS;
        uint16_t thirdLoopCount = (longCount + VECTOR_FLOATS - 1) / VECTOR_FLOATS;
        auto* counts = (__ubuf__ float*)this->mean_.GetPhyAddr();
        // Overwrite order: last segment, ordinary segments, then the longer
        // leading segments.
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> value;
            auto mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg active;
            LayerNormV4SplitRCommon::FillCountBlock(counts, value, mask, active, lastCount, splitCount, firstLoopCount,
                                                    VECTOR_FLOATS);
            LayerNormV4SplitRCommon::FillCountBlock(counts, value, mask, active, baseCount, splitCount - 1,
                                                    secondLoopCount, VECTOR_FLOATS);
            LayerNormV4SplitRCommon::FillCountBlock(counts, value, mask, active, baseCount + SPLIT_ALIGNMENT, longCount,
                                                    thirdLoopCount, VECTOR_FLOATS);
        }
    }

    __aicore__ inline int64_t SegmentStart(int64_t split) const
    {
        int64_t units = (this->td_->N + SPLIT_ALIGNMENT - 1) / SPLIT_ALIGNMENT;
        int64_t quotient = units / this->td_->splitR;
        int64_t remainder = units % this->td_->splitR;
        return (split * quotient + (split < remainder ? split : remainder)) * SPLIT_ALIGNMENT;
    }

    __aicore__ inline int64_t SegmentCount(int64_t split) const
    {
        int64_t end = SegmentStart(split + 1);
        end = end < this->td_->N ? end : this->td_->N;
        return end - SegmentStart(split);
    }

    static constexpr int64_t SPLIT_ALIGNMENT = 16;
    static constexpr int64_t PARTIAL_FLOATS = platform::GetUbBlockSize() / sizeof(float);
    static constexpr int64_t VECTOR_FLOATS = LayerNormV4Regbase::GetVRegSize() / sizeof(float);
    int64_t partialHalf_ = 0;
    int64_t row_ = 0;
    int64_t split_ = 0;
    int64_t segmentStart_ = 0;
    int64_t segmentCount_ = 0;
    GlobalTensor<float> partialGm_;
    TQue<QuePosition::VECIN, 1> partialIn_;
};

} // namespace LayerNormV4

#endif // LAYER_NORM_V4_SPLIT_R_H
