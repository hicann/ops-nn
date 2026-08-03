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
 * \file cosine_embedding_loss.h
 * \brief CosineEmbeddingLoss arch35 (DAV_3510) RegBase kernel.
 *
 * Generic scalar path matching legacy canndev semantics:
 *
 *   - x1/x2 are broadcast to a common ND shape.
 *   - dimension 1 of that common shape is reduced.
 *   - target is broadcast with the reduced x shape.
 *   - target == 1 selects 1 - cos; target == -1 selects max(0, cos - margin);
 *     any other target value contributes 0.
 *
 * Contiguous two-dimensional inputs use a UB/VF path with fp32 feature reduction. Other
 * broadcast layouts use the scalar fallback. sum/mean split rows across vector cores and merge
 * one 32B-aligned partial per core through workspace.
 */
#ifndef OPS_LOSS_COSINE_EMBEDDING_LOSS_H_
#define OPS_LOSS_COSINE_EMBEDDING_LOSS_H_

#include <math.h>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cosine_embedding_loss_tiling_data.h"

namespace NsCosineEmbeddingLoss {
using namespace AscendC;

constexpr uint32_t CEL_VF_LENGTH = 64;
constexpr uint32_t CEL_UB_BLOCK_BYTES = 32;

constexpr AscendC::Reg::CastTrait CEL_CAST_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};

template <typename X1T, typename X2T>
__simd_vf__ inline void ComputeFeaturePartialVf(__ubuf__ X1T* x1, __ubuf__ X2T* x2, __ubuf__ float* partial,
                                                uint32_t count, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> x1Reg;
    AscendC::Reg::RegTensor<float> x2Reg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> tmpReg;
    AscendC::Reg::RegTensor<float> dotReg;
    AscendC::Reg::RegTensor<float> square1Reg;
    AscendC::Reg::RegTensor<float> square2Reg;
    AscendC::Reg::RegTensor<float> reducedDot;
    AscendC::Reg::RegTensor<float> reducedSquare1;
    AscendC::Reg::RegTensor<float> reducedSquare2;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg validMask;

    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    AscendC::Reg::Duplicate(dotReg, 0.0f);
    AscendC::Reg::Duplicate(square1Reg, 0.0f);
    AscendC::Reg::Duplicate(square2Reg, 0.0f);

    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * CEL_VF_LENGTH;
        validMask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsSameType<X1T, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x1Reg, x1 + offset);
        } else if constexpr (IsSameType<X1T, half>::value) {
            AscendC::Reg::RegTensor<X1T> x1Raw;
            AscendC::Reg::DataCopy<X1T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x1Raw, x1 + offset);
            AscendC::Reg::Cast<float, X1T, CEL_CAST_TO_FP32>(x1Reg, x1Raw, validMask);
        } else {
            AscendC::Reg::RegTensor<int32_t> x1Raw;
            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
                x1Raw, reinterpret_cast<__ubuf__ int32_t*>(x1) + offset);
            AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(x1Reg, x1Raw, validMask);
        }

        if constexpr (IsSameType<X2T, float>::value) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(x2Reg, x2 + offset);
        } else if constexpr (IsSameType<X2T, half>::value) {
            AscendC::Reg::RegTensor<X2T> x2Raw;
            AscendC::Reg::DataCopy<X2T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(x2Raw, x2 + offset);
            AscendC::Reg::Cast<float, X2T, CEL_CAST_TO_FP32>(x2Reg, x2Raw, validMask);
        } else {
            AscendC::Reg::RegTensor<int32_t> x2Raw;
            AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_NORM>(
                x2Raw, reinterpret_cast<__ubuf__ int32_t*>(x2) + offset);
            AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(x2Reg, x2Raw, validMask);
        }

        // Aligned loads may read the padded tail. Select removes those lanes before accumulation.
        AscendC::Reg::Select<float>(x1Reg, x1Reg, zeroReg, validMask);
        AscendC::Reg::Select<float>(x2Reg, x2Reg, zeroReg, validMask);
        AscendC::Reg::Mul<float>(tmpReg, x1Reg, x2Reg, allMask);
        AscendC::Reg::Add<float>(dotReg, dotReg, tmpReg, allMask);
        AscendC::Reg::Mul<float>(tmpReg, x1Reg, x1Reg, allMask);
        AscendC::Reg::Add<float>(square1Reg, square1Reg, tmpReg, allMask);
        AscendC::Reg::Mul<float>(tmpReg, x2Reg, x2Reg, allMask);
        AscendC::Reg::Add<float>(square2Reg, square2Reg, tmpReg, allMask);
    }

    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedDot, dotReg, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedSquare1, square1Reg, allMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(reducedSquare2, square2Reg, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial, reducedDot, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial + 1, reducedSquare1,
                                                                                   allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(partial + 2, reducedSquare2,
                                                                                   allMask);
}

template <typename TargetT>
__simd_vf__ inline void FinalizeLossVf(__ubuf__ float* stats, __ubuf__ TargetT* target, __ubuf__ float* loss,
                                       float margin, float eps)
{
    AscendC::Reg::RegTensor<float> dotReg;
    AscendC::Reg::RegTensor<float> square1Reg;
    AscendC::Reg::RegTensor<float> square2Reg;
    AscendC::Reg::RegTensor<float> targetReg;
    AscendC::Reg::RegTensor<float> denomReg;
    AscendC::Reg::RegTensor<float> cosReg;
    AscendC::Reg::RegTensor<float> zeroReg;
    AscendC::Reg::RegTensor<float> oneReg;
    AscendC::Reg::RegTensor<float> positiveLossReg;
    AscendC::Reg::RegTensor<float> negativeLossReg;
    AscendC::Reg::RegTensor<float> clampedNegativeReg;
    AscendC::Reg::RegTensor<float> positiveOutReg;
    AscendC::Reg::RegTensor<float> negativeOutReg;
    AscendC::Reg::RegTensor<float> outputReg;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg positiveMask;
    AscendC::Reg::MaskReg negativeMask;
    AscendC::Reg::MaskReg greaterThanZeroMask;
    AscendC::Reg::MaskReg nanMask;

    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dotReg, stats);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(square1Reg, stats + 1);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(square2Reg, stats + 2);
    if constexpr (IsSameType<TargetT, float>::value) {
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(targetReg, target);
    } else if constexpr (IsSameType<TargetT, half>::value) {
        AscendC::Reg::RegTensor<TargetT> targetRaw;
        AscendC::Reg::DataCopy<TargetT, AscendC::Reg::LoadDist::DIST_BRC_B16>(targetRaw, target);
        AscendC::Reg::Cast<float, TargetT, CEL_CAST_TO_FP32>(targetReg, targetRaw, allMask);
    } else {
        AscendC::Reg::RegTensor<int32_t> targetRaw;
        AscendC::Reg::DataCopy<int32_t, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            targetRaw, reinterpret_cast<__ubuf__ int32_t*>(target));
        AscendC::Reg::Cast<float, int32_t, CEL_CAST_TO_FP32>(targetReg, targetRaw, allMask);
    }

    AscendC::Reg::Adds<float>(square1Reg, square1Reg, eps, allMask);
    AscendC::Reg::Adds<float>(square2Reg, square2Reg, eps, allMask);
    AscendC::Reg::Sqrt<float>(square1Reg, square1Reg, allMask);
    AscendC::Reg::Sqrt<float>(square2Reg, square2Reg, allMask);
    AscendC::Reg::Mul<float>(denomReg, square1Reg, square2Reg, allMask);
    AscendC::Reg::Div<float>(cosReg, dotReg, denomReg, allMask);

    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Sub<float>(positiveLossReg, oneReg, cosReg, allMask);
    AscendC::Reg::Adds<float>(negativeLossReg, cosReg, -margin, allMask);

    AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(greaterThanZeroMask, negativeLossReg, 0.0f, allMask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nanMask, negativeLossReg, negativeLossReg, allMask);
    AscendC::Reg::Select<float>(clampedNegativeReg, negativeLossReg, zeroReg, greaterThanZeroMask);
    AscendC::Reg::Select<float>(clampedNegativeReg, negativeLossReg, clampedNegativeReg, nanMask);

    AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(positiveMask, targetReg, 1.0f, allMask);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(negativeMask, targetReg, -1.0f, allMask);
    AscendC::Reg::Select<float>(positiveOutReg, positiveLossReg, zeroReg, positiveMask);
    AscendC::Reg::Select<float>(negativeOutReg, clampedNegativeReg, zeroReg, negativeMask);
    AscendC::Reg::Add<float>(outputReg, positiveOutReg, negativeOutReg, allMask);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(loss, outputReg, allMask);
}

template <typename X1T, typename X2T, typename TargetT>
class CosineEmbeddingLossKernel {
public:
    __aicore__ inline CosineEmbeddingLossKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR target, GM_ADDR y, GM_ADDR workspace,
                                const CosineEmbeddingLossTilingData* td, TPipe* pipe)
    {
        td_ = td;
        pipe_ = pipe;
        blockIdx_ = static_cast<int64_t>(GetBlockIdx());
        rowBase_ = blockIdx_ * td_->rowsPerCore;
        if (blockIdx_ >= td_->usedCoreNum) {
            rows_ = 0;
        } else if (blockIdx_ == td_->usedCoreNum - 1) {
            rows_ = td_->tailRows;
        } else {
            rows_ = td_->rowsPerCore;
        }

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ X1T*>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ X2T*>(x2));
        tgtGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TargetT*>(target));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe_->InitBuffer(scalarBuf_, CEL_UB_BLOCK_BYTES);

        if (td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH) {
            pipe_->InitBuffer(x1Queue_, 1, td_->featureTile * static_cast<int64_t>(sizeof(X1T)));
            pipe_->InitBuffer(x2Queue_, 1, td_->featureTile * static_cast<int64_t>(sizeof(X2T)));
            pipe_->InitBuffer(targetQueue_, 1, CEL_UB_BLOCK_BYTES);
            pipe_->InitBuffer(featurePartialBuf_, CEL_UB_BLOCK_BYTES);
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
            const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), workspaceElements);
            pipe_->InitBuffer(corePartialBuf_,
                              COSINE_EMBEDDING_LOSS_PARTIAL_BUF_ELEMS * static_cast<int64_t>(sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        float partial = 0.0f;
        for (int64_t row = 0; row < rows_; ++row) {
            const int64_t outputIndex = rowBase_ + row;
            const bool useFastPath = td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH;
            const float loss = useFastPath ? ComputeFastLoss(outputIndex) : ComputeGenericLoss(outputIndex);
            if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
                WriteScalarOutput(outputIndex, loss);
            } else {
                partial += loss;
            }
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
            FinishReduction(partial);
        }
    }

private:
    template <typename T>
    __aicore__ inline float ReadAsFloat(GlobalTensor<T>& gm, int64_t offset)
    {
        return static_cast<float>(gm.GetValue(static_cast<uint64_t>(offset)));
    }

    __aicore__ inline void LinearToOutputCoords(int64_t index, int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        for (int64_t axis = static_cast<int64_t>(td_->outputRank) - 1; axis >= 0; --axis) {
            const int64_t dim = td_->outputShape[static_cast<uint32_t>(axis)];
            coords[static_cast<uint32_t>(axis)] = index % dim;
            index /= dim;
        }
    }

    __aicore__ inline int64_t OffsetFromStrides(const int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                const int64_t strides[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        int64_t offset = 0;
        for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
            offset += coords[axis] * strides[axis];
        }
        return offset;
    }

    __aicore__ inline float ComputeGenericLoss(int64_t outputIndex)
    {
        int64_t coords[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        LinearToOutputCoords(outputIndex, coords);
        const int64_t x1Base = OffsetFromStrides(coords, td_->x1OutStrides);
        const int64_t x2Base = OffsetFromStrides(coords, td_->x2OutStrides);
        const int64_t targetOffset = OffsetFromStrides(coords, td_->targetOutStrides);

        float dot = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (int64_t c = 0; c < td_->d; ++c) {
            const float a = ReadAsFloat(x1Gm_, x1Base + c * td_->x1ReduceStride);
            const float b = ReadAsFloat(x2Gm_, x2Base + c * td_->x2ReduceStride);
            dot += a * b;
            s1 += a * a;
            s2 += b * b;
        }

        const float denom = sqrt(s1 + td_->eps) * sqrt(s2 + td_->eps);
        const float cos = dot / denom;
        const float target = ReadAsFloat(tgtGm_, targetOffset);
        if (target == 1.0f) {
            return 1.0f - cos;
        }
        if (target == -1.0f) {
            const float neg = cos - td_->margin;
            return (neg != neg || neg > 0.0f) ? neg : 0.0f;
        }
        return 0.0f;
    }

    __aicore__ inline float ComputeFastLoss(int64_t outputIndex)
    {
        const int64_t x1Base = outputIndex * td_->x1OutStrides[0];
        const int64_t x2Base = outputIndex * td_->x2OutStrides[0];
        float dot = 0.0f;
        float square1 = 0.0f;
        float square2 = 0.0f;

        for (int64_t featureOffset = 0; featureOffset < td_->d; featureOffset += td_->featureTile) {
            const int64_t remaining = td_->d - featureOffset;
            const int64_t current = remaining > td_->featureTile ? td_->featureTile : remaining;
            LocalTensor<X1T> x1Local = x1Queue_.template AllocTensor<X1T>();
            LocalTensor<X2T> x2Local = x2Queue_.template AllocTensor<X2T>();
            DataCopyExtParams x1Params{1, static_cast<uint32_t>(current * static_cast<int64_t>(sizeof(X1T))), 0, 0, 0};
            DataCopyExtParams x2Params{1, static_cast<uint32_t>(current * static_cast<int64_t>(sizeof(X2T))), 0, 0, 0};
            DataCopyPadExtParams<X1T> x1Pad{false, 0, 0, 0};
            DataCopyPadExtParams<X2T> x2Pad{false, 0, 0, 0};
            DataCopyPad(x1Local, x1Gm_[x1Base + featureOffset], x1Params, x1Pad);
            DataCopyPad(x2Local, x2Gm_[x2Base + featureOffset], x2Params, x2Pad);
            x1Queue_.EnQue(x1Local);
            x2Queue_.EnQue(x2Local);

            x1Local = x1Queue_.template DeQue<X1T>();
            x2Local = x2Queue_.template DeQue<X2T>();
            LocalTensor<float> partialLocal = featurePartialBuf_.Get<float>();
            auto* x1Addr = reinterpret_cast<__ubuf__ X1T*>(x1Local.GetPhyAddr());
            auto* x2Addr = reinterpret_cast<__ubuf__ X2T*>(x2Local.GetPhyAddr());
            auto* partialAddr = reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr());
            const uint16_t repeatTime = static_cast<uint16_t>((current + CEL_VF_LENGTH - 1) / CEL_VF_LENGTH);
            asc_vf_call<ComputeFeaturePartialVf<X1T, X2T>>(x1Addr, x2Addr, partialAddr, static_cast<uint32_t>(current),
                                                           repeatTime);

            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            dot += partialLocal.GetValue(0);
            square1 += partialLocal.GetValue(1);
            square2 += partialLocal.GetValue(2);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
        }

        LocalTensor<float> statsLocal = featurePartialBuf_.Get<float>();
        statsLocal.SetValue(0, dot);
        statsLocal.SetValue(1, square1);
        statsLocal.SetValue(2, square2);

        LocalTensor<TargetT> targetLocal = targetQueue_.template AllocTensor<TargetT>();
        const int64_t targetOffset = outputIndex * td_->targetOutStrides[0];
        DataCopyExtParams targetParams{1, static_cast<uint32_t>(sizeof(TargetT)), 0, 0, 0};
        DataCopyPadExtParams<TargetT> targetPad{false, 0, 0, 0};
        DataCopyPad(targetLocal, tgtGm_[targetOffset], targetParams, targetPad);
        targetQueue_.EnQue(targetLocal);
        targetLocal = targetQueue_.template DeQue<TargetT>();

        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        LocalTensor<float> lossLocal = scalarBuf_.Get<float>();
        auto* statsAddr = reinterpret_cast<__ubuf__ float*>(statsLocal.GetPhyAddr());
        auto* targetAddr = reinterpret_cast<__ubuf__ TargetT*>(targetLocal.GetPhyAddr());
        auto* lossAddr = reinterpret_cast<__ubuf__ float*>(lossLocal.GetPhyAddr());
        asc_vf_call<FinalizeLossVf<TargetT>>(statsAddr, targetAddr, lossAddr, td_->margin, td_->eps);

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        const float loss = lossLocal.GetValue(0);
        targetQueue_.FreeTensor(targetLocal);
        return loss;
    }

    __aicore__ inline void WriteScalarOutput(int64_t outputIndex, float value)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, value);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[outputIndex], stage, copyParams);
        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
    }

    __aicore__ inline void FinishReduction(float partial)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, partial);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        DataCopyExtParams workspaceParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[blockIdx_ * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE], stage, workspaceParams);

        SyncAll();
        if (blockIdx_ != 0) {
            return;
        }

        LocalTensor<float> partials = corePartialBuf_.Get<float>();
        const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
        DataCopyExtParams inputParams{1, static_cast<uint32_t>(workspaceElements * static_cast<int64_t>(sizeof(float))),
                                      0, 0, 0};
        DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
        DataCopyPad(partials, wsGm_, inputParams, inputPad);
        event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2S);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2S);

        float total = 0.0f;
        for (int64_t core = 0; core < td_->usedCoreNum; ++core) {
            total += partials.GetValue(core * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE);
        }
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_MEAN) {
            total *= td_->meanCoef;
        }
        WriteScalarOutput(0, total);
    }

    const CosineEmbeddingLossTilingData* td_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t rowBase_ = 0;
    int64_t rows_ = 0;

    GlobalTensor<X1T> x1Gm_;
    GlobalTensor<X2T> x2Gm_;
    GlobalTensor<TargetT> tgtGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> wsGm_;

    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECIN, 1> targetQueue_;
    TBuf<QuePosition::VECCALC> featurePartialBuf_;
    TBuf<QuePosition::VECCALC> scalarBuf_;
    TBuf<QuePosition::VECCALC> corePartialBuf_;
};

} // namespace NsCosineEmbeddingLoss

#endif // OPS_LOSS_COSINE_EMBEDDING_LOSS_H_
