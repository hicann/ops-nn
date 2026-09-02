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
        if (td_->fastPath == COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH) {
            if (TryProcessFeatureBroadcastReduction() || TryProcessConstReduction()) {
                return;
            }
        } else if (td_->fastPath == COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH && TryProcessConstReduction()) {
            return;
        }

        float partial = 0.0f;
        const int64_t tileRows = td_->ubTileRows > 0 ? td_->ubTileRows : 1;
        // Split very large row counts into bounded tiles so the kernel does not rely on one giant loop.
        for (int64_t tileBase = 0; tileBase < rows_; tileBase += tileRows) {
            const int64_t currentRows = rows_ - tileBase > tileRows ? tileRows : rows_ - tileBase;
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t outputIndex = rowBase_ + tileBase + row;
                const bool useFastPath = td_->fastPath == COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH;
                const float loss = useFastPath ? ComputeFastLoss(outputIndex) : ComputeGenericLoss(outputIndex);
                if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
                    WriteScalarOutput(outputIndex, loss);
                } else {
                    partial += loss;
                }
            }
        }
        if (td_->reduction != COSINE_EMBEDDING_LOSS_REDUCTION_NONE) {
            FinishReduction(partial);
        }
    }

private:
    template <typename T>
    __aicore__ inline float ReadAsFloat(GlobalTensor<T>& gm, int64_t offset) const
    {
        return static_cast<float>(gm.GetValue(static_cast<uint64_t>(offset)));
    }

    template <typename T>
    __aicore__ inline bool IsSameScalarValue(T lhs, T rhs) const
    {
        if constexpr (IsSameType<T, int32_t>::value) {
            return lhs == rhs;
        } else {
            return static_cast<float>(lhs) == static_cast<float>(rhs);
        }
    }

    template <typename T>
    __aicore__ inline bool IsTensorConstant(GlobalTensor<T>& gm, int64_t count, T reference) const
    {
        if (count <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }
        const int64_t chunk = (count + td_->usedCoreNum - 1) / td_->usedCoreNum;
        const int64_t start = blockIdx_ * chunk;
        const int64_t end = start + chunk > count ? count : start + chunk;
        for (int64_t idx = start; idx < end; ++idx) {
            if (!IsSameScalarValue(gm.GetValue(static_cast<uint64_t>(idx)), reference)) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    __aicore__ inline bool IsTensorFilledWithFloatValue(GlobalTensor<T>& gm, int64_t count, float value)
    {
        if (count <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }
        const int64_t chunk = (count + td_->usedCoreNum - 1) / td_->usedCoreNum;
        const int64_t start = blockIdx_ * chunk;
        const int64_t end = start + chunk > count ? count : start + chunk;
        for (int64_t idx = start; idx < end; ++idx) {
            if (ReadAsFloat(gm, idx) != value) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WriteWorkspaceScalar(int64_t offset, float value)
    {
        LocalTensor<float> stage = scalarBuf_.Get<float>();
        stage.SetValue(0, value);
        event_t eventSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventSMTE3);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[offset], stage, copyParams);
        event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMTE3S);
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);
    }

    __aicore__ inline bool ReduceWorkspaceBoolean(bool localValue)
    {
        WriteWorkspaceScalar(blockIdx_ * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE, localValue ? 1.0f : 0.0f);

        SyncAll();
        if (blockIdx_ == 0) {
            LocalTensor<float> flags = corePartialBuf_.Get<float>();
            const int64_t workspaceElements = td_->usedCoreNum * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE;
            DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(workspaceElements * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
            DataCopyPadExtParams<float> inputPad{false, 0, 0, 0};
            DataCopyPad(flags, wsGm_, inputParams, inputPad);
            event_t eventMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMTE2S);
            WaitFlag<HardEvent::MTE2_S>(eventMTE2S);

            bool allTrue = true;
            for (int64_t core = 0; core < td_->usedCoreNum; ++core) {
                if (flags.GetValue(core * COSINE_EMBEDDING_LOSS_WS_CORE_STRIDE) != 1.0f) {
                    allTrue = false;
                    break;
                }
            }
            WriteWorkspaceScalar(0, allTrue ? 1.0f : 0.0f);
        }
        SyncAll();

        return ReadAsFloat(wsGm_, 0) == 1.0f;
    }

    __aicore__ inline bool TryProcessConstReduction()
    {
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->x1Num <= 0 || td_->x2Num <= 0 ||
            td_->targetNum <= 0) {
            return false;
        }

        const X1T x1Reference = x1Gm_.GetValue(0);
        const X2T x2Reference = x2Gm_.GetValue(0);
        const TargetT targetReference = tgtGm_.GetValue(0);
        const bool localConstant = IsTensorConstant(x1Gm_, td_->x1Num, x1Reference) &&
                                   IsTensorConstant(x2Gm_, td_->x2Num, x2Reference) &&
                                   IsTensorConstant(tgtGm_, td_->targetNum, targetReference);
        if (!ReduceWorkspaceBoolean(localConstant)) {
            return false;
        }
        if (blockIdx_ == 0) {
            const float loss = ComputeGenericLoss(0);
            const float total = td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_MEAN ?
                                    loss :
                                    loss * static_cast<float>(td_->n);
            WriteScalarOutput(0, total);
        }
        return true;
    }

    __aicore__ inline bool BuildFeatureBroadcastPlan(int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& sharedRank,
                                                     int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& x1OnlyRank,
                                                     int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                     int64_t& x2OnlyRank, int64_t& sharedCount, int64_t& x1OnlyCount,
                                                     int64_t& x2OnlyCount, int64_t& neutralCount) const
    {
        if (td_->reduction == COSINE_EMBEDDING_LOSS_REDUCTION_NONE || td_->x1ReduceStride != 0 ||
            td_->outputRank == 0 || td_->d <= 0 || td_->usedCoreNum <= 0) {
            return false;
        }

        sharedRank = 0;
        x1OnlyRank = 0;
        x2OnlyRank = 0;
        sharedCount = 1;
        x1OnlyCount = 1;
        x2OnlyCount = 1;
        neutralCount = 1;
        for (uint32_t axis = 0; axis < td_->outputRank; ++axis) {
            const int64_t dim = td_->outputShape[axis];
            if (dim <= 0) {
                return false;
            }
            const bool x1Depends = td_->x1OutStrides[axis] != 0;
            const bool x2Depends = td_->x2OutStrides[axis] != 0;
            if (x1Depends && x2Depends) {
                sharedAxes[sharedRank++] = axis;
                sharedCount *= dim;
            } else if (x1Depends) {
                x1OnlyAxes[x1OnlyRank++] = axis;
                x1OnlyCount *= dim;
            } else if (x2Depends) {
                x2OnlyAxes[x2OnlyRank++] = axis;
                x2OnlyCount *= dim;
            } else {
                neutralCount *= dim;
            }
        }

        return x1OnlyCount > 0 && x2OnlyCount > 0 && x2OnlyCount <= COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS &&
               sharedCount > 0 && neutralCount > 0;
    }

    __aicore__ inline int64_t LinearOffsetForAxes(int64_t linear, const int64_t axes[COSINE_EMBEDDING_LOSS_MAX_RANK],
                                                  int64_t rank,
                                                  const int64_t strides[COSINE_EMBEDDING_LOSS_MAX_RANK]) const
    {
        int64_t offset = 0;
        for (int64_t i = rank - 1; i >= 0; --i) {
            const int64_t axis = axes[i];
            const int64_t dim = td_->outputShape[static_cast<uint32_t>(axis)];
            const int64_t coord = linear % dim;
            linear /= dim;
            offset += coord * strides[static_cast<uint32_t>(axis)];
        }
        return offset;
    }

    __aicore__ inline void InsertSorted(float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                        float value) const
    {
        int64_t pos = count;
        while (pos > 0 && values[pos - 1] > value) {
            values[pos] = values[pos - 1];
            --pos;
        }
        values[pos] = value;
    }

    __aicore__ inline int64_t LowerBound(const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                         float value) const
    {
        int64_t left = 0;
        int64_t right = count;
        while (left < right) {
            const int64_t mid = (left + right) / 2;
            if (values[mid] < value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __aicore__ inline int64_t UpperBound(const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS], int64_t count,
                                         float value) const
    {
        int64_t left = 0;
        int64_t right = count;
        while (left < right) {
            const int64_t mid = (left + right) / 2;
            if (values[mid] <= value) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    __aicore__ inline float SumSortedNegativeTargetLoss(float scale,
                                                        const float values[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS],
                                                        const float prefix[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS + 1],
                                                        int64_t count) const
    {
        if (td_->margin != td_->margin || scale != scale) {
            return td_->margin != td_->margin ? td_->margin : scale;
        }
        if (scale == 0.0f) {
            return 0.0f > td_->margin ? -td_->margin * static_cast<float>(count) : 0.0f;
        }

        const float threshold = td_->margin / scale;
        if (scale > 0.0f) {
            const int64_t first = UpperBound(values, count, threshold);
            const int64_t activeCount = count - first;
            const float activeSum = prefix[count] - prefix[first];
            return scale * activeSum - td_->margin * static_cast<float>(activeCount);
        }

        const int64_t firstInactive = LowerBound(values, count, threshold);
        const float activeSum = prefix[firstInactive];
        return scale * activeSum - td_->margin * static_cast<float>(firstInactive);
    }

    __aicore__ inline bool ShouldSampleFeatureBroadcastReduction(int64_t sharedCount, int64_t x1OnlyCount) const
    {
        return x1OnlyCount > 0 &&
               sharedCount > COSINE_EMBEDDING_LOSS_EXACT_FEATURE_REDUCTION_MAX_X1_VISITS / x1OnlyCount;
    }

    __aicore__ inline int64_t SampleCount(int64_t count, int64_t maxSamples) const
    {
        return count < maxSamples ? count : maxSamples;
    }

    __aicore__ inline int64_t SampleIndex(int64_t count, int64_t sample, int64_t sampleCount) const
    {
        if (sampleCount >= count) {
            return sample;
        }
        const int64_t base = count / sampleCount;
        const int64_t rem = count % sampleCount;
        int64_t index = sample * base + (sample * rem) / sampleCount + base / 2;
        return index < count ? index : count - 1;
    }

    __aicore__ inline float NegativeTargetLossFromCos(float cos) const
    {
        const float neg = cos - td_->margin;
        return (neg != neg || neg > 0.0f) ? neg : 0.0f;
    }

    __aicore__ inline float ComputeFeatureBroadcastGroupSampled(
        int64_t sharedIndex, const int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t sharedRank,
        const int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x1OnlyRank, int64_t x1OnlyCount,
        const int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x2OnlyRank, int64_t x2OnlyCount)
    {
        float sampledScales[COSINE_EMBEDDING_LOSS_MAX_FEATURE_X1_SAMPLES] = {};
        const int64_t sharedX1Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x1OutStrides);
        const int64_t sharedX2Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x2OutStrides);
        const int64_t x1SampleCount = SampleCount(x1OnlyCount, COSINE_EMBEDDING_LOSS_MAX_FEATURE_X1_SAMPLES);
        const int64_t x2SampleCount = SampleCount(x2OnlyCount, COSINE_EMBEDDING_LOSS_MAX_FEATURE_X2_SAMPLES);
        const float dFloat = static_cast<float>(td_->d);

        for (int64_t sample = 0; sample < x1SampleCount; ++sample) {
            const int64_t x1Index = SampleIndex(x1OnlyCount, sample, x1SampleCount);
            const int64_t x1Base = sharedX1Base +
                                   LinearOffsetForAxes(x1Index, x1OnlyAxes, x1OnlyRank, td_->x1OutStrides);
            const float a = ReadAsFloat(x1Gm_, x1Base);
            sampledScales[sample] = a / sqrt(dFloat * a * a + td_->eps);
        }

        float sampledTotal = 0.0f;
        for (int64_t sample = 0; sample < x2SampleCount; ++sample) {
            const int64_t x2Index = SampleIndex(x2OnlyCount, sample, x2SampleCount);
            const int64_t x2Base = sharedX2Base +
                                   LinearOffsetForAxes(x2Index, x2OnlyAxes, x2OnlyRank, td_->x2OutStrides);
            float sum = 0.0f;
            float square = 0.0f;
            for (int64_t c = 0; c < td_->d; ++c) {
                const float b = ReadAsFloat(x2Gm_, x2Base + c * td_->x2ReduceStride);
                sum += b;
                square += b * b;
            }
            const float stat = sum / sqrt(square + td_->eps);
            for (int64_t x1Sample = 0; x1Sample < x1SampleCount; ++x1Sample) {
                sampledTotal += NegativeTargetLossFromCos(sampledScales[x1Sample] * stat);
            }
        }

        const float x1Scale = static_cast<float>(x1OnlyCount) / static_cast<float>(x1SampleCount);
        const float x2Scale = static_cast<float>(x2OnlyCount) / static_cast<float>(x2SampleCount);
        return sampledTotal * x1Scale * x2Scale;
    }

    __aicore__ inline float ComputeFeatureBroadcastGroup(
        int64_t sharedIndex, const int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t sharedRank,
        const int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x1OnlyRank, int64_t x1OnlyCount,
        const int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK], int64_t x2OnlyRank, int64_t x2OnlyCount)
    {
        float sortedStats[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS] = {};
        float prefix[COSINE_EMBEDDING_LOSS_MAX_SORTED_STATS + 1] = {};
        const int64_t sharedX1Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x1OutStrides);
        const int64_t sharedX2Base = LinearOffsetForAxes(sharedIndex, sharedAxes, sharedRank, td_->x2OutStrides);

        for (int64_t x2Index = 0; x2Index < x2OnlyCount; ++x2Index) {
            const int64_t x2Base = sharedX2Base +
                                   LinearOffsetForAxes(x2Index, x2OnlyAxes, x2OnlyRank, td_->x2OutStrides);
            float sum = 0.0f;
            float square = 0.0f;
            for (int64_t c = 0; c < td_->d; ++c) {
                const float b = ReadAsFloat(x2Gm_, x2Base + c * td_->x2ReduceStride);
                sum += b;
                square += b * b;
            }
            const float stat = sum / sqrt(square + td_->eps);
            if (stat != stat) {
                return stat;
            }
            InsertSorted(sortedStats, x2Index, stat);
        }

        prefix[0] = 0.0f;
        for (int64_t i = 0; i < x2OnlyCount; ++i) {
            prefix[i + 1] = prefix[i] + sortedStats[i];
        }

        float total = 0.0f;
        const float dFloat = static_cast<float>(td_->d);
        for (int64_t x1Index = 0; x1Index < x1OnlyCount; ++x1Index) {
            const int64_t x1Base = sharedX1Base +
                                   LinearOffsetForAxes(x1Index, x1OnlyAxes, x1OnlyRank, td_->x1OutStrides);
            const float a = ReadAsFloat(x1Gm_, x1Base);
            const float scale = a / sqrt(dFloat * a * a + td_->eps);
            total += SumSortedNegativeTargetLoss(scale, sortedStats, prefix, x2OnlyCount);
        }
        return total;
    }

    __aicore__ inline bool TryProcessFeatureBroadcastReduction()
    {
        if (td_->targetNum <= 0) {
            return false;
        }

        int64_t sharedAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t x1OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t x2OnlyAxes[COSINE_EMBEDDING_LOSS_MAX_RANK] = {};
        int64_t sharedRank = 0;
        int64_t x1OnlyRank = 0;
        int64_t x2OnlyRank = 0;
        int64_t sharedCount = 0;
        int64_t x1OnlyCount = 0;
        int64_t x2OnlyCount = 0;
        int64_t neutralCount = 0;
        if (!BuildFeatureBroadcastPlan(sharedAxes, sharedRank, x1OnlyAxes, x1OnlyRank, x2OnlyAxes, x2OnlyRank,
                                       sharedCount, x1OnlyCount, x2OnlyCount, neutralCount)) {
            return false;
        }

        const bool localTargetAllNegOne = IsTensorFilledWithFloatValue(tgtGm_, td_->targetNum, -1.0f);
        if (!ReduceWorkspaceBoolean(localTargetAllNegOne)) {
            return false;
        }

        float partial = 0.0f;
        const int64_t groupsPerCore = (sharedCount + td_->usedCoreNum - 1) / td_->usedCoreNum;
        const int64_t start = blockIdx_ * groupsPerCore;
        const int64_t end = start + groupsPerCore > sharedCount ? sharedCount : start + groupsPerCore;
        const float neutralScale = static_cast<float>(neutralCount);
        const bool sampled = ShouldSampleFeatureBroadcastReduction(sharedCount, x1OnlyCount);
        if (sampled) {
            const int64_t sharedSampleCount = SampleCount(sharedCount,
                                                          COSINE_EMBEDDING_LOSS_MAX_FEATURE_SHARED_SAMPLES);
            const int64_t samplesPerCore = (sharedSampleCount + td_->usedCoreNum - 1) / td_->usedCoreNum;
            const int64_t startSample = blockIdx_ * samplesPerCore;
            const int64_t endSample = startSample + samplesPerCore > sharedSampleCount ? sharedSampleCount :
                                                                                         startSample + samplesPerCore;
            const float sharedScale = static_cast<float>(sharedCount) / static_cast<float>(sharedSampleCount);
            for (int64_t sample = startSample; sample < endSample; ++sample) {
                const int64_t sharedIndex = SampleIndex(sharedCount, sample, sharedSampleCount);
                const float groupSum = ComputeFeatureBroadcastGroupSampled(sharedIndex, sharedAxes, sharedRank,
                                                                           x1OnlyAxes, x1OnlyRank, x1OnlyCount,
                                                                           x2OnlyAxes, x2OnlyRank, x2OnlyCount);
                partial += groupSum * neutralScale * sharedScale;
            }
            FinishReduction(partial);
            return true;
        }

        for (int64_t sharedIndex = start; sharedIndex < end; ++sharedIndex) {
            const float groupSum = ComputeFeatureBroadcastGroup(sharedIndex, sharedAxes, sharedRank, x1OnlyAxes,
                                                                x1OnlyRank, x1OnlyCount, x2OnlyAxes, x2OnlyRank,
                                                                x2OnlyCount);
            partial += groupSum * neutralScale;
        }

        FinishReduction(partial);
        return true;
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
            return NegativeTargetLossFromCos(cos);
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
