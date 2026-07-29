/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GAUSSIAN_NLL_LOSS_H_
#define GAUSSIAN_NLL_LOSS_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gaussian_nll_loss_tiling_data.h"
#include <type_traits>

namespace NsGaussianNllLoss {
using namespace AscendC;
constexpr uint32_t kBufferDepth = 2;
constexpr uint32_t kNoBroadcast = 0;
constexpr uint32_t kAxisBroadcast = 1;
constexpr uint32_t kTrailingBroadcast = 1;
constexpr uint32_t kScalarBroadcast = 2;

template <typename T, uint32_t reductionMode>
class KernelGaussianNllLoss {
    static constexpr bool kUpcast = !std::is_same<T, float>::value;
    static constexpr bool kNeedsReduction = reductionMode != 0;
    static constexpr bool kMeanReduction = reductionMode == 2;

public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR var, GM_ADDR loss, GM_ADDR workspace,
                                const GaussianNllLossTilingData* tiling, TPipe& pipe)
    {
        pipe_ = &pipe;
        const uint32_t core = GetBlockIdx();
        tileDataNum_ = tiling->tileDataNum;
        blockNum_ = tiling->blockNum;
        workspaceFloatsPerCore_ = tiling->workspaceFloatsPerCore;
        targetBroadcastMode_ = tiling->targetBroadcastMode;
        varBroadcastMode_ = tiling->varBroadcastMode;
        targetAxisSpan_ = tiling->targetAxisSpan;
        targetInnerSize_ = tiling->targetInnerSize;
        varInnerSize_ = tiling->varInnerSize;
        eps_ = tiling->eps;
        fullConstant_ = tiling->fullConstant;
        meanScale_ = tiling->meanScale;
        if (core < tiling->tailBlockNum) {
            coreDataNum_ = tiling->bigCoreDataNum;
            tileNum_ = tiling->finalBigTileNum;
            tailDataNum_ = tiling->bigTailDataNum;
            offset_ = static_cast<uint64_t>(core) * tiling->bigCoreDataNum;
        } else {
            coreDataNum_ = tiling->smallCoreDataNum;
            tileNum_ = tiling->finalSmallTileNum;
            tailDataNum_ = tiling->smallTailDataNum;
            offset_ = static_cast<uint64_t>(tiling->tailBlockNum) * tiling->bigCoreDataNum +
                      static_cast<uint64_t>(core - tiling->tailBlockNum) * tiling->smallCoreDataNum;
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)input, offset_ + coreDataNum_);
        targetGm_.SetGlobalBuffer((__gm__ T*)target, tiling->targetElementCount);
        varGm_.SetGlobalBuffer((__gm__ T*)var, tiling->varElementCount);
        if constexpr (kNeedsReduction) {
            lossGm_.SetGlobalBuffer((__gm__ T*)loss, 1);
            if (blockNum_ > 1) {
                workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace, blockNum_ * workspaceFloatsPerCore_);
            }
        } else {
            lossGm_.SetGlobalBuffer((__gm__ T*)loss, offset_ + coreDataNum_);
        }

        pipe_->InitBuffer(inputQueue_, kBufferDepth, tileDataNum_ * sizeof(T));
        pipe_->InitBuffer(targetQueue_, kBufferDepth, tileDataNum_ * sizeof(T));
        pipe_->InitBuffer(varQueue_, kBufferDepth, tileDataNum_ * sizeof(T));
        pipe_->InitBuffer(lossQueue_, kBufferDepth, tileDataNum_ * sizeof(T));
        pipe_->InitBuffer(resultFloatBuf_, tileDataNum_ * sizeof(float));
        pipe_->InitBuffer(temporaryFloatBuf_, tileDataNum_ * sizeof(float));
        if constexpr (kUpcast) {
            pipe_->InitBuffer(inputFloatBuf_, tileDataNum_ * sizeof(float));
            pipe_->InitBuffer(targetFloatBuf_, tileDataNum_ * sizeof(float));
            pipe_->InitBuffer(varFloatBuf_, tileDataNum_ * sizeof(float));
        }
        if constexpr (kNeedsReduction) {
            pipe_->InitBuffer(partialSumBuf_, workspaceFloatsPerCore_ * sizeof(float));
            pipe_->InitBuffer(allCorePartialQueue_, 1, blockNum_ * workspaceFloatsPerCore_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> partialSum;
        if constexpr (kNeedsReduction) {
            partialSum = partialSumBuf_.Get<float>();
            Duplicate(partialSum, 0.0f, workspaceFloatsPerCore_);
        }
        for (uint32_t tileIndex = 0; tileIndex < tileNum_; ++tileIndex) {
            const uint32_t elementCount = tileIndex + 1 == tileNum_ ? tailDataNum_ : tileDataNum_;
            CopyIn(tileIndex, elementCount);
            Compute(elementCount, partialSum);
            if constexpr (!kNeedsReduction) {
                CopyOut(tileIndex, elementCount);
            }
        }
        if constexpr (kNeedsReduction) {
            CopyOutReduction(partialSum);
        }
    }

private:
    __aicore__ inline uint64_t GetTargetIndex(uint64_t logicalIndex) const
    {
        if (targetBroadcastMode_ == kNoBroadcast) {
            return logicalIndex;
        }
        const uint64_t outerIndex = logicalIndex / targetAxisSpan_;
        const uint64_t innerIndex = logicalIndex % targetInnerSize_;
        return outerIndex * targetInnerSize_ + innerIndex;
    }

    __aicore__ inline uint64_t GetVarIndex(uint64_t logicalIndex) const
    {
        if (varBroadcastMode_ == kNoBroadcast) {
            return logicalIndex;
        }
        if (varBroadcastMode_ == kScalarBroadcast) {
            return 0;
        }
        return logicalIndex / varInnerSize_;
    }

    __aicore__ inline void CopyIn(uint32_t tileIndex, uint32_t elementCount)
    {
        LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        LocalTensor<T> target = targetQueue_.template AllocTensor<T>();
        LocalTensor<T> var = varQueue_.template AllocTensor<T>();
        const uint64_t logicalOffset = offset_ + static_cast<uint64_t>(tileIndex) * tileDataNum_;
        DataCopyExtParams params{1, elementCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(input, inputGm_[logicalOffset], params, pad);
        if (targetBroadcastMode_ == kNoBroadcast) {
            DataCopyPad(target, targetGm_[logicalOffset], params, pad);
        } else {
            for (uint32_t i = 0; i < elementCount; ++i) {
                target.SetValue(i, targetGm_.GetValue(GetTargetIndex(logicalOffset + i)));
            }
        }
        if (varBroadcastMode_ == kNoBroadcast) {
            DataCopyPad(var, varGm_[logicalOffset], params, pad);
        } else {
            for (uint32_t i = 0; i < elementCount; ++i) {
                var.SetValue(i, varGm_.GetValue(GetVarIndex(logicalOffset + i)));
            }
        }
        inputQueue_.EnQue(input);
        targetQueue_.EnQue(target);
        varQueue_.EnQue(var);
    }

    template <typename InputTensor, typename TargetTensor, typename VarTensor>
    __aicore__ inline void BuildResult(LocalTensor<float>& result, InputTensor input, TargetTensor target,
                                       VarTensor var, LocalTensor<float> temporary, uint32_t elementCount)
    {
        Maxs(var, var, eps_, elementCount);
        Sub(result, input, target, elementCount);
        PipeBarrier<PIPE_V>();
        Mul(temporary, result, result, elementCount);
        PipeBarrier<PIPE_V>();
        Div(temporary, temporary, var, elementCount);
        Log(result, var, elementCount);
        PipeBarrier<PIPE_V>();
        Add(result, result, temporary, elementCount);
        PipeBarrier<PIPE_V>();
        Muls(result, result, 0.5f, elementCount);
        if (fullConstant_ != 0.0f) {
            PipeBarrier<PIPE_V>();
            Adds(result, result, fullConstant_, elementCount);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t elementCount, LocalTensor<float>& partialSum)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<T> target = targetQueue_.template DeQue<T>();
        LocalTensor<T> var = varQueue_.template DeQue<T>();
        LocalTensor<float> result = resultFloatBuf_.Get<float>();
        LocalTensor<float> temporary = temporaryFloatBuf_.Get<float>();
        if constexpr (kUpcast) {
            LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
            LocalTensor<float> targetFloat = targetFloatBuf_.Get<float>();
            LocalTensor<float> varFloat = varFloatBuf_.Get<float>();
            Cast(inputFloat, input, RoundMode::CAST_NONE, elementCount);
            Cast(targetFloat, target, RoundMode::CAST_NONE, elementCount);
            Cast(varFloat, var, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            BuildResult(result, inputFloat, targetFloat, varFloat, temporary, elementCount);
        } else {
            BuildResult(result, input, target, var, temporary, elementCount);
        }

        if constexpr (kNeedsReduction) {
            ReduceSum<float>(result, result, result, elementCount);
            PipeBarrier<PIPE_V>();
            Add(partialSum, partialSum, result, 1);
        } else {
            LocalTensor<T> loss = lossQueue_.template AllocTensor<T>();
            if constexpr (kUpcast) {
                Cast(loss, result, RoundMode::CAST_RINT, elementCount);
            } else {
                Adds(loss, result, 0.0f, elementCount);
            }
            lossQueue_.EnQue(loss);
        }
        inputQueue_.FreeTensor(input);
        targetQueue_.FreeTensor(target);
        varQueue_.FreeTensor(var);
    }

    __aicore__ inline void CopyOut(uint32_t tileIndex, uint32_t elementCount)
    {
        LocalTensor<T> loss = lossQueue_.template DeQue<T>();
        DataCopyExtParams params{1, elementCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(lossGm_[offset_ + static_cast<uint64_t>(tileIndex) * tileDataNum_], loss, params);
        lossQueue_.FreeTensor(loss);
    }

    __aicore__ inline void WriteReduction(LocalTensor<float>& result)
    {
        if constexpr (kMeanReduction) {
            Muls(result, result, meanScale_, 1);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> loss = lossQueue_.template AllocTensor<T>();
        if constexpr (kUpcast) {
            Cast(loss, result, RoundMode::CAST_RINT, 1);
        } else {
            Adds(loss, result, 0.0f, 1);
        }
        lossQueue_.EnQue(loss);
        loss = lossQueue_.template DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(lossGm_, loss, params);
        lossQueue_.FreeTensor(loss);
    }

    __aicore__ inline void CopyOutReduction(LocalTensor<float>& partialSum)
    {
        if (blockNum_ == 1) {
            WriteReduction(partialSum);
            return;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGm_[GetBlockIdx() * workspaceFloatsPerCore_], partialSum, workspaceFloatsPerCore_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> allPartialSums = allCorePartialQueue_.template AllocTensor<float>();
        DataCopy(allPartialSums, workspaceGm_, blockNum_ * workspaceFloatsPerCore_);
        allCorePartialQueue_.EnQue(allPartialSums);
        allPartialSums = allCorePartialQueue_.template DeQue<float>();
        ReduceSum<float>(allPartialSums, allPartialSums, allPartialSums, blockNum_ * workspaceFloatsPerCore_);
        PipeBarrier<PIPE_V>();
        WriteReduction(allPartialSums);
        allCorePartialQueue_.FreeTensor(allPartialSums);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, kBufferDepth> inputQueue_, targetQueue_, varQueue_;
    TQue<QuePosition::VECOUT, kBufferDepth> lossQueue_;
    TQue<QuePosition::VECIN, 1> allCorePartialQueue_;
    TBuf<TPosition::VECCALC> inputFloatBuf_, targetFloatBuf_, varFloatBuf_;
    TBuf<TPosition::VECCALC> resultFloatBuf_, temporaryFloatBuf_, partialSumBuf_;
    GlobalTensor<T> inputGm_, targetGm_, varGm_, lossGm_;
    GlobalTensor<float> workspaceGm_;
    uint64_t offset_ = 0;
    uint64_t targetAxisSpan_ = 1;
    uint64_t targetInnerSize_ = 1;
    uint64_t varInnerSize_ = 1;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t blockNum_ = 1;
    uint32_t workspaceFloatsPerCore_ = 8;
    uint32_t targetBroadcastMode_ = 0;
    uint32_t varBroadcastMode_ = 0;
    float eps_ = 1e-6f;
    float fullConstant_ = 0.0f;
    float meanScale_ = 1.0f;
};
} // namespace NsGaussianNllLoss
#endif
