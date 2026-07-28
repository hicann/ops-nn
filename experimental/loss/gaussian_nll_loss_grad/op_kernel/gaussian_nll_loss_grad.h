/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GAUSSIAN_NLL_LOSS_GRAD_H_
#define GAUSSIAN_NLL_LOSS_GRAD_H_

#include "kernel_operator.h"
#include "gaussian_nll_loss_grad_tiling_data.h"

namespace NsGaussianNllLossGrad {
using namespace AscendC;

constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_MEAN = 2;
constexpr uint32_t BROADCAST_TARGET_AXIS = 1;
constexpr uint32_t VAR_SAME = 0;
constexpr uint32_t GM_ALIGN_ELEMS = 8;

template <typename T>
class KernelGaussianNllLossGrad {
public:
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR input, GM_ADDR target, GM_ADDR var, GM_ADDR gradInput,
                                GM_ADDR gradVar, const GaussianNllLossGradTilingData* tiling, TPipe& pipe)
    {
        tiling_ = tiling;
        pipe_ = &pipe;
        const uint32_t core = GetBlockIdx();
        if (core < tiling->tailBlockNum) {
            coreDataNum_ = tiling->bigCoreDataNum;
            tileNum_ = tiling->finalBigTileNum;
            tailDataNum_ = tiling->bigTailDataNum;
            coreOffset_ = core * tiling->bigCoreDataNum;
        } else {
            coreDataNum_ = tiling->smallCoreDataNum;
            tileNum_ = tiling->finalSmallTileNum;
            tailDataNum_ = tiling->smallTailDataNum;
            coreOffset_ = tiling->tailBlockNum * tiling->bigCoreDataNum +
                          (core - tiling->tailBlockNum) * tiling->smallCoreDataNum;
        }

        gradOutputGm_.SetGlobalBuffer((__gm__ T*)gradOutput,
                                      tiling->reduction == REDUCTION_NONE ? tiling->totalDataNum : 1);
        inputGm_.SetGlobalBuffer((__gm__ T*)input, tiling->totalDataNum);
        targetGm_.SetGlobalBuffer((__gm__ T*)target, tiling->targetDataNum);
        varGm_.SetGlobalBuffer((__gm__ T*)var, tiling->varDataNum);
        gradInputGm_.SetGlobalBuffer((__gm__ T*)gradInput, tiling->totalDataNum);
        gradVarGm_.SetGlobalBuffer((__gm__ T*)gradVar, tiling->varDataNum);

        const uint32_t rawBytes = Align32(tiling->tileDataNum * sizeof(T)) + 32U;
        const uint32_t floatBytes = Align32(tiling->tileDataNum * sizeof(float));
        pipe_->InitBuffer(rawGradOutputBuf_, rawBytes);
        pipe_->InitBuffer(rawInputBuf_, rawBytes);
        pipe_->InitBuffer(rawTargetBuf_, rawBytes);
        pipe_->InitBuffer(rawVarBuf_, rawBytes);
        pipe_->InitBuffer(rawOutputBuf_, rawBytes);
        pipe_->InitBuffer(floatGradOutputBuf_, floatBytes);
        pipe_->InitBuffer(floatInputBuf_, floatBytes);
        pipe_->InitBuffer(floatTargetBuf_, floatBytes);
        pipe_->InitBuffer(floatVarBuf_, floatBytes);
        pipe_->InitBuffer(floatOutputBuf_, floatBytes);
        pipe_->InitBuffer(floatTmpBuf_, floatBytes);
    }

    __aicore__ inline void Process()
    {
        ProcessGradInput();
        if (tiling_->varBroadcastMode != VAR_SAME) {
            ProcessBroadcastGradVar();
        }
    }

private:
    __aicore__ inline uint32_t Align32(uint32_t bytes) const { return (bytes + 31U) / 32U * 32U; }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    template <typename U>
    __aicore__ inline void CopyIn(LocalTensor<U> dst, GlobalTensor<U> src, uint32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> padParams{true, 0, 0, static_cast<U>(0)};
        DataCopyPad(dst, src, copyParams, padParams);
    }

    template <typename U>
    __aicore__ inline void CopyOut(GlobalTensor<U> dst, LocalTensor<U> src, uint32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void ToFloat(LocalTensor<float> dst, LocalTensor<T> src, uint32_t count)
    {
        if constexpr (IsSameType<T, float>::value) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void FromFloat(LocalTensor<T> dst, LocalTensor<float> src, uint32_t count)
    {
        if constexpr (IsSameType<T, float>::value) {
            Adds(dst, src, 0.0f, count);
        } else {
            Cast(dst, src, RoundMode::CAST_RINT, count);
        }
    }

    __aicore__ inline void LoadTarget(LocalTensor<T> dst, uint32_t logicalBegin, uint32_t count)
    {
        if (tiling_->targetBroadcastMode != BROADCAST_TARGET_AXIS) {
            CopyIn(dst, targetGm_[logicalBegin], count);
            return;
        }
        const uint32_t inner = tiling_->targetInnerStride;
        const uint32_t axis = tiling_->targetBroadcastAxisSize;
        LocalTensor<T> staging = rawOutputBuf_.Get<T>();
        uint32_t copied = 0;
        while (copied < count) {
            const uint32_t logical = logicalBegin + copied;
            const uint32_t positionInInner = logical % inner;
            const uint32_t chunk = MinU32(count - copied, inner - positionInInner);
            const uint32_t outer = logical / (inner * axis);
            const uint32_t targetIndex = outer * inner + positionInInner;
            CopyIn(staging, targetGm_[targetIndex], chunk);
            PipeBarrier<PIPE_ALL>();
            for (uint32_t i = 0; i < chunk; ++i) {
                dst.SetValue(copied + i, staging.GetValue(i));
            }
            copied += chunk;
        }
    }

    __aicore__ inline void LoadVar(LocalTensor<T> dst, uint32_t logicalBegin, uint32_t count)
    {
        if (tiling_->varBroadcastMode == VAR_SAME) {
            CopyIn(dst, varGm_[logicalBegin], count);
            return;
        }
        uint32_t copied = 0;
        while (copied < count) {
            const uint32_t logical = logicalBegin + copied;
            const uint32_t varIndex = logical / tiling_->varReduceSize;
            const uint32_t position = logical % tiling_->varReduceSize;
            const uint32_t chunk = MinU32(count - copied, tiling_->varReduceSize - position);
            const T value = varGm_.GetValue(varIndex);
            for (uint32_t i = 0; i < chunk; ++i) {
                dst.SetValue(copied + i, value);
            }
            copied += chunk;
        }
    }

    __aicore__ inline void LoadGradOutput(LocalTensor<T> dst, uint32_t logicalBegin, uint32_t count)
    {
        if (tiling_->reduction == REDUCTION_NONE) {
            CopyIn(dst, gradOutputGm_[logicalBegin], count);
        } else {
            Duplicate(dst, gradOutputGm_.GetValue(0), count);
        }
    }

    __aicore__ inline void LoadTile(uint32_t logicalBegin, uint32_t count)
    {
        LocalTensor<T> rawGradOutput = rawGradOutputBuf_.Get<T>();
        LocalTensor<T> rawInput = rawInputBuf_.Get<T>();
        LocalTensor<T> rawTarget = rawTargetBuf_.Get<T>();
        LocalTensor<T> rawVar = rawVarBuf_.Get<T>();
        CopyIn(rawInput, inputGm_[logicalBegin], count);
        LoadTarget(rawTarget, logicalBegin, count);
        LoadVar(rawVar, logicalBegin, count);
        LoadGradOutput(rawGradOutput, logicalBegin, count);
        PipeBarrier<PIPE_ALL>();

        ToFloat(floatGradOutputBuf_.Get<float>(), rawGradOutput, count);
        ToFloat(floatInputBuf_.Get<float>(), rawInput, count);
        ToFloat(floatTargetBuf_.Get<float>(), rawTarget, count);
        ToFloat(floatVarBuf_.Get<float>(), rawVar, count);
        PipeBarrier<PIPE_V>();
        if (tiling_->reduction == REDUCTION_MEAN) {
            Muls(floatGradOutputBuf_.Get<float>(), floatGradOutputBuf_.Get<float>(), tiling_->meanScale, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeGradInput(uint32_t count)
    {
        LocalTensor<float> gradOutput = floatGradOutputBuf_.Get<float>();
        LocalTensor<float> input = floatInputBuf_.Get<float>();
        LocalTensor<float> target = floatTargetBuf_.Get<float>();
        LocalTensor<float> var = floatVarBuf_.Get<float>();
        LocalTensor<float> output = floatOutputBuf_.Get<float>();
        Sub(input, input, target, count);
        PipeBarrier<PIPE_V>();
        Maxs(var, var, tiling_->eps, count);
        PipeBarrier<PIPE_V>();
        Div(output, input, var, count);
        PipeBarrier<PIPE_V>();
        Mul(output, output, gradOutput, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeGradVar(uint32_t count)
    {
        LocalTensor<float> gradOutput = floatGradOutputBuf_.Get<float>();
        LocalTensor<float> difference = floatInputBuf_.Get<float>();
        LocalTensor<float> squareTerm = floatTargetBuf_.Get<float>();
        LocalTensor<float> var = floatVarBuf_.Get<float>();
        LocalTensor<float> output = floatOutputBuf_.Get<float>();
        LocalTensor<float> tmp = floatTmpBuf_.Get<float>();

        Mul(squareTerm, difference, difference, count);
        PipeBarrier<PIPE_V>();
        Mul(tmp, var, var, count);
        PipeBarrier<PIPE_V>();
        Div(squareTerm, squareTerm, tmp, count);
        PipeBarrier<PIPE_V>();
        Duplicate(output, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(output, output, var, count);
        PipeBarrier<PIPE_V>();
        Sub(output, output, squareTerm, count);
        PipeBarrier<PIPE_V>();
        Mul(output, output, gradOutput, count);
        PipeBarrier<PIPE_V>();
        Muls(output, output, 0.5f, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteVector(GlobalTensor<T> dst, uint32_t count)
    {
        LocalTensor<T> rawOutput = rawOutputBuf_.Get<T>();
        FromFloat(rawOutput, floatOutputBuf_.Get<float>(), count);
        PipeBarrier<PIPE_ALL>();
        CopyOut(dst, rawOutput, count);
        PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ProcessGradInput()
    {
        for (uint32_t tile = 0; tile < tileNum_; ++tile) {
            const uint32_t count = tile + 1 == tileNum_ ? tailDataNum_ : tiling_->tileDataNum;
            const uint32_t logicalBegin = coreOffset_ + tile * tiling_->tileDataNum;
            LoadTile(logicalBegin, count);
            ComputeGradInput(count);
            WriteVector(gradInputGm_[logicalBegin], count);
            if (tiling_->varBroadcastMode == VAR_SAME) {
                ComputeGradVar(count);
                WriteVector(gradVarGm_[logicalBegin], count);
            }
        }
    }

    __aicore__ inline float ReduceTile(uint32_t count)
    {
        LocalTensor<float> output = floatOutputBuf_.Get<float>();
        LocalTensor<float> tmp = floatTmpBuf_.Get<float>();
        ReduceSum<float>(output, output, tmp, count);
        PipeBarrier<PIPE_ALL>();
        return output.GetValue(0);
    }

    __aicore__ inline void WriteScalar(GlobalTensor<T> dst, float value)
    {
        LocalTensor<float> output = floatOutputBuf_.Get<float>();
        output.SetValue(0, value);
        PipeBarrier<PIPE_ALL>();
        WriteVector(dst, 1);
    }

    __aicore__ inline void ProcessBroadcastGradVar()
    {
        const uint64_t blockNum = GetBlockNum();
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t varBlocks = (static_cast<uint64_t>(tiling_->varDataNum) + GM_ALIGN_ELEMS - 1) / GM_ALIGN_ELEMS;
        const uint32_t varBegin = static_cast<uint32_t>(varBlocks * blockIdx / blockNum * GM_ALIGN_ELEMS);
        const uint64_t untrimmedEnd = varBlocks * (blockIdx + 1) / blockNum * GM_ALIGN_ELEMS;
        const uint32_t varEnd = static_cast<uint32_t>(untrimmedEnd < tiling_->varDataNum ? untrimmedEnd :
                                                                                           tiling_->varDataNum);
        for (uint32_t varIndex = varBegin; varIndex < varEnd; ++varIndex) {
            const uint32_t logicalBegin = varIndex * tiling_->varReduceSize;
            const uint32_t logicalCount = MinU32(tiling_->varReduceSize, tiling_->totalDataNum - logicalBegin);
            float sum = 0.0f;
            for (uint32_t offset = 0; offset < logicalCount; offset += tiling_->tileDataNum) {
                const uint32_t count = MinU32(tiling_->tileDataNum, logicalCount - offset);
                LoadTile(logicalBegin + offset, count);
                ComputeGradInput(count);
                ComputeGradVar(count);
                sum += ReduceTile(count);
            }
            WriteScalar(gradVarGm_[varIndex], sum);
        }
    }

    const GaussianNllLossGradTilingData* tiling_ = nullptr;
    GlobalTensor<T> gradOutputGm_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> targetGm_;
    GlobalTensor<T> varGm_;
    GlobalTensor<T> gradInputGm_;
    GlobalTensor<T> gradVarGm_;
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> rawGradOutputBuf_;
    TBuf<TPosition::VECCALC> rawInputBuf_;
    TBuf<TPosition::VECCALC> rawTargetBuf_;
    TBuf<TPosition::VECCALC> rawVarBuf_;
    TBuf<TPosition::VECCALC> rawOutputBuf_;
    TBuf<TPosition::VECCALC> floatGradOutputBuf_;
    TBuf<TPosition::VECCALC> floatInputBuf_;
    TBuf<TPosition::VECCALC> floatTargetBuf_;
    TBuf<TPosition::VECCALC> floatVarBuf_;
    TBuf<TPosition::VECCALC> floatOutputBuf_;
    TBuf<TPosition::VECCALC> floatTmpBuf_;
    uint32_t coreOffset_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tailDataNum_ = 0;
};
} // namespace NsGaussianNllLossGrad

#endif
