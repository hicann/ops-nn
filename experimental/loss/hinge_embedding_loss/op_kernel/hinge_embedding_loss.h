/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HINGE_EMBEDDING_LOSS_H_
#define HINGE_EMBEDDING_LOSS_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hinge_embedding_loss_tiling_data.h"
#include <type_traits>

namespace NsHingeEmbeddingLoss {
using namespace AscendC;
constexpr uint32_t kBufferNum = 2;

template <typename T, uint32_t reductionMode>
class KernelHingeEmbeddingLoss {
    static constexpr bool kUpcast = !std::is_same<T, float>::value;
    static constexpr bool kNeedsReduction = reductionMode != 0;
    static constexpr bool kMeanReduction = reductionMode == 2;

public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                const HingeEmbeddingLossTilingData* tiling)
    {
        const uint32_t core = GetBlockIdx();
        tileDataNum_ = tiling->tileDataNum;
        blockNum_ = tiling->blockNum;
        workspaceFloatsPerCore_ = tiling->workspaceFloatsPerCore;
        margin_ = tiling->margin;
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
        inputGm_.SetGlobalBuffer((__gm__ T*)input + offset_, coreDataNum_);
        targetGm_.SetGlobalBuffer((__gm__ T*)target + offset_, coreDataNum_);
        if constexpr (!kNeedsReduction) {
            lossGm_.SetGlobalBuffer((__gm__ T*)loss + offset_, coreDataNum_);
        } else {
            lossGm_.SetGlobalBuffer((__gm__ T*)loss, 1);
            if (blockNum_ > 1) {
                workspaceGm_.SetGlobalBuffer((__gm__ float*)workspace, blockNum_ * workspaceFloatsPerCore_);
            }
        }

        pipe_.InitBuffer(inputQueue_, kBufferNum, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(targetQueue_, kBufferNum, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(lossQueue_, kBufferNum, tileDataNum_ * sizeof(T));
        pipe_.InitBuffer(negativeBuf_, tileDataNum_ * sizeof(float));
        if constexpr (kNeedsReduction) {
            pipe_.InitBuffer(partialBuf_, workspaceFloatsPerCore_ * sizeof(float));
            pipe_.InitBuffer(downloadQueue_, 1, blockNum_ * workspaceFloatsPerCore_ * sizeof(float));
        }
        if constexpr (kUpcast) {
            pipe_.InitBuffer(inputFloatBuf_, tileDataNum_ * sizeof(float));
            pipe_.InitBuffer(targetFloatBuf_, tileDataNum_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> partial;
        if constexpr (kNeedsReduction) {
            partial = partialBuf_.Get<float>();
            Duplicate(partial, 0.0f, workspaceFloatsPerCore_);
        }
        for (uint32_t i = 0; i < tileNum_; ++i) {
            const uint32_t count = i + 1 == tileNum_ ? tailDataNum_ : tileDataNum_;
            CopyIn(i, count);
            Compute(count, partial);
            if constexpr (!kNeedsReduction) {
                CopyOut(i, count);
            }
        }
        if constexpr (kNeedsReduction) {
            CopyOutReduction(partial);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t count)
    {
        LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        LocalTensor<T> target = targetQueue_.template AllocTensor<T>();
        DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, 0, static_cast<T>(0)};
        DataCopyPad(input, inputGm_[progress * tileDataNum_], params, pad);
        DataCopyPad(target, targetGm_[progress * tileDataNum_], params, pad);
        inputQueue_.EnQue(input);
        targetQueue_.EnQue(target);
    }

    __aicore__ inline void Compute(uint32_t count, LocalTensor<float>& partial)
    {
        LocalTensor<T> input = inputQueue_.template DeQue<T>();
        LocalTensor<T> target = targetQueue_.template DeQue<T>();
        LocalTensor<float> negative = negativeBuf_.Get<float>();
        LocalTensor<float> result;

        if constexpr (kUpcast) {
            LocalTensor<float> inputFloat = inputFloatBuf_.Get<float>();
            LocalTensor<float> targetFloat = targetFloatBuf_.Get<float>();
            Cast(inputFloat, input, RoundMode::CAST_NONE, count);
            Cast(targetFloat, target, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            BuildResult(result, inputFloat, targetFloat, negative, count);
        } else {
            BuildResult(result, input, target, negative, count);
        }

        if constexpr (!kNeedsReduction) {
            LocalTensor<T> output = lossQueue_.template AllocTensor<T>();
            if constexpr (kUpcast) {
                Cast(output, result, RoundMode::CAST_RINT, count);
            } else {
                Adds(output, result, 0.0f, count);
            }
            lossQueue_.EnQue(output);
        } else {
            ReduceSum<float>(result, result, result, count);
            PipeBarrier<PIPE_V>();
            Add(partial, partial, result, 1);
        }
        inputQueue_.FreeTensor(input);
        targetQueue_.FreeTensor(target);
    }

    template <typename InputTensor, typename TargetTensor>
    __aicore__ inline void BuildResult(LocalTensor<float>& result, InputTensor input, TargetTensor target,
                                       LocalTensor<float> negative, uint32_t count)
    {
        Muls(negative, input, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Adds(negative, negative, margin_, count);
        PipeBarrier<PIPE_V>();
        Maxs(negative, negative, 0.0f, count);
        PipeBarrier<PIPE_V>();
        // target is constrained to {-1, 1}; (target + 1) / 2 is therefore the positive-target selector.
        Adds(target, target, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(target, target, 0.5f, count);
        PipeBarrier<PIPE_V>();
        Sub(input, input, negative, count);
        PipeBarrier<PIPE_V>();
        Mul(input, input, target, count);
        PipeBarrier<PIPE_V>();
        Add(negative, negative, input, count);
        PipeBarrier<PIPE_V>();
        result = negative;
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t count)
    {
        LocalTensor<T> output = lossQueue_.template DeQue<T>();
        DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(lossGm_[progress * tileDataNum_], output, params);
        lossQueue_.FreeTensor(output);
    }

    __aicore__ inline void WriteReduction(LocalTensor<float>& result)
    {
        if constexpr (kMeanReduction) {
            Muls(result, result, meanScale_, 1);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<T> output = lossQueue_.template AllocTensor<T>();
        if constexpr (kUpcast) {
            Cast(output, result, RoundMode::CAST_RINT, 1);
        } else {
            Adds(output, result, 0.0f, 1);
        }
        lossQueue_.EnQue(output);
        output = lossQueue_.template DeQue<T>();
        DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(lossGm_, output, params);
        lossQueue_.FreeTensor(output);
    }

    __aicore__ inline void CopyOutReduction(LocalTensor<float>& partial)
    {
        if (blockNum_ == 1) {
            WriteReduction(partial);
            return;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(workspaceGm_[GetBlockIdx() * workspaceFloatsPerCore_], partial, workspaceFloatsPerCore_);
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        if (GetBlockIdx() != 0) {
            return;
        }
        LocalTensor<float> merged = downloadQueue_.template AllocTensor<float>();
        DataCopy(merged, workspaceGm_, blockNum_ * workspaceFloatsPerCore_);
        downloadQueue_.EnQue(merged);
        merged = downloadQueue_.template DeQue<float>();
        ReduceSum<float>(merged, merged, merged, blockNum_ * workspaceFloatsPerCore_);
        PipeBarrier<PIPE_V>();
        WriteReduction(merged);
        downloadQueue_.FreeTensor(merged);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> inputQueue_, targetQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> lossQueue_;
    TQue<QuePosition::VECIN, 1> downloadQueue_;
    TBuf<TPosition::VECCALC> inputFloatBuf_, targetFloatBuf_, negativeBuf_, partialBuf_;
    GlobalTensor<T> inputGm_, targetGm_, lossGm_;
    GlobalTensor<float> workspaceGm_;
    uint64_t offset_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = 0;
    uint32_t tailDataNum_ = 0;
    uint32_t blockNum_ = 1;
    uint32_t workspaceFloatsPerCore_ = 8;
    float margin_ = 1.0f;
    float meanScale_ = 1.0f;
};
} // namespace NsHingeEmbeddingLoss
#endif
