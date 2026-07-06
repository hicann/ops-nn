/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file elu_grad_v2.h
 */
#ifndef ELU_GRAD_V2_H
#define ELU_GRAD_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "elu_grad_v2_tiling_data.h"
#include "elu_grad_v2_tiling_key.h"

namespace NsEluGradV2 {
using namespace AscendC;

constexpr int32_t MAX_BUFFER_NUM = ELU_GRAD_V2_MAX_BUFFER_NUM;

template <typename T, typename U>
struct IsSameType {
    static constexpr bool value = false;
};

template <typename T>
struct IsSameType<T, T> {
    static constexpr bool value = true;
};

template <typename T, typename U>
inline constexpr bool kIsSameType = IsSameType<T, U>::value;

template <typename T, bool IS_RESULT_MODE>
class KernelEluGradV2 {
public:
    __aicore__ inline KernelEluGradV2() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, const EluGradV2TilingData* tilingData,
                                TPipe* pipe)
    {
        uint32_t coreIdx = GetBlockIdx();
        pipe_ = pipe;

        coreNum_ = tilingData->coreNum;
        bigCoreNum_ = tilingData->bigCoreNum;
        bigCoreDataNum_ = tilingData->bigCoreDataNum;
        smallCoreDataNum_ = tilingData->smallCoreDataNum;
        lastCoreDataNum_ = tilingData->lastCoreDataNum;
        tileLength_ = tilingData->tileDataNum;
        bufferNum_ = tilingData->bufferOpen != 0U ? ELU_GRAD_V2_MAX_BUFFER_NUM : 1U;
        computeChunk_ = tilingData->computeChunk;

        if (coreIdx < bigCoreNum_) {
            startOffset_ = coreIdx * bigCoreDataNum_;
            processLength_ = bigCoreDataNum_;
        } else {
            startOffset_ = bigCoreNum_ * bigCoreDataNum_ + (coreIdx - bigCoreNum_) * smallCoreDataNum_;
            processLength_ = smallCoreDataNum_;
        }
        if (coreIdx == coreNum_ - 1U) {
            processLength_ = lastCoreDataNum_;
        }
        tileCount_ = CeilDiv(processLength_, tileLength_);
        lastTileLength_ = GetTailLength(processLength_, tileLength_);
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads + startOffset_, processLength_);
        activationsGm_.SetGlobalBuffer((__gm__ T*)activations + startOffset_, processLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset_, processLength_);

        factorPos_ = tilingData->scale;
        factorNeg_ = tilingData->alpha * tilingData->scale * tilingData->inputScale;
        inputScale_ = tilingData->inputScale;
        factorNegBias_ = (inputScale_ == 0.0F) ? 0.0F : (factorNeg_ / inputScale_);

        InitScratchBuffers();
        if (IsDirectSingleTilePath()) {
            pipe_->InitBuffer(gradsDirectBuf_, tileLength_ * sizeof(T));
            pipe_->InitBuffer(actDirectBuf_, tileLength_ * sizeof(T));
            pipe_->InitBuffer(yDirectBuf_, tileLength_ * sizeof(T));
            return;
        }

        pipe_->InitBuffer(inQueueGrads_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueAct_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueueY_, bufferNum_, tileLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (processLength_ == 0U) {
            return;
        }
        if (IsDirectSingleTilePath()) {
            ProcessSmallDirect();
            return;
        }
        if (bufferNum_ == 1U || tileCount_ <= 1U) {
            for (uint32_t i = 0; i < tileCount_; ++i) {
                uint32_t currentLength = GetTileLength(i);
                CopyIn(GetTileOffset(i), currentLength);
                Compute(currentLength);
                CopyOut(GetTileOffset(i), currentLength);
            }
            return;
        }

        CopyIn(0U, GetTileLength(0U));
        for (uint32_t i = 1U; i <= tileCount_; ++i) {
            if (i < tileCount_) {
                CopyIn(GetTileOffset(i), GetTileLength(i));
            }
            Compute(GetTileLength(i - 1U));
            if (i > 1U) {
                CopyOut(GetTileOffset(i - 2U), GetTileLength(i - 2U));
            }
        }
        CopyOut(GetTileOffset(tileCount_ - 1U), GetTileLength(tileCount_ - 1U));
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        if (divisor == 0U) {
            return 0U;
        }
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor)
    {
        if (factor == 0U) {
            return value;
        }
        return ((value + factor - 1U) / factor) * factor;
    }

    __aicore__ inline uint32_t GetTailLength(uint32_t total, uint32_t tile)
    {
        if (total == 0U) {
            return 0U;
        }
        uint32_t tail = total % tile;
        return tail == 0U ? tile : tail;
    }

    __aicore__ inline uint32_t GetExecLength(uint32_t length)
    {
        uint32_t alignNum = ELU_GRAD_V2_BLOCK_BYTES / sizeof(T);
        if (alignNum == 0U) {
            return length;
        }
        return Min(AlignUp(length, alignNum), computeChunk_);
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) { return tileIdx * tileLength_; }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx)
    {
        return (tileIdx + 1U == tileCount_) ? lastTileLength_ : tileLength_;
    }

    __aicore__ inline bool IsDirectSingleTilePath() const
    {
        return tileCount_ == 1U && processLength_ <= ELU_GRAD_V2_CORE_CHUNK;
    }

    __aicore__ inline bool IsNearlyOne(float value) const { return value > 0.999F && value < 1.001F; }

    __aicore__ inline void InitScratchBuffers()
    {
        pipe_->InitBuffer(maskBuf_, computeChunk_ * sizeof(uint8_t));
        if constexpr (kIsSameType<T, bfloat16_t> || kIsSameType<T, float> ||
                      (!kIsSameType<T, float> && !kIsSameType<T, bfloat16_t>)) {
            pipe_->InitBuffer(tmpBufF32_, computeChunk_ * sizeof(float));
        }
        if constexpr (!kIsSameType<T, float>) {
            pipe_->InitBuffer(gradsBufF32_, computeChunk_ * sizeof(float));
            pipe_->InitBuffer(actBufF32_, computeChunk_ * sizeof(float));
        }
    }

    __aicore__ inline void ProcessSmallDirect()
    {
        LocalTensor<T> gradsLocal = gradsDirectBuf_.Get<T>();
        LocalTensor<T> actLocal = actDirectBuf_.Get<T>();
        LocalTensor<T> yLocal = yDirectBuf_.Get<T>();

        if (processLength_ == tileLength_) {
            DataCopy(gradsLocal, gradsGm_[0], processLength_);
            DataCopy(actLocal, activationsGm_[0], processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(gradsLocal, gradsGm_[0], copyParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(actLocal, activationsGm_[0], copyParams, {false, 0, 0, 0});
        }

        MTE2ToVSync();
        if constexpr (kIsSameType<T, float>) {
            ComputeFloat(gradsLocal, actLocal, yLocal, processLength_);
        } else {
            ComputeMixedDirect(gradsLocal, actLocal, yLocal, processLength_);
        }
        VToMTE3Sync();

        if (processLength_ == tileLength_) {
            DataCopy(yGm_[0], yLocal, processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> gradsLocal = inQueueGrads_.AllocTensor<T>();
        LocalTensor<T> actLocal = inQueueAct_.AllocTensor<T>();
        if (length == tileLength_) {
            DataCopy(gradsLocal, gradsGm_[offset], length);
            DataCopy(actLocal, activationsGm_[offset], length);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = length * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(gradsLocal, gradsGm_[offset], copyParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(actLocal, activationsGm_[offset], copyParams, {false, 0, 0, 0});
        }
        inQueueGrads_.EnQue(gradsLocal);
        inQueueAct_.EnQue(actLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> gradsRaw = inQueueGrads_.DeQue<T>();
        LocalTensor<T> actRaw = inQueueAct_.DeQue<T>();
        LocalTensor<T> yRaw = outQueueY_.AllocTensor<T>();

        if constexpr (kIsSameType<T, float>) {
            ComputeFloat(gradsRaw, actRaw, yRaw, length);
        } else {
            ComputeMixed(gradsRaw, actRaw, yRaw, length);
        }

        outQueueY_.EnQue(yRaw);
        inQueueGrads_.FreeTensor(gradsRaw);
        inQueueAct_.FreeTensor(actRaw);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        if (length == tileLength_) {
            DataCopy(yGm_[offset], yLocal, length);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = length * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeFloat(LocalTensor<float> gradsRaw, LocalTensor<float> actRaw, LocalTensor<float> yRaw,
                                        uint32_t length)
    {
        if constexpr (IS_RESULT_MODE) {
            ComputeFloatResultMode(gradsRaw, actRaw, yRaw, length);
        } else {
            ComputeFloatExpMode(gradsRaw, actRaw, yRaw, length);
        }
    }

    __aicore__ inline void ComputeMixed(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                        uint32_t length)
    {
        if constexpr (IS_RESULT_MODE) {
            ComputeMixedResultMode(gradsRaw, actRaw, yRaw, length);
        } else {
            ComputeMixedExpMode(gradsRaw, actRaw, yRaw, length);
        }
    }

    __aicore__ inline void ComputeMixedDirect(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                              uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        uint32_t execLength = GetExecLength(length);
        const bool unitFactorPos = IsNearlyOne(factorPos_);
        const bool unitFactorNeg = IsNearlyOne(factorNeg_);
        const bool unitInputScale = IsNearlyOne(inputScale_);

        if constexpr (kIsSameType<T, bfloat16_t>) {
            Cast(gradsF32, gradsRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actRaw, RoundMode::CAST_NONE, execLength);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
        } else {
            CompareScalar(mask, actRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength);
            Cast(gradsF32, gradsRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actRaw, RoundMode::CAST_NONE, execLength);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (IS_RESULT_MODE) {
            Adds(actF32, actF32, factorNegBias_, execLength);
            if (!unitInputScale) {
                Muls(actF32, actF32, inputScale_, execLength);
            }
            Mul(actF32, actF32, gradsF32, execLength);
        } else {
            if (!unitInputScale) {
                Muls(actF32, actF32, inputScale_, execLength);
            }
            Exp(actF32, actF32, execLength);
            if (!unitFactorNeg) {
                Muls(actF32, actF32, factorNeg_, execLength);
            }
            Mul(actF32, actF32, gradsF32, execLength);
        }
        if (!unitFactorPos) {
            Muls(gradsF32, gradsF32, factorPos_, execLength);
        }
        PipeBarrier<PIPE_V>();
        Select(gradsF32, mask, actF32, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
        PipeBarrier<PIPE_V>();
        Cast(yRaw, gradsF32, RoundMode::CAST_RINT, execLength);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFloatResultMode(LocalTensor<float> gradsRaw, LocalTensor<float> actRaw,
                                                  LocalTensor<float> yRaw, uint32_t length)
    {
        LocalTensor<float> tmp = tmpBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        if (length <= computeChunk_) {
            uint32_t execLength = GetExecLength(length);
            CompareScalar(mask, actRaw, 0.0F, CMPMODE::LE, execLength);
            Adds(tmp, actRaw, factorNegBias_, execLength);
            if (!IsNearlyOne(inputScale_)) {
                Muls(tmp, tmp, inputScale_, execLength);
            }
            Mul(tmp, tmp, gradsRaw, execLength);
            Muls(yRaw, gradsRaw, factorPos_, execLength);
            Select(yRaw, mask, tmp, yRaw, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
            return;
        }

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<float> gradsChunk = gradsRaw[offset];
            LocalTensor<float> actChunk = actRaw[offset];
            LocalTensor<float> yChunk = yRaw[offset];

            CompareScalar(mask, actChunk, 0.0F, CMPMODE::LE, execLength);
            Adds(tmp, actChunk, factorNegBias_, execLength);
            if (!IsNearlyOne(inputScale_)) {
                Muls(tmp, tmp, inputScale_, execLength);
            }
            Mul(tmp, tmp, gradsChunk, execLength);
            Muls(yChunk, gradsChunk, factorPos_, execLength);
            Select(yChunk, mask, tmp, yChunk, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
        }
    }

    __aicore__ inline void ComputeFloatExpMode(LocalTensor<float> gradsRaw, LocalTensor<float> actRaw,
                                               LocalTensor<float> yRaw, uint32_t length)
    {
        LocalTensor<float> tmp = tmpBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<float> gradsChunk = gradsRaw[offset];
            LocalTensor<float> actChunk = actRaw[offset];
            LocalTensor<float> yChunk = yRaw[offset];

            CompareScalar(mask, actChunk, 0.0F, CMPMODE::LE, execLength);
            Muls(tmp, actChunk, inputScale_, execLength);
            Exp(tmp, tmp, execLength);
            Muls(tmp, tmp, factorNeg_, execLength);
            Mul(tmp, tmp, gradsChunk, execLength);
            Muls(actChunk, gradsChunk, factorPos_, execLength);
            Select(yChunk, mask, tmp, actChunk, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
        }
    }

    __aicore__ inline void ComputeMixedResultMode(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                                  uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        if (length <= computeChunk_) {
            uint32_t execLength = GetExecLength(length);
            Cast(gradsF32, gradsRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actRaw, RoundMode::CAST_NONE, execLength);
            if constexpr (kIsSameType<T, bfloat16_t>) {
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
                Adds(tmp, actF32, factorNegBias_, execLength);
                Muls(tmp, tmp, inputScale_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                Muls(actF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(gradsF32, mask, tmp, actF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yRaw, gradsF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                CompareScalar(mask, actRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength);
                PipeBarrier<PIPE_V>();
                Adds(tmp, actF32, factorNegBias_, execLength);
                Muls(tmp, tmp, inputScale_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                Muls(gradsF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(gradsF32, mask, tmp, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yRaw, gradsF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            }
            return;
        }

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<T> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<T> actChunkRaw = actRaw[offset];
            LocalTensor<T> yChunkRaw = yRaw[offset];

            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            if constexpr (kIsSameType<T, bfloat16_t>) {
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
                Adds(tmp, actF32, factorNegBias_, execLength);
                Muls(tmp, tmp, inputScale_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                Muls(actF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(gradsF32, mask, tmp, actF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                CompareScalar(mask, actChunkRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength);
                PipeBarrier<PIPE_V>();
                Adds(tmp, actF32, factorNegBias_, execLength);
                Muls(tmp, tmp, inputScale_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                Muls(gradsF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(gradsF32, mask, tmp, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeMixedExpMode(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                               uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<T> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<T> actChunkRaw = actRaw[offset];
            LocalTensor<T> yChunkRaw = yRaw[offset];

            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            if constexpr (kIsSameType<T, bfloat16_t>) {
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
                Muls(tmp, actF32, inputScale_, execLength);
                Exp(tmp, tmp, execLength);
                Muls(tmp, tmp, factorNeg_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                PipeBarrier<PIPE_V>();
                Muls(actF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(gradsF32, mask, tmp, actF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            } else {
                LocalTensor<float> tmp = tmpBufF32_.Get<float>();
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
                Muls(tmp, actF32, inputScale_, execLength);
                Exp(tmp, tmp, execLength);
                Muls(tmp, tmp, factorNeg_, execLength);
                Mul(tmp, tmp, gradsF32, execLength);
                PipeBarrier<PIPE_V>();
                Muls(gradsF32, gradsF32, factorPos_, execLength);
                PipeBarrier<PIPE_V>();
                Select(actF32, mask, tmp, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
                PipeBarrier<PIPE_V>();
                Cast(yChunkRaw, actF32, RoundMode::CAST_RINT, execLength);
                PipeBarrier<PIPE_V>();
            }
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueGrads_;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueAct_;
    TQue<QuePosition::VECOUT, MAX_BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> gradsDirectBuf_;
    TBuf<QuePosition::VECCALC> actDirectBuf_;
    TBuf<QuePosition::VECCALC> yDirectBuf_;
    TBuf<QuePosition::VECCALC> tmpBufF32_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gradsBufF32_;
    TBuf<QuePosition::VECCALC> actBufF32_;
    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> activationsGm_;
    GlobalTensor<T> yGm_;

    uint32_t coreNum_ = 0U;
    uint32_t bigCoreNum_ = 0U;
    uint32_t bigCoreDataNum_ = 0U;
    uint32_t smallCoreDataNum_ = 0U;
    uint32_t lastCoreDataNum_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t processLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t lastTileLength_ = 0U;
    uint32_t bufferNum_ = 1U;
    uint32_t computeChunk_ = ELU_GRAD_V2_FLOAT_EXP_COMPUTE_CHUNK;
    float factorPos_ = 1.0F;
    float factorNeg_ = 1.0F;
    float factorNegBias_ = 1.0F;
    float inputScale_ = 1.0F;
};

template <typename Tin, typename Tout>
class KernelEluGradV2ResultMixedOutput {
public:
    __aicore__ inline KernelEluGradV2ResultMixedOutput() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, const EluGradV2TilingData* tilingData,
                                TPipe* pipe)
    {
        uint32_t coreIdx = GetBlockIdx();
        pipe_ = pipe;

        coreNum_ = tilingData->coreNum;
        bigCoreNum_ = tilingData->bigCoreNum;
        bigCoreDataNum_ = tilingData->bigCoreDataNum;
        smallCoreDataNum_ = tilingData->smallCoreDataNum;
        lastCoreDataNum_ = tilingData->lastCoreDataNum;
        tileLength_ = tilingData->tileDataNum;
        bufferNum_ = tilingData->bufferOpen != 0U ? ELU_GRAD_V2_MAX_BUFFER_NUM : 1U;
        computeChunk_ = tilingData->computeChunk;

        if (coreIdx < bigCoreNum_) {
            startOffset_ = coreIdx * bigCoreDataNum_;
            processLength_ = bigCoreDataNum_;
        } else {
            startOffset_ = bigCoreNum_ * bigCoreDataNum_ + (coreIdx - bigCoreNum_) * smallCoreDataNum_;
            processLength_ = smallCoreDataNum_;
        }
        if (coreIdx == coreNum_ - 1U) {
            processLength_ = lastCoreDataNum_;
        }
        tileCount_ = CeilDiv(processLength_, tileLength_);
        lastTileLength_ = GetTailLength(processLength_, tileLength_);
        gradsGm_.SetGlobalBuffer((__gm__ Tin*)grads + startOffset_, processLength_);
        activationsGm_.SetGlobalBuffer((__gm__ Tin*)activations + startOffset_, processLength_);
        yGm_.SetGlobalBuffer((__gm__ Tout*)y + startOffset_, processLength_);

        factorPos_ = tilingData->scale;
        factorNeg_ = tilingData->alpha * tilingData->scale * tilingData->inputScale;
        inputScale_ = tilingData->inputScale;
        factorNegBias_ = (inputScale_ == 0.0F) ? 0.0F : (factorNeg_ / inputScale_);

        pipe_->InitBuffer(maskBuf_, computeChunk_ * sizeof(uint8_t));
        pipe_->InitBuffer(tmpBufF32_, computeChunk_ * sizeof(float));
        pipe_->InitBuffer(gradsBufF32_, computeChunk_ * sizeof(float));
        pipe_->InitBuffer(actBufF32_, computeChunk_ * sizeof(float));
        if (IsDirectSingleTilePath()) {
            pipe_->InitBuffer(gradsDirectBuf_, tileLength_ * sizeof(Tin));
            pipe_->InitBuffer(actDirectBuf_, tileLength_ * sizeof(Tin));
            pipe_->InitBuffer(yDirectBuf_, tileLength_ * sizeof(Tout));
            return;
        }

        pipe_->InitBuffer(inQueueGrads_, bufferNum_, tileLength_ * sizeof(Tin));
        pipe_->InitBuffer(inQueueAct_, bufferNum_, tileLength_ * sizeof(Tin));
        pipe_->InitBuffer(outQueueY_, bufferNum_, tileLength_ * sizeof(Tout));
    }

    __aicore__ inline void Process()
    {
        if (processLength_ == 0U) {
            return;
        }
        if (IsDirectSingleTilePath()) {
            ProcessSmallDirect();
            return;
        }
        if (bufferNum_ == 1U || tileCount_ <= 1U) {
            for (uint32_t i = 0; i < tileCount_; ++i) {
                uint32_t currentLength = GetTileLength(i);
                CopyIn(GetTileOffset(i), currentLength);
                Compute(currentLength);
                CopyOut(GetTileOffset(i), currentLength);
            }
            return;
        }

        CopyIn(0U, GetTileLength(0U));
        for (uint32_t i = 1U; i <= tileCount_; ++i) {
            if (i < tileCount_) {
                CopyIn(GetTileOffset(i), GetTileLength(i));
            }
            Compute(GetTileLength(i - 1U));
            if (i > 1U) {
                CopyOut(GetTileOffset(i - 2U), GetTileLength(i - 2U));
            }
        }
        CopyOut(GetTileOffset(tileCount_ - 1U), GetTileLength(tileCount_ - 1U));
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        if (divisor == 0U) {
            return 0U;
        }
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor)
    {
        if (factor == 0U) {
            return value;
        }
        return ((value + factor - 1U) / factor) * factor;
    }

    __aicore__ inline uint32_t GetTailLength(uint32_t total, uint32_t tile)
    {
        if (total == 0U) {
            return 0U;
        }
        uint32_t tail = total % tile;
        return tail == 0U ? tile : tail;
    }

    __aicore__ inline uint32_t GetExecLength(uint32_t length)
    {
        constexpr uint32_t elementBytes = (sizeof(Tout) > sizeof(Tin)) ? sizeof(Tout) : sizeof(Tin);
        uint32_t alignNum = ELU_GRAD_V2_BLOCK_BYTES / elementBytes;
        if (alignNum == 0U) {
            return length;
        }
        return Min(AlignUp(length, alignNum), computeChunk_);
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) { return tileIdx * tileLength_; }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx)
    {
        return (tileIdx + 1U == tileCount_) ? lastTileLength_ : tileLength_;
    }

    __aicore__ inline bool IsDirectSingleTilePath() const
    {
        return tileCount_ == 1U && processLength_ <= ELU_GRAD_V2_CORE_CHUNK;
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<Tin> gradsLocal = inQueueGrads_.AllocTensor<Tin>();
        LocalTensor<Tin> actLocal = inQueueAct_.AllocTensor<Tin>();
        if (length == tileLength_) {
            DataCopy(gradsLocal, gradsGm_[offset], length);
            DataCopy(actLocal, activationsGm_[offset], length);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = length * sizeof(Tin);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(gradsLocal, gradsGm_[offset], copyParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(actLocal, activationsGm_[offset], copyParams, {false, 0, 0, 0});
        }
        inQueueGrads_.EnQue(gradsLocal);
        inQueueAct_.EnQue(actLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<Tin> gradsRaw = inQueueGrads_.DeQue<Tin>();
        LocalTensor<Tin> actRaw = inQueueAct_.DeQue<Tin>();
        LocalTensor<Tout> yRaw = outQueueY_.AllocTensor<Tout>();
        ComputeResult(gradsRaw, actRaw, yRaw, length);
        outQueueY_.EnQue(yRaw);
        inQueueGrads_.FreeTensor(gradsRaw);
        inQueueAct_.FreeTensor(actRaw);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<Tout> yLocal = outQueueY_.DeQue<Tout>();
        if (length == tileLength_) {
            DataCopy(yGm_[offset], yLocal, length);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = length * sizeof(Tout);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        }
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessSmallDirect()
    {
        LocalTensor<Tin> gradsLocal = gradsDirectBuf_.Get<Tin>();
        LocalTensor<Tin> actLocal = actDirectBuf_.Get<Tin>();
        LocalTensor<Tout> yLocal = yDirectBuf_.Get<Tout>();

        if (processLength_ == tileLength_) {
            DataCopy(gradsLocal, gradsGm_[0], processLength_);
            DataCopy(actLocal, activationsGm_[0], processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(Tin);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(gradsLocal, gradsGm_[0], copyParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(actLocal, activationsGm_[0], copyParams, {false, 0, 0, 0});
        }

        MTE2ToVSync();
        ComputeResult(gradsLocal, actLocal, yLocal, processLength_);
        VToMTE3Sync();

        if (processLength_ == tileLength_) {
            DataCopy(yGm_[0], yLocal, processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(Tout);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

    __aicore__ inline void ComputeResult(LocalTensor<Tin> gradsRaw, LocalTensor<Tin> actRaw, LocalTensor<Tout> yRaw,
                                         uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<float> tmp = tmpBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<Tin> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<Tin> actChunkRaw = actRaw[offset];
            LocalTensor<Tout> yChunkRaw = yRaw[offset];

            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            if constexpr (kIsSameType<Tin, bfloat16_t>) {
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
            } else {
                CompareScalar(mask, actChunkRaw, static_cast<Tin>(0.0f), CMPMODE::LE, execLength);
                PipeBarrier<PIPE_V>();
            }

            Adds(tmp, actF32, factorNegBias_, execLength);
            Muls(tmp, tmp, inputScale_, execLength);
            Mul(tmp, tmp, gradsF32, execLength);
            Muls(gradsF32, gradsF32, factorPos_, execLength);
            PipeBarrier<PIPE_V>();
            Select(yChunkRaw, mask, tmp, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueGrads_;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueAct_;
    TQue<QuePosition::VECOUT, MAX_BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> gradsDirectBuf_;
    TBuf<QuePosition::VECCALC> actDirectBuf_;
    TBuf<QuePosition::VECCALC> yDirectBuf_;
    TBuf<QuePosition::VECCALC> tmpBufF32_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gradsBufF32_;
    TBuf<QuePosition::VECCALC> actBufF32_;
    GlobalTensor<Tin> gradsGm_;
    GlobalTensor<Tin> activationsGm_;
    GlobalTensor<Tout> yGm_;

    uint32_t coreNum_ = 0U;
    uint32_t bigCoreNum_ = 0U;
    uint32_t bigCoreDataNum_ = 0U;
    uint32_t smallCoreDataNum_ = 0U;
    uint32_t lastCoreDataNum_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t processLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t lastTileLength_ = 0U;
    uint32_t bufferNum_ = 1U;
    uint32_t computeChunk_ = ELU_GRAD_V2_MIXED_RESULT_COMPUTE_CHUNK;
    float factorPos_ = 1.0F;
    float factorNeg_ = 1.0F;
    float factorNegBias_ = 1.0F;
    float inputScale_ = 1.0F;
};

template <typename T, bool IS_RESULT_MODE>
class KernelEluGradV2SingleTile {
public:
    __aicore__ inline KernelEluGradV2SingleTile() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, const EluGradV2TilingData* tilingData,
                                TPipe* pipe)
    {
        uint32_t coreIdx = GetBlockIdx();
        pipe_ = pipe;

        coreNum_ = tilingData->coreNum;
        bigCoreNum_ = tilingData->bigCoreNum;
        bigCoreDataNum_ = tilingData->bigCoreDataNum;
        smallCoreDataNum_ = tilingData->smallCoreDataNum;
        lastCoreDataNum_ = tilingData->lastCoreDataNum;
        tileLength_ = tilingData->tileDataNum;

        if (coreIdx < bigCoreNum_) {
            startOffset_ = coreIdx * bigCoreDataNum_;
            processLength_ = bigCoreDataNum_;
        } else {
            startOffset_ = bigCoreNum_ * bigCoreDataNum_ + (coreIdx - bigCoreNum_) * smallCoreDataNum_;
            processLength_ = smallCoreDataNum_;
        }
        if (coreIdx == coreNum_ - 1U) {
            processLength_ = lastCoreDataNum_;
        }

        gradsGm_.SetGlobalBuffer((__gm__ T*)grads + startOffset_, processLength_);
        activationsGm_.SetGlobalBuffer((__gm__ T*)activations + startOffset_, processLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset_, processLength_);

        factorPos_ = tilingData->scale;
        factorNeg_ = tilingData->alpha * tilingData->scale * tilingData->inputScale;
        factorNegBias_ = (tilingData->inputScale == 0.0F) ? 0.0F : (factorNeg_ / tilingData->inputScale);
        inputScale_ = tilingData->inputScale;
        unitFactorPos_ = IsNearlyOne(factorPos_);
        unitFactorNeg_ = IsNearlyOne(factorNeg_);
        unitInputScale_ = IsNearlyOne(inputScale_);

        execLength_ = GetExecLength(processLength_);
        pipe_->InitBuffer(gradsDirectBuf_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(actDirectBuf_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(yDirectBuf_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(maskBuf_, execLength_ * sizeof(uint8_t));
        if constexpr (kIsSameType<T, float>) {
            pipe_->InitBuffer(tmpBufF32_, execLength_ * sizeof(float));
        }
        if constexpr (!kIsSameType<T, float>) {
            pipe_->InitBuffer(gradsBufF32_, execLength_ * sizeof(float));
            pipe_->InitBuffer(actBufF32_, execLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (processLength_ == 0U) {
            return;
        }

        LocalTensor<T> gradsLocal = gradsDirectBuf_.Get<T>();
        LocalTensor<T> actLocal = actDirectBuf_.Get<T>();
        LocalTensor<T> yLocal = yDirectBuf_.Get<T>();

        if (processLength_ == tileLength_) {
            DataCopy(gradsLocal, gradsGm_[0], processLength_);
            DataCopy(actLocal, activationsGm_[0], processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(gradsLocal, gradsGm_[0], copyParams, {false, 0, 0, 0});
            AscendC::DataCopyPad(actLocal, activationsGm_[0], copyParams, {false, 0, 0, 0});
        }

        MTE2ToVSync();
        if constexpr (kIsSameType<T, float>) {
            ComputeFloatDirect(gradsLocal, actLocal, yLocal);
        } else {
            ComputeMixedDirect(gradsLocal, actLocal, yLocal);
        }
        VToMTE3Sync();

        if (processLength_ == tileLength_) {
            DataCopy(yGm_[0], yLocal, processLength_);
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = processLength_ * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm_[0], yLocal, copyParams);
        }
    }

private:
    __aicore__ inline bool IsNearlyOne(float value) const { return value > 0.999F && value < 1.001F; }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor) const
    {
        if (factor == 0U) {
            return value;
        }
        return ((value + factor - 1U) / factor) * factor;
    }

    __aicore__ inline uint32_t GetExecLength(uint32_t length) const
    {
        uint32_t alignNum = ELU_GRAD_V2_BLOCK_BYTES / sizeof(T);
        if (alignNum == 0U) {
            return length;
        }
        return AlignUp(length, alignNum);
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void ComputeFloatDirect(LocalTensor<float> gradsRaw, LocalTensor<float> actRaw,
                                              LocalTensor<float> yRaw)
    {
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        CompareScalar(mask, actRaw, 0.0F, CMPMODE::LE, execLength_);

        if constexpr (IS_RESULT_MODE) {
            LocalTensor<float> tmp = tmpBufF32_.Get<float>();
            Adds(tmp, actRaw, factorNegBias_, execLength_);
            if (!unitInputScale_) {
                Muls(tmp, tmp, inputScale_, execLength_);
            }
            Mul(tmp, tmp, gradsRaw, execLength_);
            Muls(yRaw, gradsRaw, unitFactorPos_ ? 1.0F : factorPos_, execLength_);
            Select(yRaw, mask, tmp, yRaw, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength_);
            return;
        } else {
            Muls(yRaw, actRaw, unitInputScale_ ? 1.0F : inputScale_, execLength_);
            Exp(yRaw, yRaw, execLength_);
            if (!unitFactorNeg_) {
                Muls(yRaw, yRaw, factorNeg_, execLength_);
            }
            Mul(yRaw, yRaw, gradsRaw, execLength_);
        }
        Muls(actRaw, gradsRaw, unitFactorPos_ ? 1.0F : factorPos_, execLength_);
        Select(yRaw, mask, yRaw, actRaw, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength_);
    }

    __aicore__ inline void ComputeMixedDirect(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        if constexpr (kIsSameType<T, bfloat16_t>) {
            Cast(gradsF32, gradsRaw, RoundMode::CAST_NONE, execLength_);
            Cast(actF32, actRaw, RoundMode::CAST_NONE, execLength_);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength_);
        } else {
            CompareScalar(mask, actRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength_);
            Cast(gradsF32, gradsRaw, RoundMode::CAST_NONE, execLength_);
            Cast(actF32, actRaw, RoundMode::CAST_NONE, execLength_);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (IS_RESULT_MODE) {
            Adds(actF32, actF32, factorNegBias_, execLength_);
            if (!unitInputScale_) {
                Muls(actF32, actF32, inputScale_, execLength_);
            }
            Mul(actF32, actF32, gradsF32, execLength_);
        } else {
            if (!unitInputScale_) {
                Muls(actF32, actF32, inputScale_, execLength_);
            }
            Exp(actF32, actF32, execLength_);
            if (!unitFactorNeg_) {
                Muls(actF32, actF32, factorNeg_, execLength_);
            }
            Mul(actF32, actF32, gradsF32, execLength_);
        }
        if (!unitFactorPos_) {
            Muls(gradsF32, gradsF32, factorPos_, execLength_);
        }
        PipeBarrier<PIPE_V>();
        Select(gradsF32, mask, actF32, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength_);
        PipeBarrier<PIPE_V>();
        Cast(yRaw, gradsF32, RoundMode::CAST_RINT, execLength_);
        PipeBarrier<PIPE_V>();
    }

private:
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> gradsDirectBuf_;
    TBuf<QuePosition::VECCALC> actDirectBuf_;
    TBuf<QuePosition::VECCALC> yDirectBuf_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> tmpBufF32_;
    TBuf<QuePosition::VECCALC> gradsBufF32_;
    TBuf<QuePosition::VECCALC> actBufF32_;
    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> activationsGm_;
    GlobalTensor<T> yGm_;

    uint32_t coreNum_ = 0U;
    uint32_t bigCoreNum_ = 0U;
    uint32_t bigCoreDataNum_ = 0U;
    uint32_t smallCoreDataNum_ = 0U;
    uint32_t lastCoreDataNum_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t processLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t execLength_ = 0U;
    float factorPos_ = 1.0F;
    float factorNeg_ = 1.0F;
    float factorNegBias_ = 1.0F;
    float inputScale_ = 1.0F;
    bool unitFactorPos_ = false;
    bool unitFactorNeg_ = false;
    bool unitInputScale_ = false;
};

template <typename T, bool IS_RESULT_MODE>
class KernelEluGradV2Float16AlignedFast {
public:
    __aicore__ inline KernelEluGradV2Float16AlignedFast() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, const EluGradV2TilingData* tilingData,
                                TPipe* pipe)
    {
        uint32_t coreIdx = GetBlockIdx();
        pipe_ = pipe;

        coreNum_ = tilingData->coreNum;
        bigCoreNum_ = tilingData->bigCoreNum;
        bigCoreDataNum_ = tilingData->bigCoreDataNum;
        smallCoreDataNum_ = tilingData->smallCoreDataNum;
        lastCoreDataNum_ = tilingData->lastCoreDataNum;
        tileLength_ = tilingData->tileDataNum;
        bufferNum_ = tilingData->bufferOpen != 0U ? ELU_GRAD_V2_MAX_BUFFER_NUM : 1U;
        computeChunk_ = tilingData->computeChunk;

        if (coreIdx < bigCoreNum_) {
            startOffset_ = coreIdx * bigCoreDataNum_;
            processLength_ = bigCoreDataNum_;
        } else {
            startOffset_ = bigCoreNum_ * bigCoreDataNum_ + (coreIdx - bigCoreNum_) * smallCoreDataNum_;
            processLength_ = smallCoreDataNum_;
        }
        if (coreIdx == coreNum_ - 1U) {
            processLength_ = lastCoreDataNum_;
        }
        tileCount_ = CeilDiv(processLength_, tileLength_);
        lastTileLength_ = GetTailLength(processLength_, tileLength_);
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads + startOffset_, processLength_);
        activationsGm_.SetGlobalBuffer((__gm__ T*)activations + startOffset_, processLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset_, processLength_);

        factorPos_ = tilingData->scale;
        factorNeg_ = tilingData->alpha * tilingData->scale * tilingData->inputScale;
        factorNegBias_ = (tilingData->inputScale == 0.0F) ? 0.0F : (tilingData->alpha * tilingData->scale);
        inputScale_ = tilingData->inputScale;
        unitFactorPos_ = IsNearlyOne(factorPos_);
        unitFactorNeg_ = IsNearlyOne(factorNeg_);
        unitInputScale_ = IsNearlyOne(inputScale_);

        pipe_->InitBuffer(inQueueGrads_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueAct_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueueY_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(maskBuf_, computeChunk_ * sizeof(uint8_t));
        pipe_->InitBuffer(gradsBufF32_, computeChunk_ * sizeof(float));
        pipe_->InitBuffer(actBufF32_, computeChunk_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (processLength_ == 0U) {
            return;
        }
        if (bufferNum_ == 1U || tileCount_ <= 1U) {
            for (uint32_t i = 0; i < tileCount_; ++i) {
                uint32_t currentLength = GetTileLength(i);
                CopyIn(GetTileOffset(i), currentLength);
                Compute(currentLength);
                CopyOut(GetTileOffset(i), currentLength);
            }
            return;
        }

        CopyIn(0U, GetTileLength(0U));
        for (uint32_t i = 1U; i <= tileCount_; ++i) {
            if (i < tileCount_) {
                CopyIn(GetTileOffset(i), GetTileLength(i));
            }
            Compute(GetTileLength(i - 1U));
            if (i > 1U) {
                CopyOut(GetTileOffset(i - 2U), GetTileLength(i - 2U));
            }
        }
        CopyOut(GetTileOffset(tileCount_ - 1U), GetTileLength(tileCount_ - 1U));
    }

private:
    __aicore__ inline bool IsNearlyOne(float value) const { return value > 0.999F && value < 1.001F; }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        if (divisor == 0U) {
            return 0U;
        }
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor)
    {
        if (factor == 0U) {
            return value;
        }
        return ((value + factor - 1U) / factor) * factor;
    }

    __aicore__ inline uint32_t GetTailLength(uint32_t total, uint32_t tile)
    {
        if (total == 0U) {
            return 0U;
        }
        uint32_t tail = total % tile;
        return tail == 0U ? tile : tail;
    }

    __aicore__ inline uint32_t GetExecLength(uint32_t length)
    {
        uint32_t alignNum = ELU_GRAD_V2_BLOCK_BYTES / sizeof(T);
        if (alignNum == 0U) {
            return length;
        }
        return Min(AlignUp(length, alignNum), computeChunk_);
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) { return tileIdx * tileLength_; }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx)
    {
        return (tileIdx + 1U == tileCount_) ? lastTileLength_ : tileLength_;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> gradsLocal = inQueueGrads_.AllocTensor<T>();
        LocalTensor<T> actLocal = inQueueAct_.AllocTensor<T>();
        DataCopy(gradsLocal, gradsGm_[offset], length);
        DataCopy(actLocal, activationsGm_[offset], length);
        inQueueGrads_.EnQue(gradsLocal);
        inQueueAct_.EnQue(actLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> gradsRaw = inQueueGrads_.DeQue<T>();
        LocalTensor<T> actRaw = inQueueAct_.DeQue<T>();
        LocalTensor<T> yRaw = outQueueY_.AllocTensor<T>();

        ComputeFloat16(gradsRaw, actRaw, yRaw, length);

        outQueueY_.EnQue(yRaw);
        inQueueGrads_.FreeTensor(gradsRaw);
        inQueueAct_.FreeTensor(actRaw);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        DataCopy(yGm_[offset], yLocal, length);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeFloat16(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                          uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<T> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<T> actChunkRaw = actRaw[offset];
            LocalTensor<T> yChunkRaw = yRaw[offset];

            CompareScalar(mask, actChunkRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength);
            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            PipeBarrier<PIPE_V>();

            if constexpr (IS_RESULT_MODE) {
                Adds(actF32, actF32, factorNegBias_, execLength);
                if (!unitInputScale_) {
                    Muls(actF32, actF32, inputScale_, execLength);
                }
                Mul(actF32, actF32, gradsF32, execLength);
            } else {
                if (!unitInputScale_) {
                    Muls(actF32, actF32, inputScale_, execLength);
                }
                Exp(actF32, actF32, execLength);
                if (!unitFactorNeg_) {
                    Muls(actF32, actF32, factorNeg_, execLength);
                }
                Mul(actF32, actF32, gradsF32, execLength);
            }
            if (!unitFactorPos_) {
                Muls(gradsF32, gradsF32, factorPos_, execLength);
            }
            PipeBarrier<PIPE_V>();
            Select(gradsF32, mask, actF32, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
            PipeBarrier<PIPE_V>();
            Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueGrads_;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueAct_;
    TQue<QuePosition::VECOUT, MAX_BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gradsBufF32_;
    TBuf<QuePosition::VECCALC> actBufF32_;
    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> activationsGm_;
    GlobalTensor<T> yGm_;

    uint32_t coreNum_ = 0U;
    uint32_t bigCoreNum_ = 0U;
    uint32_t bigCoreDataNum_ = 0U;
    uint32_t smallCoreDataNum_ = 0U;
    uint32_t lastCoreDataNum_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t processLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t lastTileLength_ = 0U;
    uint32_t bufferNum_ = 1U;
    uint32_t computeChunk_ = ELU_GRAD_V2_MIXED_EXP_COMPUTE_CHUNK;
    float factorPos_ = 1.0F;
    float factorNeg_ = 1.0F;
    float factorNegBias_ = 1.0F;
    float inputScale_ = 1.0F;
    bool unitFactorPos_ = false;
    bool unitFactorNeg_ = false;
    bool unitInputScale_ = false;
};

template <typename T, bool IS_RESULT_MODE>
class KernelEluGradV2MixedAlignedFast {
public:
    __aicore__ inline KernelEluGradV2MixedAlignedFast() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR activations, GM_ADDR y, const EluGradV2TilingData* tilingData,
                                TPipe* pipe)
    {
        uint32_t coreIdx = GetBlockIdx();
        pipe_ = pipe;

        coreNum_ = tilingData->coreNum;
        bigCoreNum_ = tilingData->bigCoreNum;
        bigCoreDataNum_ = tilingData->bigCoreDataNum;
        smallCoreDataNum_ = tilingData->smallCoreDataNum;
        lastCoreDataNum_ = tilingData->lastCoreDataNum;
        tileLength_ = tilingData->tileDataNum;
        bufferNum_ = tilingData->bufferOpen != 0U ? ELU_GRAD_V2_MAX_BUFFER_NUM : 1U;
        computeChunk_ = tilingData->computeChunk;

        if (coreIdx < bigCoreNum_) {
            startOffset_ = coreIdx * bigCoreDataNum_;
            processLength_ = bigCoreDataNum_;
        } else {
            startOffset_ = bigCoreNum_ * bigCoreDataNum_ + (coreIdx - bigCoreNum_) * smallCoreDataNum_;
            processLength_ = smallCoreDataNum_;
        }
        if (coreIdx == coreNum_ - 1U) {
            processLength_ = lastCoreDataNum_;
        }
        tileCount_ = CeilDiv(processLength_, tileLength_);
        lastTileLength_ = GetTailLength(processLength_, tileLength_);
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads + startOffset_, processLength_);
        activationsGm_.SetGlobalBuffer((__gm__ T*)activations + startOffset_, processLength_);
        yGm_.SetGlobalBuffer((__gm__ T*)y + startOffset_, processLength_);

        factorPos_ = tilingData->scale;
        factorNeg_ = tilingData->alpha * tilingData->scale * tilingData->inputScale;
        factorNegBias_ = (tilingData->inputScale == 0.0F) ? 0.0F : (tilingData->alpha * tilingData->scale);
        inputScale_ = tilingData->inputScale;

        pipe_->InitBuffer(inQueueGrads_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(inQueueAct_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(outQueueY_, bufferNum_, tileLength_ * sizeof(T));
        pipe_->InitBuffer(maskBuf_, computeChunk_ * sizeof(uint8_t));
        pipe_->InitBuffer(gradsBufF32_, computeChunk_ * sizeof(float));
        pipe_->InitBuffer(actBufF32_, computeChunk_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (processLength_ == 0U) {
            return;
        }
        if (bufferNum_ == 1U || tileCount_ <= 1U) {
            for (uint32_t i = 0; i < tileCount_; ++i) {
                uint32_t currentLength = GetTileLength(i);
                CopyIn(GetTileOffset(i), currentLength);
                Compute(currentLength);
                CopyOut(GetTileOffset(i), currentLength);
            }
            return;
        }

        CopyIn(0U, GetTileLength(0U));
        for (uint32_t i = 1U; i <= tileCount_; ++i) {
            if (i < tileCount_) {
                CopyIn(GetTileOffset(i), GetTileLength(i));
            }
            Compute(GetTileLength(i - 1U));
            if (i > 1U) {
                CopyOut(GetTileOffset(i - 2U), GetTileLength(i - 2U));
            }
        }
        CopyOut(GetTileOffset(tileCount_ - 1U), GetTileLength(tileCount_ - 1U));
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        if (divisor == 0U) {
            return 0U;
        }
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t factor)
    {
        if (factor == 0U) {
            return value;
        }
        return ((value + factor - 1U) / factor) * factor;
    }

    __aicore__ inline uint32_t GetTailLength(uint32_t total, uint32_t tile)
    {
        if (total == 0U) {
            return 0U;
        }
        uint32_t tail = total % tile;
        return tail == 0U ? tile : tail;
    }

    __aicore__ inline uint32_t GetExecLength(uint32_t length)
    {
        uint32_t alignNum = ELU_GRAD_V2_BLOCK_BYTES / sizeof(T);
        if (alignNum == 0U) {
            return length;
        }
        return Min(AlignUp(length, alignNum), computeChunk_);
    }

    __aicore__ inline uint32_t GetTileOffset(uint32_t tileIdx) { return tileIdx * tileLength_; }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx)
    {
        return (tileIdx + 1U == tileCount_) ? lastTileLength_ : tileLength_;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> gradsLocal = inQueueGrads_.AllocTensor<T>();
        LocalTensor<T> actLocal = inQueueAct_.AllocTensor<T>();
        DataCopy(gradsLocal, gradsGm_[offset], length);
        DataCopy(actLocal, activationsGm_[offset], length);
        inQueueGrads_.EnQue(gradsLocal);
        inQueueAct_.EnQue(actLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> gradsRaw = inQueueGrads_.DeQue<T>();
        LocalTensor<T> actRaw = inQueueAct_.DeQue<T>();
        LocalTensor<T> yRaw = outQueueY_.AllocTensor<T>();

        if constexpr (kIsSameType<T, bfloat16_t>) {
            ComputeBf16(gradsRaw, actRaw, yRaw, length);
        } else {
            ComputeFloat16(gradsRaw, actRaw, yRaw, length);
        }

        outQueueY_.EnQue(yRaw);
        inQueueGrads_.FreeTensor(gradsRaw);
        inQueueAct_.FreeTensor(actRaw);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> yLocal = outQueueY_.DeQue<T>();
        DataCopy(yGm_[offset], yLocal, length);
        outQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeFloat16(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                          uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<T> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<T> actChunkRaw = actRaw[offset];
            LocalTensor<T> yChunkRaw = yRaw[offset];

            CompareScalar(mask, actChunkRaw, static_cast<T>(0.0f), CMPMODE::LE, execLength);
            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            PipeBarrier<PIPE_V>();
            if constexpr (IS_RESULT_MODE) {
                Adds(actF32, actF32, factorNegBias_, execLength);
                Muls(actF32, actF32, inputScale_, execLength);
            } else {
                Muls(actF32, actF32, inputScale_, execLength);
                Exp(actF32, actF32, execLength);
                Muls(actF32, actF32, factorNeg_, execLength);
            }
            Mul(actF32, actF32, gradsF32, execLength);
            Muls(gradsF32, gradsF32, factorPos_, execLength);
            PipeBarrier<PIPE_V>();
            Select(gradsF32, mask, actF32, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
            PipeBarrier<PIPE_V>();
            Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeBf16(LocalTensor<T> gradsRaw, LocalTensor<T> actRaw, LocalTensor<T> yRaw,
                                       uint32_t length)
    {
        LocalTensor<float> gradsF32 = gradsBufF32_.Get<float>();
        LocalTensor<float> actF32 = actBufF32_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        for (uint32_t offset = 0; offset < length; offset += computeChunk_) {
            uint32_t currentLength = Min(length - offset, computeChunk_);
            uint32_t execLength = GetExecLength(currentLength);

            LocalTensor<T> gradsChunkRaw = gradsRaw[offset];
            LocalTensor<T> actChunkRaw = actRaw[offset];
            LocalTensor<T> yChunkRaw = yRaw[offset];

            Cast(gradsF32, gradsChunkRaw, RoundMode::CAST_NONE, execLength);
            Cast(actF32, actChunkRaw, RoundMode::CAST_NONE, execLength);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask, actF32, 0.0F, CMPMODE::LE, execLength);
            if constexpr (IS_RESULT_MODE) {
                Adds(actF32, actF32, factorNegBias_, execLength);
                Muls(actF32, actF32, inputScale_, execLength);
            } else {
                Muls(actF32, actF32, inputScale_, execLength);
                Exp(actF32, actF32, execLength);
                Muls(actF32, actF32, factorNeg_, execLength);
            }
            Mul(actF32, actF32, gradsF32, execLength);
            Muls(gradsF32, gradsF32, factorPos_, execLength);
            PipeBarrier<PIPE_V>();
            Select(gradsF32, mask, actF32, gradsF32, SELMODE::VSEL_TENSOR_TENSOR_MODE, execLength);
            PipeBarrier<PIPE_V>();
            Cast(yChunkRaw, gradsF32, RoundMode::CAST_RINT, execLength);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueGrads_;
    TQue<QuePosition::VECIN, MAX_BUFFER_NUM> inQueueAct_;
    TQue<QuePosition::VECOUT, MAX_BUFFER_NUM> outQueueY_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> gradsBufF32_;
    TBuf<QuePosition::VECCALC> actBufF32_;
    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> activationsGm_;
    GlobalTensor<T> yGm_;

    uint32_t coreNum_ = 0U;
    uint32_t bigCoreNum_ = 0U;
    uint32_t bigCoreDataNum_ = 0U;
    uint32_t smallCoreDataNum_ = 0U;
    uint32_t lastCoreDataNum_ = 0U;
    uint32_t startOffset_ = 0U;
    uint32_t processLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t lastTileLength_ = 0U;
    uint32_t bufferNum_ = 1U;
    uint32_t computeChunk_ = ELU_GRAD_V2_MIXED_RESULT_COMPUTE_CHUNK;
    float factorPos_ = 1.0F;
    float factorNeg_ = 1.0F;
    float factorNegBias_ = 1.0F;
    float inputScale_ = 1.0F;
};

} // namespace NsEluGradV2

#endif // ELU_GRAD_V2_H
