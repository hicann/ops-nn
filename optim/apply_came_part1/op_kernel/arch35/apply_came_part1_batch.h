/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NN_APPLY_CAME_PART1_BATCH_H
#define OPS_NN_APPLY_CAME_PART1_BATCH_H

#include "apply_came_part1_common.h"

namespace ApplyCamePart1 {
// Process independent matrices on disjoint batch ranges. The host selects this
// path from UB capacity and available batch parallelism, so no matrix shape is
// hard-coded. Small matrices are grouped into one contiguous transfer; larger
// single-tile matrices reuse the normal compensated reduction in one core.
template <typename T>
class ApplyCamePart1Batch {
public:
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR eps, GM_ADDR sumGradR, GM_ADDR sumGradC, GM_ADDR sumGradRC,
                                int64_t n, int64_t m)
    {
        n_ = n;
        m_ = m;
        matrixElements_ = n * m;
        gradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(grad));
        epsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eps), 1);
        sumGradRGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradR));
        sumGradCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradC));
        sumGradRCGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumGradRC));

        pipe_.InitBuffer(gradQueue_, 1, kMaxInputElements * sizeof(T));
        pipe_.InitBuffer(epsQueue_, 1, kBlockFloats * sizeof(float));
        pipe_.InitBuffer(sumGradRQueue_, 1, kVectorTileWidth * sizeof(float));
        pipe_.InitBuffer(sumGradCQueue_, 1, kVectorTileWidth * sizeof(float));
        pipe_.InitBuffer(castBuf_, kMaxInputElements * sizeof(float));
        pipe_.InitBuffer(valueBuf_, kMaxInputElements * sizeof(float));
        pipe_.InitBuffer(outputBuf_, kMaxOutputElements * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, kReduceScratchElements * sizeof(float));
    }

    __aicore__ inline void Process(int64_t batchCount, int64_t blockCount)
    {
        if (batchCount <= 0 || blockCount <= 0 || n_ <= 0 || m_ <= 0 || matrixElements_ <= 0) {
            return;
        }
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t batchesPerBlock = CeilDivApplyCamePart1(batchCount, blockCount);
        const int64_t alignedBatchesPerBlock = CeilDivApplyCamePart1(batchesPerBlock, kBatchAlignment) *
                                               kBatchAlignment;
        const int64_t batchBegin = blockIdx * alignedBatchesPerBlock;
        const int64_t batchEnd = batchBegin + alignedBatchesPerBlock < batchCount ?
                                     batchBegin + alignedBatchesPerBlock :
                                     batchCount;
        if (batchBegin >= batchEnd) {
            return;
        }

        LocalTensor<float> epsLocal = epsQueue_.AllocTensor<float>();
        DataCopyPad(epsLocal, epsGm_, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
        epsQueue_.EnQue(epsLocal);
        epsLocal = epsQueue_.DeQue<float>();
        event_t eventMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);
        const float epsValue = epsLocal.GetValue(0);
        epsQueue_.FreeTensor(epsLocal);

        if (UseVectorSingleMatrix()) {
            for (int64_t batchIdx = batchBegin; batchIdx < batchEnd; ++batchIdx) {
                ProcessVectorMatrix(batchIdx, epsValue);
            }
            return;
        }

        const int64_t tileBatchCount = GetScalarTileBatchCount();
        for (int64_t batchOffset = batchBegin; batchOffset < batchEnd; batchOffset += tileBatchCount) {
            const int64_t tileBatches = batchEnd - batchOffset < tileBatchCount ? batchEnd - batchOffset :
                                                                                  tileBatchCount;
            ProcessScalarTile(batchOffset, tileBatches, epsValue);
        }
    }

private:
    __aicore__ inline bool UseVectorSingleMatrix() const
    {
        return n_ <= kVectorTileWidth && m_ <= kVectorTileWidth && matrixElements_ > kScalarMatrixElementLimit;
    }

    __aicore__ inline int64_t GetScalarTileBatchCount() const
    {
        int64_t byInput = kMaxInputElements / matrixElements_;
        const int64_t outputElementsPerBatch = n_ + m_ + 1;
        int64_t byOutput = kMaxOutputElements / outputElementsPerBatch;
        int64_t tileBatches = byInput < byOutput ? byInput : byOutput;
        tileBatches = tileBatches < kMaxTileBatches ? tileBatches : kMaxTileBatches;
        return tileBatches > 0 ? tileBatches : 1;
    }

    __aicore__ inline float KahanSum(LocalTensor<float> values, int64_t offset, int64_t count, int64_t stride)
    {
        float sum = 0.0f;
        float correction = 0.0f;
        for (int64_t index = 0; index < count; ++index) {
            const float value = values.GetValue(offset + index * stride);
            CompensatedAddApplyCamePart1(sum, correction, value, 0.0f, sum, correction);
        }
        return sum + correction;
    }

    __aicore__ inline void ProcessScalarTile(int64_t batchOffset, int64_t tileBatches, float epsValue)
    {
        const int64_t inputCount = tileBatches * matrixElements_;
        LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(inputCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(gradLocal, gradGm_[batchOffset * matrixElements_], copyParams, padParams);
        gradQueue_.EnQue(gradLocal);
        gradLocal = gradQueue_.DeQue<T>();

        LocalTensor<float> castLocal = castBuf_.Get<float>();
        LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        LocalTensor<float> outputLocal = outputBuf_.Get<float>();
        event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        if constexpr (sizeof(T) == sizeof(float)) {
            Muls(castLocal, gradLocal, static_cast<float>(1.0), inputCount);
        } else {
            Cast(castLocal, gradLocal, RoundMode::CAST_NONE, inputCount);
        }
        PipeBarrier<PIPE_V>();
        Mul(valueLocal, castLocal, castLocal, inputCount);
        PipeBarrier<PIPE_V>();
        Adds(valueLocal, valueLocal, epsValue, inputCount);

        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        // Every DataCopyPad source must begin at a 32B boundary.  The
        // packed layout is valid only when tileBatches*n and
        // tileBatches*(n+m) happen to be multiples of eight floats, which
        // is not true for small tail batches.  Keep each output segment
        // independently aligned so all matrix shapes use the same contract.
        const int64_t outROffset = 0;
        const int64_t outCOffset = AlignOutputElements(tileBatches * n_);
        const int64_t outRCOffset = AlignOutputElements(outCOffset + tileBatches * m_);
        LocalTensor<float> outR = outputLocal[outROffset];
        LocalTensor<float> outC = outputLocal[outCOffset];
        LocalTensor<float> outRC = outputLocal[outRCOffset];
        for (int64_t batch = 0; batch < tileBatches; ++batch) {
            const int64_t matrixBase = batch * matrixElements_;
            for (int64_t row = 0; row < n_; ++row) {
                outR.SetValue(batch * n_ + row, KahanSum(valueLocal, matrixBase + row * m_, m_, 1));
            }
            for (int64_t column = 0; column < m_; ++column) {
                outC.SetValue(batch * m_ + column, KahanSum(valueLocal, matrixBase + column, n_, m_));
            }
            outRC.SetValue(batch, KahanSum(outC, batch * m_, m_, 1));
        }
        CopyScalarOutputs(outputLocal, batchOffset, tileBatches);
        gradQueue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void CopyScalarOutputs(LocalTensor<float> outputLocal, int64_t batchOffset, int64_t tileBatches)
    {
        const int64_t outROffset = 0;
        const int64_t outCOffset = AlignOutputElements(tileBatches * n_);
        const int64_t outRCOffset = AlignOutputElements(outCOffset + tileBatches * m_);
        LocalTensor<float> outR = outputLocal[outROffset];
        LocalTensor<float> outC = outputLocal[outCOffset];
        LocalTensor<float> outRC = outputLocal[outRCOffset];
        event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
        DataCopyExtParams rCopyParams{1, static_cast<uint32_t>(tileBatches * n_ * sizeof(float)), 0, 0, 0};
        DataCopyExtParams cCopyParams{1, static_cast<uint32_t>(tileBatches * m_ * sizeof(float)), 0, 0, 0};
        DataCopyExtParams rcCopyParams{1, static_cast<uint32_t>(tileBatches * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(sumGradRGm_[batchOffset * n_], outR, rCopyParams);
        DataCopyPad<float, PaddingMode::Compact>(sumGradCGm_[batchOffset * m_], outC, cCopyParams);
        DataCopyPad<float, PaddingMode::Compact>(sumGradRCGm_[batchOffset], outRC, rcCopyParams);
        WaitOutputBeforeNextInput();
    }

    __aicore__ inline void CopyInVectorMatrix(int64_t batchIdx, LocalTensor<T> gradLocal)
    {
        const int64_t paddedElements = n_ * kVectorTileWidth;
        Duplicate(gradLocal, static_cast<T>(0), paddedElements);
        event_t eventVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(m_ * sizeof(T)), 0, 0, 0};
        constexpr int64_t blockElements = 32 / sizeof(T);
        const uint8_t rightPadding = static_cast<uint8_t>((blockElements - m_ % blockElements) % blockElements);
        // DataCopyPad rounds each GM row transfer to a 32-byte block.  Make
        // that rounded suffix explicit zero padding; otherwise an unaligned
        // row width can pull elements from the next GM row into the 64-wide
        // reduction slot.  The initial Duplicate still clears the remainder
        // between this DMA-aligned suffix and kVectorTileWidth.
        DataCopyPadExtParams<T> padParams{true, 0, rightPadding, static_cast<T>(0)};
        const int64_t inputBase = batchIdx * matrixElements_;
        for (int64_t row = 0; row < n_; ++row) {
            DataCopyPad(gradLocal[row * kVectorTileWidth], gradGm_[inputBase + row * m_], copyParams, padParams);
        }
    }

    __aicore__ inline void ComputeVectorMatrix(LocalTensor<T> gradLocal, float epsValue,
                                               LocalTensor<float>& sumGradRLocal, LocalTensor<float>& sumGradCLocal,
                                               LocalTensor<float>& sumGradRCLocal)
    {
        const int64_t paddedElements = n_ * kVectorTileWidth;
        LocalTensor<float> castLocal = castBuf_.Get<float>();
        LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        LocalTensor<float> reductionScratch = reduceBuf_.Get<float>();
        event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
        if constexpr (sizeof(T) == sizeof(float)) {
            Muls(castLocal, gradLocal, static_cast<float>(1.0), paddedElements);
        } else {
            Cast(castLocal, gradLocal, RoundMode::CAST_NONE, paddedElements);
        }
        PipeBarrier<PIPE_V>();
        Mul(valueLocal, castLocal, castLocal, paddedElements);
        PipeBarrier<PIPE_V>();
        AddEpsApplyCamePart1(valueLocal, epsValue, m_, n_);
        PipeBarrier<PIPE_V>();
        Duplicate(castLocal, static_cast<float>(0), 2 * kVectorTileWidth);
        PipeBarrier<PIPE_V>();

        ComputeRApplyCamePart1(sumGradRQueue_, n_, 0, 1, valueLocal, castLocal, reductionScratch,
                               reductionScratch[4 * kVectorTileWidth], kVectorTileWidth);
        ComputeCApplyCamePart1(sumGradCQueue_, n_, valueLocal, 1, 1, kVectorTileWidth, m_, 0);
        sumGradRLocal = sumGradRQueue_.DeQue<float>();
        sumGradCLocal = sumGradCQueue_.DeQue<float>();
        sumGradRCLocal = outputBuf_.Get<float>();
        FinishApplyCamePart1Reduction(sumGradRCLocal, sumGradCLocal, reductionScratch, kVectorTileWidth);
    }

    __aicore__ inline void CopyOutVectorMatrix(int64_t batchIdx, LocalTensor<T> gradLocal,
                                               LocalTensor<float> sumGradRLocal, LocalTensor<float> sumGradCLocal,
                                               LocalTensor<float> sumGradRCLocal)
    {
        event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
        DataCopyExtParams rCopyParams{1, static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        DataCopyExtParams cCopyParams{1, static_cast<uint32_t>(m_ * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(sumGradRGm_[batchIdx * n_], sumGradRLocal, rCopyParams);
        DataCopyPad<float, PaddingMode::Compact>(sumGradCGm_[batchIdx * m_], sumGradCLocal, cCopyParams);
        event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
        DataCopyExtParams rcCopyParams{1, sizeof(float), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(sumGradRCGm_[batchIdx], sumGradRCLocal, rcCopyParams);
        WaitOutputBeforeNextInput();

        sumGradRQueue_.FreeTensor(sumGradRLocal);
        sumGradCQueue_.FreeTensor(sumGradCLocal);
        gradQueue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void ProcessVectorMatrix(int64_t batchIdx, float epsValue)
    {
        LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
        CopyInVectorMatrix(batchIdx, gradLocal);
        gradQueue_.EnQue(gradLocal);
        gradLocal = gradQueue_.DeQue<T>();
        LocalTensor<float> sumGradRLocal;
        LocalTensor<float> sumGradCLocal;
        LocalTensor<float> sumGradRCLocal;
        ComputeVectorMatrix(gradLocal, epsValue, sumGradRLocal, sumGradCLocal, sumGradRCLocal);
        CopyOutVectorMatrix(batchIdx, gradLocal, sumGradRLocal, sumGradCLocal, sumGradRCLocal);
    }

    __aicore__ inline int64_t AlignOutputElements(int64_t elementCount) const
    {
        return (elementCount + kOutputAlignmentElements - 1) / kOutputAlignmentElements * kOutputAlignmentElements;
    }

    __aicore__ inline void WaitOutputBeforeNextInput()
    {
        event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    }

private:
    static constexpr int64_t kVectorTileWidth = 64;
    static constexpr int64_t kMaxInputElements = 8192;
    static constexpr int64_t kScalarMatrixElementLimit = 256;
    static constexpr int64_t kMaxTileBatches = 32;
    static constexpr int64_t kBatchAlignment = 8;
    static constexpr int64_t kBlockFloats = 8;
    // Output segments are independently aligned to 32B (8 float elements).
    // The extra three alignment slots cover the two segment boundaries and
    // the final tail without changing the per-tile element limit.
    static constexpr int64_t kOutputAlignmentElements = 8;
    static constexpr int64_t kMaxOutputElements = kMaxInputElements + 2 * kMaxTileBatches +
                                                  3 * kOutputAlignmentElements;
    static constexpr int64_t kReduceScratchElements = 12 * kVectorTileWidth;

    int64_t n_{0};
    int64_t m_{0};
    int64_t matrixElements_{0};
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> epsQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradRQueue_;
    TQue<QuePosition::VECOUT, 1> sumGradCQueue_;
    TBuf<QuePosition::VECCALC> castBuf_;
    TBuf<QuePosition::VECCALC> valueBuf_;
    TBuf<QuePosition::VECCALC> outputBuf_;
    TBuf<QuePosition::VECCALC> reduceBuf_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<float> epsGm_;
    GlobalTensor<float> sumGradRGm_;
    GlobalTensor<float> sumGradCGm_;
    GlobalTensor<float> sumGradRCGm_;
};

} // namespace ApplyCamePart1
#endif // OPS_NN_APPLY_CAME_PART1_BATCH_H
