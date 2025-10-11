/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_b16_b1sd.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_
#include "../platform.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

template <bool isPagedAttention, typename KV_DTYPE>
class KernelKvRmsNormRopeCacheB16B1SD
{
public:
    __aicore__ inline KernelKvRmsNormRopeCacheB16B1SD(TPipe* pipe, const KvRmsNormRopeCacheTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache)
    {
        int64_t currentBlockFactor = tilingData_->blockFactor;
        if (GetBlockIdx() == (tilingData_->blockDim - 1)) {
            currentBlockFactor = tilingData_->batchSize * tilingData_->seqLength -
                                 (tilingData_->blockDim - 1) * tilingData_->blockFactor;
        }

        ubFactor = tilingData_->ubFactor;
        ubLoop = currentBlockFactor / ubFactor;
        ubTail = currentBlockFactor - ubLoop * ubFactor;

        // init global memory
        kvGm.SetGlobalBuffer(
            (__gm__ KV_DTYPE*)kv + GetBlockIdx() * tilingData_->blockFactor * (RMS_NORM_LENGTH + ROPE_LENGTH));
        gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE*)gamma);
        cosGm.SetGlobalBuffer((__gm__ KV_DTYPE*)cos + GetBlockIdx() * tilingData_->blockFactor * ROPE_LENGTH);
        sinGm.SetGlobalBuffer((__gm__ KV_DTYPE*)sin + GetBlockIdx() * tilingData_->blockFactor * ROPE_LENGTH);
        indexGm.SetGlobalBuffer((__gm__ int64_t*)index + GetBlockIdx() * tilingData_->blockFactor);
        kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache);
        vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v_cache);

        // init pipe
        pipe_->InitBuffer(inQueueGamma, 2, RMS_NORM_LENGTH * sizeof(float));           // 2*512*4/1024=4
        pipe_->InitBuffer(inQueueX, 2, ubFactor * RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        pipe_->InitBuffer(outQueue, 2, ubFactor * RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        pipe_->InitBuffer(wsBuffer, 3 * ubFactor * RMS_NORM_LENGTH * sizeof(float));   // 3*16*512*4/1024=96
        pipe_->InitBuffer(inQueueIndex, 1, ubFactor * sizeof(int64_t));                // 16*8/1024=0.125
    }

    __aicore__ inline void Process()
    {
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma: [RMS_NORM_LENGTH]
        LocalTensor<float> gammaLocalFp32 = inQueueGamma.AllocTensor<float>();
        LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[RMS_NORM_LENGTH];
        copyParamsContinguous.blockLen = RMS_NORM_LENGTH * sizeof(KV_DTYPE);
        DataCopyPad(gammaLocal, gammaGm, copyParamsContinguous, padParams);
        inQueueGamma.EnQue(gammaLocalFp32);
        gammaLocalFp32 = inQueueGamma.DeQue<float>();
        gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[RMS_NORM_LENGTH];
        Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, RMS_NORM_LENGTH);

        LocalTensor<float> workspaceBuffer = wsBuffer.Get<float>();
        for (int64_t loopIdx = 0; loopIdx < ubLoop; ++loopIdx) {
            int64_t kvGlobalMemoryOffset = loopIdx * ubFactor * (RMS_NORM_LENGTH + ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = loopIdx * ubFactor * ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = loopIdx * ubFactor;
            int64_t startIdx = GetBlockIdx() * tilingData_->blockFactor + loopIdx * ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = inQueueX.AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[ubFactor * ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[ubFactor * ROPE_LENGTH];
            // CopyIn x/cos/sin [ubFactor, ROPE_LENGTH]
            DataCopyExtParams copyParams{/* blockCount */ static_cast<uint16_t>(ubFactor),
                                         /* blockLen (Byte) */ ROPE_LENGTH * sizeof(KV_DTYPE),
                                         /* srcStride */ RMS_NORM_LENGTH * sizeof(KV_DTYPE),
                                         /* dstStride */ 0,
                                         /* rsv */ 0};
            DataCopyPad(ropeLocal, kvGm[kvGlobalMemoryOffset + RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = ubFactor * ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            inQueueX.EnQue(ropeLocal);
            ropeLocal = inQueueX.DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[ubFactor * ROPE_LENGTH];
            sinLocal = cosLocal[ubFactor * ROPE_LENGTH];

            // CopyIn index [ubFactor,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = ubFactor * sizeof(int64_t);
            DataCopyPad(indexLocal, indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.DeQue<int64_t>();

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = outQueue.AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true, ROPE_LENGTH>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, ubFactor);
            inQueueX.FreeTensor(ropeLocal);
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdate<KV_DTYPE, isPagedAttention, ROPE_LENGTH>(kCacheGm, outLocal, indexLocal, startIdx, ubFactor);
            outQueue.FreeTensor(outLocal);

            // CopyIn x: [ubFactor, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = inQueueX.AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(ubFactor);
            copyParams.blockLen = RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = outQueue.AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE, RMS_NORM_LENGTH>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, ubFactor);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdate<KV_DTYPE, isPagedAttention, RMS_NORM_LENGTH>(
                vCacheGm, outLocal, indexLocal, startIdx, ubFactor);
            outQueue.FreeTensor(outLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        if (ubTail > 0) {
            int64_t kvGlobalMemoryOffset = ubLoop * ubFactor * (RMS_NORM_LENGTH + ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = ubLoop * ubFactor * ROPE_LENGTH;
            int64_t indexGlobalMemoryOffset = ubLoop * ubFactor;
            int64_t startIdx = GetBlockIdx() * tilingData_->blockFactor + ubLoop * ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = inQueueX.AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[ubTail * ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[ubTail * ROPE_LENGTH];
            // CopyIn x/cos/sin [ubTail, ROPE_LENGTH]
            DataCopyExtParams copyParams{/* blockCount */ static_cast<uint16_t>(ubTail),
                                         /* blockLen (Byte) */ ROPE_LENGTH * sizeof(KV_DTYPE),
                                         /* srcStride */ RMS_NORM_LENGTH * sizeof(KV_DTYPE),
                                         /* dstStride */ 0,
                                         /* rsv */ 0};
            DataCopyPad(ropeLocal, kvGm[kvGlobalMemoryOffset + RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = ubTail * ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            inQueueX.EnQue(ropeLocal);
            ropeLocal = inQueueX.DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[ubTail * ROPE_LENGTH];
            sinLocal = cosLocal[ubTail * ROPE_LENGTH];

            // CopyIn index [ubTail,]
            LocalTensor<int64_t> indexLocal = inQueueIndex.AllocTensor<int64_t>();
            copyParamsContinguous.blockLen = ubTail * sizeof(int64_t);
            DataCopyPad(indexLocal, indexGm[indexGlobalMemoryOffset], copyParamsContinguous, padParamsInt64);
            inQueueIndex.EnQue(indexLocal);
            indexLocal = inQueueIndex.DeQue<int64_t>();

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = outQueue.AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true, ROPE_LENGTH>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, ubTail);
            inQueueX.FreeTensor(ropeLocal);
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<KV_DTYPE>();

            // Scatter Update kCache
            ScatterUpdate<KV_DTYPE, isPagedAttention, ROPE_LENGTH>(kCacheGm, outLocal, indexLocal, startIdx, ubTail);
            outQueue.FreeTensor(outLocal);

            // CopyIn x: [ubTail, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = inQueueX.AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(ubTail);
            copyParams.blockLen = RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = outQueue.AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE, RMS_NORM_LENGTH>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, ubTail);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<KV_DTYPE>();

            // Scatter Update vCache
            ScatterUpdate<KV_DTYPE, isPagedAttention, RMS_NORM_LENGTH>(
                vCacheGm, outLocal, indexLocal, startIdx, ubTail);
            outQueue.FreeTensor(outLocal);
            inQueueIndex.FreeTensor(indexLocal);
        }
        inQueueGamma.FreeTensor(gammaLocalFp32);
    }

    template <typename T, bool isElementWise = true, int64_t headSize>
    __aicore__ inline void RoPE(
        const LocalTensor<T>& outLocal, const LocalTensor<T>& xLocal, const LocalTensor<T>& cosLocal,
        const LocalTensor<T>& sinLocal, const LocalTensor<float>& wsLocal, int64_t rows)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_THREE = 3;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        if constexpr (isElementWise) {
            /**
             * cosLocalFp32 : [rows * headSize * 0, rows * headSize * 1]
             * sinLocalFp32 : [rows * headSize * 1, rows * headSize * 2]
             * y0           : [rows * headSize * 2, rows * headSize * 3]
             * y1           : [rows * headSize * 3, rows * headSize * 4]
             * realLocalFp32: [rows * headSize * 4, rows * headSize * 5]
             * imagLocalFp32: [rows * headSize * 5, rows * headSize * 6]
             * realLocal    : [rows * headSize * 6, rows * headSize * 7]
             * imagLocal    : [rows * headSize * 7, rows * headSize * 8]
             */
            int64_t cosLocalFp32Offset = rows * headSize * 0;
            int64_t sinLocalFp32Offset = rows * headSize * 1;
            int64_t y0Offset = rows * headSize * 2;
            int64_t y1Offset = rows * headSize * 3;
            int64_t realLocalFp32Offset = rows * headSize * 4;
            int64_t imagLocalFp32Offset = rows * headSize * 5;
            int64_t realLocalOffset = rows * headSize * 6;
            int64_t imagLocalOffset = rows * headSize * 7;

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];
            LocalTensor<T> realLocal = wsLocal[realLocalOffset].template ReinterpretCast<T>();
            LocalTensor<T> imagLocal = wsLocal[imagLocalOffset].template ReinterpretCast<T>();

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #2: Gather out real and imag
            uint64_t rsvdCnt = 0;
            GatherMask(realLocal, xLocal, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            GatherMask(imagLocal, xLocal, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();

            // step #3: Cast realLocal and imagLocal to Fp32
            Cast(realLocalFp32, realLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            Cast(imagLocalFp32, imagLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();

            // step #4: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
            Mul(y0, realLocalFp32, cosLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(headSize >> 1)], imagLocalFp32, cosLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #5: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(headSize >> 1)], realLocalFp32, sinLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #6: y0 = y0 + y1
            Add(y0, y0, y1, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #7: outLocal = Cast(y0, T)
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

    template <typename T, bool isPA = false, int64_t headSize>
    __aicore__ inline void ScatterUpdate(
        const GlobalTensor<T>& dst, const LocalTensor<T>& outLocal, const LocalTensor<int64_t>& indexLocal,
        int64_t startIdx, int64_t rows)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, headSize * sizeof(T), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t offset = indexLocal.GetValue(i);
            SToMTE3Sync();
            if (offset >= 0) {
                if constexpr (isPA) {
                    int64_t gmOffset = headSize * offset;
                    int64_t ubOffset = headSize * i;
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                } else {
                    int64_t batchId = (startIdx + i) / tilingData_->seqLength;
                    int64_t gmOffset = batchId * tilingData_->cacheLength * headSize + offset * headSize;
                    int64_t ubOffset = i * headSize;
                    DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, int64_t headSize>
    __aicore__ inline void RmsNorm(
        const LocalTensor<T>& outLocal, const LocalTensor<T>& xLocal, const LocalTensor<float>& gammaLocal,
        const LocalTensor<float>& wsLocal, int64_t rows)
    {
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_SIXTEEN = 16;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
        int64_t xLocalFp32Offset = 0;                  // [0, rows * headSize]
        int64_t xSquareLocalOffset = rows * headSize;  // [rows * headSize, rows * headSize * 2]
        int64_t xSumLocalOffset = rows * headSize * 2; // [rows * headSize * 2, rows * headSize * 3]
        LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
        LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
        LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, rows * headSize);
        PipeBarrier<PIPE_V>();
        Mul(xSquareLocal, xLocalFp32, xLocalFp32, rows * headSize);
        PipeBarrier<PIPE_V>();
        int64_t repeatTimes = rows * (headSize >> 1) / NUM_SIXTY_FOUR;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 256
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 128
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 64
        PipeBarrier<PIPE_V>();

        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, rows, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, tilingData_->reciprocal, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, tilingData_->epsilon, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, rows);
        PipeBarrier<PIPE_V>();

        // Brcb
        int64_t block_count = (rows + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 / xSquareLocal
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Div(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], xSquareLocal[rowId * NUM_EIGHT],
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 * xSquareLocal [rows, RMS_NORM_LENGTH] * [RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], gammaLocal, headSize);
        }
        PipeBarrier<PIPE_V>();

        // Cast xLocalFp32 to outLocal
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rows * headSize);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_NONE, rows * headSize);
        }
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

private:
    constexpr static int64_t RMS_NORM_LENGTH = 512;
    constexpr static int64_t ROPE_LENGTH = 64;
    TPipe* pipe_ = nullptr;
    const KvRmsNormRopeCacheTilingData* tilingData_;
    GlobalTensor<KV_DTYPE> kvGm, gammaGm, cosGm, sinGm, kCacheGm, vCacheGm;
    GlobalTensor<int64_t> indexGm;

    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueIndex;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer;

    int64_t ubLoop = 1;
    int64_t ubFactor = 8;
    int64_t ubTail = 0;
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_B1SD_H_