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
 * \file kv_rms_norm_rope_cache_b16_mtp.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_
#include "arch35/platform.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

template <bool isPagedAttention, typename KV_DTYPE>
class KernelKvRmsNormRopeCacheB16MTP
{
public:
    __aicore__ inline KernelKvRmsNormRopeCacheB16MTP(TPipe* pipe, const KvRmsNormRopeCacheTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache)
    {
        // init global memory
        kvGm.SetGlobalBuffer(
            (__gm__ KV_DTYPE*)kv +
            GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->seqLength * (RMS_NORM_LENGTH + ROPE_LENGTH));
        gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE*)gamma);
        cosGm.SetGlobalBuffer((__gm__ KV_DTYPE*)cos + GetBlockIdx() * tilingData_->rowsPerBlock * ROPE_LENGTH);
        sinGm.SetGlobalBuffer((__gm__ KV_DTYPE*)sin + GetBlockIdx() * tilingData_->rowsPerBlock * ROPE_LENGTH);
        if constexpr (isPagedAttention) {
            indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
            kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache);
            vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v_cache);
        } else {
            indexGm.SetGlobalBuffer(
                (__gm__ int64_t*)index + GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->seqLength);
            kCacheGm.SetGlobalBuffer(
                (__gm__ KV_DTYPE*)k_cache +
                GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->cacheLength * ROPE_LENGTH);
            vCacheGm.SetGlobalBuffer(
                (__gm__ KV_DTYPE*)v_cache +
                GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->cacheLength * RMS_NORM_LENGTH);
        }

        // init pipe
        pipe_->InitBuffer(inQueueX, 1, tilingData_->rowsPerBlock * RMS_NORM_LENGTH * sizeof(float));
        pipe_->InitBuffer(inQueueGamma, 1, RMS_NORM_LENGTH * sizeof(KV_DTYPE));
        pipe_->InitBuffer(bufferXFp32, tilingData_->rowsPerBlock * RMS_NORM_LENGTH * sizeof(float));
        pipe_->InitBuffer(bufferXSquare, tilingData_->rowsPerBlock * RMS_NORM_LENGTH * sizeof(float));
        pipe_->InitBuffer(
            bufferSum, ((tilingData_->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT) * NUM_EIGHT * sizeof(float));
        pipe_->InitBuffer(outQueueV, 1, tilingData_->rowsPerBlock * RMS_NORM_LENGTH * sizeof(KV_DTYPE));
        pipe_->InitBuffer(outQueueK, 1, tilingData_->rowsPerBlock * ROPE_LENGTH * sizeof(KV_DTYPE));
    }

    __aicore__ inline void Process()
    {
        for (int64_t i = 0; i < tilingData_->seqLength; i++) {
            RopeK(i);
            PipeBarrier<PIPE_V>();
            ScatterUpdateK(i);
            PipeBarrier<PIPE_V>();
            RmsNorm(i);
            PipeBarrier<PIPE_V>();
            ScatterUpdateV(i);
        }
    }

    __aicore__ inline void RopeK(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> kLocal = inQueueX.AllocTensor<KV_DTYPE>();
        LocalTensor<KV_DTYPE> cosLocal = kLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];
        LocalTensor<KV_DTYPE> sinLocal = cosLocal[tilingData_->rowsPerBlock * ROPE_LENGTH * NUM_TWO];
        LocalTensor<KV_DTYPE> cosLocalB16 = cosLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];
        LocalTensor<KV_DTYPE> sinLocalB16 = sinLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];

        // load kLocal [tilingData_->rowsPerBlock, ROPE_LENGTH]
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(tilingData_->rowsPerBlock), ROPE_LENGTH * sizeof(KV_DTYPE),
            static_cast<uint32_t>(
                RMS_NORM_LENGTH * sizeof(KV_DTYPE) +
                (tilingData_->seqLength - 1) * (RMS_NORM_LENGTH + ROPE_LENGTH) * sizeof(KV_DTYPE)),
            0, 0};
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPad(kLocal, kvGm[seqId * (RMS_NORM_LENGTH + ROPE_LENGTH) + RMS_NORM_LENGTH], copyParams, padParams);

        // load cosLocal, sinLocal [tilingData_->rowsPerBlock, ROPE_LENGTH]
        DataCopy(cosLocalB16, cosGm, tilingData_->rowsPerBlock * ROPE_LENGTH);
        DataCopy(sinLocalB16, sinGm, tilingData_->rowsPerBlock * ROPE_LENGTH);
        inQueueX.EnQue(kLocal);
        kLocal = inQueueX.DeQue<KV_DTYPE>();
        cosLocal = kLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];
        sinLocal = cosLocal[tilingData_->rowsPerBlock * ROPE_LENGTH * NUM_TWO];
        cosLocalB16 = cosLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];
        sinLocalB16 = sinLocal[tilingData_->rowsPerBlock * ROPE_LENGTH];

        // realLocal, imagLocal comes from odd/even elements in kLocal
        LocalTensor<KV_DTYPE> k = bufferXFp32.Get<KV_DTYPE>();
        LocalTensor<KV_DTYPE> realLocal = k[0];
        LocalTensor<KV_DTYPE> imagLocal = realLocal[tilingData_->rowsPerBlock * ROPE_LENGTH_HALF * NUM_TWO];
        LocalTensor<KV_DTYPE> buf_ = imagLocal[tilingData_->rowsPerBlock * ROPE_LENGTH_HALF * NUM_TWO];
        LocalTensor<KV_DTYPE> realLocalB16 = realLocal[tilingData_->rowsPerBlock * ROPE_LENGTH_HALF];
        LocalTensor<KV_DTYPE> imagLocalB16 = imagLocal[tilingData_->rowsPerBlock * ROPE_LENGTH_HALF];

        uint64_t rsvdCnt = 0;
        GatherMask(
            realLocalB16, kLocal, 1, true, tilingData_->rowsPerBlock * ROPE_LENGTH, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
        GatherMask(
            imagLocalB16, kLocal, NUM_TWO, true, tilingData_->rowsPerBlock * ROPE_LENGTH, {1, 1, NUM_EIGHT, 0},
            rsvdCnt);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> realLocalFp32 = realLocal.template ReinterpretCast<float>();
        LocalTensor<float> imagLocalFp32 = imagLocal.template ReinterpretCast<float>();
        LocalTensor<float> cosLocalFp32 = cosLocal.template ReinterpretCast<float>();
        LocalTensor<float> sinLocalFp32 = sinLocal.template ReinterpretCast<float>();

        Cast(realLocalFp32, realLocalB16, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * ROPE_LENGTH_HALF);
        Cast(imagLocalFp32, imagLocalB16, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * ROPE_LENGTH_HALF);
        Cast(cosLocalFp32, cosLocalB16, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * ROPE_LENGTH);
        Cast(sinLocalFp32, sinLocalB16, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        uint64_t mask[NUM_TWO] = {MASK_CONFIG, 0}; // mask ROPE_LENGTH_HALF Elements
        LocalTensor<KV_DTYPE> outLocal = outQueueK.AllocTensor<KV_DTYPE>();
        LocalTensor<float> outLocalFp32 = outLocal.template ReinterpretCast<float>();
        Mul(outLocalFp32, realLocalFp32, cosLocalFp32, mask, tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(outLocalFp32[ROPE_LENGTH_HALF], imagLocalFp32, cosLocalFp32[ROPE_LENGTH_HALF], mask,
            tilingData_->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, tilingData_->rowsPerBlock * ROPE_LENGTH_HALF);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> bufFp32 = buf_.template ReinterpretCast<float>();

        Mul(bufFp32, imagLocalFp32, sinLocalFp32, mask, tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(bufFp32[ROPE_LENGTH_HALF], realLocalFp32, sinLocalFp32[ROPE_LENGTH_HALF], mask, tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(kLocal);

        Add(outLocalFp32, outLocalFp32, bufFp32, tilingData_->rowsPerBlock * ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, outLocalFp32, RoundMode::CAST_RINT, tilingData_->rowsPerBlock * ROPE_LENGTH);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, outLocalFp32, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * ROPE_LENGTH);
        }

        outQueueK.EnQue(outLocal);
    }

    __aicore__ inline void ScatterUpdateK(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> outLocal = outQueueK.DeQue<KV_DTYPE>();
        DataCopyExtParams copyParams{1, ROPE_LENGTH * sizeof(KV_DTYPE), 0, 0, 0};
        for (int64_t i = 0; i < tilingData_->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->seqLength +
                                  i * tilingData_->seqLength + seqId;
                int64_t offset = indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(kCacheGm[offset * ROPE_LENGTH], outLocal[i * ROPE_LENGTH], copyParams);
            } else {
                int64_t offset = indexGm(i * tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(
                    kCacheGm[i * tilingData_->cacheLength * ROPE_LENGTH + offset * ROPE_LENGTH],
                    outLocal[i * ROPE_LENGTH], copyParams);
            }
        }
        outQueueK.FreeTensor(outLocal);
    }

    __aicore__ inline void RmsNorm(int64_t seqId)
    {
        // load xB16 [tilingData_->rowsPerBlock, RMS_NORM_LENGTH]
        LocalTensor<KV_DTYPE> xB16Local = inQueueX.AllocTensor<KV_DTYPE>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(tilingData_->rowsPerBlock), RMS_NORM_LENGTH * sizeof(KV_DTYPE),
            static_cast<uint32_t>(
                ROPE_LENGTH * sizeof(KV_DTYPE) +
                (tilingData_->seqLength - 1) * (RMS_NORM_LENGTH + ROPE_LENGTH) * sizeof(KV_DTYPE)),
            0, 0};
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPad(xB16Local, kvGm[seqId * (RMS_NORM_LENGTH + ROPE_LENGTH)], copyParams, padParams);
        inQueueX.EnQue(xB16Local);
        xB16Local = inQueueX.DeQue<KV_DTYPE>();

        // Cast xB16 to xFp32 [tilingData_->rowsPerBlock, RMS_NORM_LENGTH]
        LocalTensor<float> xFp32Local = bufferXFp32.Get<float>();
        Cast(xFp32Local, xB16Local, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(xB16Local);

        // load gamma [RMS_NORM_LENGTH]
        LocalTensor<KV_DTYPE> gammaLocal = inQueueGamma.AllocTensor<KV_DTYPE>();
        DataCopy(gammaLocal, gammaGm, RMS_NORM_LENGTH);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<KV_DTYPE>();

        // Calc: xSquare = xFp32 * xFp32, xSquare shape is [tilingData_->rowsPerBlock, RMS_NORM_LENGTH]
        LocalTensor<float> xSquareLocal = bufferXSquare.Get<float>();
        Mul(xSquareLocal, xFp32Local, xFp32Local, tilingData_->rowsPerBlock * RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[tilingData_->rowsPerBlock, 512]
        //          --> [tilingData_->rowsPerBlock, 256]
        //          --> [tilingData_->rowsPerBlock, 128]
        //          --> [tilingData_->rowsPerBlock, 64]
        uint64_t mask[NUM_TWO] = {UINT64_MAX, UINT64_MAX};
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, NUM_FOUR * tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [tilingData_->rowsPerBlock, 256]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, NUM_TWO * tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [tilingData_->rowsPerBlock, 128]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, tilingData_->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [tilingData_->rowsPerBlock, 64]
        PipeBarrier<PIPE_V>();

        // Calc: xSum = [tilingData_->rowsPerBlock, 1]
        LocalTensor<float> xSumLocal = bufferSum.Get<float>();
        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, tilingData_->rowsPerBlock, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, tilingData_->reciprocal, tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, tilingData_->epsilon, tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, tilingData_->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[tilingData_->rowsPerBlock, 8] = brc(xSum[tilingData_->rowsPerBlock, 1])
        int64_t block_count = (tilingData_->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local / xSquareLocal
        for (int64_t rowId = 0; rowId < tilingData_->rowsPerBlock; rowId++) {
            Div(xFp32Local[rowId * RMS_NORM_LENGTH], xFp32Local[rowId * RMS_NORM_LENGTH],
                xSquareLocal[rowId * NUM_EIGHT], mask, NUM_EIGHT, {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Cast gammaLocal to xSquareLocal (b16 -> fp32) [RMS_NORM_LENGTH]
        Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();
        inQueueGamma.FreeTensor(gammaLocal);

        // Calc: xFp32Local = xFp32Local * xSquareLocal [tilingData_->rowsPerBlock, RMS_NORM_LENGTH] * [RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < tilingData_->rowsPerBlock; rowId++) {
            Mul(xFp32Local[rowId * RMS_NORM_LENGTH], xFp32Local[rowId * RMS_NORM_LENGTH], xSquareLocal,
                RMS_NORM_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        // Cast xFp32 to outLocal
        LocalTensor<KV_DTYPE> outLocal = outQueueV.AllocTensor<KV_DTYPE>();
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xFp32Local, RoundMode::CAST_RINT, tilingData_->rowsPerBlock * RMS_NORM_LENGTH);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xFp32Local, RoundMode::CAST_NONE, tilingData_->rowsPerBlock * RMS_NORM_LENGTH);
        }
        outQueueV.EnQue(outLocal);
    }

    __aicore__ inline void ScatterUpdateV(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> outLocal = outQueueV.DeQue<KV_DTYPE>();
        DataCopyExtParams copyParams{1, RMS_NORM_LENGTH * sizeof(KV_DTYPE), 0, 0, 0};
        for (int64_t i = 0; i < tilingData_->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * tilingData_->rowsPerBlock * tilingData_->seqLength +
                                  i * tilingData_->seqLength + seqId;
                int64_t offset = indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(vCacheGm[offset * RMS_NORM_LENGTH], outLocal[i * RMS_NORM_LENGTH], copyParams);
            } else {
                int64_t offset = indexGm(i * tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(
                    vCacheGm[i * tilingData_->cacheLength * RMS_NORM_LENGTH + offset * RMS_NORM_LENGTH],
                    outLocal[i * RMS_NORM_LENGTH], copyParams);
            }
        }
        outQueueV.FreeTensor(outLocal);
    }

private:
    TPipe* pipe_ = nullptr;
    const KvRmsNormRopeCacheTilingData* tilingData_;
    GlobalTensor<KV_DTYPE> kvGm;
    GlobalTensor<KV_DTYPE> gammaGm;
    GlobalTensor<KV_DTYPE> cosGm;
    GlobalTensor<KV_DTYPE> sinGm;
    GlobalTensor<int64_t> indexGm;
    GlobalTensor<KV_DTYPE> kCacheGm;
    GlobalTensor<KV_DTYPE> vCacheGm;

    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma;
    TQue<QuePosition::VECOUT, 1> outQueueK, outQueueV;
    TBuf<TPosition::VECCALC> bufferXFp32, bufferXSquare, bufferSum;

    constexpr static int64_t RMS_NORM_LENGTH = 512;
    constexpr static int64_t ROPE_LENGTH = 64;
    constexpr static int64_t ROPE_LENGTH_HALF = 32;
    constexpr static int64_t NUM_TWO = 2;
    constexpr static int64_t NUM_FOUR = 4;
    constexpr static int64_t NUM_EIGHT = 8;
    constexpr static int64_t NUM_SIXTEEN = 16;
    constexpr static int64_t NUM_SIXTY_FOUR = 64;
    constexpr static int64_t MASK_CONFIG = 0xffffffff;
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_