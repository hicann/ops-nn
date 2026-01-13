/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "embedding_common.h"

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(EMBEDDING_THREAD_NUM) inline void EmbeddingHashTableApplyAdamWB32Fn(
    uint32_t tableSize, int64_t keyNum, int64_t unusedKey, uint32_t bucketSizeByte, uint32_t xLoopSize,
    uint32_t embeddingDim, uint32_t maximize, uint32_t amsgrad, __gm__ int64_t* gmTableIn, __gm__ int64_t* gmKeys,
    __gm__ T* gmM, __gm__ T* gmV, __gm__ T* gmBeta1Power, __gm__ T* gmBeta2Power, __gm__ T* gmLr,
    __gm__ T* gmWeightDecay, __gm__ T* gmBeta1, __gm__ T* gmBeta2, __gm__ T* gmEpsilon, __gm__ T* gmGrad,
    __gm__ T* gmMaxGradNorm, __gm__ T* gmMOut, __gm__ T* gmVOut, __gm__ T* gmBeta1PowerOut, __gm__ T* gmBeta2PowerOut,
    __gm__ T* gmMaxGradNormOut)
{
    int32_t threadXIdx = Simt::GetThreadIdx<0>();
    int32_t threadYIdx = Simt::GetThreadIdx<1>();
    int32_t threadXNum = Simt::GetThreadNum<0>();
    int32_t threadYNum = Simt::GetThreadNum<1>();

    int64_t tableAddr = *(reinterpret_cast<__gm__ int64_t*>(gmTableIn[0]));
    float beta1PowerLocal = gmBeta1Power[0];
    float beta2PowerLocal = gmBeta2Power[0];
    float beta1Local = gmBeta1[0];
    float beta2Local = gmBeta2[0];
    float lrLocal = gmLr[0];
    float weightDecayLocal = gmWeightDecay[0];
    float epsilonLocal = gmEpsilon[0];

    beta1PowerLocal = beta1PowerLocal * beta1Local;
    beta2PowerLocal = beta2PowerLocal * beta2Local;

    for (uint32_t idx = block_idx * threadYNum + threadYIdx; idx < keyNum; idx += block_num * threadYNum) {
        int64_t updateKey = gmKeys[idx];
        __gm__ int64_t* key = gmKeys + idx;
        const uint32_t len = 8;
        const uint32_t seed = 0;
        uint32_t hashValue = Hashtbl::MurmurHash3(key, len, seed);
        uint32_t currentIdx = hashValue % tableSize;
        uint32_t tableOffsetByte = currentIdx * bucketSizeByte;

        for (uint32_t xLoopIdx = 0; xLoopIdx < xLoopSize; xLoopIdx += 1) {
            uint32_t xOffset = threadXNum * xLoopIdx + threadXIdx;
            if (xOffset >= embeddingDim) {
                continue;
            }

            float gtLocal = gmGrad[idx * embeddingDim + xOffset];

            if (maximize != 0) {
                gtLocal = -gtLocal;
            }
            uint32_t counter = 0;
            while (true) {
                if (counter++ >= tableSize || updateKey == unusedKey) {
                    break;
                }
                __gm__ uint8_t* currItem = reinterpret_cast<__gm__ uint8_t*>(tableAddr) + tableOffsetByte;
                __gm__ float* currItemVal =
                    reinterpret_cast<__gm__ float*>(currItem + TABLE_FLAG_NUM * sizeof(int64_t));
                int64_t currKey = *reinterpret_cast<__gm__ int64_t*>(currItem);

                if (currKey == updateKey) {
                    float value = currItemVal[xOffset];
                    value = value * (1 + (-lrLocal * weightDecayLocal));

                    float maxGradNormLocal = gmMaxGradNorm[currentIdx * embeddingDim + xOffset];
                    float mLocal = gmM[currentIdx * embeddingDim + xOffset];
                    float vLocal = gmV[currentIdx * embeddingDim + xOffset];

                    float mOutLocal = mLocal * beta1Local - (beta1Local + static_cast<float>(-1.0)) * gtLocal;
                    float vOutLocal = vLocal * beta2Local - (beta2Local + static_cast<float>(-1.0)) * gtLocal * gtLocal;

                    gmMOut[currentIdx * embeddingDim + xOffset] = mOutLocal;
                    gmVOut[currentIdx * embeddingDim + xOffset] = vOutLocal;

                    float denom = 1.0;
                    if (amsgrad != 0) {
                        maxGradNormLocal = Simt::Max(maxGradNormLocal, vOutLocal);
                        denom = Simt::Sqrt(-maxGradNormLocal / (beta2PowerLocal + (-1))) + epsilonLocal;
                    } else {
                        denom = Simt::Sqrt(-vOutLocal / (beta2PowerLocal + (-1))) + epsilonLocal;
                    }

                    value = value + (lrLocal * mOutLocal / (beta1PowerLocal + (-1))) / denom;

                    currItemVal[xOffset] = value;
                    gmMaxGradNormOut[currentIdx * embeddingDim + xOffset] = maxGradNormLocal;
                    break;
                }
                if (currentIdx == tableSize - 1) {
                    currentIdx = 0;
                } else {
                    ++currentIdx;
                }
                tableOffsetByte = currentIdx * bucketSizeByte;
            }
        }
    }
}

template <typename T>
class KernelEmbeddingHashTableApplyAdamWB32
{
public:
    __aicore__ inline KernelEmbeddingHashTableApplyAdamWB32()
    {}

    __aicore__ inline void Init(
        GM_ADDR tableIn, GM_ADDR keys, GM_ADDR m, GM_ADDR v, GM_ADDR beta1Power, GM_ADDR beta2Power, GM_ADDR lr,
        GM_ADDR weightDecay, GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad, GM_ADDR maxGradNorm,
        GM_ADDR mOut, GM_ADDR vOut, GM_ADDR beta1PowerOut, GM_ADDR beta2PowerOut, GM_ADDR maxGradNormOut,
        GM_ADDR workspace, EmbeddingHashTableApplyAdamWTilingData tilingData)
    {
        blockIdx_ = GetBlockIdx();

        keyNum_ = tilingData.keyNum;
        tableSize_ = tilingData.tableSize;
        embeddingDim_ = tilingData.embeddingDim;
        amsgrad_ = tilingData.amsgrad;
        maximize_ = tilingData.maximize;
        blockX_ = tilingData.blockX;
        blockY_ = tilingData.blockY;
        blockNum_ = tilingData.blockNum;
        // key, counter, flag are int64 data type, bucketSizeByte need to be aligned based on the int64 bit
        bucketSizeByte = ROUND_UP8(embeddingDim_ * sizeof(float) + TABLE_FLAG_NUM * sizeof(int64_t));
        xLoopSize_ = (embeddingDim_ + blockX_ - 1) / blockX_;

        gmTableIn_.SetGlobalBuffer((__gm__ int64_t*)tableIn);
        gmKeys_.SetGlobalBuffer((__gm__ int64_t*)keys);
        gmM_.SetGlobalBuffer((__gm__ T*)m);
        gmV_.SetGlobalBuffer((__gm__ T*)v);
        gmBeta1Power_.SetGlobalBuffer((__gm__ T*)beta1Power, 1);
        gmBeta2Power_.SetGlobalBuffer((__gm__ T*)beta2Power, 1);
        gmLr_.SetGlobalBuffer((__gm__ T*)lr);
        gmWeightDecay_.SetGlobalBuffer((__gm__ T*)weightDecay);
        gmBeta1_.SetGlobalBuffer((__gm__ T*)beta1, 1);
        gmBeta2_.SetGlobalBuffer((__gm__ T*)beta2, 1);
        gmEpsilon_.SetGlobalBuffer((__gm__ T*)epsilon);
        gmGrad_.SetGlobalBuffer((__gm__ T*)grad);
        gmMaxGradNorm_.SetGlobalBuffer((__gm__ T*)maxGradNorm);
        gmMOut_.SetGlobalBuffer((__gm__ T*)mOut);
        gmVOut_.SetGlobalBuffer((__gm__ T*)vOut);
        gmBeta1PowerOut_.SetGlobalBuffer((__gm__ T*)beta1PowerOut, 1);
        gmBeta2PowerOut_.SetGlobalBuffer((__gm__ T*)beta2PowerOut, 1);
        gmMaxGradNormOut_.SetGlobalBuffer((__gm__ T*)maxGradNormOut);
    }

    __aicore__ inline void PostProcess(GM_ADDR beta1PowerOut, GM_ADDR beta2PowerOut)
    {
        if (blockIdx_ >= 1) {
            return;
        }
        float beta1Power_ = gmBeta1Power_.GetValue(0);
        float beta2Power_ = gmBeta2Power_.GetValue(0);
        float beta1_ = gmBeta1_.GetValue(0);
        float beta2_ = gmBeta2_.GetValue(0);

        beta1Power_ = beta1Power_ * beta1_;
        beta2Power_ = beta2Power_ * beta2_;
        gmBeta1PowerOut_.SetValue(0, beta1Power_);
        gmBeta2PowerOut_.SetValue(0, beta2Power_);

        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(beta1PowerOut);
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(beta2PowerOut);
    }

    __aicore__ inline void Process()
    {
        Simt::VF_CALL<EmbeddingHashTableApplyAdamWB32Fn<T>>(
            Simt::Dim3{static_cast<uint32_t>(blockX_), static_cast<uint32_t>(blockY_)}, tableSize_, keyNum_, unusedKey,
            bucketSizeByte, xLoopSize_, embeddingDim_, maximize_, amsgrad_, gmTableIn_.GetPhyAddr(0),
            gmKeys_.GetPhyAddr(0), gmM_.GetPhyAddr(0), gmV_.GetPhyAddr(0), gmBeta1Power_.GetPhyAddr(0),
            gmBeta2Power_.GetPhyAddr(0), gmLr_.GetPhyAddr(0), gmWeightDecay_.GetPhyAddr(0), gmBeta1_.GetPhyAddr(0),
            gmBeta2_.GetPhyAddr(0), gmEpsilon_.GetPhyAddr(0), gmGrad_.GetPhyAddr(0), gmMaxGradNorm_.GetPhyAddr(0),
            gmMOut_.GetPhyAddr(0), gmVOut_.GetPhyAddr(0), gmBeta1PowerOut_.GetPhyAddr(0),
            gmBeta2PowerOut_.GetPhyAddr(0), gmMaxGradNormOut_.GetPhyAddr(0));
        SyncAll();
    }

private:
    GlobalTensor<int64_t> gmTableIn_;
    GlobalTensor<int64_t> gmKeys_;
    GlobalTensor<T> gmM_;
    GlobalTensor<T> gmV_;
    GlobalTensor<T> gmBeta1Power_;
    GlobalTensor<T> gmBeta2Power_;
    GlobalTensor<T> gmLr_;
    GlobalTensor<T> gmWeightDecay_;
    GlobalTensor<T> gmBeta1_;
    GlobalTensor<T> gmBeta2_;
    GlobalTensor<T> gmEpsilon_;
    GlobalTensor<T> gmGrad_;
    GlobalTensor<T> gmMaxGradNorm_;
    GlobalTensor<T> gmMOut_;
    GlobalTensor<T> gmVOut_;
    GlobalTensor<T> gmBeta1PowerOut_;
    GlobalTensor<T> gmBeta2PowerOut_;
    GlobalTensor<T> gmMaxGradNormOut_;

    uint32_t blockIdx_ = 0;

    int64_t unusedKey = -1;
    int64_t keyNum_ = 0;
    uint32_t tableSize_ = 0;
    uint32_t embeddingDim_ = 0;
    uint32_t amsgrad_ = 0;
    uint32_t maximize_ = 0;
    uint64_t blockX_ = 0;
    uint64_t blockY_ = 0;
    uint64_t blockNum_ = 0;
    // key, counter, flag are int64 data type, bucketSizeByte need to be aligned based on the int64 bit
    uint32_t bucketSizeByte = 0;
    uint32_t xLoopSize_ = 0;
};

