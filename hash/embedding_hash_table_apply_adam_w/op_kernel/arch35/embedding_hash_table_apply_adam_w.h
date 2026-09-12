
/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/vector_functions.h"

static constexpr uint8_t VALID_FLAG_MASK = 0b00000001;
// fp32 合并访存分派窗口，与 tiling 侧 IsMergeFp32Dim 逐字一致（布局契约，两侧必须同改）：
// 主窗口偶数 dim∈[6,128]\{26,62}（26/62 实测 0.975x 回退）+ 高窗口偶数 dim>736
// （合并 VF 仅在 >736 反超纯布局：768:1.08x/1024:1.22x；257~736 由 tiling 切 probe
// 摊销布局 bx=32 承担，legacy 路径 1.06~1.49x 全 dtype/奇偶生效）
static constexpr uint32_t ADAMW_MERGE_MIN_DIM = 6;
static constexpr uint32_t ADAMW_MERGE_MAX_DIM = 128;
static constexpr uint32_t ADAMW_MERGE_VF_HIGH_DIM = 736; // 合并 VF 高窗口起点（与 tiling MERGE_VF_HIGH_DIM 一致）

// SIMT 访存合并（b128/b64 短向量，见 simt_api/vector_functions.h）：每个 X 线程处理连续
// MERGE 个元素。7 条外部张量流（grad/m/v/maxGradNorm 读 + mOut/vOut/maxGradNormOut 写）
// 行基址 dim%4==0 时 16B 对齐 → float4(B128)；桶 values 区偏移 24B 恒 ≡8 (mod 16) →
// 读/写封顶 float2(B64)（24B 桶头为跨算子契约，与 lookup/import/export 相同约束）。
// 每元素的 fp32 运算表达式与标量路径逐字一致（同序同形），结果逐比特不变。
// 注意：load/compute/store 分两相以压低寄存器峰值活跃区间（7 流×4 元素同驻会 spill，
// 实测拖慢整个 VF 含 legacy 路径 ~28%），勿改回"全量装载后统一计算"形态。
template <int MERGE>
__simt_callee__ __aicore__ inline void AdamWUpdateGroupFp32(
    uint32_t j0, int64_t keyRowBase, int64_t stateRowBase, __gm__ float* currItemVal, uint32_t maximize,
    uint32_t amsgrad, float beta1PowerLocal, float beta2PowerLocal, float beta1Local, float beta2Local, float lrLocal,
    float weightDecayLocal, float epsilonLocal, const __gm__ float* gmGrad, const __gm__ float* gmM,
    const __gm__ float* gmV, const __gm__ float* gmMaxGradNorm, __gm__ float* gmMOut, __gm__ float* gmVOut,
    __gm__ float* gmMaxGradNormOut)
{
    // 第一相：grad/m/v 装载 → mOut/vOut 计算并立刻写回，只留 gt*vOut 给第二相
    float gtLocal[MERGE];
    float mOutLocal[MERGE];
    float vOutLocal[MERGE];
    if constexpr (MERGE == 4) {
        float4 g4 = *(reinterpret_cast<const __gm__ float4*>(gmGrad + keyRowBase + j0));
        gtLocal[0] = g4.x;
        gtLocal[1] = g4.y;
        gtLocal[2] = g4.z;
        gtLocal[3] = g4.w;
        float4 m4 = *(reinterpret_cast<const __gm__ float4*>(gmM + stateRowBase + j0));
        float4 v4 = *(reinterpret_cast<const __gm__ float4*>(gmV + stateRowBase + j0));
        for (int k = 0; k < MERGE; k++) {
            if (maximize != 0) {
                gtLocal[k] = -gtLocal[k];
            }
            float mLocal = (k == 0) ? m4.x : (k == 1) ? m4.y : (k == 2) ? m4.z : m4.w;
            float vLocal = (k == 0) ? v4.x : (k == 1) ? v4.y : (k == 2) ? v4.z : v4.w;
            mOutLocal[k] = mLocal * beta1Local - (beta1Local + static_cast<float>(-1.0)) * gtLocal[k];
            vOutLocal[k] = vLocal * beta2Local - (beta2Local + static_cast<float>(-1.0)) * gtLocal[k] * gtLocal[k];
        }
        *(reinterpret_cast<__gm__ float4*>(gmMOut + stateRowBase + j0)) = make_float4(mOutLocal[0], mOutLocal[1],
                                                                                      mOutLocal[2], mOutLocal[3]);
        *(reinterpret_cast<__gm__ float4*>(gmVOut + stateRowBase + j0)) = make_float4(vOutLocal[0], vOutLocal[1],
                                                                                      vOutLocal[2], vOutLocal[3]);
    } else {
        float2 g2 = *(reinterpret_cast<const __gm__ float2*>(gmGrad + keyRowBase + j0));
        gtLocal[0] = g2.x;
        gtLocal[1] = g2.y;
        float2 m2 = *(reinterpret_cast<const __gm__ float2*>(gmM + stateRowBase + j0));
        float2 v2 = *(reinterpret_cast<const __gm__ float2*>(gmV + stateRowBase + j0));
        for (int k = 0; k < MERGE; k++) {
            if (maximize != 0) {
                gtLocal[k] = -gtLocal[k];
            }
            float mLocal = (k == 0) ? m2.x : m2.y;
            float vLocal = (k == 0) ? v2.x : v2.y;
            mOutLocal[k] = mLocal * beta1Local - (beta1Local + static_cast<float>(-1.0)) * gtLocal[k];
            vOutLocal[k] = vLocal * beta2Local - (beta2Local + static_cast<float>(-1.0)) * gtLocal[k] * gtLocal[k];
        }
        *(reinterpret_cast<__gm__ float2*>(gmMOut + stateRowBase + j0)) = make_float2(mOutLocal[0], mOutLocal[1]);
        *(reinterpret_cast<__gm__ float2*>(gmVOut + stateRowBase + j0)) = make_float2(vOutLocal[0], vOutLocal[1]);
    }

    // 第二相：maxGradNorm + 桶 value 装载 → denom/value 计算 → 写回（mOut/vOut 已落 GM，
    // 活跃区间只剩 value/mgn/mOut/vOut 四组）
    float valueLocal[MERGE];
    float maxGradNormLocal[MERGE];
    if constexpr (MERGE == 4) {
        float4 mgn4 = *(reinterpret_cast<const __gm__ float4*>(gmMaxGradNorm + stateRowBase + j0));
        maxGradNormLocal[0] = mgn4.x;
        maxGradNormLocal[1] = mgn4.y;
        maxGradNormLocal[2] = mgn4.z;
        maxGradNormLocal[3] = mgn4.w;
        const __gm__ float2* pBucket = reinterpret_cast<const __gm__ float2*>(currItemVal + j0);
        float2 vlo = pBucket[0];
        float2 vhi = pBucket[1];
        valueLocal[0] = vlo.x;
        valueLocal[1] = vlo.y;
        valueLocal[2] = vhi.x;
        valueLocal[3] = vhi.y;
        for (int k = 0; k < MERGE; k++) {
            valueLocal[k] = valueLocal[k] * (1 + (-lrLocal * weightDecayLocal));
            float denom = 1.0;
            if (amsgrad != 0) {
                maxGradNormLocal[k] = fmaxf(maxGradNormLocal[k], vOutLocal[k]);
                denom = sqrtf(-maxGradNormLocal[k] / (beta2PowerLocal + (-1))) + epsilonLocal;
            } else {
                denom = sqrtf(-vOutLocal[k] / (beta2PowerLocal + (-1))) + epsilonLocal;
            }
            valueLocal[k] = valueLocal[k] + (lrLocal * mOutLocal[k] / (beta1PowerLocal + (-1))) / denom;
        }
        *(reinterpret_cast<__gm__ float4*>(gmMaxGradNormOut + stateRowBase + j0)) = make_float4(
            maxGradNormLocal[0], maxGradNormLocal[1], maxGradNormLocal[2], maxGradNormLocal[3]);
        __gm__ float2* pBucketOut = reinterpret_cast<__gm__ float2*>(currItemVal + j0);
        pBucketOut[0] = make_float2(valueLocal[0], valueLocal[1]);
        pBucketOut[1] = make_float2(valueLocal[2], valueLocal[3]);
    } else {
        float2 mgn2 = *(reinterpret_cast<const __gm__ float2*>(gmMaxGradNorm + stateRowBase + j0));
        maxGradNormLocal[0] = mgn2.x;
        maxGradNormLocal[1] = mgn2.y;
        float2 vb = *(reinterpret_cast<const __gm__ float2*>(currItemVal + j0));
        valueLocal[0] = vb.x;
        valueLocal[1] = vb.y;
        for (int k = 0; k < MERGE; k++) {
            valueLocal[k] = valueLocal[k] * (1 + (-lrLocal * weightDecayLocal));
            float denom = 1.0;
            if (amsgrad != 0) {
                maxGradNormLocal[k] = fmaxf(maxGradNormLocal[k], vOutLocal[k]);
                denom = sqrtf(-maxGradNormLocal[k] / (beta2PowerLocal + (-1))) + epsilonLocal;
            } else {
                denom = sqrtf(-vOutLocal[k] / (beta2PowerLocal + (-1))) + epsilonLocal;
            }
            valueLocal[k] = valueLocal[k] + (lrLocal * mOutLocal[k] / (beta1PowerLocal + (-1))) / denom;
        }
        *(reinterpret_cast<__gm__ float2*>(gmMaxGradNormOut + stateRowBase + j0)) = make_float2(maxGradNormLocal[0],
                                                                                                maxGradNormLocal[1]);
        *(reinterpret_cast<__gm__ float2*>(currItemVal + j0)) = make_float2(valueLocal[0], valueLocal[1]);
    }
}

// fp32 合并访存版主循环（独立 VF 函数，Process 按 dtype+dim 分派；fp16 实例化完全不含
// 此函数——寄存器分配按 VF 函数独立，legacy ComputeAdamW 代码生成不受任何影响）。
// probe/哈希与 ComputeAdamW 逐字一致；values 更新走 AdamWUpdateGroupFp32 合并访存。
__simt_vf__ __aicore__ LAUNCH_BOUND(EMBEDDING_THREAD_NUM) inline void ComputeAdamWMergedFp32(
    uint32_t tableSize, int64_t keyNum, int64_t unusedKey, uint32_t bucketSizeByte, uint32_t embeddingDim,
    uint32_t maximize, uint32_t amsgrad, __gm__ int64_t* gmTableIn, __gm__ int64_t* gmKeys, __gm__ float* gmM,
    __gm__ float* gmV, __gm__ float* gmBeta1Power, __gm__ float* gmBeta2Power, __gm__ float* gmLr,
    __gm__ float* gmWeightDecay, __gm__ float* gmBeta1, __gm__ float* gmBeta2, __gm__ float* gmEpsilon,
    __gm__ float* gmGrad, __gm__ float* gmMaxGradNorm, __gm__ float* gmMOut, __gm__ float* gmVOut,
    __gm__ float* gmMaxGradNormOut)
{
    int32_t threadXIdx = threadIdx.x;
    int32_t threadYIdx = threadIdx.y;
    int32_t threadXNum = blockDim.x;
    int32_t threadYNum = blockDim.y;

    int64_t tableAddr = *(reinterpret_cast<__gm__ int64_t*>(gmTableIn[0]));
    __gm__ uint8_t* table = reinterpret_cast<__gm__ uint8_t*>(tableAddr);

    float beta1PowerLocal = static_cast<float>(gmBeta1Power[0]);
    float beta2PowerLocal = static_cast<float>(gmBeta2Power[0]);
    float beta1Local = static_cast<float>(gmBeta1[0]);
    float beta2Local = static_cast<float>(gmBeta2[0]);
    float lrLocal = static_cast<float>(gmLr[0]);
    float weightDecayLocal = static_cast<float>(gmWeightDecay[0]);
    float epsilonLocal = static_cast<float>(gmEpsilon[0]);

    beta1PowerLocal = beta1PowerLocal * beta1Local;
    beta2PowerLocal = beta2PowerLocal * beta2Local;

    // merge 为 kernel 级不变量，warp 内无分化
    uint32_t merge = (embeddingDim % 4 == 0) ? 4 : 2;
    if (merge == 4) {
        // dim%4==0 时行偏移恒为 16B 倍数，仅依赖各张量基址；极端非对齐兜底落 MERGE=2
        bool aligned16 = ((reinterpret_cast<uintptr_t>(gmGrad) | reinterpret_cast<uintptr_t>(gmM) |
                           reinterpret_cast<uintptr_t>(gmV) | reinterpret_cast<uintptr_t>(gmMaxGradNorm) |
                           reinterpret_cast<uintptr_t>(gmMOut) | reinterpret_cast<uintptr_t>(gmVOut) |
                           reinterpret_cast<uintptr_t>(gmMaxGradNormOut)) &
                          15) == 0;
        if (!aligned16) {
            merge = 2;
        }
    }

    for (uint32_t idx = block_idx * threadYNum + threadYIdx; idx < keyNum; idx += block_num * threadYNum) {
        int64_t updateKey = gmKeys[idx];
        if (updateKey == unusedKey) {
            continue;
        }

        size_t currentIdx = Hashtbl::MurmurHash3(gmKeys + idx, sizeof(int64_t), 0) % tableSize;
        size_t tableOffsetByte = currentIdx * bucketSizeByte;
        __gm__ uint8_t* currItem = table + tableOffsetByte;

        // 查找该更新的索引位置currentIdx
        bool found = false;
        size_t detectCounts = 0;
        while (detectCounts++ < tableSize) {
            int64_t currKey = *reinterpret_cast<__gm__ int64_t*>(currItem);
            uint8_t currFlag = *(currItem + TABLE_FLAG_NUM * sizeof(int64_t) - 1); // 取第23字节的flag位
            if ((currFlag & VALID_FLAG_MASK) == 0) {
                // 该位置validFlag=0，为空bucket，说明该key已经无法在表内找到了
                break;
            } else if (currKey == updateKey) {
                found = true;
                break;
            }
            currentIdx = (currentIdx + 1) % tableSize;
            tableOffsetByte = currentIdx * bucketSizeByte;
            currItem = table + tableOffsetByte;
        }
        // 如果查到了
        if (found) {
            // lane 块连续映射 j0 = threadXIdx*merge，步长 threadXNum*merge；
            // threadXNum*merge >= dim 时尾部 lane 自然空转（j0 >= dim 不进循环）
            __gm__ float* currItemVal = reinterpret_cast<__gm__ float*>(
                currItem + TABLE_FLAG_NUM * sizeof(int64_t)); // 这是currItem+24，第24字节开始是values
            int64_t keyRowBase = static_cast<int64_t>(idx) * embeddingDim;
            int64_t stateRowBase = static_cast<int64_t>(currentIdx) * embeddingDim;
            for (uint32_t j0 = threadXIdx * merge; j0 < embeddingDim; j0 += threadXNum * merge) {
                if (merge == 4) {
                    AdamWUpdateGroupFp32<4>(j0, keyRowBase, stateRowBase, currItemVal, maximize, amsgrad,
                                            beta1PowerLocal, beta2PowerLocal, beta1Local, beta2Local, lrLocal,
                                            weightDecayLocal, epsilonLocal, gmGrad, gmM, gmV, gmMaxGradNorm, gmMOut,
                                            gmVOut, gmMaxGradNormOut);
                } else {
                    AdamWUpdateGroupFp32<2>(j0, keyRowBase, stateRowBase, currItemVal, maximize, amsgrad,
                                            beta1PowerLocal, beta2PowerLocal, beta1Local, beta2Local, lrLocal,
                                            weightDecayLocal, epsilonLocal, gmGrad, gmM, gmV, gmMaxGradNorm, gmMOut,
                                            gmVOut, gmMaxGradNormOut);
                }
            }
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(EMBEDDING_THREAD_NUM) inline void ComputeAdamW(
    uint32_t tableSize, int64_t keyNum, int64_t unusedKey, uint32_t bucketSizeByte, uint32_t xLoopSize,
    uint32_t embeddingDim, uint32_t maximize, uint32_t amsgrad, __gm__ int64_t* gmTableIn, __gm__ int64_t* gmKeys,
    __gm__ T* gmM, __gm__ T* gmV, __gm__ T* gmBeta1Power, __gm__ T* gmBeta2Power, __gm__ T* gmLr,
    __gm__ T* gmWeightDecay, __gm__ T* gmBeta1, __gm__ T* gmBeta2, __gm__ T* gmEpsilon, __gm__ T* gmGrad,
    __gm__ T* gmMaxGradNorm, __gm__ T* gmMOut, __gm__ T* gmVOut, __gm__ T* gmBeta1PowerOut, __gm__ T* gmBeta2PowerOut,
    __gm__ T* gmMaxGradNormOut)
{
    int32_t threadXIdx = threadIdx.x;
    int32_t threadYIdx = threadIdx.y;
    int32_t threadXNum = blockDim.x;
    int32_t threadYNum = blockDim.y;

    int64_t tableAddr = *(reinterpret_cast<__gm__ int64_t*>(gmTableIn[0]));
    __gm__ uint8_t* table = reinterpret_cast<__gm__ uint8_t*>(tableAddr);

    float beta1PowerLocal = static_cast<float>(gmBeta1Power[0]);
    float beta2PowerLocal = static_cast<float>(gmBeta2Power[0]);
    float beta1Local = static_cast<float>(gmBeta1[0]);
    float beta2Local = static_cast<float>(gmBeta2[0]);
    float lrLocal = static_cast<float>(gmLr[0]);
    float weightDecayLocal = static_cast<float>(gmWeightDecay[0]);
    float epsilonLocal = static_cast<float>(gmEpsilon[0]);

    beta1PowerLocal = beta1PowerLocal * beta1Local;
    beta2PowerLocal = beta2PowerLocal * beta2Local;

    for (uint32_t idx = block_idx * threadYNum + threadYIdx; idx < keyNum; idx += block_num * threadYNum) {
        int64_t updateKey = gmKeys[idx];
        if (updateKey == unusedKey) {
            continue;
        }

        size_t currentIdx = Hashtbl::MurmurHash3(gmKeys + idx, sizeof(int64_t), 0) % tableSize;
        size_t tableOffsetByte = currentIdx * bucketSizeByte;
        __gm__ uint8_t* currItem = table + tableOffsetByte;

        // 查找该更新的索引位置currentIdx
        bool found = false;
        size_t detectCounts = 0;
        while (detectCounts++ < tableSize) {
            int64_t currKey = *reinterpret_cast<__gm__ int64_t*>(currItem);
            uint8_t currFlag = *(currItem + TABLE_FLAG_NUM * sizeof(int64_t) - 1); // 取第23字节的flag位
            if ((currFlag & VALID_FLAG_MASK) == 0) {
                // 该位置validFlag=0，为空bucket，说明该key已经无法在表内找到了
                break;
            } else if (currKey == updateKey) {
                found = true;
                break;
            }
            currentIdx = (currentIdx + 1) % tableSize;
            tableOffsetByte = currentIdx * bucketSizeByte;
            currItem = table + tableOffsetByte;
        }
        // 如果查到了
        if (found) {
            for (uint32_t xLoopIdx = 0; xLoopIdx < xLoopSize; xLoopIdx += 1) {
                size_t xOffset = static_cast<size_t>(threadXNum) * xLoopIdx + threadXIdx;
                if (xOffset >= embeddingDim) {
                    // threadXNum比embedingDim要大，多出来的线程不做任何操作
                    break;
                }
                __gm__ float* currItemVal = reinterpret_cast<__gm__ float*>(
                    currItem + TABLE_FLAG_NUM * sizeof(int64_t)); // 这是currItem+24，第24字节开始是values

                float gtLocal = static_cast<float>(gmGrad[idx * embeddingDim + xOffset]);
                if (maximize != 0) {
                    gtLocal = -gtLocal;
                }

                float value = static_cast<float>(currItemVal[xOffset]);
                value = value * (1 + (-lrLocal * weightDecayLocal));

                float maxGradNormLocal = static_cast<float>(gmMaxGradNorm[currentIdx * embeddingDim + xOffset]);
                float mLocal = static_cast<float>(gmM[currentIdx * embeddingDim + xOffset]);
                float vLocal = static_cast<float>(gmV[currentIdx * embeddingDim + xOffset]);

                float mOutLocal = mLocal * beta1Local - (beta1Local + static_cast<float>(-1.0)) * gtLocal;
                float vOutLocal = vLocal * beta2Local - (beta2Local + static_cast<float>(-1.0)) * gtLocal * gtLocal;

                gmMOut[currentIdx * embeddingDim + xOffset] = static_cast<T>(mOutLocal);
                gmVOut[currentIdx * embeddingDim + xOffset] = static_cast<T>(vOutLocal);

                float denom = 1.0;
                if (amsgrad != 0) {
                    maxGradNormLocal = fmaxf(maxGradNormLocal, vOutLocal);
                    denom = sqrtf(-maxGradNormLocal / (beta2PowerLocal + (-1))) + epsilonLocal;
                } else {
                    denom = sqrtf(-vOutLocal / (beta2PowerLocal + (-1))) + epsilonLocal;
                }

                value = value + (lrLocal * mOutLocal / (beta1PowerLocal + (-1))) / denom;
                currItemVal[xOffset] = static_cast<T>(value);
                gmMaxGradNormOut[currentIdx * embeddingDim + xOffset] = static_cast<T>(maxGradNormLocal);
            }
        }
    }
}

template <typename T>
class KernelEmbeddingHashTableApplyAdamW {
public:
    __aicore__ inline KernelEmbeddingHashTableApplyAdamW() {}

    __aicore__ inline void Init(GM_ADDR tableIn, GM_ADDR keys, GM_ADDR m, GM_ADDR v, GM_ADDR beta1Power,
                                GM_ADDR beta2Power, GM_ADDR lr, GM_ADDR weightDecay, GM_ADDR beta1, GM_ADDR beta2,
                                GM_ADDR epsilon, GM_ADDR grad, GM_ADDR maxGradNorm, GM_ADDR mOut, GM_ADDR vOut,
                                GM_ADDR beta1PowerOut, GM_ADDR beta2PowerOut, GM_ADDR maxGradNormOut, GM_ADDR workspace,
                                EmbeddingHashTableApplyAdamWTilingData tilingData)
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
        float beta1Power = static_cast<float>(gmBeta1Power_.GetValue(0));
        float beta2Power = static_cast<float>(gmBeta2Power_.GetValue(0));
        float beta1 = static_cast<float>(gmBeta1_.GetValue(0));
        float beta2 = static_cast<float>(gmBeta2_.GetValue(0));

        beta1Power = beta1Power * beta1;
        beta2Power = beta2Power * beta2;

        gmBeta1PowerOut_.SetValue(0, static_cast<T>(beta1Power));
        gmBeta2PowerOut_.SetValue(0, static_cast<T>(beta2Power));

        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gmBeta1PowerOut_);
        DataCacheCleanAndInvalid<T, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gmBeta2PowerOut_);
    }

    __aicore__ inline void Process()
    {
        // fp32 且 dim 在合并窗口（[6,128]\{26,62} 或 >736）走合并访存 VF
        // （与 tiling IsMergeFp32Dim 同条件）；其余（含全部 fp16）走原 ComputeAdamW，路径零改动。
        // dim 257~736 的加速由 tiling 侧 probe 摊销布局承担（legacy 路径，本 kernel 无感知）
        if constexpr (sizeof(T) == 4) {
            if (embeddingDim_ % 2 == 0 &&
                ((embeddingDim_ >= ADAMW_MERGE_MIN_DIM && embeddingDim_ <= ADAMW_MERGE_MAX_DIM && embeddingDim_ != 26 &&
                  embeddingDim_ != 62) ||
                 embeddingDim_ > ADAMW_MERGE_VF_HIGH_DIM)) {
                asc_vf_call<ComputeAdamWMergedFp32>(
                    dim3{static_cast<uint32_t>(blockX_), static_cast<uint32_t>(blockY_)}, tableSize_, keyNum_,
                    unusedKey, bucketSizeByte, embeddingDim_, maximize_, amsgrad_, gmTableIn_.GetPhyAddr(0),
                    gmKeys_.GetPhyAddr(0), gmM_.GetPhyAddr(0), gmV_.GetPhyAddr(0), gmBeta1Power_.GetPhyAddr(0),
                    gmBeta2Power_.GetPhyAddr(0), gmLr_.GetPhyAddr(0), gmWeightDecay_.GetPhyAddr(0),
                    gmBeta1_.GetPhyAddr(0), gmBeta2_.GetPhyAddr(0), gmEpsilon_.GetPhyAddr(0), gmGrad_.GetPhyAddr(0),
                    gmMaxGradNorm_.GetPhyAddr(0), gmMOut_.GetPhyAddr(0), gmVOut_.GetPhyAddr(0),
                    gmMaxGradNormOut_.GetPhyAddr(0));
                SyncAll();
                return;
            }
        }
        asc_vf_call<ComputeAdamW<T>>(
            dim3{static_cast<uint32_t>(blockX_), static_cast<uint32_t>(blockY_)}, tableSize_, keyNum_, unusedKey,
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
