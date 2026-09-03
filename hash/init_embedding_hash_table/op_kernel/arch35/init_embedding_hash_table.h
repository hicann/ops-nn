/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file init_embedding_hash_table.h
 * \brief Ascendc InitEmbeddingHashTable kernel implement
 */

#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_INIT_EMBEDDING_HASHTABLE_INIT_EMBEDDING_HASH_TABLE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_INIT_EMBEDDING_HASHTABLE_INIT_EMBEDDING_HASH_TABLE_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/vector_functions.h"

namespace InitEmbeddingHashTable {
using namespace AscendC;
constexpr uint32_t THREAD_NUM = 512;

constexpr int64_t DEFAULT_KEY = -1;
constexpr int64_t DEFAULT_COUNTER = 0;
constexpr int64_t DEFAULT_FLAG = 0;

constexpr int64_t INT64_PER_BYTE = 8;
constexpr int64_t FLOAT_PER_BYTE = 4;

constexpr int64_t RANDOM_MOD = 0;
constexpr int64_t CONST_MOD = 1;

constexpr int64_t KEY_OFFSET = 0;
constexpr int64_t COUNTER_OFFSET = 1;
constexpr int64_t FLAG_OFFSET = 2;
constexpr int64_t VALUES_OFFSET = 3;

// SIMT 访存合并（b64/b128 短向量，见 simt_api/vector_functions.h；init 为 fp32 单 dtype）：
// 桶头 24B = key(8B) + counter(8B) + flag(8B)，values 区偏移 24B；桶 stride 仅 8B 对齐，
// 桶基址 mod16 ∈ {0,8}（bucketLength 为奇时逐桶交替），head 为对齐到 16B 网格的偏移。
//   基址≡0：key+counter 一笔 16B 常数写 {-1,0}，flag 折进 values 首块 {0,0,v0,v1}；
//   基址≡8：key 单写 8B + counter|flag 一笔 16B 零写，values 从 24B 起天然 16B 对齐。
// 两条路径主体均为 float4(B128)，尾块 float2(B64)→float 逐级收尾。
template <typename Tvalue>
__simt_callee__ __aicore__ inline void InitBucket(__gm__ uint8_t* bucket, uint64_t head, int64_t bucketIdx,
                                                  int64_t embeddingDim, int64_t initializerMode, Tvalue constantValue,
                                                  __gm__ Tvalue* sampledValuesGm)
{
    if (head == 0) {
        *reinterpret_cast<__gm__ longlong2*>(bucket) = make_longlong2(DEFAULT_KEY, DEFAULT_COUNTER);
    } else {
        *reinterpret_cast<__gm__ int64_t*>(bucket) = DEFAULT_KEY;
        *reinterpret_cast<__gm__ longlong2*>(bucket + INT64_PER_BYTE) = make_longlong2(0, 0);
    }

    __gm__ uint8_t* valuesBegin = bucket + VALUES_OFFSET * INT64_PER_BYTE;
    // 16B 网格主体窗口 [bodyBegin, valuesBegin+dim*4)，jBase 为窗口起点对应的 value 下标（可为 -2）
    __gm__ uint8_t* bodyBegin = bucket + FLAG_OFFSET * INT64_PER_BYTE + head;
    const int64_t bodyBytes = (valuesBegin + embeddingDim * FLOAT_PER_BYTE) - bodyBegin;
    const int64_t jBase = (bodyBegin - valuesBegin) / FLOAT_PER_BYTE;
    const __gm__ Tvalue* row = sampledValuesGm + bucketIdx * embeddingDim; // 仅 RANDOM_MOD 解引用
    const int64_t nFull = bodyBytes / 16;
    for (int64_t k = 0; k < nFull; ++k) {
        const int64_t j = jBase + k * 4; // j<0 仅出现在 head==0 首块（j==-2，flag 折叠位）
        float4 v;
        if (initializerMode == RANDOM_MOD) {
            v = make_float4(j >= 0 ? row[j] : 0.0f, j + 1 >= 0 ? row[j + 1] : 0.0f, row[j + 2], row[j + 3]);
        } else {
            v = make_float4(constantValue, constantValue, constantValue, constantValue);
            if (j < 0) {
                v.x = 0.0f;
                v.y = 0.0f;
            }
        }
        *reinterpret_cast<__gm__ float4*>(bodyBegin + k * 16) = v;
    }

    // 尾块 0/4/8/12B：float2(B64) + float 逐级收尾
    __gm__ uint8_t* tail = bodyBegin + nFull * 16;
    const int64_t tailBytes = bodyBytes - nFull * 16;
    int64_t jt = jBase + nFull * 4;
    if (tailBytes >= 8) {
        const Tvalue a = jt >= 0 ? (initializerMode == RANDOM_MOD ? row[jt] : constantValue) : (Tvalue)0;
        const Tvalue b = jt + 1 >= 0 ? (initializerMode == RANDOM_MOD ? row[jt + 1] : constantValue) : (Tvalue)0;
        *reinterpret_cast<__gm__ float2*>(tail) = make_float2(a, b);
        tail += 8;
        jt += 2;
    }
    if ((tailBytes & 4) != 0) {
        *reinterpret_cast<__gm__ Tvalue*>(tail) = initializerMode == RANDOM_MOD ? row[jt] : constantValue;
    }
}

template <typename Tkey, typename Tvalue>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void InitCompute(
    int64_t embeddingDim, int64_t bucketSize, int64_t bucketLength, int64_t initializerMode, Tvalue constantValue,
    uint32_t blockIdx, uint32_t blockNum, __gm__ int64_t* tableHanldeGm, __gm__ Tvalue* sampledValuesGm,
    __gm__ uint8_t* outputGm)
{
    // 桶基址 mod16 = tableAlign + 8*(i&1)*(bucketLength&1)；grid-stride 为偶数 → 线程对齐类全程不变，
    // head 提到循环外只算一次（warp 内仍可能两路分化，但循环体无分支）。
    const uint64_t tableAlign = reinterpret_cast<uint64_t>(outputGm) & 15;
    const int64_t tid = blockIdx * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockNum * blockDim.x;

    // 奇桶长 + 小 dim（3/4）时每桶写数太少，宽写的收益抵不过两路分化的开销，
    // 该区间回退标量逐字段写（原始实现），其余区间走宽存储。
    if ((bucketLength & 1) != 0 && embeddingDim <= 4) {
        for (int64_t i = tid; i < bucketSize; i += stride) {
            const int64_t bucketBase = bucketLength * i * INT64_PER_BYTE;
            *reinterpret_cast<__gm__ int64_t*>(outputGm + bucketBase + KEY_OFFSET * INT64_PER_BYTE) = DEFAULT_KEY;
            *reinterpret_cast<__gm__ int64_t*>(outputGm + bucketBase +
                                               COUNTER_OFFSET * INT64_PER_BYTE) = DEFAULT_COUNTER;
            *reinterpret_cast<__gm__ uint64_t*>(outputGm + bucketBase + FLAG_OFFSET * INT64_PER_BYTE) = DEFAULT_FLAG;
            __gm__ Tvalue* values = reinterpret_cast<__gm__ Tvalue*>(outputGm + bucketBase +
                                                                     VALUES_OFFSET * INT64_PER_BYTE);
            for (int64_t j = 0; j < embeddingDim; ++j) {
                values[j] = initializerMode == RANDOM_MOD ? sampledValuesGm[i * embeddingDim + j] : constantValue;
            }
        }
        return;
    }

    const uint64_t myHead = (16 - (tableAlign + (((bucketLength & 1) && (tid & 1)) ? 8ULL : 0ULL))) & 15;
    for (int64_t i = tid; i < bucketSize; i += stride) {
        InitBucket(outputGm + (bucketLength * i) * INT64_PER_BYTE, myHead, i, embeddingDim, initializerMode,
                   constantValue, sampledValuesGm);
    }
}

template <typename Tkey, typename Tvalue>
class KernelInitEmbeddingHashTable {
public:
    __aicore__ inline KernelInitEmbeddingHashTable(){};

    __aicore__ inline void Init(GM_ADDR tableHandle, GM_ADDR sampledValues, int64_t bucketSize, int64_t embeddingDim,
                                int64_t initializerMode, Tvalue constantValue, int64_t bucketLength,
                                uint32_t useThreadNum)
    {
        tableHanldeGm.SetGlobalBuffer((__gm__ int64_t*)(tableHandle));
        sampledValuesGm.SetGlobalBuffer((__gm__ Tvalue*)(sampledValues));
        __gm__ int64_t* table = reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHanldeGm(0)));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(*table));
        this->initializerMode = initializerMode;
        this->bucketSize = bucketSize;
        this->embeddingDim = embeddingDim;
        this->constantValue = constantValue;
        this->bucketLength = bucketLength;
        this->useThreadNum = useThreadNum;

        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
    }
    __aicore__ inline void Process()
    {
        asc_vf_call<InitCompute<Tkey, Tvalue>>(dim3{static_cast<uint32_t>(useThreadNum)}, embeddingDim, bucketSize,
                                               bucketLength, initializerMode, constantValue, blockIdx, blockNum,
                                               tableHanldeGm.GetPhyAddr(0), sampledValuesGm.GetPhyAddr(0),
                                               outputGm.GetPhyAddr(0));
    }

private:
    AscendC::GlobalTensor<int64_t> tableHanldeGm;
    AscendC::GlobalTensor<Tvalue> sampledValuesGm;
    AscendC::GlobalTensor<uint8_t> outputGm;

    int64_t embeddingDim{1};
    int64_t bucketSize{1};
    int64_t bucketLength{0};
    int64_t initializerMode{0};
    Tvalue constantValue;

    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t useThreadNum;
};

} // namespace InitEmbeddingHashTable

#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_INIT_EMBEDDING_HASHTABLE_INIT_EMBEDDING_HASH_TABLE_H
