/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

template <typename Tkey, typename Tvalue>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void InitCompute(
    int64_t embeddingDim, int64_t bucketSize, int64_t bucketLength, int64_t initializerMode, Tvalue constantValue,
    uint32_t blockIdx, uint32_t blockNum, __gm__ int64_t* tableHanldeGm, __gm__ Tvalue* sampledValuesGm,
    __gm__ uint8_t* outputGm)
{
    for (int64_t i = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < bucketSize;
         i += blockNum * Simt::GetThreadNum()) {
        // SetKey(-1)
        int64_t keyOffset = (bucketLength * i + KEY_OFFSET) * INT64_PER_BYTE;
        __gm__ int64_t* key = reinterpret_cast<__gm__ int64_t*>(outputGm + keyOffset);
        *key = DEFAULT_KEY;

        // SetCounter
        int64_t counterOffset = (bucketLength * i + COUNTER_OFFSET) * INT64_PER_BYTE;
        __gm__ int64_t* counter = reinterpret_cast<__gm__ int64_t*>(outputGm + counterOffset);
        *counter = DEFAULT_COUNTER;

        // SetFlag
        int64_t flagOffset = (bucketLength * i + FLAG_OFFSET) * INT64_PER_BYTE;
        __gm__ uint64_t* flag = reinterpret_cast<__gm__ uint64_t*>(outputGm + flagOffset);
        *flag = DEFAULT_FLAG;

        int64_t valueOffset = (bucketLength * i + VALUES_OFFSET) * INT64_PER_BYTE;
        for (int64_t j = 0; j < embeddingDim; ++j) {
            int64_t offset = valueOffset + j * FLOAT_PER_BYTE;
            __gm__ Tvalue* tableValue = reinterpret_cast<__gm__ Tvalue*>(outputGm + offset);
            if (initializerMode == RANDOM_MOD) {
                *tableValue = sampledValuesGm[i * embeddingDim + j];
            } else {
                *tableValue = constantValue;
            }
        }
    }
}

template <typename Tkey, typename Tvalue>
class KernelInitEmbeddingHashTable
{
public:
    __aicore__ inline KernelInitEmbeddingHashTable(){};

    __aicore__ inline void Init(
        GM_ADDR tableHandle, GM_ADDR sampledValues, int64_t bucketSize, int64_t embeddingDim, int64_t initializerMode,
        Tvalue constantValue, int64_t bucketLength, uint32_t useThreadNum)
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
        Simt::VF_CALL<InitCompute<Tkey, Tvalue>>(
            Simt::Dim3{static_cast<uint32_t>(useThreadNum)}, embeddingDim, bucketSize, bucketLength, initializerMode,
            constantValue, blockIdx, blockNum, tableHanldeGm.GetPhyAddr(0), sampledValuesGm.GetPhyAddr(0),
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
