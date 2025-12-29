/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_lookup_or_insert.h
 * \brief embedding_hash_table_lookup_or_insert
 */

#ifndef ASCENDC_EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_H_
#define ASCENDC_EMBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_H_

#include "kernel_operator.h"
#include "../../inc/hashtable_common.h"

namespace Hashtbl {
using namespace AscendC;

constexpr uint32_t FLOAT_TYPR_BYTES = 4;
constexpr uint32_t INT64_TYPE_BYTES = 8;
constexpr int64_t INT64_ONE = 1;
constexpr int64_t HANDLE_SIZE_ALL_OFFSET = 2;
constexpr int64_t HANDLE_SIZE_ALL_NOEXPORT_OFFSET = 4;
constexpr int64_t TABLE_FLAG_OFFSET = 20;
constexpr int64_t TABLE_STATE_OFFSET = 16;
constexpr int32_t BIG_ENDIAN_ONE = 16777216;
constexpr int32_t EVICTED_FLAG_MASK = 1 << 3;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
constexpr int64_t VALUE_OFFSET = sizeof(int64_t) * 3; // key + counter + flags

/* *
 * current Bucket contains: int64_t key, int64_t count, int64_t flags, float value[]; // value 8Byte align
 */

class KernelLookupOrInsert
{
public:
    __aicore__ inline KernelLookupOrInsert(){};
    __aicore__ inline void Init(
        GM_ADDR tableHandle, GM_ADDR keys, GM_ADDR values, const LookupOtInsertTilingData tilingData);
    __aicore__ inline void Process();
    __aicore__ inline uint32_t ROUND_UP8(const uint32_t x) const;

private:
    AscendC::GlobalTensor<int64_t> tableHandle_;
    AscendC::GlobalTensor<int64_t> keys_;
    AscendC::GlobalTensor<float> values_;
    AscendC::GlobalTensor<int8_t> table_;
    __gm__ int64_t* tableHandleAddr_;
    uint32_t tableSize_{1};
    uint32_t embeddingDim_{1};
    uint32_t filterFreq_{0};
    uint32_t keyNum_{1};
    int64_t unusedKey{-1};
    uint32_t filterMode_{0};
    uint32_t bucketSize_{0};
    int64_t defaultKey_{0};
    float defaultValue_{0};
    uint32_t defaultKeyOrValue_{0};
    uint32_t filterKeyFlag_{0};
    int64_t filterKey_{-1};

    uint32_t blockId_;
    uint32_t blockNums_;
};

__aicore__ inline void KernelLookupOrInsert::Init(
    GM_ADDR tableHandle, GM_ADDR keys, GM_ADDR values, const LookupOtInsertTilingData tilingData)
{
    tableHandle_.SetGlobalBuffer((__gm__ int64_t*)(tableHandle));
    keys_.SetGlobalBuffer((__gm__ int64_t*)(keys));
    values_.SetGlobalBuffer((__gm__ float*)(values));
    tableHandleAddr_ = reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHandle_(0)));
    table_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(*tableHandleAddr_));

    tableSize_ = tilingData.size;
    embeddingDim_ = tilingData.embeddingDim;
    filterFreq_ = tilingData.filterFreq;
    keyNum_ = tilingData.keyNum;
    filterMode_ = tilingData.filterMode;
    defaultKey_ = tilingData.defaultKey;
    defaultValue_ = tilingData.defaultValue;
    defaultKeyOrValue_ = tilingData.defaultKeyOrValue;
    filterKeyFlag_ = tilingData.filterKeyFlag;
    filterKey_ = tilingData.filterKey;
    bucketSize_ = ROUND_UP8(VALUE_OFFSET + embeddingDim_ * FLOAT_TYPR_BYTES);

    this->blockId_ = GetBlockIdx();
    this->blockNums_ = GetBlockNum();
}

__aicore__ inline uint32_t KernelLookupOrInsert::ROUND_UP8(const uint32_t x) const
{
    constexpr uint32_t ROUND_SIZE = 8;
    if (x % ROUND_SIZE != 0) {
        return (x / ROUND_SIZE + 1) * ROUND_SIZE;
    }
    return x;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void GetInsertCompute(
    uint32_t tableSize, int32_t embeddingDim, uint32_t bucketSize, int64_t filterFreq, int64_t defaultKey,
    float defaultValue, uint32_t defaultKeyOrValue, uint32_t filterKeyFlag, int64_t filterKey, uint32_t blockId,
    uint32_t blockNums, uint32_t keyNum, __gm__ int64_t* tableHandleAddr, __gm__ int64_t* keys, __gm__ float* values,
    __gm__ int8_t* table)
{
    for (uint32_t i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < keyNum;
         i = i + blockNums * Simt::GetThreadNum()) {
        int64_t insertKey = keys[i];
        if ((filterKeyFlag != 0 && (filterKey == insertKey))) {
            if (defaultKeyOrValue == 0) { // return defaultValue
                for (uint32_t j = 0; j < embeddingDim; ++j) {
                    values[i * embeddingDim + j] = defaultValue;
                }
                continue;
            }
            keys[i] = defaultKey;
            insertKey = defaultKey; // return defaultKey -> value
        }

        uint32_t hashValue = MurmurHash3(keys + i, INT64_TYPE_BYTES, 0);

        uint64_t currentIdx = hashValue % tableSize;
        uint64_t tableOffset = currentIdx * bucketSize;

        bool insertSuccess = false;
        uint32_t counter = 0;
        while (false == insertSuccess) {
            if (counter++ >= tableSize) {
                break;
            }
            const int32_t originalFlag = Simt::AtomicCas(
                reinterpret_cast<__gm__ int32_t*>(table + tableOffset + TABLE_FLAG_OFFSET), static_cast<int32_t>(0),
                BIG_ENDIAN_ONE);

            if (0 == originalFlag) {
                *reinterpret_cast<__gm__ int64_t*>(table + tableOffset) = insertKey;
                Simt::ThreadFence();
                *reinterpret_cast<__gm__ int32_t*>(table + tableOffset + TABLE_STATE_OFFSET) = 1;
                insertSuccess = true;
                Simt::AtomicAdd(tableHandleAddr + HANDLE_SIZE_ALL_OFFSET, INT64_ONE);
                Simt::AtomicAdd(tableHandleAddr + HANDLE_SIZE_ALL_NOEXPORT_OFFSET, INT64_ONE);
                break;
            } else {
                while (*reinterpret_cast<__gm__ volatile int32_t*>(table + tableOffset + TABLE_STATE_OFFSET) != 1) {
                }
                int64_t currentKey = *reinterpret_cast<__gm__ volatile int64_t*>(table + tableOffset);
                if (currentKey == insertKey) {
                    auto flag = *reinterpret_cast<__gm__ volatile int32_t*>(table + tableOffset + TABLE_FLAG_OFFSET);

                    if ((flag & EVICTED_FLAG_MASK) != 0) {
                        auto newFlag = flag ^ EVICTED_FLAG_MASK;
                        auto oldFlag = Simt::AtomicCas(
                            reinterpret_cast<__gm__ int32_t*>(table + tableOffset + TABLE_FLAG_OFFSET),
                            static_cast<int32_t>(flag), newFlag);

                        if ((oldFlag & EVICTED_FLAG_MASK) != 0) {
                            Simt::AtomicAdd(tableHandleAddr + HANDLE_SIZE_ALL_OFFSET, INT64_ONE);
                            Simt::AtomicAdd(tableHandleAddr + HANDLE_SIZE_ALL_NOEXPORT_OFFSET, INT64_ONE);
                        }
                    }

                    insertSuccess = true;
                    break;
                }
            }

            if (currentIdx == tableSize - 1) {
                currentIdx = 0;
            } else {
                ++currentIdx;
            }
            tableOffset = currentIdx * bucketSize;
        }
        if (false == insertSuccess) {
            // trap error?  table is full
            break;
        }
        Simt::AtomicAdd(reinterpret_cast<__gm__ int64_t*>(table + tableOffset) + 1, INT64_ONE);
        for (uint32_t j = 0; j < embeddingDim; ++j) {
            __gm__ float* tableValue =
                reinterpret_cast<__gm__ float*>(table + tableOffset + VALUE_OFFSET + j * FLOAT_TYPR_BYTES);
            values[i * embeddingDim + j] = *tableValue;
        }
    }
}

__aicore__ inline void KernelLookupOrInsert::Process()
{
    Simt::VF_CALL<GetInsertCompute>(
        Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)}, tableSize_, embeddingDim_, bucketSize_, filterFreq_, defaultKey_,
        defaultValue_, defaultKeyOrValue_, filterKeyFlag_, filterKey_, blockId_, blockNums_, keyNum_, tableHandleAddr_,
        keys_.GetPhyAddr(0), values_.GetPhyAddr(0), table_.GetPhyAddr(0));
}
} // namespace Hashtbl

#endif // ASCENDC_MBEDDING_HASH_TABLE_LOOKUP_OR_INSERT_H_
