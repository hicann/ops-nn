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
 * \file embedding_hash_table_import.h
 * \brief
 */
#ifndef ASCENDC_EMBEDDING_HASH_TABLE_IMPORT_H_
#define ASCENDC_EMBEDDING_HASH_TABLE_IMPORT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../inc/hashtable_common.h"

namespace EmbeddingHashTable {
using namespace AscendC;

constexpr int64_t TABLE_FLAG_OFFSET = 20;
constexpr int64_t TABLE_STATE_OFFSET = 16;
constexpr int32_t BIG_ENDIAN_ONE = 16777216;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;

constexpr int64_t INT64_TYPE_BYTES = 8;
constexpr int64_t ROUND_UP_BYTE8 = 8;
constexpr int64_t ARRAY_SIZE = 10;

// 参数offset偏移
constexpr int64_t HANDLE_SIZE_ALL_OFFSET = 2;
constexpr int64_t HANDLE_SIZE_ALL_NOEXPORT_OFFSET = 4;
constexpr int64_t INT64_ONE = 1;

constexpr int64_t KEY_OFFSET = 0;
constexpr int64_t COUNTER_OFFSET = 1;
constexpr int64_t FLAG_OFFSET = 2;
constexpr int64_t VALUE_OFFSET = 3;
constexpr uint8_t FILTER_FLAG_MASK = 0b00000101;
constexpr uint8_t EXPORT_FLAG_MASK = 0b00000100;

template <typename T>
class EmbeddingHashTableImport
{
public:
    __aicore__ inline EmbeddingHashTableImport(){};
    __aicore__ inline void Init(
        GM_ADDR tableHandles, GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys, GM_ADDR counters,
        GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace,
        const EmbeddingHashTableImportTilingData* __restrict tilingData);
    __aicore__ inline void ParseTilingData(const EmbeddingHashTableImportTilingData* __restrict tilingData);
    __aicore__ inline int64_t AlignUpByte8(const int64_t x) const;
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<int64_t> tableHandlesGm_;
    AscendC::GlobalTensor<int64_t> embeddingDimsGm_;
    AscendC::GlobalTensor<int64_t> bucketSizesGm_;
    AscendC::GlobalTensor<int64_t> keyGm_;
    AscendC::GlobalTensor<uint64_t> counterGm_;
    AscendC::GlobalTensor<uint8_t> filterFlagGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<uint8_t> tableGm_;
    AscendC::GlobalTensor<int64_t> tableHandleGm_;

    AscendC::TensorDesc<int64_t> keysDesc_;
    AscendC::TensorDesc<uint64_t> countersDesc_;
    AscendC::TensorDesc<uint64_t> filterFlagsDesc_;
    AscendC::TensorDesc<T> valuesDesc_;

    ListTensorDesc keysListGm_;
    ListTensorDesc countersListGm_;
    ListTensorDesc filterFlagsListGm_;
    ListTensorDesc valuesListGm_;

    int64_t embeddingDim_{0};
    int64_t blockSize_{0};
    int64_t bucketSize_{0};
    int64_t tableOffset_{0};

    int64_t bitWidth_{1};
    int64_t tablesCount_{1};
    int64_t unusedKey_{-1};
    int64_t blockIdx_{0};
    int64_t blockNum_{0};

    uint64_t buf_[ARRAY_SIZE];
};

template <typename T>
__aicore__ inline int64_t EmbeddingHashTableImport<T>::AlignUpByte8(const int64_t x) const
{
    return (x + ROUND_UP_BYTE8 - 1) / ROUND_UP_BYTE8 * ROUND_UP_BYTE8;
}

template <typename T>
__aicore__ inline void EmbeddingHashTableImport<T>::ParseTilingData(
    const EmbeddingHashTableImportTilingData* __restrict tilingData)
{
    tablesCount_ = tilingData->tablesCount;
    bitWidth_ = tilingData->bitWidth;
    blockNum_ = tilingData->blockNum;
}

template <typename T>
__aicore__ inline void EmbeddingHashTableImport<T>::Init(
    GM_ADDR tableHandles, GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys, GM_ADDR counters,
    GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace,
    const EmbeddingHashTableImportTilingData* __restrict tilingData)
{
    // parse tiling
    ParseTilingData(tilingData);

    // SetGlobalBuffer
    tableHandlesGm_.SetGlobalBuffer((__gm__ int64_t*)(tableHandles));
    embeddingDimsGm_.SetGlobalBuffer((__gm__ int64_t*)(embeddingDims));
    bucketSizesGm_.SetGlobalBuffer((__gm__ int64_t*)(bucketSizes));

    // ListTensorDesc
    keysListGm_ = ListTensorDesc(
        reinterpret_cast<__gm__ void*>(reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(keys))));
    countersListGm_ = ListTensorDesc(
        reinterpret_cast<__gm__ void*>(reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(counters))));
    filterFlagsListGm_ = ListTensorDesc(
        reinterpret_cast<__gm__ void*>(
            reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(filterFlags))));
    valuesListGm_ = ListTensorDesc(
        reinterpret_cast<__gm__ void*>(reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(values))));

    // SetShapeAddr
    keysDesc_.SetShapeAddr(&buf_[0]);
    countersDesc_.SetShapeAddr(&buf_[0]);
    filterFlagsDesc_.SetShapeAddr(&buf_[0]);
    valuesDesc_.SetShapeAddr(&buf_[0]);

    // base param
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SingleTableImportCompute(
    int64_t tableIdx, int64_t keyNum, int64_t embeddingDim, int64_t blockSize, int64_t bucketSize, int64_t bitWidth,
    int64_t unusedKey, int64_t blockIdx, int64_t blockNum, __gm__ int64_t* tableHandlesGm, __gm__ int64_t* keyGm,
    __gm__ uint64_t* counterGm, __gm__ uint8_t* filterFlagGm, __gm__ T* valueGm, __gm__ uint8_t* tableGm)
{   
    __gm__ int64_t* tableHandle =
            reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHandlesGm[tableIdx]));
    for (int64_t i = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < keyNum;
         i = i + blockNum * Simt::GetThreadNum()) {
        int64_t insertKey = keyGm[i];
        uint32_t hashValue = Hashtbl::MurmurHash3(keyGm + i, INT64_TYPE_BYTES, 0);
        int64_t hashTabIdx = hashValue % bucketSize;
        int64_t blockOffset = hashTabIdx * blockSize;

        bool isInsertSucc = false;
        bool isNewKey = false;
        uint32_t counter = 0;
        while (false == isInsertSucc) {
            if (counter++ >= bucketSize) {
                break;
            }
            // 插入key值序列
            const int32_t originalFlag = Simt::AtomicCas(
                reinterpret_cast<__gm__ int32_t*>(tableGm + blockOffset + TABLE_FLAG_OFFSET), static_cast<int32_t>(0),
                BIG_ENDIAN_ONE);

            int64_t keyOffset = blockOffset + (KEY_OFFSET * INT64_TYPE_BYTES);
            if (0 == originalFlag) {
                *reinterpret_cast<__gm__ int64_t*>(tableGm + keyOffset) = insertKey;
                Simt::ThreadFence();
                *reinterpret_cast<__gm__ int32_t*>(tableGm + keyOffset + TABLE_STATE_OFFSET) = 1;
                isInsertSucc = true;
                isNewKey = true;
                break;
            } else {
                while (*reinterpret_cast<__gm__ volatile int32_t*>(tableGm + keyOffset + TABLE_STATE_OFFSET) != 1) {
                }
                int64_t currentKey = *reinterpret_cast<__gm__ volatile int64_t*>(tableGm + keyOffset);
                if (currentKey == insertKey) {
                    isInsertSucc = true;
                    break;
                }
            }

            if (hashTabIdx == bucketSize - 1) {
                hashTabIdx = 0;
            } else {
                ++hashTabIdx;
            }
            blockOffset = hashTabIdx * blockSize;
        }
        if (true == isInsertSucc) {
            // 更新tableHandle struct
            if (isNewKey) {
                // 刷新total_hash_addr地址 --> (key不存在， tablesize++, noexportsize不变)
                AscendC::Simt::AtomicAdd(tableHandle + HANDLE_SIZE_ALL_OFFSET, INT64_ONE);
            } else {
                // 刷新no_export_hash_addr地址 --> (key存在， tablesize不变, noexportsize--)
                int64_t flagOffset = blockOffset + ((FLAG_OFFSET + 1) * INT64_TYPE_BYTES - 1);
                __gm__ uint8_t* filterFlagValue = reinterpret_cast<__gm__ uint8_t*>(tableGm + flagOffset);
                if (!(*filterFlagValue & EXPORT_FLAG_MASK)) { // means change flag from 0 to 1(1 means cannot be exported)
                    AscendC::Simt::AtomicSub(tableHandle + HANDLE_SIZE_ALL_NOEXPORT_OFFSET, INT64_ONE); 
                }
            }
            // 插入counter值
            int64_t counterOffset = blockOffset + (COUNTER_OFFSET * INT64_TYPE_BYTES);
            __gm__ uint64_t* counterValue = reinterpret_cast<__gm__ uint64_t*>(tableGm + counterOffset);
            *counterValue = counterGm[i];

            // 插入flag值(valid_flag, filter_flag, export_flag)
            int64_t flagOffset = blockOffset + ((FLAG_OFFSET + 1) * INT64_TYPE_BYTES - 1);
            __gm__ uint8_t* filterFlagValue = reinterpret_cast<__gm__ uint8_t*>(tableGm + flagOffset);
            if (isInsertSucc) {
                *filterFlagValue |= FILTER_FLAG_MASK;
            } else {
                *filterFlagValue = filterFlagGm[i];   // actually not exist
            }

            // 插入value值
            for (int64_t j = 0; j < embeddingDim; ++j) {
                int64_t valueOffset = blockOffset + (VALUE_OFFSET * INT64_TYPE_BYTES) + (j * bitWidth);
                __gm__ T* tableValue = reinterpret_cast<__gm__ T*>(tableGm + valueOffset);
                *tableValue = valueGm[i * embeddingDim + j];
            }
        } // need consider else, which means table is full
    }
}

template <typename T>
__aicore__ inline void EmbeddingHashTableImport<T>::Process()
{
    for (int64_t idx = 0; idx < tablesCount_; idx++) {
        __gm__ int64_t* tableHandle =
            reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHandlesGm_.GetValue(idx)));
        tableHandleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tableHandle));
        __gm__ int64_t* table = reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHandleGm_(0)));
        tableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(table));

        embeddingDim_ = embeddingDimsGm_.GetValue(idx);
        bucketSize_ = bucketSizesGm_.GetValue(idx);

        // 单表中block大小
        blockSize_ = AlignUpByte8(VALUE_OFFSET * INT64_TYPE_BYTES + embeddingDim_ * bitWidth_);

        // 获取key的数量
        keysListGm_.GetDesc(keysDesc_, idx);
        int64_t keyNum = keysDesc_.GetShape(0);

        // 获取Tensorlist中tensor
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(keysListGm_.GetDataPtr<int64_t>(idx)));
        counterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(countersListGm_.GetDataPtr<int64_t>(idx)));
        filterFlagGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(filterFlagsListGm_.GetDataPtr<uint8_t>(idx)));
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(valuesListGm_.GetDataPtr<T>(idx)));

        Simt::VF_CALL<SingleTableImportCompute<T>>(
            Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)}, idx, keyNum, embeddingDim_, blockSize_, bucketSize_,
            bitWidth_, unusedKey_, blockIdx_, blockNum_, tableHandlesGm_.GetPhyAddr(0), keyGm_.GetPhyAddr(0),
            counterGm_.GetPhyAddr(0), filterFlagGm_.GetPhyAddr(0), valueGm_.GetPhyAddr(0), tableGm_.GetPhyAddr(0));
    }
}
} // namespace EmbeddingHashTable

#endif // ASCENDC_EMBEDDING_HASH_TABLE_IMPORT_H_