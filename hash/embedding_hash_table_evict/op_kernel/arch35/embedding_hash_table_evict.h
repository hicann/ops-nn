/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_evict.h
 * \brief embedding_hash_table_evict
 */

#pragma once

#include "kernel_operator.h"
#include "../../inc/hashtable_common.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"

namespace Hashtbl {
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 128;
#else
constexpr uint32_t THREAD_NUM = 512;
#endif
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;

constexpr uint32_t FLOAT_TYPE_BYTES = 4;
constexpr uint32_t INT64_TYPE_BYTES = 8;

constexpr int64_t HANDLE_SIZE_ALL_OFFSET = 2;
constexpr int64_t HANDLE_SIZE_ALL_NOEXPORT_OFFSET = 4;

constexpr int64_t BUCKET_COUNT_OFFSET = 8;
constexpr int64_t BUCKET_STATE_OFFSET = 16;
constexpr int64_t BUCKET_FLAG_OFFSET = 20;
constexpr int64_t BUCKET_VALUE_OFFSET = sizeof(int64_t) * 3;

constexpr int32_t EVICTED_FLAG_MASK = 1 << 3;

constexpr int64_t INIT_MODE_CONST = 0;
constexpr int64_t INIT_MODE_RANDOM = 1;

class KernelEvict {
public:
    __aicore__ inline KernelEvict(){};

    __aicore__ inline void Init(GM_ADDR tableHandle, GM_ADDR keys, GM_ADDR sampledValues,
                                const EvictTilingData tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t ROUND_UP8(const uint32_t x) const;

private:
    uint32_t blockId_;
    uint32_t blockNum_;

    int64_t tableCap_;
    int64_t embeddingDim_;
    int64_t initMode_;
    float constVal_;
    uint32_t keyNum_;
    uint32_t bucketSize_;

    __gm__ int64_t* tableHandleAddr_;

    AscendC::GlobalTensor<int64_t> tableHandle_;
    AscendC::GlobalTensor<int8_t> table_;
    AscendC::GlobalTensor<int64_t> keys_;
    AscendC::GlobalTensor<float> sampledValues_;
};

__aicore__ inline void KernelEvict::Init(GM_ADDR tableHandle, GM_ADDR keys, GM_ADDR sampledValues,
                                         const EvictTilingData tilingData)
{
    blockId_ = AscendC::GetBlockIdx();
    blockNum_ = AscendC::GetBlockNum();

    tableCap_ = tilingData.tableCap;
    embeddingDim_ = tilingData.embeddingDim;
    initMode_ = tilingData.initMode;
    constVal_ = tilingData.constVal;
    keyNum_ = tilingData.keyNum;
    bucketSize_ = ROUND_UP8(BUCKET_VALUE_OFFSET + embeddingDim_ * FLOAT_TYPE_BYTES);

    tableHandle_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(tableHandle));
    keys_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(keys));
    sampledValues_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sampledValues));

    tableHandleAddr_ = reinterpret_cast<__gm__ int64_t*>(reinterpret_cast<__gm__ uint8_t*>(tableHandle_(0)));
    table_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(*tableHandleAddr_));
}

__aicore__ inline uint32_t KernelEvict::ROUND_UP8(const uint32_t x) const
{
    constexpr uint32_t ROUND_SIZE = 8;
    if (x % ROUND_SIZE != 0) {
        return (x / ROUND_SIZE + 1) * ROUND_SIZE;
    }
    return x;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void EvictCompute(
    uint32_t blockId, uint32_t blockNum, int64_t tableCap, int64_t embeddingDim, int64_t initMode, float constVal,
    uint32_t keyNum, uint32_t bucketSize, __gm__ int64_t* tableHandleAddr, __gm__ int8_t* table, __gm__ int64_t* keys,
    __gm__ float* sampledValues)
{
    for (auto i = blockId * blockDim.x + threadIdx.x; i < keyNum; i += blockNum * blockDim.x) {
        int64_t key = keys[i];
        uint32_t hashValue = MurmurHash3(keys + i, INT64_TYPE_BYTES, 0);

        uint64_t currIdx = hashValue % tableCap;
        uint64_t tableOffset = currIdx * bucketSize;

        uint32_t counter = 0;
        while (counter < tableCap) {
            auto keyInTable = *reinterpret_cast<__gm__ volatile int64_t*>(table + tableOffset);
            if (keyInTable == key) {
                break;
            }

            currIdx = (currIdx == (tableCap - 1)) ? 0 : currIdx + 1;
            tableOffset = currIdx * bucketSize;
            counter += 1;
        }
        if (counter >= tableCap) {
            continue;
        }

        auto bucketFlag = *reinterpret_cast<__gm__ volatile int32_t*>(table + tableOffset + BUCKET_FLAG_OFFSET);
        if ((bucketFlag & EVICTED_FLAG_MASK) != 0) {
            continue;
        }

        *reinterpret_cast<__gm__ volatile int32_t*>(table + tableOffset + BUCKET_FLAG_OFFSET) = bucketFlag ^
                                                                                                EVICTED_FLAG_MASK;
        *reinterpret_cast<__gm__ volatile int64_t*>(table + tableOffset + BUCKET_COUNT_OFFSET) = 0;

        asc_atomic_sub(tableHandleAddr + HANDLE_SIZE_ALL_OFFSET, static_cast<int64_t>(1));
        asc_atomic_sub(tableHandleAddr + HANDLE_SIZE_ALL_NOEXPORT_OFFSET, static_cast<int64_t>(1));

        for (auto j = 0; j < embeddingDim; j++) {
            auto valPtr = reinterpret_cast<__gm__ volatile float*>(table + tableOffset + BUCKET_VALUE_OFFSET +
                                                                   j * FLOAT_TYPE_BYTES);
            if (initMode == INIT_MODE_CONST) {
                *valPtr = constVal;
            } else {
                *valPtr = sampledValues[i * embeddingDim + j];
            }
        }
    }
}

__aicore__ inline void KernelEvict::Process()
{
    asc_vf_call<EvictCompute>(dim3{static_cast<uint32_t>(THREAD_NUM)}, blockId_, blockNum_, tableCap_, embeddingDim_,
                              initMode_, constVal_, keyNum_, bucketSize_, tableHandleAddr_, table_.GetPhyAddr(0),
                              keys_.GetPhyAddr(0), sampledValues_.GetPhyAddr(0));
}

} // namespace Hashtbl
