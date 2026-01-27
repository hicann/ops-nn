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
 * \file kernel_lookup_or_insert_general.h
 * \brief kernel_lookup_or_insert_general
 */
#ifndef __KERNEL_LOOKUP_OR_INSERT_GENERAL_H__
#define __KERNEL_LOOKUP_OR_INSERT_GENERAL_H__

#include "lookup_or_insert_base.h"

namespace Hashtbl {
using namespace AscendC;

template <bool WITH_FILTERING_LOGIC = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) void ComputeLookupOrInsert(
    uint32_t blockIdx, uint32_t blockNum, size_t bucketSize, int64_t tableSize, int64_t embeddingDim, int64_t keyNum,
    uint32_t defaultKeyOrValue, int64_t defaultKey, float defaultValue, int64_t filterKey, __gm__ int64_t* pTableHandle,
    __gm__ uint8_t* pTable, __gm__ int64_t* pKeys, __gm__ float* pValues, __ubuf__ int64_t* pThreadInsertCounts)
{
    // 每core线程划分为(x,y)，每threadXNum个x对应1个y，共启动threadXNum*threadYNum个线程
    uint32_t threadXIdx = static_cast<uint32_t>(Simt::GetThreadIdx<0>());
    uint32_t threadYIdx = static_cast<uint32_t>(Simt::GetThreadIdx<1>());
    uint32_t threadXNum = static_cast<uint32_t>(Simt::GetThreadNum<0>());
    uint32_t threadYNum = static_cast<uint32_t>(Simt::GetThreadNum<1>());

    int64_t insertCounts = 0; // 各线程自有变量，记录insert的次数
    for (uint32_t i = threadYIdx + blockIdx * threadYNum; i < keyNum; i += blockNum * threadYNum) {
        int64_t insertKey = pKeys[i];
        if constexpr (WITH_FILTERING_LOGIC) {
            if (insertKey == filterKey) {
                if (defaultKeyOrValue == 0) {
                    for (size_t j = threadXIdx; j < embeddingDim; j += threadXNum) {
                        pValues[i * embeddingDim + j] = defaultValue; // 写回i行j列
                    }
                    continue;
                } else {
                    if (threadXIdx == 0) {
                        pKeys[i] = defaultKey;
                    }
                    insertKey = defaultKey;
                }
            }
        }

        size_t currIdx = 0;
        __gm__ uint8_t* pCurrBucket = nullptr;
        bool succ = false;
        size_t detectCounts = 0;
        if (threadXIdx == 0) {
            // 一组threadX里只有第一条线程执行控制查找操作
            currIdx = static_cast<size_t>(MurmurHash3(pKeys + i, sizeof(int64_t), 0) % tableSize);
            pCurrBucket = pTable + currIdx * bucketSize;
            while (detectCounts < tableSize) {
                detectCounts++;

                // 由于AtmoicCas限制，用int32来cas第20~23字节的BIG_ENDIAN_ONE那个位置
                const int32_t casOrigFlag = AscendC::Simt::AtomicCas(
                    reinterpret_cast<__gm__ int32_t*>(pCurrBucket + TABLE_FLAG_OFFSET_FOR_B32), static_cast<int32_t>(0),
                    BIG_ENDIAN_ONE);

                if (casOrigFlag == 0) {
                    // 可以插入
                    *reinterpret_cast<__gm__ int64_t*>(pCurrBucket) = insertKey;
                    Simt::ThreadFence();
                    *reinterpret_cast<__gm__ int32_t*>(pCurrBucket + TABLE_STATE_OFFSET) = 1;
                    succ = true;
                    insertCounts++;
                    break;
                } else {
                    while (*reinterpret_cast<__gm__ volatile int32_t*>(pCurrBucket + TABLE_STATE_OFFSET) != 1) {
                        // 自旋等待casOrigFlag==0的分支解除占用
                    }
                    int64_t currentKey = *reinterpret_cast<__gm__ volatile int64_t*>(pCurrBucket);
                    if (currentKey == insertKey) {
                        // 可以查到
                        succ = true;

                        // 处理evict调用后的逻辑，这块与evict算子的逻辑相照应
                        auto currFlag =
                            *reinterpret_cast<__gm__ volatile int32_t*>(pCurrBucket + TABLE_FLAG_OFFSET_FOR_B32);
                        if ((currFlag & EVICTED_FLAG_MASK) != 0) {
                            auto newFlag = currFlag ^ EVICTED_FLAG_MASK;
                            auto oldFlag = Simt::AtomicCas(
                                reinterpret_cast<__gm__ int32_t*>(pCurrBucket + TABLE_FLAG_OFFSET_FOR_B32),
                                static_cast<int32_t>(currFlag), newFlag);
                            if ((oldFlag & EVICTED_FLAG_MASK) != 0) {
                                insertCounts++;
                            }
                        }

                        break;
                    }
                }
                currIdx = (currIdx + 1) % tableSize;
                pCurrBucket = pTable + currIdx * bucketSize;
            } // while循环
        }     // if控制线程

        succ = __shfl(succ, 0, static_cast<int>(threadXNum)); // 从控制线程取succ值
        if (succ) {
            // 从控制线程取待返回的bucket的地址
            currIdx = __shfl(currIdx, 0, static_cast<int>(threadXNum));
            pCurrBucket = pTable + currIdx * bucketSize;
            if (threadXIdx == 0) {
                //  由控制线程来执行bucket的counter++操作
                Simt::AtomicAdd(
                    reinterpret_cast<__gm__ int64_t*>(pCurrBucket + COUNTER_OFFSET), static_cast<int64_t>(1));
            }
            for (size_t j = threadXIdx; j < embeddingDim; j += threadXNum) {
                __gm__ float* pCurrValue =
                    reinterpret_cast<__gm__ float*>(pCurrBucket + VALUES_OFFSET) + j; // 读取pCurrBucket的j列
                pValues[i * embeddingDim + j] = *pCurrValue;                          // 写回i行j列
            }
        }
    } // threadY的for循环

    // 把当前Y线程的insertCounts记录到UB里对应的位置去
    if (threadXIdx == 0) {
        pThreadInsertCounts[threadYIdx] = insertCounts;
    }
}

class KernelLookupOrInsertGeneral : public KernelLookupOrInsertBase {
public:
    __aicore__ KernelLookupOrInsertGeneral(TPipe* pipe) : KernelLookupOrInsertBase(pipe)
    {}

    __aicore__ void Process()
    {
        LocalTensor<int64_t> threadInsertCountsLocal = threadInsertCountsBuf_.Get<int64_t>();
        __ubuf__ int64_t* pThreadInsertCounts =
            reinterpret_cast<__ubuf__ int64_t*>(threadInsertCountsLocal.GetPhyAddr());

        if (filterKeyFlag_) {
            Simt::VF_CALL<ComputeLookupOrInsert<true>>(
                Simt::Dim3{threadXNum_, threadYNum_}, blockIdx_, blockNum_, bucketSize_, tableSize_, embeddingDim_,
                keyNum_, defaultKeyOrValue_, defaultKey_, defaultValue_, filterKey_, pTableHandle_, pTable_, pKeys_,
                pValues_, pThreadInsertCounts);
        } else {
            Simt::VF_CALL<ComputeLookupOrInsert<false>>(
                Simt::Dim3{threadXNum_, threadYNum_}, blockIdx_, blockNum_, bucketSize_, tableSize_, embeddingDim_,
                keyNum_, defaultKeyOrValue_, defaultKey_, defaultValue_, filterKey_, pTableHandle_, pTable_, pKeys_,
                pValues_, pThreadInsertCounts);
        }

        // SIMD汇总写回tableHandle的那几个统计字段的值
        uint16_t vfLoopNum = static_cast<uint16_t>(Ops::Base::CeilDiv<uint32_t>(threadYNum_, VL_FOR_B64));
        VF_CALL<ComputeInplaceReduceSumB64>(pThreadInsertCounts, VL_FOR_B64, threadYNum_, vfLoopNum);
        SetWaitFlag<HardEvent::V_S>();
        int64_t insertCounts = threadInsertCountsLocal.GetValue(0);
        if (insertCounts > 0) {
            AtomicAdd<int64_t>(pTableHandle_ + HANDLE_SIZE_ALL_OFFSET, insertCounts);
            AtomicAdd<int64_t>(pTableHandle_ + HANDLE_SIZE_ALL_NOEXPORT_OFFSET, insertCounts);
        }
    }
};

} // namespace Hashtbl

#endif // __KERNEL_LOOKUP_OR_INSERT_GENERAL_H__
