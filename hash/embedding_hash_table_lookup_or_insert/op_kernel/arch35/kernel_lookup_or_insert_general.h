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
#include "simt_api/vector_functions.h"

namespace Hashtbl {
using namespace AscendC;

// SIMT 访存合并（b64/b128 短向量）：按 merge 个连续 float 为一组拷贝，
// merge 由调用方按 dim 对齐给出（dim%4==0→4 / dim%2==0→2 / 否则→1，m 必整除 dim）。
// 桶 values 区偏移 24B、stride 仅 8B 对齐 → 读侧封顶 float2(B64)；
// pValues 行首 i*dim*4 在 dim%4==0 时 16B 对齐 → 写侧可用 float4(B128)。
template <int MERGE>
__simt_callee__ __aicore__ inline void CopyBucketValuesMerged(__gm__ uint8_t* pBucket, __gm__ float* pOutRow,
                                                              size_t embeddingDim, uint32_t threadXIdx,
                                                              uint32_t threadXNum)
{
    for (size_t j0 = threadXIdx * MERGE; j0 < embeddingDim; j0 += threadXNum * MERGE) {
        if constexpr (MERGE == 4) {
            __gm__ float2* pSrc = reinterpret_cast<__gm__ float2*>(pBucket + VALUES_OFFSET + j0 * sizeof(float));
            float2 lo = pSrc[0];
            float2 hi = pSrc[1];
            *reinterpret_cast<__gm__ float4*>(pOutRow + j0) = make_float4(lo.x, lo.y, hi.x, hi.y);
        } else if constexpr (MERGE == 2) {
            *reinterpret_cast<__gm__ float2*>(pOutRow + j0) = *(
                reinterpret_cast<__gm__ float2*>(pBucket + VALUES_OFFSET + j0 * sizeof(float)));
        } else {
            pOutRow[j0] = *reinterpret_cast<__gm__ float*>(pBucket + VALUES_OFFSET + j0 * sizeof(float));
        }
    }
}

template <int MERGE>
__simt_callee__ __aicore__ inline void FillValuesMerged(__gm__ float* pOutRow, size_t embeddingDim, float value,
                                                        uint32_t threadXIdx, uint32_t threadXNum)
{
    for (size_t j0 = threadXIdx * MERGE; j0 < embeddingDim; j0 += threadXNum * MERGE) {
        if constexpr (MERGE == 4) {
            *reinterpret_cast<__gm__ float4*>(pOutRow + j0) = make_float4(value, value, value, value);
        } else if constexpr (MERGE == 2) {
            *reinterpret_cast<__gm__ float2*>(pOutRow + j0) = make_float2(value, value);
        } else {
            pOutRow[j0] = value;
        }
    }
}

template <bool WITH_FILTERING_LOGIC = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) void ComputeLookupOrInsert(
    uint32_t blockIdx, uint32_t blockNum, size_t bucketSize, int64_t tableSize, int64_t embeddingDim, int64_t keyNum,
    uint32_t defaultKeyOrValue, int64_t defaultKey, float defaultValue, int64_t filterKey, __gm__ int64_t* pTableHandle,
    __gm__ uint8_t* pTable, __gm__ int64_t* pKeys, __gm__ float* pValues, __ubuf__ int64_t* pThreadInsertCounts)
{
    // 每core线程划分为(x,y)，每threadXNum个x对应1个y，共启动threadXNum*threadYNum个线程
    uint32_t threadXIdx = static_cast<uint32_t>(threadIdx.x);
    uint32_t threadYIdx = static_cast<uint32_t>(threadIdx.y);
    uint32_t threadXNum = static_cast<uint32_t>(blockDim.x);
    uint32_t threadYNum = static_cast<uint32_t>(blockDim.y);

    // 访存合并档位（与 tiling threadXNum 公式同源）：m 必整除 dim，无需尾部标量
    const uint32_t merge = (embeddingDim % 4 == 0) ? 4 : ((embeddingDim % 2 == 0) ? 2 : 1);

    int64_t insertCounts = 0; // 各线程自有变量，记录insert的次数
    for (uint32_t i = threadYIdx + blockIdx * threadYNum; i < keyNum; i += blockNum * threadYNum) {
        int64_t insertKey = pKeys[i];
        if constexpr (WITH_FILTERING_LOGIC) {
            if (insertKey == filterKey) {
                if (defaultKeyOrValue == 0) {
                    if (merge == 4) {
                        FillValuesMerged<4>(pValues + i * embeddingDim, embeddingDim, defaultValue, threadXIdx,
                                            threadXNum);
                    } else if (merge == 2) {
                        FillValuesMerged<2>(pValues + i * embeddingDim, embeddingDim, defaultValue, threadXIdx,
                                            threadXNum);
                    } else {
                        FillValuesMerged<1>(pValues + i * embeddingDim, embeddingDim, defaultValue, threadXIdx,
                                            threadXNum);
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
                const int32_t casOrigFlag = asc_atomic_cas(
                    reinterpret_cast<__gm__ int32_t*>(pCurrBucket + TABLE_FLAG_OFFSET_FOR_B32), static_cast<int32_t>(0),
                    BIG_ENDIAN_ONE);

                if (casOrigFlag == 0) {
                    // 可以插入
                    *reinterpret_cast<__gm__ int64_t*>(pCurrBucket) = insertKey;
                    __threadfence();
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
                        auto currFlag = *reinterpret_cast<__gm__ volatile int32_t*>(pCurrBucket +
                                                                                    TABLE_FLAG_OFFSET_FOR_B32);
                        if ((currFlag & EVICTED_FLAG_MASK) != 0) {
                            auto newFlag = currFlag ^ EVICTED_FLAG_MASK;
                            auto oldFlag = asc_atomic_cas(
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
        } // if控制线程

        succ = __shfl(succ, 0, static_cast<int>(threadXNum)); // 从控制线程取succ值
        if (succ) {
            // 从控制线程取待返回的bucket的地址
            currIdx = __shfl(currIdx, 0, static_cast<int>(threadXNum));
            pCurrBucket = pTable + currIdx * bucketSize;
            if (threadXIdx == 0) {
                //  由控制线程来执行bucket的counter++操作
                asc_atomic_add(reinterpret_cast<__gm__ int64_t*>(pCurrBucket + COUNTER_OFFSET),
                               static_cast<int64_t>(1));
            }
            if (merge == 4) {
                CopyBucketValuesMerged<4>(pCurrBucket, pValues + i * embeddingDim, embeddingDim, threadXIdx,
                                          threadXNum);
            } else if (merge == 2) {
                CopyBucketValuesMerged<2>(pCurrBucket, pValues + i * embeddingDim, embeddingDim, threadXIdx,
                                          threadXNum);
            } else {
                CopyBucketValuesMerged<1>(pCurrBucket, pValues + i * embeddingDim, embeddingDim, threadXIdx,
                                          threadXNum);
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
    __aicore__ KernelLookupOrInsertGeneral(TPipe* pipe) : KernelLookupOrInsertBase(pipe) {}

    __aicore__ void Process()
    {
        LocalTensor<int64_t> threadInsertCountsLocal = threadInsertCountsBuf_.Get<int64_t>();
        __ubuf__ int64_t* pThreadInsertCounts = reinterpret_cast<__ubuf__ int64_t*>(
            threadInsertCountsLocal.GetPhyAddr());

        if (filterKeyFlag_) {
            asc_vf_call<ComputeLookupOrInsert<true>>(dim3{threadXNum_, threadYNum_}, blockIdx_, blockNum_, bucketSize_,
                                                     tableSize_, embeddingDim_, keyNum_, defaultKeyOrValue_,
                                                     defaultKey_, defaultValue_, filterKey_, pTableHandle_, pTable_,
                                                     pKeys_, pValues_, pThreadInsertCounts);
        } else {
            asc_vf_call<ComputeLookupOrInsert<false>>(dim3{threadXNum_, threadYNum_}, blockIdx_, blockNum_, bucketSize_,
                                                      tableSize_, embeddingDim_, keyNum_, defaultKeyOrValue_,
                                                      defaultKey_, defaultValue_, filterKey_, pTableHandle_, pTable_,
                                                      pKeys_, pValues_, pThreadInsertCounts);
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
