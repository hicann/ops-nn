/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file embedding_hash_table_export.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_EXPORT_H_
#define EMBEDDING_HASH_TABLE_EXPORT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/vector_functions.h"

namespace EmbeddingHashTableExportAicore {
using namespace AscendC;

/* *
 * current Bucket contains: int64_t key, uint64_t count, uint8 flag, int64_t value[embeddingDims];
 */
constexpr int64_t SIMT_THREAD_LAUNCH_BOUND = 256;
constexpr int64_t BUFFER_LENGTH = 2;
constexpr int64_t KEY_FLAG_OFFSET_OF_BYTE = 23;
constexpr int64_t KEY_VALUE_OFFSET_OF_BYTE = 24;
constexpr int64_t BYTE_COUNT_8 = 8;
constexpr int64_t SIZE_ALL_NO_EXPORT_IDX = 4;

constexpr uint8_t EVICTED_FLAG_MASK = 0b00001000;
constexpr uint8_t EXPORT_FLAG_MASK = 0b00000100;
constexpr uint8_t FILTER_FLAG_MASK = 0b00000010;
constexpr uint8_t VALID_FLAG_MASK = 0b00000001;

// SIMT 访存合并（b64/b128 短向量，见 simt_api/vector_functions.h）：写侧按 MERGE 个连续
// float 为一组写，MERGE 由调用方按对齐给出（dim%4==0 且行基址 16B 对齐→4 / dim%2==0→2 /
// 否则→1，MERGE 必整除 dim 无尾部）。桶 values 区偏移 24B、仅 8B 对齐 → 读侧封顶
// float2(B64)（24B 桶头为跨算子契约，与 lookup/import 相同约束）。
template <int MERGE>
__simt_callee__ __aicore__ inline void CopyExportValuesMerged(__gm__ uint8_t* pBucketValues, __gm__ float* pDstRow,
                                                              int64_t embeddingDim)
{
    for (int64_t j0 = 0; j0 < embeddingDim; j0 += MERGE) {
        if constexpr (MERGE == 4) {
            __gm__ float2* pSrc = reinterpret_cast<__gm__ float2*>(pBucketValues + j0 * sizeof(float));
            float2 lo = pSrc[0];
            float2 hi = pSrc[1];
            *reinterpret_cast<__gm__ float4*>(pDstRow + j0) = make_float4(lo.x, lo.y, hi.x, hi.y);
        } else if constexpr (MERGE == 2) {
            *reinterpret_cast<__gm__ float2*>(pDstRow + j0) = *(
                reinterpret_cast<__gm__ float2*>(pBucketValues + j0 * sizeof(float)));
        } else {
            pDstRow[j0] = *reinterpret_cast<__gm__ float*>(pBucketValues + j0 * sizeof(float));
        }
    }
}

template <typename T>
class EmbeddingHashTableExport {
public:
    __aicore__ inline EmbeddingHashTableExport(){};
    __aicore__ inline void Init(GM_ADDR tableHandles, GM_ADDR tableSizes, GM_ADDR embeddingDims, GM_ADDR bucketSizes,
                                GM_ADDR keys, GM_ADDR counters, GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace,
                                EmbeddingHashTableExportTilingData tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void SingleTableCompute(int64_t tableIndex);

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ubBuf1_;
    TBuf<QuePosition::VECCALC> ubBuf2_;
    TBuf<QuePosition::VECCALC> ubBuf3_;

    int64_t tableNum_{1};
    int64_t exportMode_{0}; // 0: all export, 1: new export
    int64_t filteredExportFlag_{1};

    int64_t blockIdx_;

    int64_t maxCoreNum_;
    int64_t maxThreadNum_;

    int64_t usedCoreNum_;
    int64_t normalCoreProcessKeys_;
    int64_t tailCoreProcessKeys_;
    int64_t curCoreProcessKeys_;

    int64_t normalThreadProcessKeys_;
    int64_t usedThreadNum_;
    int64_t tailThreadProcessKeys_;

    GlobalTensor<int64_t> tableHandleStructGm_;
    GlobalTensor<int64_t> tableHandlesGm_;
    GlobalTensor<int64_t> embeddingDimsGm;
    GlobalTensor<int64_t> bucketSizesGm;

    GlobalTensor<int64_t> coreSyncWorkspaceGm_;
    LocalTensor<int64_t> threadCountKeysToExportUB_;
    LocalTensor<int64_t> threadCountReFreshExportFlagUB_;
    LocalTensor<int64_t> threadCountKeysToExportSumUB_;

    int64_t tableAddr_;
    int64_t embeddingDims_;
    int64_t keyWidthByte_;
    int64_t bucketSize_;

    ListTensorDesc keysList_;
    ListTensorDesc countersList_;
    ListTensorDesc filterFlagsList_;
    ListTensorDesc valuesList_;

    GlobalTensor<int64_t> outKeyGm_;
    GlobalTensor<uint64_t> outCounterGm_;
    GlobalTensor<uint8_t> outFilterFlagGm_;
    GlobalTensor<T> outValueGm_;

    int64_t keyWidthByteD8_;
    int64_t keyWidthByteDT_;
};

template <typename T>
__aicore__ inline void EmbeddingHashTableExport<T>::Init(GM_ADDR tableHandles, GM_ADDR tableSizes,
                                                         GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys,
                                                         GM_ADDR counters, GM_ADDR filterFlags, GM_ADDR values,
                                                         GM_ADDR workspace,
                                                         EmbeddingHashTableExportTilingData tilingData)
{
    blockIdx_ = GetBlockIdx();

    maxCoreNum_ = tilingData.maxCoreNum;
    maxThreadNum_ = tilingData.maxThreadNum;

    tableNum_ = tilingData.tableNum;
    exportMode_ = tilingData.exportMode;
    filteredExportFlag_ = tilingData.filteredExportFlag;

    tableHandlesGm_.SetGlobalBuffer((__gm__ int64_t*)tableHandles);
    embeddingDimsGm.SetGlobalBuffer((__gm__ int64_t*)embeddingDims);
    bucketSizesGm.SetGlobalBuffer((__gm__ int64_t*)bucketSizes);

    keysList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(keys));
    countersList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(counters));
    filterFlagsList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(filterFlags));
    valuesList_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(values));

    coreSyncWorkspaceGm_.SetGlobalBuffer((__gm__ int64_t*)workspace);
    pipe_.InitBuffer(ubBuf1_, sizeof(int64_t) * maxThreadNum_ * BUFFER_LENGTH);
    pipe_.InitBuffer(ubBuf2_, sizeof(int64_t) * maxThreadNum_ * BUFFER_LENGTH);
    pipe_.InitBuffer(ubBuf3_, sizeof(int64_t) * maxThreadNum_ * BUFFER_LENGTH);

    threadCountKeysToExportUB_ = ubBuf1_.Get<int64_t>();
    threadCountReFreshExportFlagUB_ = ubBuf2_.Get<int64_t>();
    threadCountKeysToExportSumUB_ = ubBuf3_.Get<int64_t>();
}

template <typename T>
__aicore__ inline void EmbeddingHashTableExport<T>::SingleTableCompute(int64_t tableIndex)
{
    GM_ADDR tableHandleStruct = reinterpret_cast<__gm__ uint8_t*>(tableHandlesGm_.GetValue(tableIndex));

    tableHandleStructGm_.SetGlobalBuffer((__gm__ int64_t*)tableHandleStruct);

    tableAddr_ = tableHandleStructGm_.GetValue(0);
    embeddingDims_ = embeddingDimsGm.GetValue(tableIndex);
    keyWidthByte_ = KEY_VALUE_OFFSET_OF_BYTE +
                    (sizeof(T) * embeddingDims_ + BYTE_COUNT_8 - 1) / BYTE_COUNT_8 * BYTE_COUNT_8;
    keyWidthByteD8_ = keyWidthByte_ / BYTE_COUNT_8;
    keyWidthByteDT_ = keyWidthByte_ / sizeof(T);
    bucketSize_ = bucketSizesGm.GetValue(tableIndex);

    int64_t singleCoreProcessNum = (bucketSize_ + maxCoreNum_ - 1) / maxCoreNum_;
    normalCoreProcessKeys_ = singleCoreProcessNum <= maxThreadNum_ ? maxThreadNum_ : singleCoreProcessNum;
    usedCoreNum_ = (bucketSize_ + normalCoreProcessKeys_ - 1) / normalCoreProcessKeys_;
    tailCoreProcessKeys_ = bucketSize_ - (usedCoreNum_ - 1) * normalCoreProcessKeys_;
    curCoreProcessKeys_ = blockIdx_ < (usedCoreNum_ - 1) ? normalCoreProcessKeys_ : tailCoreProcessKeys_;

    normalThreadProcessKeys_ = (curCoreProcessKeys_ + maxThreadNum_ - 1) / maxThreadNum_;
    usedThreadNum_ = (curCoreProcessKeys_ + normalThreadProcessKeys_ - 1) / normalThreadProcessKeys_;
    tailThreadProcessKeys_ = curCoreProcessKeys_ - (usedThreadNum_ - 1) * normalThreadProcessKeys_;

    outKeyGm_.SetGlobalBuffer(keysList_.GetDataPtr<int64_t>(tableIndex));
    outCounterGm_.SetGlobalBuffer(countersList_.GetDataPtr<uint64_t>(tableIndex));
    outFilterFlagGm_.SetGlobalBuffer(filterFlagsList_.GetDataPtr<uint8_t>(tableIndex));
    outValueGm_.SetGlobalBuffer(valuesList_.GetDataPtr<T>(tableIndex));
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void SaveToCoreSyncWorkspace(int64_t maxCoreNum, int64_t maxThreadNum,
                                                                           int64_t tableIndx, int64_t blockIdx,
                                                                           int64_t usedCoreNum,
                                                                           __gm__ int64_t* coreSyncWorkspaceGm,
                                                                           __ubuf__ int64_t* threadCountKeysToExportUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (threadIdx.x == 0) {
        coreSyncWorkspaceGm[tableIndx * maxCoreNum + blockIdx] = threadCountKeysToExportUB[maxThreadNum];
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void AtomicSubToGm(int64_t maxCoreNum, int64_t maxThreadNum,
                                                                 int64_t blockIdx, int64_t usedCoreNum,
                                                                 __gm__ int64_t* tableHandleStructGm,
                                                                 __ubuf__ int64_t* threadCountReFreshExportFlagUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (threadIdx.x == 0) {
        asc_atomic_sub(tableHandleStructGm + SIZE_ALL_NO_EXPORT_IDX, threadCountReFreshExportFlagUB[maxThreadNum]);
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_LAUNCH_BOUND) inline void CountPerThread(
    int64_t maxCoreNum, int64_t maxThreadNum, int64_t blockIdx, int64_t usedCoreNum, int64_t usedThreadNum,
    int64_t normalThreadProcessKeys, int64_t tailThreadProcessKeys, int64_t tableAddr, int64_t keyWidthByte,
    int64_t normalCoreProcessKeys, int64_t exportMode, __ubuf__ int64_t* threadCountKeysToExportUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (threadIdx.x >= maxThreadNum) {
        return;
    }

    int64_t curThreadProcessKeys = threadIdx.x < (usedThreadNum - 1) ? normalThreadProcessKeys : tailThreadProcessKeys;
    int64_t keysNumToExport = 0;
    if (threadIdx.x < usedThreadNum) {
        __gm__ uint8_t* tableAddrU8 = reinterpret_cast<__gm__ uint8_t*>(tableAddr);

        for (int64_t i = 0; i < curThreadProcessKeys; i++) {
            uint8_t flag = tableAddrU8[keyWidthByte * (blockIdx * normalCoreProcessKeys +
                                                       threadIdx.x * normalThreadProcessKeys + i) +
                                       KEY_FLAG_OFFSET_OF_BYTE];

            if ((flag & VALID_FLAG_MASK) && !(flag & EVICTED_FLAG_MASK) &&
                (exportMode != 1 || !(flag & EXPORT_FLAG_MASK))) {
                keysNumToExport++;
            }
        }
    }
    threadCountKeysToExportUB[threadIdx.x] = keysNumToExport;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_LAUNCH_BOUND) inline void CalcOffset(
    int64_t maxCoreNum, int64_t maxThreadNum, int64_t tableIndx, int64_t blockIdx, __gm__ int64_t* coreSyncWorkspaceGm,
    __ubuf__ int64_t* threadCountKeysToExportUB, __ubuf__ int64_t* threadCountKeysToExportSumUB)
{
    int64_t offset = 0;
    for (int32_t i = 0; i < blockIdx; i++) {
        offset += coreSyncWorkspaceGm[tableIndx * maxCoreNum + i];
    }
    for (int32_t i = 0; i < threadIdx.x; i++) {
        offset += threadCountKeysToExportUB[i];
    }
    threadCountKeysToExportSumUB[threadIdx.x] = offset;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_LAUNCH_BOUND) inline void ExportPerThread(
    int64_t blockIdx, int64_t usedCoreNum, int64_t usedThreadNum, int64_t normalThreadProcessKeys,
    int64_t tailThreadProcessKeys, int64_t tableAddr, int64_t keyWidthByte, int64_t normalCoreProcessKeys,
    int64_t exportMode, int64_t keyWidthByteD8, int64_t keyWidthByteDT, int64_t embeddingDims,
    __gm__ int64_t* coreSyncWorkspaceGm, __ubuf__ int64_t* threadCountKeysToExportUB, __gm__ int64_t* outKeyGm,
    __gm__ uint64_t* outCounterGm, __gm__ uint8_t* outFilterFlagGm, __gm__ T* outValueGm,
    __ubuf__ int64_t* threadCountReFreshExportFlagUB, __ubuf__ int64_t* threadCountKeysToExportSumUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (threadIdx.x >= usedThreadNum) {
        return;
    }

    int64_t offset = threadCountKeysToExportSumUB[threadIdx.x];

    __gm__ int64_t* tableAddrI64 = reinterpret_cast<__gm__ int64_t*>(tableAddr);
    __gm__ uint64_t* tableAddrU64 = reinterpret_cast<__gm__ uint64_t*>(tableAddr);
    __gm__ uint8_t* tableAddrU8 = reinterpret_cast<__gm__ uint8_t*>(tableAddr);

    int64_t curThreadRefreshExportFlagNum = 0;
    int64_t positionIndex = 0;

    int64_t curThreadProcessKeys = threadIdx.x < (usedThreadNum - 1) ? normalThreadProcessKeys : tailThreadProcessKeys;
    for (int64_t i = 0; i < curThreadProcessKeys; i++) {
        uint8_t flag = tableAddrU8[keyWidthByte *
                                       (blockIdx * normalCoreProcessKeys + threadIdx.x * normalThreadProcessKeys + i) +
                                   KEY_FLAG_OFFSET_OF_BYTE];
        if ((flag & VALID_FLAG_MASK) && !(flag & EVICTED_FLAG_MASK) &&
            (exportMode != 1 || !(flag & EXPORT_FLAG_MASK))) {
            int64_t key = tableAddrI64[keyWidthByteD8 *
                                       (blockIdx * normalCoreProcessKeys + threadIdx.x * normalThreadProcessKeys + i)];
            outKeyGm[offset + positionIndex] = key;
            uint64_t counter = tableAddrU64[keyWidthByteD8 * (blockIdx * normalCoreProcessKeys +
                                                              threadIdx.x * normalThreadProcessKeys + i) +
                                            1];
            outCounterGm[offset + positionIndex] = counter;

            if (FILTER_FLAG_MASK & flag) {
                outFilterFlagGm[offset + positionIndex] = 1;
            } else {
                outFilterFlagGm[offset + positionIndex] = 0;
            }
            // 拷出 values（访存合并；merge 档位逐 key 相同、warp 内无分化。
            // binary 仅 fp32 单 bin，T 恒为 float，按 4B 元素处理。
            // dim<=4 走原始标量路径：小 dim 宽访存收益抵不过开销——
            // 与 import 算子 dim=4 实测回退 16% 同因，直接带fallback）
            int64_t bucketByteBase = keyWidthByte *
                                     (blockIdx * normalCoreProcessKeys + threadIdx.x * normalThreadProcessKeys + i);
            __gm__ float* pDstRow = reinterpret_cast<__gm__ float*>(outValueGm) +
                                    (offset + positionIndex) * embeddingDims;
            const bool dstAligned16 = (reinterpret_cast<uintptr_t>(pDstRow) & 15) == 0;
            if (embeddingDims % 4 == 0 && dstAligned16 && embeddingDims > 4) {
                CopyExportValuesMerged<4>(tableAddrU8 + bucketByteBase + KEY_VALUE_OFFSET_OF_BYTE, pDstRow,
                                          embeddingDims);
            } else if (embeddingDims % 2 == 0 && embeddingDims > 4) {
                CopyExportValuesMerged<2>(tableAddrU8 + bucketByteBase + KEY_VALUE_OFFSET_OF_BYTE, pDstRow,
                                          embeddingDims);
            } else {
                CopyExportValuesMerged<1>(tableAddrU8 + bucketByteBase + KEY_VALUE_OFFSET_OF_BYTE, pDstRow,
                                          embeddingDims);
            }
            // 刷新导出flag, 只在第一次导出时刷新
            if (!(flag & EXPORT_FLAG_MASK)) {
                tableAddrU8[keyWidthByte *
                                (blockIdx * normalCoreProcessKeys + threadIdx.x * normalThreadProcessKeys + i) +
                            KEY_FLAG_OFFSET_OF_BYTE] |= EXPORT_FLAG_MASK;
                curThreadRefreshExportFlagNum++;
            }
            positionIndex++;
        }
    }
    threadCountReFreshExportFlagUB[threadIdx.x] = curThreadRefreshExportFlagNum;
}

template <typename T>
__aicore__ inline void EmbeddingHashTableExport<T>::Process()
{
    for (int64_t tableIndx = 0; tableIndx < tableNum_; tableIndx++) {
        Duplicate(threadCountKeysToExportUB_, int64_t(0), maxThreadNum_ * BUFFER_LENGTH);
        Duplicate(threadCountReFreshExportFlagUB_, int64_t(0), maxThreadNum_ * BUFFER_LENGTH);
        SingleTableCompute(tableIndx);
        asc_vf_call<CountPerThread<T>>(dim3{static_cast<uint32_t>(maxThreadNum_)}, maxCoreNum_, maxThreadNum_,
                                       blockIdx_, usedCoreNum_, usedThreadNum_, normalThreadProcessKeys_,
                                       tailThreadProcessKeys_, tableAddr_, keyWidthByte_, normalCoreProcessKeys_,
                                       exportMode_, (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr());
        ReduceSum<int64_t>(threadCountKeysToExportUB_[maxThreadNum_], threadCountKeysToExportUB_,
                           threadCountReFreshExportFlagUB_, usedThreadNum_);
        asc_vf_call<SaveToCoreSyncWorkspace<T>>(dim3{static_cast<uint32_t>(1)}, maxCoreNum_, maxThreadNum_, tableIndx,
                                                blockIdx_, usedCoreNum_, coreSyncWorkspaceGm_.GetPhyAddr(0),
                                                (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr());
        SyncAll();
        asc_vf_call<CalcOffset<T>>(dim3{static_cast<uint32_t>(maxThreadNum_)}, maxCoreNum_, maxThreadNum_, tableIndx,
                                   blockIdx_, coreSyncWorkspaceGm_.GetPhyAddr(0),
                                   (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr(),
                                   (__ubuf__ int64_t*)threadCountKeysToExportSumUB_.GetPhyAddr());
        asc_vf_call<ExportPerThread<T>>(
            dim3{static_cast<uint32_t>(maxThreadNum_)}, blockIdx_, usedCoreNum_, usedThreadNum_,
            normalThreadProcessKeys_, tailThreadProcessKeys_, tableAddr_, keyWidthByte_, normalCoreProcessKeys_,
            exportMode_, keyWidthByteD8_, keyWidthByteDT_, embeddingDims_, coreSyncWorkspaceGm_.GetPhyAddr(0),
            (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr(), outKeyGm_.GetPhyAddr(0),
            outCounterGm_.GetPhyAddr(0), outFilterFlagGm_.GetPhyAddr(0), outValueGm_.GetPhyAddr(0),
            (__ubuf__ int64_t*)threadCountReFreshExportFlagUB_.GetPhyAddr(),
            (__ubuf__ int64_t*)threadCountKeysToExportSumUB_.GetPhyAddr());
        ReduceSum<int64_t>(threadCountReFreshExportFlagUB_[maxThreadNum_], threadCountReFreshExportFlagUB_,
                           threadCountKeysToExportUB_, usedThreadNum_);
        asc_vf_call<AtomicSubToGm<T>>(dim3{static_cast<uint32_t>(1)}, maxCoreNum_, maxThreadNum_, blockIdx_,
                                      usedCoreNum_, tableHandleStructGm_.GetPhyAddr(0),
                                      (__ubuf__ int64_t*)threadCountReFreshExportFlagUB_.GetPhyAddr());
        SyncAll();
    }
}
} // namespace EmbeddingHashTableExportAicore

#endif // EMBEDDING_HASH_TABLE_EXPORT_H_
