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
 * \file embedding_hash_table_export.h
 * \brief
 */

#ifndef EMBEDDING_HASH_TABLE_EXPORT_H_
#define EMBEDDING_HASH_TABLE_EXPORT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

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

template <typename T>
class EmbeddingHashTableExport
{
public:
    __aicore__ inline EmbeddingHashTableExport(){};
    __aicore__ inline void Init(
        GM_ADDR tableHandles, GM_ADDR tableSizes, GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys,
        GM_ADDR counters, GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace,
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
__aicore__ inline void EmbeddingHashTableExport<T>::Init(
    GM_ADDR tableHandles, GM_ADDR tableSizes, GM_ADDR embeddingDims, GM_ADDR bucketSizes, GM_ADDR keys,
    GM_ADDR counters, GM_ADDR filterFlags, GM_ADDR values, GM_ADDR workspace,
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
    keyWidthByte_ =
        KEY_VALUE_OFFSET_OF_BYTE + (sizeof(T) * embeddingDims_ + BYTE_COUNT_8 - 1) / BYTE_COUNT_8 * BYTE_COUNT_8;
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
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void SaveToCoreSyncWorkspace(
    int64_t maxCoreNum, int64_t maxThreadNum, int64_t tableIndx, int64_t blockIdx, int64_t usedCoreNum,
    __gm__ int64_t* coreSyncWorkspaceGm, __ubuf__ int64_t* threadCountKeysToExportUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (Simt::GetThreadIdx() == 0) {
        coreSyncWorkspaceGm[tableIndx * maxCoreNum + blockIdx] = threadCountKeysToExportUB[maxThreadNum];
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void AtomicSubToGm(
    int64_t maxCoreNum, int64_t maxThreadNum, int64_t blockIdx, int64_t usedCoreNum,
    __gm__ int64_t* tableHandleStructGm, __ubuf__ int64_t* threadCountReFreshExportFlagUB)
{
    if (blockIdx >= usedCoreNum) {
        return;
    }

    if (Simt::GetThreadIdx() == 0) {
        Simt::AtomicSub(tableHandleStructGm + SIZE_ALL_NO_EXPORT_IDX, threadCountReFreshExportFlagUB[maxThreadNum]);
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

    if (Simt::GetThreadIdx() >= maxThreadNum) {
        return;
    }

    int64_t curThreadProcessKeys =
        Simt::GetThreadIdx() < (usedThreadNum - 1) ? normalThreadProcessKeys : tailThreadProcessKeys;
    int64_t keysNumToExport = 0;
    if (Simt::GetThreadIdx() < usedThreadNum) {
        __gm__ uint8_t* tableAddrU8 = reinterpret_cast<__gm__ uint8_t*>(tableAddr);

        for (int64_t i = 0; i < curThreadProcessKeys; i++) {
            uint8_t flag = tableAddrU8
                [keyWidthByte *
                     (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i) +
                 KEY_FLAG_OFFSET_OF_BYTE];

            if ((flag & VALID_FLAG_MASK) && !(flag & EVICTED_FLAG_MASK) &&
                (exportMode != 1 || !(flag & EXPORT_FLAG_MASK))) {
                keysNumToExport++;
            }
        }
    }
    threadCountKeysToExportUB[Simt::GetThreadIdx()] = keysNumToExport;
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
    for (int32_t i = 0; i < Simt::GetThreadIdx(); i++) {
        offset += threadCountKeysToExportUB[i];
    }
    threadCountKeysToExportSumUB[Simt::GetThreadIdx()] = offset;
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

    if (Simt::GetThreadIdx() >= usedThreadNum) {
        return;
    }

    int64_t offset = threadCountKeysToExportSumUB[Simt::GetThreadIdx()];

    __gm__ int64_t* tableAddrI64 = reinterpret_cast<__gm__ int64_t*>(tableAddr);
    __gm__ uint64_t* tableAddrU64 = reinterpret_cast<__gm__ uint64_t*>(tableAddr);
    __gm__ uint8_t* tableAddrU8 = reinterpret_cast<__gm__ uint8_t*>(tableAddr);
    __gm__ T* tableAddrT = reinterpret_cast<__gm__ T*>(tableAddrU8);

    int64_t curThreadRefreshExportFlagNum = 0;
    int64_t positionIndex = 0;

    int64_t curThreadProcessKeys =
        Simt::GetThreadIdx() < (usedThreadNum - 1) ? normalThreadProcessKeys : tailThreadProcessKeys;
    for (int64_t i = 0; i < curThreadProcessKeys; i++) {
        uint8_t flag = tableAddrU8
            [keyWidthByte * (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i) +
             KEY_FLAG_OFFSET_OF_BYTE];
        if ((flag & VALID_FLAG_MASK) && !(flag & EVICTED_FLAG_MASK) &&
            (exportMode != 1 || !(flag & EXPORT_FLAG_MASK))) {
            int64_t key = tableAddrI64
                [keyWidthByteD8 *
                 (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i)];
            outKeyGm[offset + positionIndex] = key;
            uint64_t counter = tableAddrU64
                [keyWidthByteD8 *
                     (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i) +
                 1];
            outCounterGm[offset + positionIndex] = counter;

            if (FILTER_FLAG_MASK & flag) {
                outFilterFlagGm[offset + positionIndex] = 1;
            } else {
                outFilterFlagGm[offset + positionIndex] = 0;
            }
            for (int64_t j = 0; j < embeddingDims; j++) {
                outValueGm[(offset + positionIndex) * embeddingDims + j] = tableAddrT
                    [keyWidthByteDT *
                         (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i) +
                     KEY_VALUE_OFFSET_OF_BYTE / sizeof(T) + j];
            }
            // 刷新导出flag, 只在第一次导出时刷新
            if (!(flag & EXPORT_FLAG_MASK)) {
                tableAddrU8
                    [keyWidthByte *
                         (blockIdx * normalCoreProcessKeys + Simt::GetThreadIdx() * normalThreadProcessKeys + i) +
                     KEY_FLAG_OFFSET_OF_BYTE] |= EXPORT_FLAG_MASK;
                curThreadRefreshExportFlagNum++;
            }
            positionIndex++;
        }
    }
    threadCountReFreshExportFlagUB[Simt::GetThreadIdx()] = curThreadRefreshExportFlagNum;
}

template <typename T>
__aicore__ inline void EmbeddingHashTableExport<T>::Process()
{
    for (int64_t tableIndx = 0; tableIndx < tableNum_; tableIndx++) {
        Duplicate(threadCountKeysToExportUB_, int64_t(0), maxThreadNum_ * BUFFER_LENGTH);
        Duplicate(threadCountReFreshExportFlagUB_, int64_t(0), maxThreadNum_ * BUFFER_LENGTH);
        SingleTableCompute(tableIndx);
        Simt::VF_CALL<CountPerThread<T>>(
            Simt::Dim3{static_cast<uint32_t>(maxThreadNum_)}, maxCoreNum_, maxThreadNum_, blockIdx_,
            usedCoreNum_, usedThreadNum_, normalThreadProcessKeys_, tailThreadProcessKeys_, tableAddr_, keyWidthByte_,
            normalCoreProcessKeys_, exportMode_, (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr());
        ReduceSum<int64_t>(
            threadCountKeysToExportUB_[maxThreadNum_], threadCountKeysToExportUB_, threadCountReFreshExportFlagUB_,
            usedThreadNum_);
        Simt::VF_CALL<SaveToCoreSyncWorkspace<T>>(
            Simt::Dim3{static_cast<uint32_t>(1)}, maxCoreNum_, maxThreadNum_, tableIndx, blockIdx_,
            usedCoreNum_, coreSyncWorkspaceGm_.GetPhyAddr(0),
            (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr());
        SyncAll();
        Simt::VF_CALL<CalcOffset<T>>(
            Simt::Dim3{static_cast<uint32_t>(maxThreadNum_)}, maxCoreNum_, maxThreadNum_, tableIndx,
            blockIdx_, coreSyncWorkspaceGm_.GetPhyAddr(0), (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr(),
            (__ubuf__ int64_t*)threadCountKeysToExportSumUB_.GetPhyAddr());
        Simt::VF_CALL<ExportPerThread<T>>(
            Simt::Dim3{static_cast<uint32_t>(maxThreadNum_)}, blockIdx_, usedCoreNum_, usedThreadNum_,
            normalThreadProcessKeys_, tailThreadProcessKeys_, tableAddr_, keyWidthByte_, normalCoreProcessKeys_,
            exportMode_, keyWidthByteD8_, keyWidthByteDT_, embeddingDims_, coreSyncWorkspaceGm_.GetPhyAddr(0),
            (__ubuf__ int64_t*)threadCountKeysToExportUB_.GetPhyAddr(), outKeyGm_.GetPhyAddr(0),
            outCounterGm_.GetPhyAddr(0), outFilterFlagGm_.GetPhyAddr(0), outValueGm_.GetPhyAddr(0),
            (__ubuf__ int64_t*)threadCountReFreshExportFlagUB_.GetPhyAddr(),
            (__ubuf__ int64_t*)threadCountKeysToExportSumUB_.GetPhyAddr());
        ReduceSum<int64_t>(
            threadCountReFreshExportFlagUB_[maxThreadNum_], threadCountReFreshExportFlagUB_, threadCountKeysToExportUB_,
            usedThreadNum_);
        Simt::VF_CALL<AtomicSubToGm<T>>(
            Simt::Dim3{static_cast<uint32_t>(1)}, maxCoreNum_, maxThreadNum_, blockIdx_, usedCoreNum_,
            tableHandleStructGm_.GetPhyAddr(0), (__ubuf__ int64_t*)threadCountReFreshExportFlagUB_.GetPhyAddr());
        SyncAll();
    }
}
} // namespace EmbeddingHashTableExportAicore

#endif // EMBEDDING_HASH_TABLE_EXPORT_H_
