/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCENDC_APPLY_CAME_PART3_POST_H_
#define ASCENDC_APPLY_CAME_PART3_POST_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_came_part3_common.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart3Post {
public:
    __aicore__ inline ApplyCamePart3Post(){};
    __aicore__ inline void Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                                const ApplyCamePart3TilingData* tiling_data);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ApplyCamePart3TilingData* tiling_data);
    __aicore__ inline void Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m);
    __aicore__ inline void ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m);
    __aicore__ inline void CalcSumC(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx, int64_t preN,
                                    int64_t calcN, int64_t calcM);
    __aicore__ inline void CalcSumCLarge(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx, int64_t preN,
                                         int64_t calcN, int64_t calcM);
    __aicore__ inline void CalcSumCAligned(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx, int64_t preN,
                                           int64_t calcN, int64_t calcM);
    __aicore__ inline void ReduceSumUR();
    __aicore__ inline void ReduceSumUC();
    __aicore__ inline void MergeCompensated(LocalTensor<float> high, LocalTensor<float> low,
                                            LocalTensor<float> highRight, LocalTensor<float> lowRight,
                                            LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp, int64_t count);
    __aicore__ inline void ReduceSumURChunk(LocalTensor<float> high, LocalTensor<float> low,
                                            LocalTensor<float> highRight, LocalTensor<float> lowRight,
                                            LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp, int64_t rowCore,
                                            int64_t outputBase, int64_t offset, int64_t count);
    __aicore__ inline void ReduceSumUCSingleColumn(LocalTensor<float> high, LocalTensor<float> low);
    __aicore__ inline void ReduceSumUCChunk(LocalTensor<float> high, LocalTensor<float> low,
                                            LocalTensor<float> highRight, LocalTensor<float> lowRight,
                                            LocalTensor<float> sumTmp, LocalTensor<float> recoveredTmp, int64_t offset,
                                            int64_t count, float& totalHigh, float& totalLow);
    __aicore__ inline int64_t DivCeil(int64_t a, int64_t b);
    __aicore__ inline int64_t Ceil(int64_t a, int64_t b);
    __aicore__ inline void L0ReduceSum(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> worklocal,
                                       int64_t size);
    __aicore__ inline void StoreAccumulated(LocalTensor<float> inputLocal, int64_t outputOffset, int64_t count);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> inputBuf;

    GlobalTensor<float> gmSumGradR_;
    GlobalTensor<float> gmSumGradC_;
    GlobalTensor<float> gmSumGradRC_;

    GlobalTensor<float> workspaceSumGradCHigh_;
    GlobalTensor<float> workspaceSumGradCLow_;
    GlobalTensor<float> workspaceSumGradRHigh_;
    GlobalTensor<float> workspaceSumGradRLow_;

    // multi-core sync
    GlobalTensor<int32_t> syncGlobal_;
    GM_ADDR workspaceAddr_;

    // tiling params
    int64_t usedCoreNum_{0};
    int64_t curN{0};
    int64_t curM{0};
    int64_t rNumCalc_{0};
    int64_t cNumCalc_{0};
    int64_t baseN{0};
    int64_t baseM{0};
    int64_t rCoreNum_{0};
    int64_t cCoreNum_{0};

    int64_t offset{0};
    bool isGlobalShape{false};
    bool useFirstMoment{false};

    // part1 params
    int64_t baseCSize{0};
    int64_t workspaceRStride{0};
    int64_t mLoopNumCore_{0};

    const int64_t ONCE_HANDLE_NUM64{64};
    const int64_t ONCE_HANDLE_NUM512{512};
    const int64_t ONCE_ONE_SIZE8{8};
    const int64_t ONCE_ALGN_NUM{32 / sizeof(float)};
    int64_t MAX_BUF_SIZE{MAX_POST_BUFFER_SIZE};
    int64_t MAX_BLOCK_LEN{65535};
    int64_t MAX_BLOCK_LEN_SIZE_FP32{16376};
    int64_t MAX_DATA_COPY_BLOCK_COUNT{4096};
    int64_t CAME_ONE_BLOCK_SIZE_FP32{8};
    // inputBuf starts at a 32 KiB byte offset (8 KiB FP32 elements).  Four
    // compensation planes must fit in the remaining 24 KiB elements.
    int64_t COMPENSATED_REDUCE_SIZE{
        (ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 - (MAX_BUF_SIZE * 2) / static_cast<int64_t>(sizeof(float))) / 4};

    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
};

template <typename T>
__aicore__ inline int64_t ApplyCamePart3Post<T>::Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline int64_t ApplyCamePart3Post<T>::DivCeil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ParseTilingData(const ApplyCamePart3TilingData* tiling_data)
{
    // 总维度[curN, curM]
    curN = tiling_data->curN;
    curM = tiling_data->curM;
    rNumCalc_ = tiling_data->rNumCalc;
    cNumCalc_ = tiling_data->cNumCalc;
    baseN = tiling_data->baseN;
    baseM = tiling_data->baseM;
    rCoreNum_ = tiling_data->rCoreNum;
    cCoreNum_ = tiling_data->cCoreNum;

    // 使用核数 [usedCoreNum_]
    usedCoreNum_ = tiling_data->usedCoreNum;

    // 行列方向的核 [rCoreNum_, cCoreNum_]
    rCoreNum_ = tiling_data->rCoreNum;
    cCoreNum_ = tiling_data->cCoreNum;
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                                                   const ApplyCamePart3TilingData* tiling_data)
{
    // 初始化tiling
    ParseTilingData(tiling_data);

    // gm输出
    gmSumGradR_.SetGlobalBuffer((__gm__ float*)(camePart3InOut.sumUR));
    gmSumGradC_.SetGlobalBuffer((__gm__ float*)(camePart3InOut.sumUC));
    gmSumGradRC_.SetGlobalBuffer((__gm__ float*)(camePart3InOut.sumURC));

    // workspace vars
    int64_t rTailNumCalc = curN - rNumCalc_ * (rCoreNum_ - 1);
    int64_t rBlockNum = DivCeil(rNumCalc_, baseN) * (rCoreNum_ - 1) + DivCeil(rTailNumCalc, baseN);
    baseCSize = rBlockNum;

    // workspace地址
    const int64_t workspaceCElements = rBlockNum * curM;
    const int64_t workspaceRElements = usedCoreNum_ * Ceil(rNumCalc_, FP32_ONE_BLOCK_COUNT);
    auto* workspaceRBase = (__gm__ float*)workspace + DET_WORKSPACE_SIZE;
    workspaceSumGradRHigh_.SetGlobalBuffer(workspaceRBase);
    workspaceSumGradRLow_.SetGlobalBuffer(workspaceRBase + workspaceRElements);
    auto* workspaceCBase = workspaceRBase + workspaceRElements * 2;
    workspaceSumGradCHigh_.SetGlobalBuffer(workspaceCBase);
    workspaceSumGradCLow_.SetGlobalBuffer(workspaceCBase + workspaceCElements);
    workspaceRStride = Ceil(rNumCalc_, FP32_ONE_BLOCK_COUNT);
    syncGlobal_.SetGlobalBuffer((__gm__ int32_t*)workspace);

    pipe.InitBuffer(inputBuf, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::StoreAccumulated(LocalTensor<float> inputLocal, int64_t outputOffset,
                                                               int64_t count)
{
    // Only block 0 executes the post stage.  Use atomic block copies for the
    // aligned body and scalar atomics for the prefix/tail so no DMA writes
    // outside the logical output tensor.
    int64_t i = 0;
    bool hasScalarWrite = false;
    while (i < count && (outputOffset + i) % FP32_ONE_BLOCK_COUNT != 0) {
        AtomicAdd(gmSumGradC_.GetPhyAddr(outputOffset + i), inputLocal.GetValue(i));
        hasScalarWrite = true;
        ++i;
    }
    const int64_t alignedCount = (count - i) / FP32_ONE_BLOCK_COUNT * FP32_ONE_BLOCK_COUNT;
    if (alignedCount > 0) {
        SetAtomicAdd<float>();
        for (int64_t copied = 0; copied < alignedCount; copied += MAX_BLOCK_LEN_SIZE_FP32) {
            const int64_t copyCount = alignedCount - copied < MAX_BLOCK_LEN_SIZE_FP32 ? alignedCount - copied :
                                                                                        MAX_BLOCK_LEN_SIZE_FP32;
            DataCopy(gmSumGradC_[outputOffset + i + copied], inputLocal[i + copied], copyCount);
        }
        SetAtomicNone();
        event_t eventMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMte3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventMte3ToS);
    }
    i += alignedCount;
    while (i < count) {
        AtomicAdd(gmSumGradC_.GetPhyAddr(outputOffset + i), inputLocal.GetValue(i));
        hasScalarWrite = true;
        ++i;
    }
    if (hasScalarWrite) {
        // Complete the transfer fence before reusing the UB region.
        event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::L0ReduceSum(LocalTensor<float> dst, LocalTensor<float> src,
                                                          LocalTensor<float> worklocal, int64_t size)
{
    if (size <= ONCE_HANDLE_NUM64) {
        ReduceSum(dst, src, worklocal, size);
    } else if (size % ONCE_HANDLE_NUM64) {
        int64_t repeat = size / ONCE_HANDLE_NUM64;
        int64_t tail = size % ONCE_HANDLE_NUM64;

        ReduceSum(dst, src, worklocal, ONCE_HANDLE_NUM64, repeat, 8);
        PipeBarrier<PIPE_V>();
        ReduceSum(dst[1], src[repeat * ONCE_HANDLE_NUM64], worklocal, tail, 1, 8);
        PipeBarrier<PIPE_V>();
        ReduceSum(dst, dst, worklocal, 2, 1, 8);
    } else {
        int64_t repeat = size / ONCE_HANDLE_NUM64;
        ReduceSum(dst, src, worklocal, ONCE_HANDLE_NUM64, repeat, 8);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Process()
{
    SyncAll();
    if (GetBlockIdx() != 0) {
        return;
    }

    if (cCoreNum_ > 1) {
        ReduceSumUR();
    }
    ReduceSumUC();
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::MergeCompensated(LocalTensor<float> high, LocalTensor<float> low,
                                                               LocalTensor<float> highRight,
                                                               LocalTensor<float> lowRight, LocalTensor<float> sumTmp,
                                                               LocalTensor<float> recoveredTmp, int64_t count)
{
    Add(sumTmp, high, highRight, count);
    PipeBarrier<PIPE_V>();
    Sub(recoveredTmp, sumTmp, high, count);
    PipeBarrier<PIPE_V>();
    Sub(highRight, highRight, recoveredTmp, count);
    Sub(recoveredTmp, sumTmp, recoveredTmp, count);
    PipeBarrier<PIPE_V>();
    Sub(high, high, recoveredTmp, count);
    PipeBarrier<PIPE_V>();
    Add(high, high, highRight, count);
    Add(low, low, lowRight, count);
    PipeBarrier<PIPE_V>();
    Add(low, low, high, count);
    Adds(high, sumTmp, static_cast<float>(0), count);
    PipeBarrier<PIPE_V>();
    ApplyCamePart3NormalizeCompensation(high, low, highRight, recoveredTmp, count);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceSumURChunk(LocalTensor<float> high, LocalTensor<float> low,
                                                               LocalTensor<float> highRight,
                                                               LocalTensor<float> lowRight, LocalTensor<float> sumTmp,
                                                               LocalTensor<float> recoveredTmp, int64_t rowCore,
                                                               int64_t outputBase, int64_t offset, int64_t count)
{
    const int64_t alignedCount = Ceil(count, FP32_ONE_BLOCK_COUNT);
    const uint8_t rightPadding = static_cast<uint8_t>(alignedCount - count);
    const int64_t firstBlockOffset = rowCore * cCoreNum_ * workspaceRStride + offset;
    DataCopyPad(high, workspaceSumGradRHigh_[firstBlockOffset],
                {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
    DataCopyPad(low, workspaceSumGradRLow_[firstBlockOffset],
                {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    for (int64_t core = 1; core < cCoreNum_; ++core) {
        const int64_t blockOffset = (rowCore * cCoreNum_ + core) * workspaceRStride + offset;
        DataCopyPad(highRight, workspaceSumGradRHigh_[blockOffset],
                    {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
        DataCopyPad(lowRight, workspaceSumGradRLow_[blockOffset],
                    {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
        eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        MergeCompensated(high, low, highRight, lowRight, sumTmp, recoveredTmp, alignedCount);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    Add(high, high, low, alignedCount);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(gmSumGradR_[outputBase + offset], high, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceSumUR()
{
    constexpr int64_t reduceChunk = 4096;
    LocalTensor<float> inputLocal = inputBuf.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    LocalTensor<float> high = inputLocal;
    LocalTensor<float> low = high[reduceChunk];
    LocalTensor<float> highRight = low[reduceChunk];
    LocalTensor<float> lowRight = highRight[reduceChunk];
    LocalTensor<float> sumTmp = lowRight[reduceChunk];
    LocalTensor<float> recoveredTmp = sumTmp[reduceChunk];
    for (int64_t rowCore = 0; rowCore < rCoreNum_; ++rowCore) {
        const int64_t outputBase = rowCore * rNumCalc_;
        const int64_t rowCount = curN - outputBase < rNumCalc_ ? curN - outputBase : rNumCalc_;
        for (int64_t offset = 0; offset < rowCount; offset += reduceChunk) {
            const int64_t count = rowCount - offset < reduceChunk ? rowCount - offset : reduceChunk;
            ReduceSumURChunk(high, low, highRight, lowRight, sumTmp, recoveredTmp, rowCore, outputBase, offset, count);
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceSumUCSingleColumn(LocalTensor<float> high, LocalTensor<float> low)
{
    constexpr int64_t reduceChunk = 4096;
    float totalHigh = 0.0f;
    float totalLow = 0.0f;
    for (int64_t rowBase = 0; rowBase < baseCSize; rowBase += reduceChunk) {
        const int64_t count = baseCSize - rowBase < reduceChunk ? baseCSize - rowBase : reduceChunk;
        DataCopyPad(high, workspaceSumGradCHigh_[rowBase], {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(low, workspaceSumGradCLow_[rowBase], {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0});
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        for (int64_t row = 0; row < count; ++row) {
            float nextHigh = 0.0f;
            float nextLow = 0.0f;
            ApplyCamePart3CompensatedAdd(totalHigh, totalLow, high.GetValue(row), low.GetValue(row), nextHigh, nextLow);
            totalHigh = nextHigh;
            totalLow = nextLow;
        }
        if (rowBase + count < baseCSize) {
            event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
            WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        }
    }
    const float total = totalHigh + totalLow;
    gmSumGradC_.SetValue(0, total);
    gmSumGradRC_.SetValue(0, total);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceSumUCChunk(LocalTensor<float> high, LocalTensor<float> low,
                                                               LocalTensor<float> highRight,
                                                               LocalTensor<float> lowRight, LocalTensor<float> sumTmp,
                                                               LocalTensor<float> recoveredTmp, int64_t offset,
                                                               int64_t count, float& totalHigh, float& totalLow)
{
    const int64_t alignedCount = Ceil(count, FP32_ONE_BLOCK_COUNT);
    const uint8_t rightPadding = static_cast<uint8_t>(alignedCount - count);
    DataCopyPad(high, workspaceSumGradCHigh_[offset], {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                {true, 0, rightPadding, 0});
    DataCopyPad(low, workspaceSumGradCLow_[offset], {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0},
                {true, 0, rightPadding, 0});
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    for (int64_t row = 1; row < baseCSize; ++row) {
        const int64_t workspaceOffset = row * curM + offset;
        DataCopyPad(highRight, workspaceSumGradCHigh_[workspaceOffset],
                    {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
        DataCopyPad(lowRight, workspaceSumGradCLow_[workspaceOffset],
                    {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0}, {true, 0, rightPadding, 0});
        eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        MergeCompensated(high, low, highRight, lowRight, sumTmp, recoveredTmp, alignedCount);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    Add(highRight, high, low, alignedCount);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(gmSumGradC_[offset], highRight, {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    Duplicate(low, static_cast<float>(0), alignedCount);
    PipeBarrier<PIPE_V>();
    ApplyCamePart3CompensatedReduce(highRight, low, sumTmp, recoveredTmp, count);
    float nextHigh = 0.0f;
    float nextLow = 0.0f;
    ApplyCamePart3CompensatedAdd(totalHigh, totalLow, highRight.GetValue(0), low.GetValue(0), nextHigh, nextLow);
    totalHigh = nextHigh;
    totalLow = nextLow;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceSumUC()
{
    constexpr int64_t reduceChunk = 4096;
    LocalTensor<float> inputLocal = inputBuf.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);
    LocalTensor<float> high = inputLocal;
    LocalTensor<float> highRight = high[reduceChunk];
    LocalTensor<float> low = highRight[reduceChunk];
    LocalTensor<float> lowRight = low[reduceChunk];
    LocalTensor<float> sumTmp = lowRight[reduceChunk];
    LocalTensor<float> recoveredTmp = sumTmp[reduceChunk];
    if (curM == 1) {
        ReduceSumUCSingleColumn(high, low);
        return;
    }
    float totalHigh = 0.0f;
    float totalLow = 0.0f;
    for (int64_t offset = 0; offset < curM; offset += reduceChunk) {
        const int64_t count = curM - offset < reduceChunk ? curM - offset : reduceChunk;
        ReduceSumUCChunk(high, low, highRight, lowRight, sumTmp, recoveredTmp, offset, count, totalHigh, totalLow);
    }
    gmSumGradRC_.SetValue(0, totalHigh + totalLow);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
}

#include "apply_came_part3_post_calc.h"

#endif // _ASCENDC_APPLY_CAME_PART3_POST_H_
