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
    __aicore__ inline void CalcSumURC();
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

    GlobalTensor<float> workspaceSumGradRC_;
    GlobalTensor<float> workspaceSumGradC_;

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
    int64_t baseRCSize{0};
    int64_t baseCSize{0};
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
    int64_t cTailNumCalc = curM - cNumCalc_ * (cCoreNum_ - 1);
    int64_t cBlockNum = DivCeil(cNumCalc_, baseM) * (cCoreNum_ - 1) + DivCeil(cTailNumCalc, baseM);
    int64_t rTailNumCalc = curN - rNumCalc_ * (rCoreNum_ - 1);
    int64_t rBlockNum = DivCeil(rNumCalc_, baseN) * (rCoreNum_ - 1) + DivCeil(rTailNumCalc, baseN);
    baseRCSize = cBlockNum * rBlockNum;
    baseCSize = rBlockNum;

    int64_t workspaceRCSize = DivCeil(cNumCalc_, baseM) * rCoreNum_ * DivCeil(rNumCalc_, baseN) * cCoreNum_;

    // workspace地址
    workspaceSumGradRC_.SetGlobalBuffer((__gm__ float*)workspace + DET_WORKSPACE_SIZE);
    workspaceSumGradC_.SetGlobalBuffer((__gm__ float*)workspace + workspaceRCSize + DET_WORKSPACE_SIZE);

    pipe.InitBuffer(inputBuf, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 * sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::StoreAccumulated(LocalTensor<float> inputLocal, int64_t outputOffset,
                                                               int64_t count)
{
    // Only block 0 executes the post stage.  Use atomic block copies for the
    // aligned body and scalar atomics for the prefix/tail so no DMA writes
    // outside the logical output tensor.
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVToS);
    WaitFlag<HardEvent::V_S>(eventVToS);
    int64_t i = 0;
    while (i < count && (outputOffset + i) % FP32_ONE_BLOCK_COUNT != 0) {
        AtomicAdd(gmSumGradC_.GetPhyAddr(outputOffset + i), inputLocal.GetValue(i));
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
        ++i;
    }
    event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
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
    // The producer phase writes partial C/RC values from every active block.
    // Keep the post phase behind a device-wide barrier so dynamic and binary
    // launches observe the same completed workspace contents.
    SyncAll();
    if (GetBlockIdx() != 0) {
        return;
    }

    mLoopNumCore_ = DivCeil(curM, ONCE_HANDLE_NUM64);
    uint64_t core_loop = DivCeil(mLoopNumCore_, ONCE_HANDLE_NUM512 - 1);
    uint64_t pre_core_m = mLoopNumCore_ / core_loop * ONCE_HANDLE_NUM64;
    uint64_t last_core_m = (mLoopNumCore_ - pre_core_m * (core_loop - 1)) * ONCE_HANDLE_NUM64;
    uint64_t gmOffsets = 0;
    uint64_t base_m = 0;

    for (int64_t core_loop_idx = 0; core_loop_idx < core_loop - 1; core_loop_idx++) {
        gmOffsets = core_loop_idx * pre_core_m;
        Pre_Core_Compute(gmOffsets, pre_core_m);
    }
    gmOffsets = (core_loop - 1) * pre_core_m;
    base_m = curM - pre_core_m * (core_loop - 1);
    Pre_Core_Compute(gmOffsets, base_m);
    CalcSumURC();
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::Pre_Core_Compute(uint64_t gmOffsets, uint64_t cal_m)
{
    uint64_t pre_loop_n = 1;
    while (pre_loop_n < baseCSize && pre_loop_n < MAX_DATA_COPY_BLOCK_COUNT) {
        pre_loop_n = pre_loop_n << 1;
        if (pre_loop_n * Ceil(cal_m, CAME_ONE_BLOCK_SIZE_FP32) > ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64 ||
            pre_loop_n >= MAX_DATA_COPY_BLOCK_COUNT) {
            pre_loop_n = pre_loop_n >> 1;
            break;
        }
        if (pre_loop_n >= baseCSize) {
            break;
        }
    }
    uint64_t loop_time = DivCeil(baseCSize, pre_loop_n);
    uint64_t last_loop_n = baseCSize - (loop_time - 1) * pre_loop_n;
    LocalTensor<float> inputLocal = inputBuf.Get<float>(ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);

    event_t eventMte3toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventMte3toS);
    WaitFlag<HardEvent::MTE3_S>(eventMte3toS);

    constexpr float scalarValue = 0;
    Duplicate(inputLocal, scalarValue, ONCE_HANDLE_NUM512 * ONCE_HANDLE_NUM64);

    event_t eventS2Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventS2Mte2);
    WaitFlag<HardEvent::S_MTE2>(eventS2Mte2);

    for (int64_t i = 0; i < loop_time - 1; i++) {
        CalcSumC(inputLocal, gmOffsets, i, pre_loop_n, pre_loop_n, cal_m);

        event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    }

    if (loop_time > 1) {
        SetFlag<HardEvent::MTE3_S>(eventMte3toS);
        WaitFlag<HardEvent::MTE3_S>(eventMte3toS);

        Duplicate(inputLocal, scalarValue, pre_loop_n * Ceil(cal_m, FP32_ONE_BLOCK_COUNT));

        SetFlag<HardEvent::S_MTE2>(eventS2Mte2);
        WaitFlag<HardEvent::S_MTE2>(eventS2Mte2);
    }

    CalcSumC(inputLocal, gmOffsets, loop_time - 1, pre_loop_n, last_loop_n, cal_m);

    event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumC(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx,
                                                       int64_t preN, int64_t calcN, int64_t calcM)
{
    /*
    workspace -> UB -> reduceAdd -> GM
    */
    int64_t calcSize = calcM * sizeof(float);
    if (calcSize > MAX_BLOCK_LEN) {
        int64_t loop = calcM / MAX_BLOCK_LEN_SIZE_FP32;
        int64_t tail = calcM - loop * MAX_BLOCK_LEN_SIZE_FP32;
        // Each workspace C row spans the complete logical M dimension.  The
        // post kernel processes one M tile at a time, so the row stride is
        // curM rather than the tile width calcM.
        for (int64_t row = 0; row < calcN; ++row) {
            const int64_t rowOffset = (idx * preN + row) * curM + gmOffsets;
            for (int32_t i = 0; i < loop; i++) {
                DataCopy(inputLocal[i * MAX_BLOCK_LEN_SIZE_FP32],
                         workspaceSumGradC_[rowOffset + i * MAX_BLOCK_LEN_SIZE_FP32], MAX_BLOCK_LEN_SIZE_FP32);
            }
            DataCopyPad(inputLocal[loop * MAX_BLOCK_LEN_SIZE_FP32],
                        workspaceSumGradC_[rowOffset + loop * MAX_BLOCK_LEN_SIZE_FP32],
                        {static_cast<uint16_t>(1), static_cast<uint16_t>(tail * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
            event_t eventMte2toMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);
            StoreAccumulated(inputLocal, gmOffsets, calcM);
            event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        }
    } else {
        const int64_t alignedM = Ceil(calcM, CAME_ONE_BLOCK_SIZE_FP32);
        const uint8_t rightPadding = alignedM - calcM;
        // Vector reduction needs an aligned row stride.  Copy each logical row
        // from its full-M workspace row into its padded UB slot explicitly.
        for (int64_t row = 0; row < calcN; ++row) {
            const int64_t rowOffset = (idx * preN + row) * curM + gmOffsets;
            DataCopyPad(inputLocal[row * alignedM], workspaceSumGradC_[rowOffset],
                        {1, static_cast<uint16_t>(calcSize), 0, 0}, {true, 0, rightPadding, 0});
        }

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);

        ReduceAdd(inputLocal, calcN, alignedM);

        event_t eventV2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2Mte3);
        WaitFlag<HardEvent::V_MTE3>(eventV2Mte3);

        StoreAccumulated(inputLocal, gmOffsets, calcM);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumURC()
{
    LocalTensor<float> inputLocal = inputBuf.Get<float>(MAX_BUF_SIZE * 2);
    LocalTensor<float> high = inputLocal;
    LocalTensor<float> low = inputLocal[COMPENSATED_REDUCE_SIZE];
    LocalTensor<float> sumTmp = inputLocal[COMPENSATED_REDUCE_SIZE * 2];
    LocalTensor<float> recoveredTmp = inputLocal[COMPENSATED_REDUCE_SIZE * 3];
    float totalHigh = 0.0f;
    float totalLow = 0.0f;

    const uint64_t loopTime = DivCeil(curM, COMPENSATED_REDUCE_SIZE);
    for (uint64_t loopIndex = 0; loopIndex < loopTime; ++loopIndex) {
        const uint64_t offset = loopIndex * COMPENSATED_REDUCE_SIZE;
        const uint64_t count = curM - offset < COMPENSATED_REDUCE_SIZE ? curM - offset : COMPENSATED_REDUCE_SIZE;
        const uint64_t alignedCount = Ceil(count, FP32_ONE_BLOCK_COUNT);
        const uint8_t rightPadding = alignedCount - count;
        DataCopyPad(high, gmSumGradC_[offset], {1, static_cast<uint16_t>(count * sizeof(float)), 0, 0},
                    {true, 0, rightPadding, 0});
        Duplicate(low, static_cast<float>(0), alignedCount);

        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
        PipeBarrier<PIPE_V>();
        ApplyCamePart3CompensatedReduce(high, low, sumTmp, recoveredTmp, count);

        const float chunkHigh = high.GetValue(0);
        const float chunkLow = low.GetValue(0);
        const float sum = totalHigh + chunkHigh;
        const float split = sum - totalHigh;
        const float error = (totalHigh - (sum - split)) + (chunkHigh - split);
        totalHigh = sum;
        totalLow += chunkLow + error;
    }
    gmSumGradRC_.SetValue(0, totalHigh + totalLow);
    event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    ApplyCamePart3ReduceRows(accuUb, n, m);
}

#endif // _ASCENDC_APPLY_CAME_PART3_POST_H_
