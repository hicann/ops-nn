/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_APPLY_CAME_PART3_POST_CALC_H_
#define ASCENDC_APPLY_CAME_PART3_POST_CALC_H_

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

    for (int64_t i = 0; i < loop_time - 1; i++) {
        CalcSumC(inputLocal, gmOffsets, i, pre_loop_n, pre_loop_n, cal_m);

        event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    }

    CalcSumC(inputLocal, gmOffsets, loop_time - 1, pre_loop_n, last_loop_n, cal_m);

    event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumCLarge(LocalTensor<float> inputLocal, int64_t gmOffsets,
                                                            int64_t idx, int64_t preN, int64_t calcN, int64_t calcM)
{
    const int64_t loop = calcM / MAX_BLOCK_LEN_SIZE_FP32;
    const int64_t tail = calcM - loop * MAX_BLOCK_LEN_SIZE_FP32;
    for (int64_t row = 0; row < calcN; ++row) {
        const int64_t rowOffset = (idx * preN + row) * curM + gmOffsets;
        for (int32_t i = 0; i < loop; i++) {
            DataCopyPad(inputLocal[i * MAX_BLOCK_LEN_SIZE_FP32],
                        workspaceSumGradCHigh_[rowOffset + i * MAX_BLOCK_LEN_SIZE_FP32],
                        {1, static_cast<uint16_t>(MAX_BLOCK_LEN_SIZE_FP32 * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        }
        if (tail > 0) {
            DataCopyPad(inputLocal[loop * MAX_BLOCK_LEN_SIZE_FP32],
                        workspaceSumGradCHigh_[rowOffset + loop * MAX_BLOCK_LEN_SIZE_FP32],
                        {static_cast<uint16_t>(1), static_cast<uint16_t>(tail * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
        }
        event_t eventMte2toMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventMte2toMte3);
        event_t eventMte2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2toS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2toS);
        StoreAccumulated(inputLocal, gmOffsets, calcM);
        for (int32_t i = 0; i < loop; i++) {
            DataCopyPad(inputLocal[i * MAX_BLOCK_LEN_SIZE_FP32],
                        workspaceSumGradCLow_[rowOffset + i * MAX_BLOCK_LEN_SIZE_FP32],
                        {1, static_cast<uint16_t>(MAX_BLOCK_LEN_SIZE_FP32 * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        }
        if (tail > 0) {
            DataCopyPad(inputLocal[loop * MAX_BLOCK_LEN_SIZE_FP32],
                        workspaceSumGradCLow_[rowOffset + loop * MAX_BLOCK_LEN_SIZE_FP32],
                        {static_cast<uint16_t>(1), static_cast<uint16_t>(tail * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
        }
        eventMte2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2toS);
        WaitFlag<HardEvent::MTE2_S>(eventMte2toS);
        StoreAccumulated(inputLocal, gmOffsets, calcM);
        event_t eventMte3toMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3toMte2);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumCAligned(LocalTensor<float> inputLocal, int64_t gmOffsets,
                                                              int64_t idx, int64_t preN, int64_t calcN, int64_t calcM)
{
    const int64_t calcSize = calcM * sizeof(float);
    const int64_t alignedM = Ceil(calcM, CAME_ONE_BLOCK_SIZE_FP32);
    const uint8_t rightPadding = alignedM - calcM;
    for (int64_t row = 0; row < calcN; ++row) {
        const int64_t rowOffset = (idx * preN + row) * curM + gmOffsets;
        DataCopyPad(inputLocal[row * alignedM], workspaceSumGradCHigh_[rowOffset],
                    {1, static_cast<uint16_t>(calcSize), 0, 0}, {true, 0, rightPadding, 0});
    }
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    ReduceAdd(inputLocal, calcN, alignedM);
    event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV2S);
    WaitFlag<HardEvent::V_S>(eventV2S);
    StoreAccumulated(inputLocal, gmOffsets, calcM);
    for (int64_t row = 0; row < calcN; ++row) {
        const int64_t rowOffset = (idx * preN + row) * curM + gmOffsets;
        DataCopyPad(inputLocal[row * alignedM], workspaceSumGradCLow_[rowOffset],
                    {1, static_cast<uint16_t>(calcSize), 0, 0}, {true, 0, rightPadding, 0});
    }
    event_t eventMte2Low = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2Low);
    WaitFlag<HardEvent::MTE2_V>(eventMte2Low);
    ReduceAdd(inputLocal, calcN, alignedM);
    event_t eventLowV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventLowV2S);
    WaitFlag<HardEvent::V_S>(eventLowV2S);
    StoreAccumulated(inputLocal, gmOffsets, calcM);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::CalcSumC(LocalTensor<float> inputLocal, int64_t gmOffsets, int64_t idx,
                                                       int64_t preN, int64_t calcN, int64_t calcM)
{
    if (calcM * static_cast<int64_t>(sizeof(float)) > MAX_BLOCK_LEN) {
        CalcSumCLarge(inputLocal, gmOffsets, idx, preN, calcN, calcM);
        return;
    }
    CalcSumCAligned(inputLocal, gmOffsets, idx, preN, calcN, calcM);
}

template <typename T>
__aicore__ inline void ApplyCamePart3Post<T>::ReduceAdd(LocalTensor<float> accuUb, int64_t n, int64_t m)
{
    ApplyCamePart3ReduceRows(accuUb, n, m);
}
#endif // ASCENDC_APPLY_CAME_PART3_POST_CALC_H_
