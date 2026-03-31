/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common.h
 * \brief 
 */

#ifndef COMMON_H
#define COMMON_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace ScatterElementsV2NS {
constexpr uint32_t CACHE_CAPACITY = 16384;
constexpr uint32_t BYTE_ALIGNMENT = 32;
constexpr uint32_t HALF_BYTE_ALIGNMENT = 16;
constexpr uint32_t BASE_TILE_SIZE = 128;
constexpr uint32_t HALF_TILE_SIZE = 64;

constexpr uint64_t X_LOCAL_LENGTH = 40000;
constexpr uint64_t INDICES_LOCAL_LENGTH = 2048;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t MTE_UPDATES_MODE = 2; // updates dimvalue超过indices，但仍批量搬运

constexpr uint32_t MAX_BATCH_PARTS = 5;           // 批次分割的最大parts数
constexpr uint32_t GATHER_BATCH_SIZE = 32;        // gather批量大小
constexpr uint32_t TRANSPOSE_WEIGHT_UNIT = 8;     // 转置权重单元
constexpr uint32_t TRANSPOSE_TASK_UNIT = 16;      // 转置任务单元
constexpr uint32_t OFFSET_TABLE_SIZE = 128;       // 偏移表大小
constexpr uint32_t ALL_UB_SIZE = CACHE_CAPACITY * 3 * 4;  // UB总大小 192KB
constexpr uint32_t LOOP_UNROLL_SIZE = 8;          // 循环展开大小
constexpr uint32_t AGG_INDICES_NUM = 1024;           // 聚合indices数

using namespace AscendC;
// cpu等待vector计算单元完成计算
__aicore__ inline void PIPE_V_S() {
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}
// cpu等待MTE2搬运单元完成搬运
__aicore__ inline void PIPE_MTE2_S() {
    event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
}
// cpu等待MTE3搬运单元完成搬运
__aicore__ inline void PIPE_MTE3_S() {
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}
// mte3等待cpu完成计算
__aicore__ inline void PIPE_S_MTE3() {
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}
// mte2等待cpu完成计算
__aicore__ inline void PIPE_S_MTE2() {
    event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
}
// V等待cpu完成计算
__aicore__ inline void PIPE_S_V() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

__aicore__ inline void TransposeFloat(
        LocalTensor<float>& dstLocal, LocalTensor<float>& srcLocal, const uint64_t& h, const uint64_t& w)
{
    uint64_t srcList[HALF_BYTE_ALIGNMENT];
    uint64_t dstList[HALF_BYTE_ALIGNMENT];
    uint64_t hAlign = (h + HALF_BYTE_ALIGNMENT - 1) / HALF_BYTE_ALIGNMENT * HALF_BYTE_ALIGNMENT;
    uint64_t blockNum = BYTE_ALIGNMENT / sizeof(float);
    uint64_t wAlign = (w + blockNum - 1) / blockNum * blockNum;
    uint64_t blockPerTransLen = HALF_BYTE_ALIGNMENT / blockNum;

    for (size_t j = 0; j < hAlign / HALF_BYTE_ALIGNMENT; j++) {
        for (size_t i = 0; i < HALF_BYTE_ALIGNMENT; i++) {
            srcList[i] = (uint64_t)(srcLocal[j * HALF_BYTE_ALIGNMENT * wAlign + i * wAlign].GetPhyAddr());
            dstList[i] =
                (uint64_t)(dstLocal[j * HALF_BYTE_ALIGNMENT + i / blockPerTransLen * hAlign + i % blockPerTransLen * blockNum]
                                .GetPhyAddr());
        }
        TransDataTo5HDParams transDataParamsFloat;
        transDataParamsFloat.repeatTimes = wAlign / blockNum;
        if (transDataParamsFloat.repeatTimes == 1) {
            transDataParamsFloat.srcRepStride = 0;
            transDataParamsFloat.dstRepStride = 0;
        } else {
            transDataParamsFloat.srcRepStride = 1;
            transDataParamsFloat.dstRepStride = hAlign;
        }
        TransDataTo5HD<float>(dstList, srcList, transDataParamsFloat);
    }
}

__aicore__ inline void TransposeHalf(
        LocalTensor<half>& dstLocal, LocalTensor<half>& srcLocal, const uint64_t& h, const uint64_t& w)
{
    uint64_t srcList[HALF_BYTE_ALIGNMENT];
    uint64_t dstList[HALF_BYTE_ALIGNMENT];
    uint64_t hAlign = (h + HALF_BYTE_ALIGNMENT - 1) / HALF_BYTE_ALIGNMENT * HALF_BYTE_ALIGNMENT;
    uint64_t blockNum = BYTE_ALIGNMENT / sizeof(half);
    uint64_t wAlign = (w + blockNum - 1) / blockNum * blockNum;

    for (size_t j = 0; j < hAlign / HALF_BYTE_ALIGNMENT; j++) {
        for (size_t i = 0; i < HALF_BYTE_ALIGNMENT; i++) {
            srcList[i] = (uint64_t)(srcLocal[j * HALF_BYTE_ALIGNMENT * wAlign + i * wAlign].GetPhyAddr());
            dstList[i] = (uint64_t)(dstLocal[j * HALF_BYTE_ALIGNMENT + i * hAlign].GetPhyAddr());
        }
        TransDataTo5HDParams transDataParamsHalf;
        transDataParamsHalf.repeatTimes = wAlign / blockNum;
        if (transDataParamsHalf.repeatTimes == 1) {
            transDataParamsHalf.srcRepStride = 0;
            transDataParamsHalf.dstRepStride = 0;
        } else {
            transDataParamsHalf.srcRepStride = 1;
            transDataParamsHalf.dstRepStride = hAlign;
        }
        TransDataTo5HD<half>(dstList, srcList, transDataParamsHalf);
    }
}
}
#endif