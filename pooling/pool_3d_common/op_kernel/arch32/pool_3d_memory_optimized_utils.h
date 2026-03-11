/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pool_3d_memory_optimized_utils.h
 * \brief
 */

#ifndef POOL_3D_MEMORY_OPTIMIZED_UTILS_H_
#define POOL_3D_MEMORY_OPTIMIZED_UTILS_H_

#include "kernel_tiling/kernel_tiling.h"

namespace Pool3dMemCommon {
using namespace AscendC;
struct PoolVars {
    uint32_t cBlockIdx = 0;

    int64_t N = 1;
    int64_t C = 1;
    int64_t Di = 1;
    int64_t Hi = 1;
    int64_t Wi = 1;
    int64_t Do = 1;
    int64_t Ho = 1;
    int64_t Wo = 1;
    int64_t DiHiWi = 1;
    int64_t HiWi = 1;
    const int32_t VL_NUM = 64; // Vector calculate length / float size

    // 多核切分的整尾块
    int64_t ncFactor = 0;
    int64_t doFactor = 0;
    int64_t hoFactor = 0;
    int64_t woFactor = 0;
    int64_t ncTail = 0;
    int64_t doTail = 0;
    int64_t hoTail = 0;
    int64_t woTail = 0;

    // 多核切分的数量
    int64_t ncOuter = 0;
    int64_t doOuter = 0;
    int64_t hoOuter = 0;
    int64_t woOuter = 0;

    int64_t totalIdx = 0;    // 总UB计算块
    int64_t blockFactor = 0; // 每个核最多计算的UB块
    int64_t useCoreNum = 0;  // 使用核数
    int64_t blockTail = 0;   // 多核尾块

    int64_t beginIdx = 0; // 当前核计算块起始id
    int64_t endIdx = 0;   // 当前核计算块终止id
};

struct BlockVar {
    int64_t curNcIdx = 0;     // 当前nc块索引
    int64_t curNcFactor = 0;  // 当前nc块因子
    int64_t curDoIdx = 0;     // 当前do块索引
    int64_t curDoFactor = 0;  // 当前do块因子
    int64_t curHoIdx = 0;     // 当前ho块索引
    int64_t curHoFactor = 0;  // 当前ho块因子
    int64_t curWoIdx = 0;     // 当前wo块索引
    int64_t curWoFactor = 0;  // 当前wo块因子
};

template <typename TilingDataT>
__aicore__ inline void InitPoolVars(PoolVars& poolVars, const TilingDataT* __restrict__ tiling) {
    poolVars.totalIdx = tiling->totalIdx;
    poolVars.blockFactor = tiling->blockFactor;
    poolVars.blockTail = tiling->blockTail;
    poolVars.ncFactor = tiling->ncFactor;
    poolVars.woFactor = tiling->woFactor;
    poolVars.hoFactor = tiling->hoFactor;
    poolVars.doFactor = tiling->doFactor;
    poolVars.doOuter = tiling->doOuter;
    poolVars.doTail = tiling->doTail;
    poolVars.hoOuter = tiling->hoOuter;
    poolVars.hoTail = tiling->hoTail;
    poolVars.woOuter = tiling->woOuter;
    poolVars.woTail = tiling->woTail;
    poolVars.ncOuter = tiling->ncOuter;
    poolVars.ncTail = tiling->ncTail;
}

__aicore__ inline BlockVar 
CalcBlockVar(int64_t curIdx, const PoolVars& poolVars) {
    BlockVar indices;
    indices.curNcIdx = curIdx / (poolVars.doOuter * poolVars.hoOuter * poolVars.woOuter);
    indices.curNcFactor = indices.curNcIdx == (poolVars.ncOuter - 1) ? poolVars.ncTail : poolVars.ncFactor;
    auto tmpIdx = curIdx % (poolVars.doOuter * poolVars.hoOuter * poolVars.woOuter);
    indices.curDoIdx = tmpIdx / (poolVars.hoOuter * poolVars.woOuter);
    indices.curDoFactor = indices.curDoIdx == (poolVars.doOuter - 1) ? poolVars.doTail : poolVars.doFactor;
    tmpIdx = tmpIdx % (poolVars.hoOuter * poolVars.woOuter);
    indices.curHoIdx = tmpIdx / poolVars.woOuter;
    indices.curHoFactor = indices.curHoIdx == (poolVars.hoOuter - 1) ? poolVars.hoTail : poolVars.hoFactor;
    indices.curWoIdx = tmpIdx % poolVars.woOuter;
    indices.curWoFactor = indices.curWoIdx == (poolVars.woOuter - 1) ? poolVars.woTail : poolVars.woFactor;
    return indices;
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void CopyInputData(
    TQue<QuePosition::VECIN, QUEUE_DEPTH>& inputQue, 
    GlobalTensor<T>& xGm, 
    int64_t curNcFactor, 
    uint8_t diFactor, 
    uint8_t hiFactor, 
    uint8_t wiFactor, 
    int64_t xGmOffset, 
    const PoolVars& poolVars) {
    LocalTensor<T> xLocal = inputQue.template AllocTensor<T>();
    const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams paramsIn;
    paramsIn.blockCount = hiFactor;
    paramsIn.blockLen = wiFactor * sizeof(T);
    paramsIn.srcStride = (poolVars.Wi - wiFactor) * sizeof(T);
    paramsIn.dstStride = 0;
    for (int64_t ncCopyi = 0; ncCopyi < curNcFactor; ncCopyi++) {
        for (int64_t dCopyi = 0; dCopyi < diFactor; dCopyi++) {
            auto srcAddr = xGmOffset + ncCopyi * poolVars.DiHiWi + dCopyi * poolVars.HiWi;
            auto dstAddr = (ncCopyi * diFactor + dCopyi) * hiFactor * wiFactor16Align;
            DataCopyPad(xLocal[dstAddr], xGm[srcAddr], paramsIn, padParams);
        }
    }
    inputQue.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void OutTranspose(
    LocalTensor<float> xLocalTrans, LocalTensor<float> xLocal, int32_t rowNum, int32_t colNum) {
    LocalTensor<float> dstList[16];
    LocalTensor<float> srcList[16];

    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    if (colNum == 8) {
        transDataParams.repeatTimes = rowNum / 16;
        transDataParams.dstRepStride = 2;
        transDataParams.srcRepStride = 16;

        for (int32_t i = 0; i < 16; i++) {
            srcList[i] = xLocal[i * 8];
        }

        for (int32_t i = 0; i < 8; i++) {
            dstList[i * 2] = xLocalTrans[i * rowNum];
            dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8];
        }

        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        TransDataTo5HD<float>(dstList, srcList, transDataParams);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
    } else {
        transDataParams.repeatTimes = colNum / 8;
        transDataParams.dstRepStride = rowNum;
        transDataParams.srcRepStride = 1;
        for (int32_t j = 0; j < rowNum / 16; j++) {
            for (int32_t i = 0; i < 16; i++) {
                srcList[i] = xLocal[i * colNum + j * 16 * colNum];
            }

            for (int32_t i = 0; i < 8; i++) {
                dstList[i * 2] = xLocalTrans[i * rowNum + j * 16];
                dstList[i * 2 + 1] = xLocalTrans[i * rowNum + 8 + j * 16];
            }

            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            TransDataTo5HD<float>(dstList, srcList, transDataParams);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
        }
    }
}

template <typename T, int32_t QUEUE_DEPTH>
__aicore__ inline void TransposeInput(
    TQue<QuePosition::VECIN, QUEUE_DEPTH>& inputQue, 
    TBuf<TPosition::VECCALC>& inputTransBuffer, 
    TBuf<TPosition::VECCALC>& mulWBuffer, 
    int64_t curNcFactor, 
    uint8_t diFactor, 
    uint8_t hiFactor, 
    uint8_t wiFactor, 
    const PoolVars& poolVars) {
    const uint8_t wiFactor16Align = Ceil(wiFactor, 32 / sizeof(T)) * 32 / sizeof(T);
    const uint8_t wiFactorAlign = Ceil(wiFactor, 8) * 8;
    LocalTensor<T> xLocal = inputQue.template DeQue<T>();
    LocalTensor<float> xLocalTransVL = inputTransBuffer.Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        OutTranspose<T>(xLocalTransVL, xLocal, poolVars.VL_NUM, diFactor * hiFactor * wiFactorAlign);
    } else {
        LocalTensor<float> xLocalCast = mulWBuffer.Get<float>();
        UnaryRepeatParams repeatCastParams{
            (uint16_t)(wiFactorAlign / 8), (uint16_t)(wiFactor16Align / 8),
            (uint8_t)(wiFactorAlign / 8 * Ceil(diFactor * hiFactor, 2)),
            (uint8_t)(wiFactor16Align / 8 * Ceil(diFactor * hiFactor, 2))};

        Cast(xLocalCast, xLocal, RoundMode::CAST_NONE, wiFactor16Align * curNcFactor * diFactor * hiFactor);
        PipeBarrier<PIPE_V>();
        Adds(
            xLocalCast, xLocalCast, float(0.0), uint8_t(wiFactorAlign * Ceil(diFactor * hiFactor, 2)), curNcFactor * 2,
            repeatCastParams);

        PipeBarrier<PIPE_V>();
        OutTranspose<T>(xLocalTransVL, xLocalCast, poolVars.VL_NUM, diFactor * hiFactor * wiFactorAlign);
    }
    PipeBarrier<PIPE_V>();
    inputQue.FreeTensor(xLocal);
}

} // namespace Pool3dMemCommon

#endif // POOL_3D_MEMORY_OPTIMIZED_UTILS_H_
