/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEGMENT_SUM_SIMD_DETERM_H
#define SEGMENT_SUM_SIMD_DETERM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "segment_sum_struct.h"

namespace SegmentSum {
using namespace AscendC;

constexpr uint32_t DOUBLE = 2;


template <typename T1, typename T2>
class SegmentSumSimdDeterm
{
public:
    __aicore__ inline SegmentSumSimdDeterm(void){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR y, GM_ADDR workspace, TPipe& pipeIn, const SegmentSumSimdTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(int32_t copyCount, int32_t burstLen, int64_t xGmOffset);
    __aicore__ inline void CopyInSegmentIds(int32_t burstLen, int64_t segmentIdsGmOffset);
    __aicore__ inline void CopyOutY(LocalTensor<T1>& yLocal, int32_t burstLen, T2 id, int64_t colOffset);
    __aicore__ inline void ComputeSumAndCopyOut(LocalTensor<T1>& yLocal, int32_t curLoopOutters, int32_t curLoopInners, int32_t curLoopInnersAlign, int64_t colOffset, T2& curId);
    __aicore__ inline void CopyOutSegIdWorkspace(LocalTensor<T2>& tmpLocal);
    __aicore__ inline void CopyOutSumWorkspace(LocalTensor<T1>& yLocal, int32_t burstLen, int64_t colOffset, int32_t writePostion);


private:
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> segmentIdsGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> sumWorkspace_;
    GlobalTensor<T2> segIdWorkspace_;
    
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> segmentIdsQue_;
    TBuf<QuePosition::VECCALC> yBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    const SegmentSumSimdTilingData* tilingData_;

    uint32_t blockIdx_ = 0;
    uint32_t rowCoreIdx_ = 0; // 行核idx
    uint32_t colCoreIdx_ = 0; // 列核idx

    int64_t rowGmOffset_ = 0; // 当前核处理的数据块在GM行上的偏移
    int64_t colGmOffset_ = 0; // 当前核处理的数据块在GM列上的偏移

    int64_t rowUbLoop_ = 0; // 当前核的ub在行上的循环次数
    int64_t colUbLoop_ = 0; // 当前核的ub在列上的循环次数

    int64_t normalLoopOutters_ = 0; // 当前核ub正常循环一次处理的行数
    int64_t tailLoopOutters_ = 0; // 当前核ub尾循环一次处理的行数
    int64_t normalLoopInners_ = 0; // 当前核ub正常循环一次处理的列数
    int64_t tailLoopInners_ = 0; // 当前核ub尾循环一次处理的列数

    T2 preId_ = -1;
    T2 position0_ = -1;
    T2 position1_ = -1;
    bool isStartRowCore_ = false;
    bool isEndRowCore_ = false;
    bool isFirstId_ = true;
    constexpr static int32_t blockNumT1_ = platform::GetUbBlockSize() / sizeof(T1);
    // constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();

};

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::Init(
    GM_ADDR x, GM_ADDR segmentIds, GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn, const SegmentSumSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    rowCoreIdx_ = blockIdx_ / tilingData_->blockNumInCol;
    colCoreIdx_ = blockIdx_ % tilingData_->blockNumInCol;
    isStartRowCore_ = rowCoreIdx_ == 0; // 首行核
    isEndRowCore_ = rowCoreIdx_ == tilingData_->blockNumInRow - 1; // 尾行核

    rowGmOffset_ = rowCoreIdx_ * tilingData_->normalCoreOutterNum;
    colGmOffset_ = colCoreIdx_ * tilingData_->normalCoreInnerNum;

    rowUbLoop_ = rowCoreIdx_ == tilingData_->blockNumInRow - 1 ? tilingData_->tailCoreRowUbLoop : tilingData_->normalCoreRowUbLoop;
    colUbLoop_ = colCoreIdx_ == tilingData_->blockNumInCol - 1 ? tilingData_->tailCoreColUbLoop : tilingData_->normalCoreColUbLoop;

    normalLoopOutters_ = rowCoreIdx_ == tilingData_->blockNumInRow - 1 ? tilingData_->tailCoreNormalLoopOutters : tilingData_->normalCoreNormalLoopOutters;
    tailLoopOutters_ = rowCoreIdx_ == tilingData_->blockNumInRow - 1 ? tilingData_->tailCoreTailLoopOutters : tilingData_->normalCoreTailLoopOutters;
    normalLoopInners_ = colCoreIdx_ == tilingData_->blockNumInCol - 1 ? tilingData_->tailCoreNormalLoopInners : tilingData_->normalCoreNormalLoopInners;
    tailLoopInners_ = colCoreIdx_ == tilingData_->blockNumInCol - 1 ? tilingData_->tailCoreTailLoopInners : tilingData_->normalCoreTailLoopInners;
    uint32_t segIdAddrOffset = (tilingData_->blockNumInRow * DOUBLE * tilingData_->innerDim * sizeof(T1) + sizeof(T2) - 1) / sizeof(T2);
    
    xGm_.SetGlobalBuffer((__gm__ T1*)x + rowGmOffset_ * tilingData_->innerDim + colGmOffset_);
    segmentIdsGm_.SetGlobalBuffer((__gm__ T2*)segmentIds + rowGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ T1*)y + colGmOffset_);
    sumWorkspace_.SetGlobalBuffer((__gm__ T1*)workspace + rowCoreIdx_ * DOUBLE * tilingData_->innerDim + colGmOffset_);
    segIdWorkspace_.SetGlobalBuffer((__gm__ T2*)workspace + segIdAddrOffset + rowCoreIdx_ * DOUBLE);

    pipeIn.InitBuffer(xQue_, BUFFER_NUM, tilingData_->xBufferSize);
    pipeIn.InitBuffer(segmentIdsQue_, BUFFER_NUM, tilingData_->segmentIdBufferSize);
    pipeIn.InitBuffer(yBuf_, tilingData_->yBufferSize);
    pipeIn.InitBuffer(tmpBuf_, platform::GetUbBlockSize()); // 放头尾id

}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::CopyInX(int32_t copyCount, int32_t burstLen, int64_t xGmOffset)
{
    int64_t gmStride = tilingData_->innerDim - burstLen;
    LocalTensor<T1> xLocal = xQue_.AllocTensor<T1>();
    DataCopyPadExtParams<T1> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = copyCount;
    dataCoptExtParams.blockLen = burstLen * sizeof(T1);
    dataCoptExtParams.srcStride = gmStride * sizeof(T1);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[xGmOffset], dataCoptExtParams, dataCopyPadExtParams);
    xQue_.EnQue(xLocal);
}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::CopyInSegmentIds(int32_t burstLen, int64_t segmentIdsGmOffset)
{
    LocalTensor<T2> segmentIdsLocal = segmentIdsQue_.AllocTensor<T2>();
    DataCopyPadExtParams<T2> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(T2);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(segmentIdsLocal, segmentIdsGm_[segmentIdsGmOffset], dataCoptExtParams, dataCopyPadExtParams);
    segmentIdsQue_.EnQue(segmentIdsLocal);
}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::CopyOutY(LocalTensor<T1>& yLocal, int32_t burstLen, T2 id, int64_t colOffset)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(T1);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(yGm_[id * tilingData_->innerDim + colOffset], yLocal, dataCoptExtParams);
}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::CopyOutSegIdWorkspace(LocalTensor<T2>& tmpLocal)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = DOUBLE * sizeof(T2);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(segIdWorkspace_, tmpLocal, dataCoptExtParams);
}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::CopyOutSumWorkspace(LocalTensor<T1>& yLocal, int32_t burstLen, int64_t colOffset, int32_t writePostion)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(T1);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(sumWorkspace_[writePostion * tilingData_->innerDim + colOffset], yLocal, dataCoptExtParams);
}

template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::ComputeSumAndCopyOut(LocalTensor<T1>& yLocal, int32_t curLoopOutters, int32_t curLoopInners, int32_t curLoopInnersAlign, int64_t colOffset, T2& curId)
{
    LocalTensor<T1> xLocal = xQue_.DeQue<T1>();
    LocalTensor<T2> segmentIdsLocal = segmentIdsQue_.DeQue<T2>();

    // GetValue获取id前需要插同步
    event_t eventId3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId3);
    WaitFlag<HardEvent::MTE2_S>(eventId3);

    for (int32_t i = 0; i < curLoopOutters; i++) {
        curId = segmentIdsLocal.GetValue(i);
        if (curId == preId_) {
            Add(yLocal, yLocal, xLocal[i * curLoopInnersAlign], curLoopInners);
        } else if (curId != preId_ && preId_ == -1) {
            Copy(yLocal, xLocal[i * curLoopInnersAlign], curLoopInners);
            preId_ = curId;
        } else { // curId != preId_
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            if (isFirstId_ && !isStartRowCore_) {
                CopyOutSumWorkspace(yLocal, curLoopInners, colOffset, 0);
                position0_ = preId_;
                isFirstId_ = false;
            } else {
                CopyOutY(yLocal, curLoopInners, preId_, colOffset);
            }
            event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventId1);
            WaitFlag<HardEvent::MTE3_V>(eventId1);

            preId_ = curId;
            Copy(yLocal, xLocal[i * curLoopInnersAlign], curLoopInners);
        }
    }
    xQue_.FreeTensor(xLocal);
    segmentIdsQue_.FreeTensor(segmentIdsLocal);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumSimdDeterm<T1, T2>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }

    LocalTensor<T1> yLocal = yBuf_.Get<T1>();

    int64_t curLoopInners;
    int64_t curLoopInnersAlign;
    int64_t colOffset;
    int64_t xGmOffset;
    int64_t segmentIdsGmOffset;
    int64_t curLoopOutters;
    T2 curId;

    for (int64_t col = 0; col < colUbLoop_; col++) {
        isFirstId_ = true;
        preId_ = -1;
        curLoopInners = col == colUbLoop_ - 1 ? tailLoopInners_ : normalLoopInners_;
        curLoopInnersAlign = (curLoopInners + blockNumT1_ - 1) / blockNumT1_ * blockNumT1_;
        colOffset = col * normalLoopInners_;
        for (int64_t row = 0; row < rowUbLoop_; row++) {
            curLoopOutters = row == rowUbLoop_ - 1 ? tailLoopOutters_ : normalLoopOutters_;
            xGmOffset = row * normalLoopOutters_ * tilingData_->innerDim + col * normalLoopInners_;
            segmentIdsGmOffset = row * normalLoopOutters_;

            CopyInX(curLoopOutters, curLoopInners, xGmOffset);
            CopyInSegmentIds(curLoopOutters, segmentIdsGmOffset);

            ComputeSumAndCopyOut(yLocal, curLoopOutters, curLoopInners, curLoopInnersAlign, colOffset, curId);
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        // PipeBarrier<PIPE_ALL>();
        if (isEndRowCore_ && !isFirstId_) {
            CopyOutY(yLocal, curLoopInners, curId, colOffset);
        } else {
            CopyOutSumWorkspace(yLocal, curLoopInners, colOffset, 1);
            position1_ = curId;
        }
    }
    LocalTensor<T2> tmpLocal = tmpBuf_.Get<T2>();
    tmpLocal.SetValue(0, position0_);
    tmpLocal.SetValue(1, position1_);
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    CopyOutSegIdWorkspace(tmpLocal);
}


}
#endif