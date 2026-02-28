/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEGMENT_SUM_SIMD_MULT_CORE_ADD_H
#define SEGMENT_SUM_SIMD_MULT_CORE_ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "segment_sum_struct.h"

namespace SegmentSum {
using namespace AscendC;

template <typename T1, typename T2>
class SegmentSumMultiCoreAdd
{
public:
    __aicore__ inline SegmentSumMultiCoreAdd(void){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR workspace, TPipe& pipeIn, const SegmentSumSimdTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInIds(LocalTensor<T2>& segmentIdsLocal);
    __aicore__ inline void CopyInSum(int32_t burstLen, int64_t colOffset);
    __aicore__ inline void CopyOutY(LocalTensor<T1>& yLocal, int32_t burstLen, T2 id, int64_t colOffset);
    __aicore__ inline void ComputeAndCopyOut(LocalTensor<T2> segmentIdsLocal, LocalTensor<T1>& yLocal, int32_t curLoopInners, int32_t curLoopInnersAlign, int64_t colOffset);

private:
    GlobalTensor<T1> yGm_;
    GlobalTensor<T1> sumWorkspace_;
    GlobalTensor<T2> segIdWorkspace_;

    TQue<QuePosition::VECIN, X_BUFFER_NUM> xQue_;
    TQue<QuePosition::VECOUT, TMP_BUFFER_NUM> tmpQue_;
    TBuf<QuePosition::VECCALC> segmentIdsBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;

    const SegmentSumSimdTilingData* tilingData_;

    uint32_t blockIdx_ = 0;

    int64_t colGmOffset_ = 0; // 当前核处理的数据块在GM列上的偏移

    int64_t rowUbLoop_ = 0; // 当前核的ub在行上的循环次数
    int64_t colUbLoop_ = 0; // 当前核的ub在列上的循环次数

    int64_t normalLoopInners_ = 0; // 当前核ub正常循环一次处理的列数
    int64_t tailLoopInners_ = 0; // 当前核ub尾循环一次处理的列数

    constexpr static int32_t blockNumT1_ = platform::GetUbBlockSize() / sizeof(T1);
};

template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::Init(GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn, const SegmentSumSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->usedCoreNumForMultAdd) {
        return;
    }

    colGmOffset_ = blockIdx_ * tilingData_->normalCoreMultAddInners;
    rowUbLoop_ = DOUBLE * tilingData_->blockNumInRow; // no need
    colUbLoop_ = blockIdx_ == tilingData_->usedCoreNumForMultAdd - 1 ? tilingData_->tailCoreMultAddInnerLoop : tilingData_->normalCoreMultAddInnerLoop;

    normalLoopInners_ = blockIdx_ == tilingData_->usedCoreNumForMultAdd - 1 ? tilingData_->tailCoreMultAddNormalLoopInners : tilingData_->normalCoreMultAddNormalLoopInners;
    tailLoopInners_ = blockIdx_ == tilingData_->usedCoreNumForMultAdd - 1 ? tilingData_->tailCoreMultAddTailLoopInners : tilingData_->normalCoreMultAddTailLoopInners;
    uint32_t segIdAddrOffset = (tilingData_->blockNumInRow * DOUBLE * tilingData_->innerDim * sizeof(T1) + sizeof(T2) - 1) / sizeof(T2);

    yGm_.SetGlobalBuffer((__gm__ T1*)y + colGmOffset_);
    sumWorkspace_.SetGlobalBuffer((__gm__ T1*)workspace + colGmOffset_);
    segIdWorkspace_.SetGlobalBuffer((__gm__ T2*)workspace + segIdAddrOffset);

    pipeIn.InitBuffer(xQue_, X_BUFFER_NUM, tilingData_->multAddXBufferSize);
    pipeIn.InitBuffer(tmpQue_, TMP_BUFFER_NUM, tilingData_->multAddYBufferSize);
    pipeIn.InitBuffer(segmentIdsBuf_, tilingData_->multAddIdsBufferSize);
    pipeIn.InitBuffer(yBuf_, tilingData_->multAddYBufferSize);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::CopyInIds(LocalTensor<T2>& segmentIdsLocal)
{
    DataCopyPadExtParams<T2> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = tilingData_->blockNumInRow * DOUBLE * sizeof(T2);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(segmentIdsLocal, segIdWorkspace_, dataCoptExtParams, dataCopyPadExtParams);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::CopyInSum(int32_t burstLen, int64_t colOffset)
{
    int64_t gmStride = tilingData_->innerDim - burstLen;
    LocalTensor<T1> xLocal = xQue_.AllocTensor<T1>();
    DataCopyPadExtParams<T1> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = tilingData_->blockNumInRow * DOUBLE;
    dataCoptExtParams.blockLen = burstLen * sizeof(T1);
    dataCoptExtParams.srcStride = gmStride * sizeof(T1);
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, sumWorkspace_[colOffset], dataCoptExtParams, dataCopyPadExtParams);
    xQue_.EnQue(xLocal);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::CopyOutY(LocalTensor<T1>& yLocal, int32_t burstLen, T2 id, int64_t colOffset)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(T1);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(yGm_[id * tilingData_->innerDim + colOffset], yLocal, dataCoptExtParams);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::ComputeAndCopyOut(LocalTensor<T2> segmentIdsLocal, LocalTensor<T1>& yLocal, int32_t curLoopInners, int32_t curLoopInnersAlign, int64_t colOffset)
{
    LocalTensor<T1> xLocal = xQue_.DeQue<T1>();
    Copy(yLocal, xLocal[curLoopInnersAlign], curLoopInners);
    
    // GetValue获取id前需要插同步
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId1);
    WaitFlag<HardEvent::MTE2_S>(eventId1);

    T2 preId = segmentIdsLocal.GetValue(1);

    for (int32_t i = 2; i < tilingData_->blockNumInRow * DOUBLE; i++) {
        T2 curId = segmentIdsLocal.GetValue(i);
        if (curId == preId) {
            Add(yLocal, yLocal, xLocal[i * curLoopInnersAlign], curLoopInners);
        } else if (curId == -1) {
            continue;
        } else { // curId != preId
            LocalTensor<T1> tmpLocal = tmpQue_.AllocTensor<T1>();
            Copy(tmpLocal, yLocal, curLoopInners);
            tmpQue_.EnQue(tmpLocal);
            LocalTensor<T1> outLocal = tmpQue_.DeQue<T1>();

            CopyOutY(outLocal, curLoopInners, preId, colOffset);
            tmpQue_.FreeTensor(outLocal);

            preId = curId;
            Copy(yLocal, xLocal[i * curLoopInnersAlign], curLoopInners);
        }
    }
    event_t eventId4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId4);
    WaitFlag<HardEvent::V_MTE3>(eventId4);
    
    CopyOutY(yLocal, curLoopInners, preId, colOffset);
    xQue_.FreeTensor(xLocal);
}


template <typename T1, typename T2>
__aicore__ inline void SegmentSumMultiCoreAdd<T1, T2>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNumForMultAdd) {
        return;
    }

    LocalTensor<T2> segmentIdsLocal = segmentIdsBuf_.Get<T2>();
    LocalTensor<T1> yLocal = yBuf_.Get<T1>();
    CopyInIds(segmentIdsLocal);

    int64_t curLoopInners;
    int64_t curLoopInnersAlign;
    int64_t colOffset;

    for (int64_t col = 0; col < colUbLoop_; col++) {
        curLoopInners = col == colUbLoop_ - 1 ? tailLoopInners_ : normalLoopInners_;
        curLoopInnersAlign = (curLoopInners + blockNumT1_ - 1) / blockNumT1_ * blockNumT1_;
        colOffset = col * normalLoopInners_;

        CopyInSum(curLoopInners, colOffset);
        ComputeAndCopyOut(segmentIdsLocal, yLocal, curLoopInners, curLoopInnersAlign, colOffset);
    }
}

}
#endif