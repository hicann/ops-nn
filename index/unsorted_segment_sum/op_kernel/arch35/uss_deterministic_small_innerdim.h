/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file uss_deterministic_small_innerdim.h
 * \brief uss_deterministic_small_innerdim - single core single thread for innerDim=1
 */
#pragma once
#ifndef USS_DETERMINISTIC_SMALL_INNERDIM_H_
#define USS_DETERMINISTIC_SMALL_INNERDIM_H_

#include "unsorted_segment_base.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace UnsortedSegmentSum {

using namespace AscendC;

constexpr uint32_t SINGLE_THREAD_NUM = 1;

template <typename X_T, typename IDS_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SINGLE_THREAD_NUM) inline void SimtSmallInnerDimAccumulate(
    __gm__ X_T* outputGm, __local_mem__ X_T* xTensor, __local_mem__ IDS_T* sortedIdTensor,
    __local_mem__ uint32_t* sortedIdIndexTensor, uint32_t rows, uint64_t outputOuterDim)
{
    int64_t startSegId = -1;
    float sumValue = 0.0f;

    for (uint32_t i = 0; i < rows; i++) {
        int64_t curSegId = static_cast<int64_t>(sortedIdTensor[i]);

        if (curSegId < 0 || curSegId >= static_cast<int64_t>(outputOuterDim)) {
            if (i == rows - 1 && startSegId >= 0) {
                asc_atomic_add(outputGm + static_cast<uint64_t>(startSegId), static_cast<X_T>(sumValue));
            }
            continue;
        }

        float tmpValue = static_cast<float>(xTensor[sortedIdIndexTensor[i]]);

        if (startSegId < 0) {
            startSegId = curSegId;
            sumValue = tmpValue;
        } else if (curSegId == startSegId) {
            sumValue += tmpValue;
        } else {
            asc_atomic_add(outputGm + static_cast<uint64_t>(startSegId), static_cast<X_T>(sumValue));
            startSegId = curSegId;
            sumValue = tmpValue;
        }

        if (i == rows - 1) {
            asc_atomic_add(outputGm + static_cast<uint64_t>(startSegId), static_cast<X_T>(sumValue));
        }
    }
}

template <typename X_T, typename IDS_T>
class USSKernelDeterministicSmallInnerDim {
public:
    __aicore__ inline USSKernelDeterministicSmallInnerDim(const UnsortedSegmentSumDetermSmallInnerDimTilingData* tiling,
                                                          TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInData(uint32_t rowNums, uint64_t segmentIdOffset);

private:
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSumDetermSmallInnerDimTilingData* td_;
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> yGm_;
    GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> segmentIdQueue_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf;
    TBuf<QuePosition::VECCALC> sortedIdBuf;
    TBuf<QuePosition::VECCALC> sortedIdIndexBuf;

    uint64_t inputOuterDim_{0};
    uint64_t outputOuterDim_{0};
    uint64_t rowsNumInUb_{0};
    uint32_t sortSharedBufSize_{0};
    uint64_t innerDim_{1};
};

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicSmallInnerDim<X_T, IDS_T>::Init(GM_ADDR x, GM_ADDR segmentIds,
                                                                             GM_ADDR output)
{
    inputOuterDim_ = td_->inputOuterDim;
    outputOuterDim_ = td_->outputOuterDim;
    rowsNumInUb_ = td_->rowsNumInUB;
    sortSharedBufSize_ = td_->sortSharedBufSize;
    innerDim_ = td_->innerDim;

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));

    InitGlobalMemory(yGm_, outputOuterDim_ * innerDim_, (X_T)0);

    pipe_->Reset();

    uint32_t xQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_ * innerDim_ * sizeof(X_T)),
                                               UB_AGLIN_VALUE);
    uint32_t segmentIdQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_ * sizeof(IDS_T)),
                                                       UB_AGLIN_VALUE);
    uint32_t sortedIdIndexBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_ * sizeof(uint32_t)),
                                                         UB_AGLIN_VALUE);

    pipe_->InitBuffer(xQueue_, 1, xQueueSize);
    pipe_->InitBuffer(segmentIdQueue_, 1, segmentIdQueueSize);
    pipe_->InitBuffer(sharedTmpBuf, sortSharedBufSize_);
    pipe_->InitBuffer(sortedIdBuf, segmentIdQueueSize);
    pipe_->InitBuffer(sortedIdIndexBuf, sortedIdIndexBufSize);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicSmallInnerDim<X_T, IDS_T>::CopyInData(uint32_t rowNums,
                                                                                   uint64_t segmentIdOffset)
{
    LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
    uint64_t inputOffset = segmentIdOffset * innerDim_;
    DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(rowNums * innerDim_ * sizeof(X_T)), 0, 0, 0};
    DataCopyPadExtParams<X_T> dataPadParam{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[inputOffset], dataCopyParam, dataPadParam);
    xQueue_.EnQue<X_T>(xLocal);

    LocalTensor<IDS_T> segmentIdLocal = segmentIdQueue_.AllocTensor<IDS_T>();
    DataCopyExtParams idCopyParam{(uint16_t)1, (uint32_t)(rowNums * sizeof(IDS_T)), 0, 0, 0};
    DataCopyPadExtParams<IDS_T> idPadParam{false, 0, 0, 0};
    DataCopyPad(segmentIdLocal, idsGm_[segmentIdOffset], idCopyParam, idPadParam);
    segmentIdQueue_.EnQue<IDS_T>(segmentIdLocal);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicSmallInnerDim<X_T, IDS_T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    uint32_t loopCnt = Ops::Base::CeilDiv(static_cast<uint32_t>(inputOuterDim_), static_cast<uint32_t>(rowsNumInUb_));
    for (uint32_t loop = 0; loop < loopCnt; ++loop) {
        uint32_t curProcessRowsNum = loop == loopCnt - 1 ?
                                         static_cast<uint32_t>(inputOuterDim_ - (loopCnt - 1) * rowsNumInUb_) :
                                         static_cast<uint32_t>(rowsNumInUb_);
        uint64_t segmentIdOffset = loop * rowsNumInUb_;

        CopyInData(curProcessRowsNum, segmentIdOffset);

        LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();
        LocalTensor<IDS_T> segmentIdLocal = segmentIdQueue_.DeQue<IDS_T>();

        LocalTensor<IDS_T> sortedIdTensor = sortedIdBuf.Get<IDS_T>();
        LocalTensor<uint32_t> sortedIdIndexTensor = sortedIdIndexBuf.Get<uint32_t>();
        LocalTensor<uint8_t> sharedTmpTensor = sharedTmpBuf.Get<uint8_t>();

        AscendC::Sort<IDS_T, false, sortConfig>(sortedIdTensor, sortedIdIndexTensor, segmentIdLocal, sharedTmpTensor,
                                                curProcessRowsNum);

        asc_vf_call<SimtSmallInnerDimAccumulate<X_T, IDS_T>>(
            dim3(SINGLE_THREAD_NUM), (__gm__ X_T*)(yGm_.GetPhyAddr()), (__local_mem__ X_T*)(xLocal.GetPhyAddr()),
            (__local_mem__ IDS_T*)(sortedIdTensor.GetPhyAddr()),
            (__local_mem__ uint32_t*)(sortedIdIndexTensor.GetPhyAddr()), curProcessRowsNum, outputOuterDim_);

        xQueue_.FreeTensor(xLocal);
        segmentIdQueue_.FreeTensor(segmentIdLocal);
    }
}

} // namespace UnsortedSegmentSum
#endif