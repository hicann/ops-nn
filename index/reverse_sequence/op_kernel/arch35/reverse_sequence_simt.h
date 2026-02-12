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
 * \file reverse_sequence_simt.h
 * \brief
 */
#ifndef REVERSE_SEQUENCE_SIMT_H_
#define REVERSE_SEQUENCE_SIMT_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "reverse_sequence_struct.h"

namespace ReverseSequence {
using namespace AscendC;
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t PARAM_DIV_NUM = 8;
constexpr uint32_t DOUBLE = 2;

template <typename CompType>
struct ReverseSequenceQuickDivParam {
    CompType m0{1};
    CompType m1{1};
    CompType m2{1};
    CompType m3{1};
    CompType shift0{1};
    CompType shift1{1};
    CompType shift2{1};
    CompType shift3{1};
};

template <typename T, typename SeqType, typename CompType>
class ReverseSequenceSimt
{
public:
    __aicore__ inline ReverseSequenceSimt(TPipe *pipe, const ReverseSequenceSimtTilingData4RegBase *tilingData)
                        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ReverseCompute(CompType curOffset, CompType xUBSize);
    __aicore__ inline void CopyOut(CompType offset, int64_t blockLen);

private:
    
    TPipe *pipe_;
    const ReverseSequenceSimtTilingData4RegBase *tilingData_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<SeqType> seqGm_;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    int64_t seqLen_ = 0;
    int64_t blockIdx_ = 0;
    int64_t blockNum_ = 0;
    int64_t curUbLoop_ = 0;
    int64_t tailUbSize_ = 0;
};

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSimt<T, SeqType, CompType>::Init(GM_ADDR x, GM_ADDR seqLengths, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    seqGm_.SetGlobalBuffer((__gm__ SeqType*)seqLengths);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    
    pipe_->InitBuffer(outQue_, BUFFER_NUM, tilingData_->xUbFactor * sizeof(T));

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    curUbLoop_ = tilingData_->xUbLoop;
    tailUbSize_ = tilingData_->xTailUbLoopSize;
    if (blockIdx_ == tilingData_->usedCoreNums - 1) {
        curUbLoop_ = tilingData_->xTailCoreLoop;
        tailUbSize_ = tilingData_->xTailCoreTailLoopSize;
    }
}

template <typename T, typename SeqType, typename CompType>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void ReverseSimtCompute(
    __gm__ T* xGm, __gm__ SeqType* seqGm, __local_mem__ T* outLocal, CompType curOffset, CompType xUbSize,
    CompType batchDim, CompType seqDim, CompType reverseSize, CompType m0, CompType m1, CompType m2, CompType m3, CompType shift0, CompType shift1, CompType shift2, CompType shift3)
{
    for (CompType i = Simt::GetThreadIdx(); i < xUbSize; i += Simt::GetThreadNum()) {
        CompType xOffset = curOffset + i;
        CompType batchPreAxis = Simt::UintDiv(xOffset, m0, shift0);
        CompType batchIdx = Simt::UintDiv(batchPreAxis, m1, shift1);
        CompType batchDimIdx = batchPreAxis - batchIdx * batchDim; // xOffSet / batchSize % batchDim

        CompType reverseNum = static_cast<CompType>(seqGm[batchDimIdx]);
        CompType seqPreAxis = Simt::UintDiv(xOffset, m2, shift2); // xOffSet / seqSize
        CompType reverseSizeMod = xOffset - seqPreAxis * reverseSize; // xOffSet % seqSize
        CompType seqIdx = Simt::UintDiv(seqPreAxis, m3, shift3);  // xOffSet / seqSize / seqDim
        CompType seqDimIdx = seqPreAxis - seqIdx * seqDim; // xOffSet / seqSize % seqDim
        
        if (reverseNum > seqDim) {
            reverseNum = seqDim;
        }

        if (seqDimIdx < reverseNum) {
            CompType reverseOffset = (reverseNum - seqDimIdx - 1) * reverseSize + seqIdx * seqDim * reverseSize + reverseSizeMod; // seqDim:2,batchDim:3 (a,b,c,d) -> c余*d + ab*c*d + d余
            outLocal[i] = xGm[reverseOffset];
        }

        if (seqDimIdx >= reverseNum) {
            outLocal[i] = xGm[xOffset];
        }
    }
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSimt<T, SeqType, CompType>::ReverseCompute(CompType curOffset, CompType xUBSize)
{
    LocalTensor<T> outLocal = outQue_.AllocTensor<T>();

    ReverseSequenceQuickDivParam<CompType> params;
    GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<CompType>(tilingData_->batchSize));
    GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<CompType>(tilingData_->batchDim));
    GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<CompType>(tilingData_->reverseSize));
    GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<CompType>(tilingData_->seqDim));

    Simt::VF_CALL<ReverseSimtCompute<T, SeqType, CompType>>(
        Simt::Dim3(USED_THREAD), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ SeqType*)(seqGm_.GetPhyAddr()),
        (__local_mem__ T*)(outLocal.GetPhyAddr()), curOffset, xUBSize, tilingData_->batchDim, tilingData_->seqDim, 
        tilingData_->reverseSize, params.m0, params.m1, params.m2, params.m3, params.shift0, params.shift1, params.shift2, params.shift3);
    
    outQue_.EnQue(outLocal);
}

template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSimt<T, SeqType, CompType>::CopyOut(CompType offset, int64_t blockLen)
{
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = blockLen * sizeof(T);
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    LocalTensor<T> yLocal = outQue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, copyExtParams);
    outQue_.FreeTensor(yLocal);
}
 	 
template <typename T, typename SeqType, typename CompType>
__aicore__ inline void ReverseSequenceSimt<T, SeqType, CompType>::Process()
{
    if (blockIdx_ > blockNum_) {
        return;
    }
    
    CompType curCoreOffset = blockIdx_ * tilingData_->perCoreHandleNums;
    CompType curOffset = curCoreOffset;
    if (blockIdx_ < tilingData_->usedCoreNums) {
        for (CompType i = 0; i < curUbLoop_ - 1; i++) {
            auto vWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
            WaitFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
            ReverseCompute(curOffset, tilingData_->xUbFactor);
            auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
            WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
            CopyOut(curOffset, tilingData_->xUbFactor);
            curOffset += tilingData_->xUbFactor;
        }
        auto vWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
        WaitFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
        ReverseCompute(curOffset, tailUbSize_);
        auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
        CopyOut(curOffset, tailUbSize_);
    }
}

}
#endif