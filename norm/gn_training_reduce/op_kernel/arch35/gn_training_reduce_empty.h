/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gn_training_reduce_empty.h
 * \brief GNTrainingReduce empty tensor kernel implementation.
 */

#ifndef GN_TRAINING_REDUCE_EMPTY_H
#define GN_TRAINING_REDUCE_EMPTY_H

#include "gn_training_reduce_base.h"

namespace NsGNTrainingReduce {

constexpr float EMPTY_R_OUTPUT_VALUE = 0.0f; // 空归约和 = 0

__simd_vf__ inline void DuplicateEmptyROutputVfImpl(__ubuf__ float* outPtr, float value, uint32_t totalElems,
                                                    uint16_t repeatTime)
{
    constexpr uint32_t repPerVf = REP_F32;
    AscendC::Reg::RegTensor<float> dReg;
    AscendC::Reg::Duplicate(dReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        const int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(repPerVf);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(outPtr + off, dReg, mask);
    }
}

template <typename DType>
class GNTrainingReduceEmptyKernel {
public:
    using DT = DType;

    __aicore__ inline GNTrainingReduceEmptyKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, const GNTrainingReduceEmptyTilingData* td,
                                AscendC::TPipe* pipe)
    {
        (void)x; // EMPTY 分支不读输入
        td_ = td;
        pipe_ = pipe;
        sumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sum));
        squareSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(squareSum));
        if (td_->usedCoreNum == 0) {
            return; // EMPTY_A：全核早退，kPhysNodes = 0（不分配 buffer）
        }
        pipe_->InitBuffer(outBuf_, static_cast<uint32_t>(td_->postBufSize));
        evVtoMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        evMte3toV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process(int32_t processIdx)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return; // EMPTY_A：usedCoreNum=0 → 全核早退
        }

        int64_t aStart = 0;
        int64_t aEnd = 0;
        if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
            aStart = blockIdx * td_->aBigCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aBigCoreLoopCnt * td_->aUbFactor;
        } else {
            aStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt * td_->aUbFactor +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt * td_->aUbFactor;
            aEnd = aStart + td_->aSmallCoreLoopCnt * td_->aUbFactor;
        }
        if (aEnd > td_->aTotal) {
            aEnd = td_->aTotal;
        }
        if (aStart >= aEnd) {
            return;
        }

        // WAR：非首轮（processIdx==1）等上一轮 CopyOut 读完 outBuf_ 后再重写（§5.5）
        if (processIdx != 0) {
            WaitFlag<HardEvent::MTE3_V>(evMte3toV_);
        }

        DuplicateEmptyROutputVf();
        SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
        WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);

        for (int64_t aOff = aStart; aOff < aEnd; aOff += td_->aUbFactor) {
            const int64_t aLen = (aOff + td_->aUbFactor > aEnd) ? (aEnd - aOff) : td_->aUbFactor;
            DataCopyExtParams outParams;
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            outParams.blockCount = 1;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            if (processIdx == 0) {
                DataCopyPad(sumGm_[aOff], outBuf_.Get<float>(), outParams);
            } else {
                DataCopyPad(squareSumGm_[aOff], outBuf_.Get<float>(), outParams);
            }
        }

        // WAR：非末轮（processIdx==0）置位，供下一轮 Duplicate 前等待（§5.5）
        if (processIdx != N_REDUCES - 1) {
            SetFlag<HardEvent::MTE3_V>(evMte3toV_);
        }
    }

private:
    __aicore__ inline void DuplicateEmptyROutputVf()
    {
        __ubuf__ float* outPtr = reinterpret_cast<__ubuf__ float*>(outBuf_.Get<float>().GetPhyAddr());
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor);
        const uint16_t repeatTime = static_cast<uint16_t>(
            Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
        asc_vf_call<DuplicateEmptyROutputVfImpl>(outPtr, EMPTY_R_OUTPUT_VALUE, totalElems, repeatTime);
    }

    const GNTrainingReduceEmptyTilingData* td_ = nullptr;
    TPipe* pipe_ = nullptr;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    TBuf<QuePosition::VECCALC> outBuf_;
    event_t evVtoMTE3_;
    event_t evMte3toV_;
};

} // namespace NsGNTrainingReduce

#endif // GN_TRAINING_REDUCE_EMPTY_H
