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
 * \file conv3d_v2_instr_dequant_impl.h
 * \brief
 */

#ifndef CONV3D_V2_INSTR_DEQUANT_IMPL_H
#define CONV3D_V2_INSTR_DEQUANT_IMPL_H
 
#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"

namespace Conv3dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class DeQuantL0C2UBTools {
public:
    __aicore__ inline DeQuantL0C2UBTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        mStride = self_->ctx.convTiling->mUB * BLOCK_L0_N;
        isFirst = true;

        fixpipeParams.params.ndNum = 1;
        fixpipeParams.dstStride = self_->ctx.convTiling->nUB;
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0 = m;
        currentNL0 = n;
    }

    __aicore__ inline void CopyOut()
    {
        fixpipeParams.srcStride = self_->ctx.currentML0Align;

        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            maxMIter = CeilDiv(currentML0, self_->ctx.convTiling->mUB) - 1;
            mUbTail = currentML0 % self_->ctx.convTiling->mUB;
        } else {
            maxWoUbLoopTimes = CeilDiv(self_->ctx.currentWoL0, self_->ctx.convTiling->mUB);
            maxWoUbIter = maxWoUbLoopTimes - 1;
            maxMIter = maxWoUbLoopTimes * self_->ctx.currentHoL0 - 1;
            mUbTail = self_->ctx.currentWoL0 % self_->ctx.convTiling->mUB;
        }
        mUbTail = mUbTail == 0 ? self_->ctx.convTiling->mUB : mUbTail;
        
        fixpipeParams.nSize = self_->ctx.currentNL0Align;

        uint32_t srcOffset = 0;
        for (uint32_t mIter = 0; mIter <= maxMIter; ++mIter) {
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                self_->ctx.currentMUb = mIter == maxMIter ? mUbTail : self_->ctx.convTiling->mUB;
            } else {
                woUbIter = mIter % maxWoUbLoopTimes;
                self_->ctx.currentMUb = woUbIter == maxWoUbIter ? mUbTail : self_->ctx.convTiling->mUB;
                srcOffset = woUbIter * mStride + (mIter / maxWoUbLoopTimes) * self_->ctx.currentWoL0 * BLOCK_L0_N;
            }
            fixpipeParams.mSize = self_->ctx.currentMUb;
            fixpipeParams.subBlockId = subBlockId;

            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_FIX>(subBlockId * VEC_ID_MAX + CV_SYNC_ID_MTE3_FIXP);
            Fixpipe<typename Intf::L0cT, typename Intf::L0cT, CFG_ROW_MAJOR_UB>(
                self_->ctx.mmadResUbTensor, self_->ctx.cl0[srcOffset], fixpipeParams);
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_FIX>(subBlockId * VEC_ID_MAX + CV_SYNC_ID_FIXP_V);

            subBlockId ^= 1;
            isFirst = false;
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                srcOffset += mStride;
            }
        }
    }

private:
    Intf *self_ = nullptr;

    FixpipeParamsC310<CFG_ROW_MAJOR_UB.format> fixpipeParams;

    uint32_t currentML0 = 0;
    uint32_t currentNL0 = 0;
    uint32_t maxMIter = 0;
    uint32_t woUbIter = 0;
    uint32_t maxWoUbIter = 0;
    uint32_t maxWoUbLoopTimes = 0;
    uint32_t mUbTail = 0;
    uint32_t mStride = 0;
    uint8_t subBlockId = 0;
    bool isFirst = false;
};

template <class Intf>
class DeQuantLoadParamsTools {
public:
    __aicore__ inline DeQuantLoadParamsTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
    }

    __aicore__ inline void LoadParams()
    {
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);

        copyParams.blockLen = self_->ctx.currentNUb * Intf::sizeOfScale;
        uint64_t srcOffset = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                             self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;
        DataCopyPad<typename Intf::ScaleT>(self_->ctx.scaleTensor, self_->ctx.scalegm[srcOffset], copyParams, padParams);

        copyParams.blockLen = self_->ctx.currentNUb * Intf::sizeOfBias;
        if constexpr (AscendC::IsSameType<typename Intf::BiasT, float>::value) {
            DataCopyPad<typename Intf::BiasT>(self_->ctx.biasB32Tensor, self_->ctx.biasgm[srcOffset], copyParams, padParams);
        } else {
            DataCopyPad<typename Intf::BiasT>(self_->ctx.biasB16Tensor, self_->ctx.biasgm[srcOffset], copyParams, padParams);

            eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Cast<float, typename Intf::BiasT>(self_->ctx.biasB32Tensor, self_->ctx.biasB16Tensor,
                RoundMode::CAST_NONE, self_->ctx.currentNUb);
        }
    }

private:
    Intf *self_ = nullptr;

    DataCopyParams copyParams;
    DataCopyPadParams padParams;
};

template <class Intf>
class DeQuantCalcTools {
public:
    __aicore__ inline DeQuantCalcTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        ubCalcTensor = self_->ctx.mmadResUbBuf.template Get<float>();

        calcCount = self_->ctx.convTiling->mUB * self_->ctx.convTiling->nUB;

        uint32_t repStride = self_->ctx.convTiling->nUB / COUNT_ONE_BLK_B32;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = repStride;
        repeatParams.src0RepStride = repStride;
        repeatParams.src1RepStride = 0;
    }

    __aicore__ inline void DeQuantCalc()
    {
        Cast<float, typename Intf::L0cT>(ubCalcTensor, self_->ctx.mmadResUbTensor, RoundMode::CAST_RINT, calcCount);

        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        PipeBarrier<PIPE_V>();

        uint32_t maxMIter = CeilDiv(self_->ctx.currentMUb, UB_MAX_REPEAT_TIMES) - 1;
        uint32_t maxNIter = CeilDiv(self_->ctx.currentNUb, UB_CALC_ONE_REPEAT) - 1;
        uint32_t repeatTimeTail = self_->ctx.currentMUb % UB_MAX_REPEAT_TIMES;
        repeatTimeTail = repeatTimeTail == 0 ? UB_MAX_REPEAT_TIMES : repeatTimeTail;
        uint32_t calcCountTail = self_->ctx.currentNUb % UB_CALC_ONE_REPEAT;
        calcCountTail = calcCountTail == 0 ? UB_CALC_ONE_REPEAT : calcCountTail;

        for (uint32_t mIter = 0; mIter <= maxMIter; ++mIter) {
            uint32_t repeatTime = mIter == maxMIter ? repeatTimeTail : UB_MAX_REPEAT_TIMES;
            uint32_t mOffset = mIter * UB_MAX_REPEAT_TIMES * self_->ctx.convTiling->nUB;
            for (uint32_t nIter = 0; nIter <= maxNIter; ++nIter) {
                uint32_t calcCount = nIter == maxNIter ? calcCountTail : UB_CALC_ONE_REPEAT;
                uint32_t nOffset = nIter * UB_CALC_ONE_REPEAT;
                uint32_t offset = mOffset + nOffset;

                Mul<float>(ubCalcTensor[offset], ubCalcTensor[offset], self_->ctx.scaleTensor[nOffset],
                    calcCount, repeatTime, repeatParams);
                PipeBarrier<PIPE_V>();
                Add<float>(ubCalcTensor[offset], ubCalcTensor[offset], self_->ctx.biasB32Tensor[nOffset],
                    calcCount, repeatTime, repeatParams);
            }
        }

        PipeBarrier<PIPE_V>();
        Cast<typename Intf::OutputT, float>(self_->ctx.outputResUbTensor, ubCalcTensor,
            RoundMode::CAST_RINT, calcCount);
    }

private:
    Intf *self_ = nullptr;

    uint32_t calcCount = 0;

    LocalTensor<float> ubCalcTensor;

    BinaryRepeatParams repeatParams;
};

template <class Intf>
class DeQuantUB2GMTools {
public:
    __aicore__ inline DeQuantUB2GMTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        doStride = self_->ctx.orgHo * self_->ctx.orgWo * self_->ctx.orgCo;
    }

    __aicore__ inline void SetOutputGm(const GlobalTensor<typename Intf::OutputT> &output)
    {
        outputGm.SetGlobalBuffer(output.GetPhyAddr(0), output.GetSize());
    }

    __aicore__ inline void CopyUB2GM()
    {
        copyParams.blockCount = self_->ctx.currentMUb;
        copyParams.blockLen = self_->ctx.currentNUb * Intf::sizeOfOutput;
        copyParams.srcStride = (self_->ctx.convTiling->nUB - self_->ctx.currentNL0Align) / blockSize;
        copyParams.dstStride = (self_->ctx.orgCo - self_->ctx.currentNUb) * Intf::sizeOfOutput;

        uint64_t dstOffset = self_->ctx.batchIter * self_->ctx.outputOneBatchSize + self_->ctx.dOutIter * doStride +
            self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 + self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;

        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            dstOffset += self_->ctx.mUbIter * self_->ctx.convTiling->mUB * self_->ctx.orgCo +
                (self_->ctx.mAL1Iter * self_->ctx.mAL1 + self_->ctx.mL0Iter * self_->ctx.mL0) * self_->ctx.orgCo;
        } else {
            uint64_t offsetH = self_->ctx.hoAL1Iter * self_->ctx.convTiling->hoL1 +
                               self_->ctx.hoL0Iter * self_->ctx.convTiling->hoL0;
            uint64_t offsetW;
            if (self_->ctx.woL1SmallTail == 0) {
                offsetW = self_->ctx.woAL1Iter * self_->ctx.convTiling->woL1 +
                        self_->ctx.woL0Iter * self_->ctx.convTiling->woL0;
            } else {
                if (self_->ctx.woAL1Iter == self_->ctx.maxWoL1Iter) {
                    offsetW = ((self_->ctx.woAL1Iter - 1) * self_->ctx.convTiling->woL1 + self_->ctx.woAL1Tail) +
                              self_->ctx.woL0Iter * self_->ctx.convTiling->woL0;
                } else {
                    offsetW = self_->ctx.woAL1Iter * self_->ctx.convTiling->woL1 +
                              self_->ctx.woL0Iter * self_->ctx.convTiling->woL0;
                }
            }
            dstOffset += offsetH * self_->ctx.orgWo * self_->ctx.orgCo + offsetW * self_->ctx.orgCo;

            uint32_t woUbIter = self_->ctx.mUbIter % self_->ctx.l0C2UbLoopWo;
            uint32_t hoUbIter = self_->ctx.mUbIter / self_->ctx.l0C2UbLoopWo;
            dstOffset += hoUbIter * self_->ctx.orgWo * self_->ctx.orgCo +
                         woUbIter * self_->ctx.convTiling->mUB * self_->ctx.orgCo;
        }

        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        DataCopyPad<typename Intf::OutputT>(outputGm[dstOffset], self_->ctx.outputResUbTensor, copyParams);
    }

private:
    Intf *self_ = nullptr;

    GlobalTensor<typename Intf::OutputT> outputGm;
    DataCopyParams copyParams;

    constexpr static uint32_t blockSize = C0_SIZE / Intf::sizeOfOutput;
    uint64_t doStride = 0;
};

} // namespace Conv3dFunc

#endif // CONV3D_V2_INSTR_DEQUANT_IMPL_H