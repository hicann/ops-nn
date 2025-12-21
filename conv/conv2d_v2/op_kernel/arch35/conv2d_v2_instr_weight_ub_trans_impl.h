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
 * \file conv2d_v2_instr_weight_ub_trans_impl.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_WEIHGT_UB_TRANS_IMPL_H
#define CONV2D_V2_INSTR_WEIHGT_UB_TRANS_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace Conv2dFunc {
using namespace AscendC;

template <class Intf>
class WeightLoadGM2UBTools {
public:
    __aicore__ inline WeightLoadGM2UBTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadGM2UB()
    {
        if (self_->ctx.convTiling->singleCoreCi % Intf::k0 == 0 && self_->ctx.convTiling->singleCoreCo % BLOCK_L0_N == 0) {
            LoadGM2UBAlign();
        } else {
            LoadGM2UBWithPad();
        }
    }

private:
    __aicore__ inline void LoadGM2UBWithPad()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            // NDDMA Loop0 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
            // NDDMA Loop1 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTiling->coutOffsetBlock;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTiling->bUbKStep;
        }
        // NDDMA Loop0 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = self_->ctx.currentUbKStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP0_INDEX] = self_->ctx.currentKLoopRpSize;
        // NDDMA Loop1 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.currentUbNStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP1_INDEX] = self_->ctx.currentNLoopRpSize;

        uint64_t srcOffset = (self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
            self_->ctx.vecNIter * self_->ctx.convTiling->bUbNStep) * self_->ctx.convTiling->coutOffsetBlock +
            self_->ctx.kBL1Iter * self_->ctx.convTiling->kBL1 + self_->ctx.vecKIter * self_->ctx.convTiling->bUbKStep;

        DataCopy<typename Intf::WeightT, NDDMA_DIMS_NO_TRANS, kDefaultMultiCopyConfig>(
            self_->ctx.ndTensor, self_->ctx.bgm[srcOffset], copyParams);
    }

    __aicore__ inline void LoadGM2UBAlign()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            repeatParams.blockLen = self_->ctx.convTiling->bUbKStep / Intf::k0;
            repeatParams.srcStride = (self_->ctx.convTiling->singleCoreCi * self_->ctx.convTiling->kernelHxkernelW -
                self_->ctx.convTiling->bUbKStep) / Intf::k0;
            repeatParams.dstStride = 0;
        }
        repeatParams.blockCount = self_->ctx.currentUbNStep;

        uint64_t srcOffset = (self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
            self_->ctx.vecNIter * self_->ctx.convTiling->bUbNStep) * self_->ctx.convTiling->coutOffsetBlock +
            self_->ctx.kBL1Iter * self_->ctx.convTiling->kBL1 + self_->ctx.vecKIter * self_->ctx.convTiling->bUbKStep;

        DataCopy<typename Intf::WeightT>(self_->ctx.ndTensor, self_->ctx.bgm[srcOffset], repeatParams);
    }

private:
    Intf *self_ = nullptr;
    DataCopyParams repeatParams;
    MultiCopyParams<typename Intf::WeightT, NDDMA_DIMS_NO_TRANS> copyParams;
};

template <class Intf>
class WeightND2NZTools {
public:
    __aicore__ inline WeightND2NZTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        indexTensor = self_->ctx.indexUbBuf.template Get<IndexT>();
    }

    __aicore__ inline void TransND2NZ()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            SetIndex();
        }

        uint16_t ciLoopTimes = self_->ctx.convTiling->bUbKStep / self_->ctx.convTiling->kernelHxkernelW / Intf::k0;
        uint16_t coLoopTimes = self_->ctx.currentUbNStepAilgn / BLOCK_L0_N * CO0_LOOP_TIMES;
        uint16_t khkwLoopTimes = self_->ctx.convTiling->kernelHxkernelW;
        uint32_t srcCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0;
        uint32_t srcCoStride = coPerReg * self_->ctx.convTiling->bUbKStep;
        uint32_t dstCiStride = self_->ctx.convTiling->kernelHxkernelW * Intf::k0 * self_->ctx.currentUbNStepAilgn;
        uint32_t dstKhKwStride = Intf::k0 * self_->ctx.currentUbNStepAilgn;
        uint32_t dstCoStride = coPerReg * Intf::k0;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<typename Intf::WeightT> gatherReg;
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<typename Intf::WeightT, MicroAPI::MaskPattern::ALL>();

            __local_mem__ typename Intf::WeightT *srcAddr =
                (__local_mem__ typename Intf::WeightT *)self_->ctx.ndTensor.GetPhyAddr();
            __local_mem__ typename Intf::WeightT *dstAddr =
                (__local_mem__ typename Intf::WeightT *)self_->ctx.nzTensor.GetPhyAddr();
            __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();

            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);

            for (uint16_t ci1OptIndex = 0; ci1OptIndex < ciLoopTimes; ++ci1OptIndex) {
                for (uint16_t khkwIndex = 0; khkwIndex < khkwLoopTimes; ++khkwIndex) {
                    for (uint16_t coOptIndex = 0; coOptIndex < coLoopTimes; ++coOptIndex) {
                        uint32_t srcOffset = ci1OptIndex * srcCiStride + khkwIndex + coOptIndex * srcCoStride;
                        uint32_t dstOffset = ci1OptIndex * dstCiStride + khkwIndex * dstKhKwStride +
                                             coOptIndex * dstCoStride;

                        MicroAPI::DataCopyGather<typename Intf::WeightT, typename Intf::WeightT, IndexT>(
                            gatherReg, srcAddr + srcOffset, indexReg, maskReg);

                        MicroAPI::DataCopy<typename Intf::WeightT>(
                            dstAddr + dstOffset, (MicroAPI::RegTensor<typename Intf::WeightT> &)gatherReg, maskReg);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void SetIndex()
    {
        IndexT curValue = 0;
        for (uint8_t idx = 0; idx < Intf::k0; ++idx) {
            indexTensor.SetValue(idx, curValue);
            curValue += self_->ctx.convTiling->kernelHxkernelW;
        }
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        __local_mem__ IndexT *indexAddr = (__local_mem__ IndexT *)indexTensor.GetPhyAddr();
        uint16_t repeatTimes = static_cast<uint16_t>(REG_SIZE / sizeof(IndexT) / Intf::k0 - 1);
        uint8_t dstOffset = Intf::k0;
        uint8_t elesPerRepeat = Intf::k0;
        uint32_t maskL = Intf::k0;
        IndexT nStride = static_cast<IndexT>(self_->ctx.convTiling->bUbKStep);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<IndexT> indexReg;
            MicroAPI::DataCopy<IndexT>(indexReg, indexAddr);
            MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<IndexT>(maskL);

            for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                MicroAPI::Adds<IndexT, IndexT>(indexReg, indexReg, nStride, maskReg);
                MicroAPI::DataCopy<IndexT>(indexAddr + dstOffset, indexReg, maskReg);
                dstOffset += elesPerRepeat;
            }
        }
    }

private:
    Intf *self_ = nullptr;

    using IndexT = typename Conditional<
        AscendC::IsSameType<typename Intf::WeightT, float>::value, uint32_t, uint16_t>::type;

    LocalTensor<IndexT> indexTensor;

    const static uint16_t coPerReg = BLOCK_L0_N / CO0_LOOP_TIMES;
};

template <class Intf>
class WeightUB2L1Tools {
public:
    __aicore__ inline WeightUB2L1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadUB2L1()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            copyParams.blockCount = self_->ctx.convTiling->bUbKStep / Intf::k0;
            copyParams.srcStride = 0;
        }
        copyParams.blockLen = self_->ctx.currentUbNStepAilgn;
        copyParams.dstStride = self_->ctx.convTiling->nBL1 - self_->ctx.currentUbNStepAilgn;

        uint64_t dstOffset = self_->ctx.vecKIter * self_->ctx.convTiling->nBL1 * self_->ctx.convTiling->bUbKStep +
                             self_->ctx.vecNIter * self_->ctx.convTiling->bUbNStep * Intf::k0;

        if (self_->ctx.vecId == 1) {
            dstOffset += self_->ctx.bL1SpaceSize;
        }

        DataCopy<typename Intf::WeightT>(self_->ctx.bl1[dstOffset], self_->ctx.nzTensor, copyParams);
    }

private:
    Intf *self_ = nullptr;

    DataCopyParams copyParams;
};

}; // namespace Conv2dFunc

#endif // CONV2D_V2_INSTR_WEIHGT_UB_TRANS_IMPL_H