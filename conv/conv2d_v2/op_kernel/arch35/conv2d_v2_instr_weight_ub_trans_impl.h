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
#include "../../common/arch35/conv_instr_nd2nz_vf.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class WeightLoadGM2UBTools {
public:
    __aicore__ inline WeightLoadGM2UBTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadGM2UB()
    {
        if (self_->ctx.convTilingData->singleCoreCi % Intf::k0 == 0 &&
            self_->ctx.convTilingData->singleCoreCo % BLOCK_L0_N == 0) {
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
            copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
            // NDDMA Loop1 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            // NDDMA Loop2 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->coutOffsetBlock;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->bUbKStep;
            copyParams.constantValue = 0;
        }
        // NDDMA Loop0 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.currentUbKStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP1_INDEX] = self_->ctx.currentKLoopRpSize;
        // NDDMA Loop1 params
        copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.currentUbNStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP2_INDEX] = self_->ctx.currentNLoopRpSize;

        uint64_t srcOffset = (self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                              self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep) *
                                 self_->ctx.convTilingData->coutOffsetBlock +
                             self_->ctx.kBL1Iter * self_->ctx.convTilingData->kBL1 +
                             self_->ctx.vecKIter * self_->ctx.convTilingData->bUbKStep;

        LocalTensor<NddmaT> ndTensorNddma = self_->ctx.ndUbBuf.template Get<NddmaT>();
        if constexpr (sizeof(typename Intf::WeightT) == DTYPE_SIZE_B8) {
            GlobalTensor<NddmaT> bgmNddma;
            bgmNddma.SetGlobalBuffer((__gm__ NddmaT*)self_->ctx.bgm.GetPhyAddr());
            DataCopy<NddmaT, NDDMA_DIMS, kDefaultMultiCopyConfig>(ndTensorNddma, bgmNddma[srcOffset], copyParams);
        } else {
            DataCopy<NddmaT, NDDMA_DIMS, kDefaultMultiCopyConfig>(ndTensorNddma, self_->ctx.bgm[srcOffset], copyParams);
        }
    }

    __aicore__ inline void LoadGM2UBAlign()
    {
        self_->ctx.currentUbKStep *= self_->ctx.convTilingData->kernelHxkernelW;
        repeatParams.blockLen = self_->ctx.currentUbKStep / Intf::k0;
        repeatParams.blockCount = self_->ctx.currentUbNStep;
        repeatParams.srcStride = (self_->ctx.convTilingData->singleCoreCi * self_->ctx.convTilingData->kernelHxkernelW -
                                  self_->ctx.currentUbKStep) /
                                 Intf::k0;
        repeatParams.dstStride = (self_->ctx.convTilingData->bUbKStep - self_->ctx.currentUbKStep) / Intf::k0;

        uint64_t srcOffset = (self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                              self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep) *
                                 self_->ctx.convTilingData->coutOffsetBlock +
                             self_->ctx.kBL1Iter * self_->ctx.convTilingData->kBL1 +
                             self_->ctx.vecKIter * self_->ctx.convTilingData->bUbKStep;

        DataCopy<typename Intf::WeightT>(self_->ctx.ndTensor, self_->ctx.bgm[srcOffset], repeatParams);
    }

private:
    Intf* self_ = nullptr;
    using NddmaT = typename Conditional<(sizeof(typename Intf::WeightT) == DTYPE_SIZE_B8), uint8_t,
                                        typename Intf::WeightT>::type;
    DataCopyParams repeatParams;
    NdDmaParams<NddmaT, NDDMA_DIMS> copyParams;
};

template <class Intf>
class WeightND2NZTools {
public:
    __aicore__ inline WeightND2NZTools() {}

    __aicore__ inline void SetParams(Intf* self)
    {
        self_ = self;

        indexTensor = self_->ctx.indexUbBuf.template Get<IndexT>();
    }

    __aicore__ inline void TransND2NZ()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            SetIndex();
        }

        TransND2NZVfParams<SrcT, DstT, IndexT> params;
        params.ciLoopTimes = self_->ctx.convTilingData->bUbKStep / self_->ctx.convTilingData->kernelHxkernelW /
                             Intf::k0;
        params.coLoopTimes = self_->ctx.currentUbNStepAilgn / BLOCK_L0_N * co0LoopTimes;
        params.khkwLoopTimes = self_->ctx.convTilingData->kernelHxkernelW;
        params.srcCiStride = self_->ctx.convTilingData->kernelHxkernelW * Intf::k0;
        params.srcKhKwStride = 1;
        params.srcCoStride = coPerReg * self_->ctx.convTilingData->bUbKStep;
        params.dstCiStride = self_->ctx.convTilingData->kernelHxkernelW * Intf::k0 * self_->ctx.currentUbNStepAilgn;
        params.dstKhKwStride = Intf::k0 * self_->ctx.currentUbNStepAilgn;
        params.dstCoStride = coPerReg * Intf::k0;
        params.srcAddr = (__ubuf__ SrcT*)self_->ctx.ndTensor.GetPhyAddr();
        params.dstAddr = (__ubuf__ DstT*)self_->ctx.nzTensor.GetPhyAddr();
        params.indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
        TransND2NZVf<SrcT, DstT, RegT, IndexT, Intf::isQuantScene>(params);
    }

private:
    __aicore__ inline void SetIndex()
    {
        IndexT curValue = 0;
        for (uint8_t idx = 0; idx < Intf::k0; ++idx) {
            indexTensor.SetValue(idx, curValue);
            curValue += self_->ctx.convTilingData->kernelHxkernelW;
        }
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        __ubuf__ IndexT* indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
        uint16_t repeatTimes = static_cast<uint16_t>(REG_SIZE / sizeof(IndexT) / Intf::k0 - 1);
        IndexT nStride = static_cast<IndexT>(self_->ctx.convTilingData->bUbKStep);
        SetIndexVf(indexAddr, repeatTimes, nStride, static_cast<uint8_t>(Intf::k0));
    }

private:
    Intf* self_ = nullptr;

    using SrcT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using DstT = typename Conditional<Intf::isQuantScene, int8_t, typename Intf::WeightT>::type;
    using RegT = typename Conditional<Intf::isQuantScene, int16_t, typename Intf::WeightT>::type;
    using IndexT = typename Conditional<AscendC::IsSameType<typename Intf::WeightT, float>::value, uint32_t,
                                        uint16_t>::type;

    LocalTensor<IndexT> indexTensor;

    const static uint16_t co0LoopTimes = (Intf::isQuantScene) ? B8_CO0_LOOP_TIMES : CO0_LOOP_TIMES;
    const static uint16_t coPerReg = BLOCK_L0_N / co0LoopTimes;
};

template <class Intf>
class WeightUB2L1Tools {
public:
    __aicore__ inline WeightUB2L1Tools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadUB2L1()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            copyParams.blockCount = self_->ctx.convTilingData->bUbKStep / Intf::k0;
            copyParams.srcStride = 0;
        }
        copyParams.blockLen = self_->ctx.currentUbNStepAilgn;
        copyParams.dstStride = self_->ctx.convTilingData->nBL1 - self_->ctx.currentUbNStepAilgn;

        uint64_t dstOffset = self_->ctx.vecKIter * self_->ctx.convTilingData->nBL1 *
                                 self_->ctx.convTilingData->bUbKStep +
                             self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep * Intf::k0;

        if (self_->ctx.vecId == 1) {
            dstOffset += self_->ctx.bL1SpaceSize;
        }

        DataCopy<typename Intf::WeightT>(self_->ctx.bl1[dstOffset], self_->ctx.nzTensor, copyParams);
    }

private:
    Intf* self_ = nullptr;

    DataCopyParams copyParams;
};

}; // namespace Conv2dFunc

#endif // CONV2D_V2_INSTR_WEIHGT_UB_TRANS_IMPL_H
