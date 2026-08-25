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
 * \file conv2d_v2_instr_c04_impl.h
 * \brief
 */

#ifndef CONV2D_2V_INSTR_C04_IMPL_H
#define CONV2D_2V_INSTR_C04_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "../../common/arch35/conv_instr_nd2nz_vf.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class C04DupTools {
public:
    __aicore__ inline C04DupTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void KAlignDup()
    {
        if (unlikely(self_->ctx.kSizeC04 % Intf::k0 != 0)) {
            Duplicate<typename Intf::WeightT>(self_->ctx.ndTensor, 0, self_->ctx.ubBufSize);
        }
    }

private:
    Intf* self_ = nullptr;
};

template <class Intf>
class C04LoadGM2UBTools {
public:
    __aicore__ inline C04LoadGM2UBTools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadGM2UB()
    {
        if constexpr (Intf::formatWeight == ConvFormat::NCHW) {
            SetNCHWGM2UBParams();
        } else if constexpr (Intf::formatWeight == ConvFormat::HWCN) {
            SetHWCNGM2UBParams();
        }

        DataCopy<typename Intf::WeightT, NDDMA_DIMS, kDefaultMultiCopyConfig>(self_->ctx.ndTensor,
                                                                              self_->ctx.bgm[srcOffset], copyParams);
    }

private:
    __aicore__ inline void SetNCHWGM2UBParams()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            // NDDMA Loop0 params
            copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = self_->ctx.convTilingData->singleCoreCi;
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
            copyParams.loopInfo.loopRpSize[NDDMA_LOOP0_INDEX] = C04_CIN_SIZE - self_->ctx.convTilingData->singleCoreCi;
            // NDDMA Loop1 params
            copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = 1;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = C04_CIN_SIZE;
            // NDDMA Loop2 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->kernelHxkernelW *
                                                                   self_->ctx.convTilingData->singleCoreCi;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->kBL1;
            copyParams.constantValue = 0;
        }
        copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.currentUbNStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP2_INDEX] = self_->ctx.currentNLoopRpSize;

        srcOffset = (self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                     self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep) *
                    self_->ctx.convTilingData->coutOffsetBlock;
        if constexpr (!Intf::bL1DBFlag) {
            if (self_->ctx.vecId == 1) {
                srcOffset += self_->ctx.nBL1Vec0 * self_->ctx.convTilingData->coutOffsetBlock;
            }
        }
    }

    __aicore__ inline void SetHWCNGM2UBParams()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            // NDDMA Loop0 params
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = self_->ctx.convTilingData->kBL1;
            // NDDMA Loop1 params
            copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] = self_->ctx.convTilingData->singleCoreCi;
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTilingData->orgCo;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = 1;
            copyParams.loopInfo.loopRpSize[NDDMA_LOOP1_INDEX] = C04_CIN_SIZE - self_->ctx.convTilingData->singleCoreCi;
            // NDDMA Loop2 params
            copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->kernelHxkernelW;
            copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTilingData->orgCo *
                                                                   self_->ctx.convTilingData->singleCoreCi;
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = C04_CIN_SIZE;
            copyParams.constantValue = 0;
        }
        copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = self_->ctx.currentUbNStep;
        copyParams.loopInfo.loopRpSize[NDDMA_LOOP0_INDEX] = self_->ctx.currentNLoopRpSize;

        srcOffset = self_->ctx.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                    self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep;
        if constexpr (!Intf::bL1DBFlag) {
            if (self_->ctx.vecId == 1) {
                srcOffset += self_->ctx.nBL1Vec0;
            }
        }
    }

private:
    Intf* self_ = nullptr;
    NdDmaParams<typename Intf::WeightT, NDDMA_DIMS> copyParams;
    uint64_t srcOffset = 0;
};

template <class Intf>
class C04TransFractalZC04Tools {
public:
    __aicore__ inline C04TransFractalZC04Tools() {}

    __aicore__ inline void SetParams(Intf* self)
    {
        self_ = self;

        indexTensor = self_->ctx.indexUbBuf.template Get<IndexT>();
    }

    __aicore__ inline void TransFractalZC04()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            SetIndex();
        }

        TransFractalZVfParams<typename Intf::WeightT, IndexT> params;
        params.kLoopTimes = self_->ctx.convTilingData->kBL1 / Intf::k0;
        params.nLoopTimes = self_->ctx.currentUbNStepAilgn / BLOCK_L0_N * CO0_LOOP_TIMES;
        params.srcKStride = Intf::k0;
        params.srcNStride = coPerReg * self_->ctx.convTilingData->kBL1;
        params.dstKStride = Intf::k0 * self_->ctx.currentUbNStepAilgn;
        params.dstNStride = Intf::k0 * coPerReg;
        params.srcAddr = (__ubuf__ typename Intf::WeightT*)self_->ctx.ndTensor.GetPhyAddr();
        params.dstAddr = (__ubuf__ typename Intf::WeightT*)self_->ctx.nzTensor.GetPhyAddr();
        params.indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
        TransFractalZVf(params);
    }

private:
    __aicore__ inline void SetIndex()
    {
        IndexT curValue = 0;
        for (uint8_t idx = 0; idx < Intf::k0; ++idx) {
            indexTensor.SetValue(idx, curValue);
            curValue += 1;
        }
        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);

        __ubuf__ IndexT* indexAddr = (__ubuf__ IndexT*)indexTensor.GetPhyAddr();
        uint16_t repeatTimes = static_cast<uint16_t>(REG_SIZE / sizeof(IndexT) / Intf::k0 - 1);
        IndexT nStride = static_cast<IndexT>(self_->ctx.convTilingData->kBL1);
        SetIndexVf(indexAddr, repeatTimes, nStride, static_cast<uint8_t>(Intf::k0));
    }

private:
    Intf* self_ = nullptr;

    using IndexT = typename Conditional<AscendC::IsSameType<typename Intf::WeightT, float>::value, uint32_t,
                                        uint16_t>::type;

    LocalTensor<IndexT> indexTensor;

    const static uint16_t coPerReg = BLOCK_L0_N / CO0_LOOP_TIMES;
};

template <class Intf>
class C04LoadUB2L1Tools {
public:
    __aicore__ inline C04LoadUB2L1Tools() {}

    __aicore__ inline void SetParams(Intf* self) { self_ = self; }

    __aicore__ inline void LoadUB2L1()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            copyParams.blockCount = self_->ctx.convTilingData->kBL1 / Intf::k0;
            copyParams.srcStride = 0;
        }
        copyParams.blockLen = self_->ctx.currentUbNStepAilgn;
        copyParams.dstStride = self_->ctx.convTilingData->nBL1 - self_->ctx.currentUbNStepAilgn;

        uint64_t dstOffset = self_->ctx.vecNIter * self_->ctx.convTilingData->bUbNStep * Intf::k0;
        if (self_->ctx.vecId == 1) {
            if constexpr (Intf::bL1DBFlag) {
                dstOffset += self_->ctx.bL1SpaceSize;
            } else {
                dstOffset += self_->ctx.nBL1Vec0 * Intf::k0;
            }
        }

        DataCopy<typename Intf::WeightT>(self_->ctx.bl1[dstOffset], self_->ctx.nzTensor, copyParams);
    }

private:
    Intf* self_ = nullptr;

    DataCopyParams copyParams;
};

}; // namespace Conv2dFunc

#endif // CONV2D_2V_INSTR_C04_IMPL_H
