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
 * \file conv_instr_impl_51.h
 * \brief
 */

#ifndef CONV_INSTR_IMPL_5162_H
#define CONV_INSTR_IMPL_5162_H

#include "../arch35/conv_instr_hw_mode_impl.h"
#include "../arch35/conv_instr_fz_impl.h"

constexpr uint64_t BURST_LEN_UNIT = 32;
constexpr uint64_t QUANT_PRE_FACTOR = 4;
constexpr uint64_t RELU_PRE_FACTOR = 2;

namespace ConvFunc {
using namespace AscendC;

template <class Intf, typename ChannelWiseT>
__aicore__ inline void LoadChannelWiseL1Tools<Intf, ChannelWiseT>::LoadChannelWiseL1(
    const LocalTensor<ChannelWiseT>& tensorL1, const GlobalTensor<ChannelWiseT>& tensorGm, IterateParam& iterParam)
{
    if constexpr (Intf::isKL1NL0FullLoad) {
        LoadChannelWiseL1FullLoad(tensorL1, tensorGm, currentNL0_, 0);
    } else {
        uint64_t tensorGmOffset = iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                                  iterParam.nL0Iter * self_->ctx.convTilingData->nL0;
        LoadChannelWiseL1FullLoad(tensorL1, tensorGm, currentNL0_, tensorGmOffset);
    }
}

template <class Intf, typename ChannelWiseT>
__aicore__ inline void LoadChannelWiseFBTools<Intf, ChannelWiseT>::LoadChannelWiseFB(
    const LocalTensor<ChannelWiseT>& tensorFB, const LocalTensor<ChannelWiseT>& tensorL1, IterateParam& iterParam,
    const PostProcBufBlockMode& postProcBufBlockMode)
{
    constexpr uint64_t elemSize = sizeof(ChannelWiseT);
    uint64_t fullSize = self_->ctx.convTilingData->nL0 * elemSize;
    uint64_t copySize = (iterParam.nL0Iter != iterParam.maxNL0Iter) ? fullSize : iterParam.nL0Tail * elemSize;
    uint64_t offset = 0;
    if (self_->ctx.convTilingData->fixpParamsFullLoadFlag) {
        offset = iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                 iterParam.nL0Iter * self_->ctx.convTilingData->nL0;
    }

    DataCopyParams param;
    param.blockCount = 1;
    param.blockLen = CeilDiv(copySize, BURST_LEN_UNIT);
    param.srcStride = 0;
    param.dstStride = 0;
    if (postProcBufBlockMode == PostProcBufBlockMode::PRE_QUANT) {
        param.blockLen = AlignB(param.blockLen, QUANT_PRE_FACTOR);
    }
    param.postProcBufBlock = static_cast<uint8_t>(postProcBufBlockMode);
    DataCopy<ChannelWiseT>(tensorFB, tensorL1[offset], param);
}

template <class Intf, typename BiasL1T, typename BiasBtT>
__aicore__ inline void LoadBiasBtTools<Intf, BiasL1T, BiasBtT>::LoadBiasBt(const LocalTensor<BiasBtT>& biasBt,
                                                                           const LocalTensor<BiasL1T>& biasL1,
                                                                           IterateParam& iterParam)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint32_t offset = 0;

    if (self_->ctx.convTilingData->biasFullLoadFlag) {
        if constexpr (!Intf::isKL1NL0FullLoad) {
            offset += iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                      iterParam.nL0Iter * self_->ctx.convTilingData->nL0;
        }
    }

    // fixed-point multiplication should set cvt_mode = 2 and fix_val, which is encapsulated by basic api.
    DataCopyParams biasBtCopyParams(1, currentNL0_ * Intf::sizeOfBias / BT_BLOCK_SIZE, 0, 0);
    DataCopy<typename Intf::L0cT, typename Intf::BiasT>(biasBt, biasL1[offset], biasBtCopyParams);
}

template <class Intf>
__aicore__ inline void LoadBL0Tools<Intf>::SetN(IterateParam& iterParam)
{
    if constexpr (Intf::isKL1NL0FullLoad) {
        ratioOfNToN0 = iterParam.currentNL0Align / BLOCK_L0_N;
    } else {
        ratioOfNToN0 = (iterParam.nBL1Iter == iterParam.maxNBL1Iter && iterParam.nL0Iter == iterParam.maxNL0Iter) ?
                           iterParam.currentNL0Align / BLOCK_L0_N :
                           self_->ctx.convTilingData->nStep;
    }
}

template <class Intf>
__aicore__ inline void LoadBL0Tools<Intf>::SetFirst(IterateParam& iterParam)
{
    if constexpr (Intf::convToL1Mode != static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS)) {
        if constexpr (Intf::isKL1NL0FullLoad) {
            param_.SetMStartPosition(static_cast<uint32_t>(0));
        } else {
            param_.SetMStartPosition(static_cast<uint32_t>(iterParam.nL0Iter * nStep_));
        }
        param_.SetMStep(static_cast<uint16_t>(ratioOfNToN0));
        param_.SetSrcStride(static_cast<int32_t>(self_->ctx.convTilingData->nL1DivBlockSize));
        param_.SetDstStride(static_cast<uint16_t>(ratioOfNToN0));
        param_.SetIfTranspose(false);
    } else {
        weightByPassParam_.ifTranspose = false;
        uint64_t srcStride = CeilDiv(self_->ctx.convTilingData->orgCo, BLOCK_L0_N);
        if constexpr (Intf::groupOptNZFlag) {
            srcStride = CeilDiv(self_->ctx.coOpt, BLOCK_L0_N);
        }
        weightByPassParam_.srcStride = srcStride;
        weightByPassParam_.dstStride = static_cast<uint16_t>(ratioOfNToN0);
        if constexpr (Intf::isKL1NL0FullLoad) {
            weightByPassParam_.mStartPosition = static_cast<uint32_t>(0);
        } else {
            weightByPassParam_.mStartPosition = static_cast<uint32_t>(iterParam.nL0Iter * nStep_);
        }
        weightByPassParam_.mStep = static_cast<uint16_t>(ratioOfNToN0);
    }
}

template <class Intf>
__aicore__ inline uint64_t LoadBL0Tools<Intf>::CalcOffsetForWeightByPass(IterateParam& iterParam)
{
    uint64_t orgCoAlignN0 = AlignB(self_->ctx.convTilingData->orgCo, BLOCK_L0_N);
    if constexpr (Intf::groupOptNZFlag) {
        self_->ctx.coPerGroup = self_->ctx.convTilingData->orgCo / self_->ctx.convTilingData->groups;
        self_->ctx.coOpt = self_->ctx.coPerGroup * self_->ctx.convTilingData->enlarge;
        orgCoAlignN0 = AlignB(self_->ctx.coOpt, BLOCK_L0_N);
    }

    uint64_t bL1GmOffset;
    iterParam.kBL1Iter = iterParam.kIter / iterParam.multiKBL1;
    if constexpr (!Intf::hasNL1IterFlag) {
        bL1GmOffset = iterParam.kBL1Iter * self_->ctx.convTilingData->kBL1 * orgCoAlignN0;
    } else {
        bL1GmOffset = iterParam.kBL1Iter * self_->ctx.convTilingData->kBL1 * orgCoAlignN0 +
                      iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 * Intf::k0;
    }
    return bL1GmOffset;
}

template <class Intf>
__aicore__ inline void LoadBL0Tools<Intf>::LoadBL0(const uint64_t& kStartPosition, const uint64_t& kStep,
                                                   const LocalTensor<typename Intf::WeightT>& bl0, uint64_t bgmOffset)
{
    if constexpr (Intf::convToL1Mode != static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS)) {
        param_.SetKStartPosition(static_cast<uint32_t>(kStartPosition));
        param_.SetKStep(static_cast<uint16_t>(kStep));
        LoadData<TPosition::B2, TPosition::B1, typename Intf::WeightT>(bl0, self_->ctx.bl1, param_);
    } else {
        weightByPassParam_.kStartPosition = static_cast<uint32_t>(kStartPosition);
        weightByPassParam_.kStep = static_cast<uint16_t>(kStep);
        event_t eventId1 = static_cast<event_t>(self_->ctx.pipe.FetchEventID(AscendC::HardEvent::M_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(eventId1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(eventId1);

        LoadData<typename Intf::WeightT>(bl0, self_->ctx.bgm[bgmOffset], weightByPassParam_);

        event_t eventId2 = static_cast<event_t>(self_->ctx.pipe.FetchEventID(AscendC::HardEvent::MTE2_M));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(eventId2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(eventId2);
    }
}

template <class Intf>
__aicore__ inline void LoadBL1FZTools<Intf>::LoadBL1(IterateParam& iterParam)
{
    uint64_t bL1GmOffset;
    if constexpr (!Intf::hasNL1IterFlag) {
        bL1GmOffset = iterParam.kBL1Iter * self_->ctx.convTilingData->kBL1 * orgCoAlignN0;
    } else {
        bL1GmOffset = iterParam.kBL1Iter * self_->ctx.convTilingData->kBL1 * orgCoAlignN0 +
                      iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 * Intf::k0;
        iterParam.currentNBL1 = iterParam.nBL1Iter == iterParam.maxNBL1Iter ? iterParam.nBL1Tail :
                                                                              self_->ctx.convTilingData->nBL1;
    }

    uint64_t currentKBL1 = iterParam.kBL1Iter == iterParam.maxKBL1Iter ? iterParam.kBL1AlignK0Tail :
                                                                         self_->ctx.convTilingData->kBL1;
    DataCopyPadExtParams<typename Intf::WeightT> padParams;
    DataCopyExtParams dataCopyParams;
    if (orgCoAlignN0 == iterParam.currentNBL1) {
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = iterParam.currentNBL1 * currentKBL1 * Intf::sizeOfWeight;
        dataCopyParams.srcStride = 0;
    } else {
        dataCopyParams.blockCount = CeilDiv(currentKBL1, Intf::k0);
        dataCopyParams.blockLen = iterParam.currentNBL1 * C0_SIZE;
        dataCopyParams.srcStride = (orgCoAlignN0 - iterParam.currentNBL1) * C0_SIZE;
        dataCopyParams.dstStride = self_->ctx.convTilingData->nBL1 - self_->ctx.currentNBL1;
    }
    DataCopyPad<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], dataCopyParams, padParams);
}

template <class Intf>
__aicore__ inline void MMadTools<Intf>::Mad(const LocalTensor<typename Intf::FmapT>& al0,
                                            const LocalTensor<typename Intf::WeightT>& bl0,
                                            const ConvFixBitModeParams& convFixParams, IterateParam& iterParam)
{
    bool isKL0Tail = iterParam.kIter == iterParam.maxKL0Iter;
    bool isKhTail = iterParam.khIter == iterParam.maxKhIter;
    if (Intf::kl0FullLoadFlag || (iterParam.maxKL0Iter == 0 && iterParam.maxKhIter == 0)) {
        uint32_t tempOffset = (self_->ctx.l1ResPingPongFlag & 0x1) * L0C_HALF_SIZE / sizeof(typename Intf::OutputT);
        if (self_->ctx.l1ResPingPongFlag > 1) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((self_->ctx.l1ResPingPongFlag ^ 0x1) & 0x1);
        }
        if (!self_->ctx.convTilingData->hasBias) {
            ConvFix<typename Intf::OutputT, typename Intf::FmapT, typename Intf::WeightT>(
                self_->ctx.wholeL1ResTensor[tempOffset], self_->ctx.al1, bl0, convFixParams);
        } else {
            ConvFix<typename Intf::OutputT, typename Intf::FmapT, typename Intf::WeightT, typename Intf::L0cT>(
                self_->ctx.wholeL1ResTensor[tempOffset], self_->ctx.al1, bl0, self_->ctx.biasBT, convFixParams);
        }
    } else {
        uint32_t tempOffset = (self_->ctx.l1ResPingPongFlag & 0x1) * L0C_HALF_SIZE / Intf::sizeOfL0c;
        uint32_t tempPreOffset = ((self_->ctx.l1ResPingPongFlag + 1) & 0x1) * L0C_HALF_SIZE / Intf::sizeOfL0c;
        uint32_t l1ResOffsetFactor = Intf::sizeOfL0c / sizeof(typename Intf::OutputT);
        uint64_t firstIterNum = iterParam.maxKL0Iter * iterParam.maxKhIter + iterParam.maxKL0Iter + iterParam.maxKhIter;
        if ((iterParam.maxKL0Iter == 0 && self_->ctx.l1ResPingPongFlag > 1 && iterParam.maxKhIter == 0) ||
            (((iterParam.kIter == 0 && iterParam.khIter == 1) || (iterParam.kIter == 1 && iterParam.maxKhIter == 0)) &&
             self_->ctx.l1ResPingPongFlag > firstIterNum)) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((self_->ctx.l1ResPingPongFlag ^ 0x1) & 0x1);
        }
        if (isKL0Tail && isKhTail) {
            ConvFix<typename Intf::OutputT, typename Intf::FmapT, typename Intf::WeightT, typename Intf::L0cT>(
                self_->ctx.wholeL1ResTensor[tempOffset * l1ResOffsetFactor], al0, bl0,
                self_->ctx.wholeCl0Tensor[tempPreOffset], convFixParams);
        } else {
            if (iterParam.kIter == 0 && iterParam.khIter == 0 && self_->ctx.convTilingData->hasBias) {
                ConvFix<typename Intf::L0cT, typename Intf::FmapT, typename Intf::WeightT, typename Intf::L0cT>(
                    self_->ctx.wholeCl0Tensor[tempOffset], al0, bl0, self_->ctx.biasBT, convFixParams);
            } else {
                ConvFix<typename Intf::L0cT, typename Intf::FmapT, typename Intf::WeightT, typename Intf::L0cT>(
                    self_->ctx.wholeCl0Tensor[tempOffset], al0, bl0, self_->ctx.wholeCl0Tensor[tempPreOffset],
                    convFixParams);
            }
        }
    }
    self_->ctx.l1ResPingPongFlag++;
    if (isKL0Tail && isKhTail) {
        event_t eventId = static_cast<event_t>(self_->ctx.l1ResPingPongFlag & 0x1);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(eventId);
    }
}

template <class Intf, typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline uint64_t CopyOutToolsHWMode<Intf, OutputT, FixpipeIdx>::CalcFixpipeOffset(IterateParam& iterParam)
{
    uint64_t offset = iterParam.batchIter * self_->ctx.outputOneBatchSize;
    uint64_t offsetH = iterParam.hoAL1Iter * self_->ctx.convTilingData->hoL1 +
                       iterParam.hoL0Iter * self_->ctx.convTilingData->hoL0;
    uint64_t offsetW;
    if (iterParam.woL1SmallTail == 0) {
        offsetW = iterParam.woAL1Iter * self_->ctx.convTilingData->woL1 +
                  iterParam.woL0Iter * self_->ctx.convTilingData->woL0;
    } else {
        if (iterParam.woAL1Iter == iterParam.maxWoL1Iter) {
            offsetW = ((iterParam.woAL1Iter - 1) * self_->ctx.convTilingData->woL1 + iterParam.woAL1Tail) +
                      iterParam.woL0Iter * self_->ctx.convTilingData->woL0;
        } else {
            offsetW = iterParam.woAL1Iter * self_->ctx.convTilingData->woL1 +
                      iterParam.woL0Iter * self_->ctx.convTilingData->woL0;
        }
    }

    uint64_t offsetCout = 0U;
    if constexpr (!Intf::isKL1NL0FullLoad) {
        offsetCout = iterParam.nBL1Iter * self_->ctx.convTilingData->nBL1 +
                     iterParam.nL0Iter * self_->ctx.convTilingData->nL0;
    }

    if constexpr (Intf::formatOutput == ConvFormat::NCHW) {
        offset += offsetCout * valueHoWo_ + offsetH * self_->ctx.convTilingData->orgWo + offsetW;
    }

    return offset;
}

template <class Intf, typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline void CopyOutToolsHWMode<Intf, OutputT, FixpipeIdx>::SetFixpipeIntriParamsConvToL1(
    FixpipeParamsV516& intriParams, IterateParam& iterParam)
{
    if (likely(self_->ctx.convTilingData->woL0 >= self_->ctx.singleCoreWo)) {
        intriParams.mSize = currentML0_;
        intriParams.srcNdStride = 0;
        intriParams.ndNum = 1;
        intriParams.dstNdStride = 0;
    } else {
        intriParams.mSize = self_->ctx.currentWoL0;
        intriParams.srcNdStride = self_->ctx.currentWoL0;
        intriParams.ndNum = self_->ctx.currentHoL0;
        intriParams.dstNdStride = self_->ctx.singleCoreWo;
    }

    intriParams.nSize = currentNL0_;
    intriParams.srcStride = currentML0_;
    intriParams.dstStride = valueHoWo_;
}

template <class Intf, typename OutputT, uint64_t FixpipeIdx>
template <template <typename> class TensorTypeT, const FixpipeConfig& config>
__aicore__ inline void CopyOutToolsHWMode<Intf, OutputT, FixpipeIdx>::CopyOut(const TensorTypeT<OutputT>& output,
                                                                              IterateParam& iterParam,
                                                                              CopyUbInfo* ubInfo)
{
    uint64_t offset = 0;
    if constexpr (Intf::posOutput == TPosition::GM) {
        offset = CalcFixpipeOffset(iterParam);
    }

    FixpipeParamsV516 intriParams;
    SetFixpipeIntriParamsConvToL1(intriParams, iterParam);
    uint32_t tempOffset = ((self_->ctx.l1ResPingPongFlag & 0x1) ^ 0x1) * L0C_HALF_SIZE / Intf::sizeOfL0c;
    uint32_t l1ResOffsetFactor = Intf::sizeOfL0c / sizeof(typename Intf::OutputT);
    Fixpipe<OutputT, OutputT, config>(output[offset], self_->ctx.wholeL1ResTensor[tempOffset * l1ResOffsetFactor],
                                      intriParams);
    event_t eventId = static_cast<event_t>(self_->ctx.l1ResPingPongFlag & 0x1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventId);
}

} // namespace ConvFunc
#endif // CONV_INSTR_IMPL_5162_H
