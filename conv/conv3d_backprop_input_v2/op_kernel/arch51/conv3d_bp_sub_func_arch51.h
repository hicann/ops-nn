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
 * \file conv3d_bp_sub_func_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_SUB_FUNC_ARCH51_H
#define CONV3D_BP_SUB_FUNC_ARCH51_H

namespace Convolution3DBackpropFunc {

constexpr uint8_t V516_FIXPIPE_SHIFT_VALUE = 42;

template <class Intf, typename DstT>
static __aicore__ inline QuantMode_t GetQuantPre(Intf* self)
{
    if constexpr (std::is_same<typename Intf::L0cT, int32_t>::value && std::is_same<DstT, half>::value) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            return QuantMode_t::VDEQF16;
        }
        return QuantMode_t::DEQF16;
    } else if constexpr (std::is_same<typename Intf::L0cT, int32_t>::value && std::is_same<DstT, int8_t>::value) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            return QuantMode_t::VREQ8;
        }
        return QuantMode_t::REQ8;
    }
    return QuantMode_t::NoQuant;
}

static __aicore__ inline pre_quant_t MapQuantModeToPreQuant(QuantMode_t quantMode)
{
    if (quantMode == QuantMode_t::VDEQF16) {
        return pre_quant_t::DEQF16_Vector;
    }
    if (quantMode == QuantMode_t::DEQF16) {
        return pre_quant_t::DEQF16_Scalar;
    }
    if (quantMode == QuantMode_t::VREQ8) {
        return pre_quant_t::REQS8_Vector;
    }
    if (quantMode == QuantMode_t::REQ8) {
        return pre_quant_t::REQS8_Scalar;
    }
    return static_cast<pre_quant_t>(0);
}

template <class Intf>
static __aicore__ inline void LoadScalePPB(Intf* self)
{
    if (self->ctx.scaleFbStagedCin_ == self->ctx.curCinStartIdx_) {
        return;
    }
    self->ctx.scaleFbStagedCin_ = self->ctx.curCinStartIdx_;
    const bool isVectorQuant = self->ctx.tiling_->quantMode0 ==
                               static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT);
    FixPipeConfigBitModeParams params;
    params.SetShiftValue(V516_FIXPIPE_SHIFT_VALUE);
    if (isVectorQuant) {
        // per-channel scale 表 L1→PPB；blockLen 需为 4 的倍数（32B 单位），拷入长度按 128B 对齐
        self->ctx.scaleFB_ = self->ctx.scaleFBTBuf_.template Get<typename Intf::ScaleT0>();
        uint64_t copySize = AlignUp(self->ctx.singleShapeCin_ * sizeof(typename Intf::ScaleT0),
                                    V516_FBUF_PRE_QUANT_BLOCK_BYTES);
        DataCopyParams fbCopyParams(1, copySize / V516_FBUF_BURST_UNIT_BYTES, 0, 0);
        fbCopyParams.postProcBufBlock = static_cast<uint8_t>(PostProcBufBlockMode::PRE_QUANT);
        DataCopy(self->ctx.scaleFB_, self->ctx.scale0L1Buf_, fbCopyParams);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        params.SetPreQuantAddr(DivCeil(self->ctx.scaleFB_.GetPhyAddr(), V516_FBUF_PRE_QUANT_BLOCK_BYTES));
    } else {
        // perTensor：scale 表项低 32 位为 fp32，按 float 视图读第 0 元素；非量化恒 1.0
        FixPipePreQuantParams<half> fixpipeprequantparams;
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT)) {
            fixpipeprequantparams.scalarValue = self->ctx.scale0Global_.template ReinterpretCast<float>().GetValue(0);
        } else {
            fixpipeprequantparams.scalarValue = 1.0f;
        }
        SetFixpipePreQuantFlag(fixpipeprequantparams);
    }
    const bool isQuantScene = isVectorQuant || self->ctx.tiling_->quantMode0 ==
                                                   static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT);
    params.SetRestorationCtrl(static_cast<uint8_t>(isQuantScene ? 0 : 1));
    SetFixPipeConfig(params);
}

template <class Intf>
static __aicore__ inline uint32_t CalcWoExpand(Intf* self)
{
    uint32_t woExpand = ((self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW) + 1;
    if (self->ctx.tiling_->backpropPadLeft < 0 || self->ctx.tiling_->backpropPadRight < 0) {
        const uint32_t woStart = self->ctx.tiling_->backpropPadLeft < 0 ?
                                     DivCeil(abs(self->ctx.tiling_->backpropPadLeft), self->ctx.tiling_->strideW) :
                                     0;
        const uint32_t woEnd = self->ctx.tiling_->backpropPadRight < 0 ?
                                   DivCeil(woExpand + self->ctx.tiling_->backpropPadRight, self->ctx.tiling_->strideW) :
                                   DivCeil(woExpand, self->ctx.tiling_->strideW);
        woExpand = woEnd == woStart ? 1 : (woEnd - woStart - 1) * self->ctx.tiling_->strideW + 1;
    }
    return woExpand;
}

template <class Intf>
static __aicore__ inline void ConvParams(Intf* self)
{
    ConvKernelBitModeParams kernelParam_;
    kernelParam_.SetWStride(1);
    kernelParam_.SetHStride(1);
    kernelParam_.SetWKernel(self->ctx.tiling_->wk);
    kernelParam_.SetHKernel(self->ctx.tiling_->hk);
    kernelParam_.SetWDilation(self->ctx.tiling_->dilationW);
    kernelParam_.SetHDilation(self->ctx.tiling_->dilationH);
    kernelParam_.SetKernelNum(self->ctx.baseUseN_);
    SetConvKernel(kernelParam_);

    const uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.tiling_->wi);
    const uint32_t woExpand = CalcWoExpand<Intf>(self);
    CalcSetFmatrixParams<Intf>(self, self->ctx.curHoSize_, woExpand);

    ConvFMatrixBitModeParams convFMatrixParams;
    convFMatrixParams.SetWFmap(self->ctx.load3d_.l1W);
    convFMatrixParams.SetHFmap(self->ctx.load3d_.l1H);
    convFMatrixParams.SetLeftPaddingSize(self->ctx.load3d_.padList[0]);
    convFMatrixParams.SetRightPaddingSize(self->ctx.load3d_.padList[1]);
    convFMatrixParams.SetTopPaddingSize(self->ctx.load3d_.padList[2]);
    convFMatrixParams.SetBottomPaddingSize(self->ctx.load3d_.padList[3]);
    SetConvFMatrix(convFMatrixParams);

    typename Intf::SrcAT paddingValue = static_cast<typename Intf::SrcAT>(0);
    if constexpr (std::is_same<typename Intf::SrcAT, int8_t>::value) {
        paddingValue = static_cast<typename Intf::SrcAT>(self->ctx.tiling_->offsetX);
    }
    ConvPaddingBitModeParams<typename Intf::SrcAT> convPaddingParams;
    convPaddingParams.SetPaddingValue(paddingValue);
    SetConvPadding<typename Intf::SrcAT>(convPaddingParams);

    ConvFMatrixStepPosBitModeParams convFMatrixStepPosParams;
    convFMatrixStepPosParams.SetHStartPos(0);
    convFMatrixStepPosParams.SetWStartPos(0);
    convFMatrixStepPosParams.SetHStep(hiCal);
    convFMatrixStepPosParams.SetWStep(self->ctx.tiling_->wi);
    SetConvFMatrixStepPos(convFMatrixStepPosParams);

    if (self->ctx.kIter_ != 1) {
        constexpr uint32_t strideFactor = 2;
        const uint32_t currentWoL0 = self->ctx.tiling_->wi;
        ConvStrideBitModeParams strideParams;
        strideParams.SetLoop3DstStride(currentWoL0 * strideFactor);
        strideParams.SetLoop4DstStride(currentWoL0 * strideFactor * hiCal);
        SetConvStridePara(strideParams);
    }

    ConvFixBitModeParams convFixParams;
    if (self->ctx.hasBias_) {
        convFixParams.SetBiasMatrixInitVal(false);
        convFixParams.SetBiasMatrixBroadcast(true);
    } else {
        convFixParams.SetBiasMatrixInitVal(true);
        convFixParams.SetBiasMatrixBroadcast(false);
    }
    convFixParams.SetDependEnable(false);
    convFixParams.SetPreReluMode(ReluMode_t::NoRelu);
    convFixParams.SetCin(self->ctx.tiling_->baseK / (self->ctx.tiling_->hk * self->ctx.tiling_->wk));
    convFixParams.SetPreQuantMode(pre_quant_t::No_Quant);
    self->ctx.convFixParams_ = convFixParams;
    self->ctx.curKIdx_ = 0;
}

template <class Intf, bool hasBias>
static __aicore__ inline void ConvToL1Local(Intf* self, const LocalTensor<typename Intf::SrcAT>& l1a,
                                            const LocalTensor<typename Intf::SrcBT>& l0b)
{
    const int64_t curKIdx = static_cast<int64_t>(self->ctx.curKIdx_);
    const bool isFirstK = (curKIdx == 0);
    const bool isLastK = (curKIdx == static_cast<int64_t>(self->ctx.kIter_) - 1);
    const bool kFullLoad = (self->ctx.kIter_ == 1);
    if (curKIdx > 0) {
        self->ctx.convFixParams_.SetBiasMatrixInitVal(false);
        self->ctx.convFixParams_.SetBiasMatrixBroadcast(false);
        self->ctx.convFixParams_.SetDependEnable(true);
    }
    const uint8_t flag = self->ctx.l1ResPingPongFlag_ & 1;
    constexpr uint32_t halfOffDst = V516_L1_RES_HALF_BYTES / sizeof(typename Intf::DstT);
    constexpr uint32_t halfOffL0c = V516_L1_RES_HALF_BYTES / sizeof(typename Intf::L0cT);
    const uint32_t l1ResOffsetFactor = sizeof(typename Intf::L0cT) / sizeof(typename Intf::DstT);
    uint32_t tempOffset = 0;
    uint32_t tempPreOffset = 0;
    if (kFullLoad) {
        tempOffset = flag ? halfOffDst : 0;
    } else {
        tempOffset = flag ? halfOffL0c : 0;
        tempPreOffset = flag ? 0 : halfOffL0c;
    }

    const uint32_t woExpand = self->ctx.load3d_.l1W;
    const uint32_t tempL1CinOffset = (self->ctx.curKIdx_ % self->ctx.curStepKa_) *
                                     (self->ctx.tiling_->baseK / (self->ctx.tiling_->hk * self->ctx.tiling_->wk)) *
                                     (self->ctx.curHoSize_ * woExpand);
    LocalTensor<typename Intf::SrcAT> convInputA = l1a[tempL1CinOffset];

    self->ctx.convFixParams_.SetCin(self->ctx.baseUseK_ / (self->ctx.tiling_->hk * self->ctx.tiling_->wk));
    if (isFirstK && self->ctx.fixmEventSeen_[flag]) {
        WaitFlag<HardEvent::FIX_M>(flag);
        self->ctx.fixmEventSeen_[flag] = false;
    }

    if (kFullLoad) {
        ConvStrideBitModeParams strideParams;
        strideParams.SetLoop3DstStride(self->ctx.tiling_->wi);
        const uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.tiling_->wi);
        strideParams.SetLoop4DstStride(self->ctx.tiling_->wi * hiCal);
        SetConvStridePara(strideParams);
        auto quantPre = GetQuantPre<Intf, typename Intf::DstT>(self);
        auto preQuantMode = MapQuantModeToPreQuant(quantPre);
        self->ctx.convFixParams_.SetPreQuantMode(preQuantMode);
        if (hasBias) {
            ConvFix<typename Intf::DstT, typename Intf::SrcAT, typename Intf::SrcBT, typename Intf::L0cT>(
                self->ctx.wholeL1ResTensor_[tempOffset], convInputA, l0b, self->ctx.biasBTBuf_,
                self->ctx.convFixParams_);
        } else {
            ConvFix<typename Intf::DstT, typename Intf::SrcAT, typename Intf::SrcBT>(
                self->ctx.wholeL1ResTensor_[tempOffset], convInputA, l0b, self->ctx.convFixParams_);
        }
        SetFlag<HardEvent::M_FIX>(flag);
    } else if (isFirstK) {
        if (hasBias) {
            ConvFix<typename Intf::L0cT, typename Intf::SrcAT, typename Intf::SrcBT, typename Intf::L0cT>(
                self->ctx.wholeCl0Tensor_[tempOffset], convInputA, l0b, self->ctx.biasBTBuf_, self->ctx.convFixParams_);
        } else {
            ConvFix<typename Intf::L0cT, typename Intf::SrcAT, typename Intf::SrcBT>(
                self->ctx.wholeCl0Tensor_[tempOffset], convInputA, l0b, self->ctx.convFixParams_);
        }
    } else if (isLastK) {
        ConvStrideBitModeParams strideParams;
        const uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.tiling_->wi);
        strideParams.SetLoop3DstStride(self->ctx.tiling_->wi);
        strideParams.SetLoop4DstStride(self->ctx.tiling_->wi * hiCal);
        SetConvStridePara(strideParams);
        auto quantPre = GetQuantPre<Intf, typename Intf::DstT>(self);
        auto preQuantMode = MapQuantModeToPreQuant(quantPre);
        self->ctx.convFixParams_.SetPreQuantMode(preQuantMode);
        ConvFix<typename Intf::DstT, typename Intf::SrcAT, typename Intf::SrcBT, typename Intf::L0cT>(
            self->ctx.wholeL1ResTensor_[tempOffset * l1ResOffsetFactor], convInputA, l0b,
            self->ctx.wholeCl0Tensor_[tempPreOffset], self->ctx.convFixParams_);
        SetFlag<HardEvent::M_FIX>(flag);
    } else {
        ConvFix<typename Intf::L0cT, typename Intf::SrcAT, typename Intf::SrcBT, typename Intf::L0cT>(
            self->ctx.wholeCl0Tensor_[tempOffset], convInputA, l0b, self->ctx.wholeCl0Tensor_[tempPreOffset],
            self->ctx.convFixParams_);
    }

    if (hasBias && curKIdx == 0 && !self->ctx.computeBiasOnce_) {
        self->ctx.computeBiasOnce_ = true;
    }
}

template <class Intf>
static __aicore__ inline void UpdateL1KParams(Intf* self, const uint64_t kIdx, uint32_t& curStepKa, uint32_t& curStepKb)
{
    if (unlikely(kIdx == 0)) {
        self->ctx.curLoadKbl1_ = self->ctx.curStepKb_ * self->ctx.tiling_->baseK;
        self->ctx.baseUseK_ = self->ctx.tiling_->baseK;
    }
    if (unlikely(kIdx + 1 == self->ctx.kIter_)) {
        self->ctx.baseUseK_ = self->ctx.tailK_;
    }
    if (unlikely(kIdx == self->ctx.kIterStepKaTail)) {
        curStepKa = self->ctx.stepKaTail;
    }
    if (unlikely(kIdx == self->ctx.kIterStepKbTail)) {
        curStepKb = self->ctx.stepKbTail;
        self->ctx.curLoadKbl1_ = (curStepKb - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    }
}

} // namespace Convolution3DBackpropFunc

#endif // CONV3D_BP_SUB_FUNC_ARCH51_H
