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
 * \file conv_bp_filter_sub_func.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_H
#define CONV3D_BP_FILTER_SUB_FUNC_H
#include "conv_bp_filter_sub_func_load_gm_to_l1.h"
#include "conv_bp_sub_func_deterministic_common.h"
#include "conv_bp_filter_sub_func_load_l1_to_l0.h"
#include "conv_bp_filter_sub_func_store_l0c.h"
#include "conv_bp_filter_sub_func_store_l0c_dispatch.h"

namespace ConvolutionBackpropFunc {
constexpr uint16_t MMAD_THRESHOLD = 2560;

template <class Intf>
__aicore__ inline void CheckTiling(Intf *self)
{
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.tiling_->batch > 0),
        "orignal batch is %d , which should be larger than 0", self->ctx.tiling_->batch);
    ascendc_assert((self->ctx.tiling_->cin > 0),
        "orignal cin is %d , which should be larger than 0", self->ctx.tiling_->cin);
    ascendc_assert((self->ctx.tiling_->cout > 0),
        "orignal cout is %d , which should be larger than 0", self->ctx.tiling_->cout);
    ascendc_assert((self->ctx.tiling_->cin1G > 0),
        "orignal cin1G is %d , which should be larger than 0", self->ctx.tiling_->cin1G);
    ascendc_assert((self->ctx.tiling_->cout1G > 0),
        "orignal cout1G is %d , which should be larger than 0", self->ctx.tiling_->cout1G);
    ascendc_assert((self->ctx.tiling_->dout > 0),
        "orignal dout is %d , which should be larger than 0", self->ctx.tiling_->dout);
    ascendc_assert((self->ctx.tiling_->ho > 0),
        "orignal ho is %d , which should be larger than 0", self->ctx.tiling_->ho);
    ascendc_assert((self->ctx.tiling_->wo > 0),
        "orignal wo is %d , which should be larger than 0", self->ctx.tiling_->wo);
    ascendc_assert((self->ctx.tiling_->di > 0),
        "orignal di is %d , which should be larger than 0", self->ctx.tiling_->di);
    ascendc_assert((self->ctx.tiling_->hi > 0),
        "orignal hi is %d , which should be larger than 0", self->ctx.tiling_->hi);
    ascendc_assert((self->ctx.tiling_->wi > 0),
        "orignal wi is %d , which should be larger than 0", self->ctx.tiling_->wi);
    ascendc_assert((self->ctx.tiling_->dk > 0),
        "orignal dk is %d , which should be larger than 0", self->ctx.tiling_->dk);
    ascendc_assert((self->ctx.tiling_->hk > 0),
        "orignal hk is %d , which should be larger than 0", self->ctx.tiling_->hk);
    ascendc_assert((self->ctx.tiling_->wk > 0),
        "orignal wk is %d , which should be larger than 0", self->ctx.tiling_->wk);
    ascendc_assert((self->ctx.tiling_->singleCoreBatch > 0),
        "singleCoreBatch is %d , which should be larger than 0", self->ctx.tiling_->singleCoreBatch);
    ascendc_assert((self->ctx.tiling_->singleCoreCout > 0),
        "singleCoreCout is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCout);
    ascendc_assert((self->ctx.tiling_->singleCoreHo > 0),
        "singleCoreHo is %d , which should be larger than 0", self->ctx.tiling_->singleCoreHo);
    ascendc_assert((self->ctx.tiling_->singleCoreCin > 0),
        "singleCoreCin is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCin);
    ascendc_assert((self->ctx.tiling_->baseM > 0),
        "baseM is %d , which should be larger than 0", self->ctx.tiling_->baseM);
    ascendc_assert((self->ctx.tiling_->baseK > 0),
        "baseK is %d , which should be larger than 0", self->ctx.tiling_->baseK);
    ascendc_assert((self->ctx.tiling_->baseN > 0),
        "baseN is %d , which should be larger than 0", self->ctx.tiling_->baseN);
    ascendc_assert((self->ctx.tiling_->stepKa > 0),
        "stepKa is %d , which should be larger than 0", self->ctx.tiling_->stepKa);
    ascendc_assert((self->ctx.tiling_->stepKb > 0),
        "stepKb is %d , which should be larger than 0", self->ctx.tiling_->stepKb);
#endif
}

template <class Intf>
__aicore__ inline void InitParamsPart2(Intf *self)
{
    self->ctx.stepKaRound = 0;
    self->ctx.stepKbRound = 0;
    self->ctx.tailN_ = 0;
    self->ctx.tailK_ = 0;
    self->ctx.tailM_ = 0;
    self->ctx.baseUseM_ = 0;
    self->ctx.baseUseN_ = 0;
    self->ctx.baseUseK_ = 0;
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.mIter_ = 0;
    self->ctx.nIter_ = 0;
    self->ctx.kIter_ = 0;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    self->ctx.bL1cin1CopyLen = 0;
    self->ctx.singleShapeDk_ = 0;
    self->ctx.curSingleCoreDk_ = 0;
    self->ctx.cinHkWkLoop_ = 0;
    self->ctx.cinRemainLen_ = 0;
    self->ctx.nLoopCinRemainLen_ = 0;
    self->ctx.head_ = 0;
    self->ctx.tail_ = 0;
    self->ctx.nLoopHead_ = 0;
    self->ctx.lastNIdx_ = -1;
#endif
}

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // streamK场景，AIV需要申请UB空间，扩维场景AIC,AIV需要申请UB空间
    self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, AscendC::TOTAL_UB_SIZE);
#endif
    if ASCEND_IS_AIC {
        uint32_t al1BoundByteSize = self->ctx.tiling_->al1Bound * sizeof(typename Intf::SrcT);
        self->ctx.pipe_.InitBuffer(self->ctx.a1Ping_, 1, al1BoundByteSize);
        if (self->ctx.tiling_->al1Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.a1Pong_, 1, al1BoundByteSize);
        }

        uint32_t bl1BoundByteSize = self->ctx.tiling_->bl1Bound * sizeof(typename Intf::SrcT);
        self->ctx.pipe_.InitBuffer(self->ctx.b1Ping_, 1, bl1BoundByteSize);
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.b1Pong_, 1, bl1BoundByteSize);
        }

        uint32_t cMatrixByteSize = self->ctx.baseMN_ * sizeof(typename Intf::L0cT);
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPing_, 1, cMatrixByteSize);
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.l0cPong_, 1, cMatrixByteSize);
        }

        self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
        self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
    }
}

template <class Intf>
static __aicore__ inline void InitLoadToA2Params(Intf* self)
{
    self->ctx.dstL12L0aOffset_ = 0;
    self->ctx.srcL12L0aOffset_ = 0;
    self->ctx.alignedL1UseKa_ = 0;
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.mStep = 0;
    self->ctx.load2dv2_.kStep = 0;
    self->ctx.load2dv2_.srcStride = 0;
    self->ctx.load2dv2_.dstStride = 0;
    self->ctx.load2dv2_.ifTranspose = 1;
    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        self->ctx.alignedL1UseM_ = 0;
    }
}

// 计算Load2B2的指令参数
template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf* self)
{
    // load3dStepK
    self->ctx.load3d_.kExtension = self->ctx.tiling_->baseN;
    // load3dStepM
    self->ctx.load3d_.mExtension = self->ctx.tiling_->baseM;
    // posK
    self->ctx.load3d_.kStartPt = 0;
    // posM
    self->ctx.load3d_.mStartPt = 0;
    self->ctx.load3d_.strideW = self->ctx.tiling_->strideW;
    self->ctx.load3d_.strideH = self->ctx.tiling_->strideH;
    self->ctx.load3d_.filterW = self->ctx.tiling_->wk;
    self->ctx.load3d_.filterH = self->ctx.tiling_->hk;
    self->ctx.load3d_.dilationFilterW = self->ctx.tiling_->dilationW;
    self->ctx.load3d_.dilationFilterH = self->ctx.tiling_->dilationH;
    self->ctx.load3d_.filterSizeW = (self->ctx.tiling_->wk >> 8) & 255;
    self->ctx.load3d_.filterSizeH = (self->ctx.tiling_->hk >> 8) & 255;
    self->ctx.load3d_.enTranspose = 0;
    self->ctx.load3d_.fMatrixCtrl = 0;
    self->ctx.load3d_.channelSize = 16;
}

template <class Intf>
static __aicore__ inline void InitSetFmatrixParams(Intf* self)
{
    self->ctx.load3d_.l1W = self->ctx.tiling_->wi;
    self->ctx.load3d_.l1H = 1;
    self->ctx.load3d_.padList[0] = self->ctx.tiling_->padLeft;
    self->ctx.load3d_.padList[1] = self->ctx.tiling_->padRight;
    // padUp now is set in Load BL1
    self->ctx.load3d_.padList[3] = 255; // 255: set max pad_bottom for compatibility
}

template <class Intf>
static __aicore__ inline void InitMmadParams(Intf* self)
{
    self->ctx.dstL0cOffset_ = 0;
    self->ctx.srcL0aOffset_ = 0;
    self->ctx.srcL0bOffset_ = 0;
    self->ctx.mmad_.m = self->ctx.tiling_->baseM;
    self->ctx.mmad_.k = self->ctx.tiling_->baseK;
    self->ctx.mmad_.n = self->ctx.tiling_->baseN;
    self->ctx.mmad_.unitFlag = 0;
    self->ctx.mmad_.kDirectionAlign = 0;
    self->ctx.mmad_.cmatrixSource = 0;
    self->ctx.mmad_.cmatrixInitVal = 0;

    // cinG表示每组cin的个数
    self->ctx.cinG_ = self->ctx.tiling_->cin;
    if (self->ctx.tiling_->group != 1) {
        // 扩维情况和depthwise情况均不会使用此参数
        self->ctx.cinG_ = self->ctx.tiling_->cin1G;
    }
}

template <class Intf>
__aicore__ inline void InitParams(Intf *self)
{
    self->ctx.baseMN_ = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
    self->ctx.hwK_ = static_cast<uint64_t>(self->ctx.tiling_->hk) * self->ctx.tiling_->wk;
    self->ctx.hwO_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.hwI_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.dhwK_ = self->ctx.hwK_ * self->ctx.tiling_->dk;
    self->ctx.dhwO_ = self->ctx.hwO_ * self->ctx.tiling_->dout;
    self->ctx.dhwI_ = self->ctx.hwI_ * self->ctx.tiling_->di;
    self->ctx.kal1_ = self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
    self->ctx.kbl1_ = self->ctx.tiling_->stepKb * self->ctx.tiling_->baseK;

    // simplify the calculation of hi = (ho - 1) * strideh + (hk - 1) * dilationh + 1
    self->ctx.strideKernelDilationH = static_cast<int64_t>(self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH
        + 1 - self->ctx.tiling_->strideH;
    // simplify the calculation of wi = (wo - 1) * stridew + (wk - 1) * dilationw + 1
    self->ctx.strideKernelDilationW = static_cast<int64_t>(self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW
        + 1 - self->ctx.tiling_->strideW;
    self->ctx.isFirstIter_ = true;
    self->ctx.l0aPingPongFlag_ = 0;
    self->ctx.l0bPingPongFlag_ = 0;
    self->ctx.l0cPingPongFlag_ = 1;
    self->ctx.useL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);

    self->ctx.singleShapeWo_ = 0;
    self->ctx.singleShapeCout_ = 0;
    self->ctx.singleShapeBatch_ = 0;
    self->ctx.singleShapeCin_ = 0;
    self->ctx.singleShapeHo_ = 0;
    self->ctx.batchDoutStartIdx_ = 0;
    self->ctx.hoStartIdx_ = 0;
    self->ctx.hiStartIdx_ = 0;
    self->ctx.dkStartIdx_ = 0;
    self->ctx.isSplitWo_ = !(self->ctx.tiling_->splitWo == self->ctx.tiling_->wo);
    InitParamsPart2<Intf>(self);
    InitLoadToB2Params<Intf>(self);
    InitLoadToA2Params<Intf>(self);
    InitSetFmatrixParams<Intf>(self);
    InitMmadParams<Intf>(self);
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf *self)
{
    self->ctx.mIter_ = Ceil(self->ctx.singleShapeCout_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeCout_ - self->ctx.tiling_->baseM * (self->ctx.mIter_ - 1);
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.mIter_ > 0),
        "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    uint64_t singleCoreHoWo = static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.tiling_->wo;
    uint64_t kIter = Ceil(singleCoreHoWo, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = singleCoreHoWo - self->ctx.tiling_->baseK * (kIter - 1);
    self->ctx.stepKaRound = Ceil(kIter, self->ctx.tiling_->stepKa);
    self->ctx.stepKbRound = Ceil(kIter, self->ctx.tiling_->stepKb);
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.kIter_ > 0),
        "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDC1HWC0) {
        self->ctx.nIter_ = Ceil(ShiftCeilChannelSize<Intf>(self->ctx.singleShapeCin_, self->ctx.tiling_->channelSize) *
            self->ctx.tiling_->channelSize * self->ctx.hwK_, self->ctx.tiling_->baseN);
        self->ctx.tailN_ =
            self->ctx.singleShapeCin_ * self->ctx.hwK_ - self->ctx.tiling_->baseN * (self->ctx.nIter_ - 1);
    } else {
        uint64_t curNWithoutDk = ShiftCeilM0(self->ctx.singleShapeCin_, self->ctx.tiling_->n0) *
            self->ctx.tiling_->n0 * self->ctx.hwK_;
        self->ctx.nIter_ = Ceil(curNWithoutDk * self->ctx.singleShapeDk_, self->ctx.tiling_->baseN);
        self->ctx.tailN_ = curNWithoutDk % self->ctx.tiling_->baseN;
        bool baseNIncludeDkNoCin1Flag1 = (self->ctx.nIter_ > self->ctx.singleShapeDk_ &&
                                    self->ctx.nIter_ % self->ctx.singleShapeDk_ != 0);
        bool baseNIncludeDkNoCin1Flag2 = (self->ctx.nIter_ < self->ctx.singleShapeDk_ &&
                                    self->ctx.singleShapeDk_ % self->ctx.nIter_ != 0);
        // 当baseN上不包含cin1但包含全部的dk时，需手动干预，使baseN上先载cin后载dk
        if (baseNIncludeDkNoCin1Flag1 || baseNIncludeDkNoCin1Flag2) {
            self->ctx.nIter_ = Ceil(curNWithoutDk, self->ctx.tiling_->baseN) * self->ctx.singleShapeDk_;
        }
        // nIter_包含cinhkwk循环和dk循环，d上有padding时不计算
        self->ctx.cinHkWkLoop_ = Ceil(self->ctx.nIter_, self->ctx.singleShapeDk_);
        // 当SingleCoreDk_大于nIter_时，表明每次循环时SingleCoreDk_>1，否则SingleCoreDk_=1
        self->ctx.curSingleCoreDk_ = Ceil(self->ctx.singleShapeDk_, self->ctx.nIter_);
        self->ctx.tailN_ = self->ctx.tailN_ * self->ctx.curSingleCoreDk_;
    }
    if (self->ctx.tailN_ == 0) {
        self->ctx.tailN_ = self->ctx.tiling_->baseN;
    }
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.nIter_ > 0),
        "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_);
#endif
}

template <class Intf>
__aicore__ inline void UpdateMNIdxFirstIterate(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.isFirstIter_ = false;

    out2L1Params.isLoad2L1A = true;
    out2L1Params.isLoad2L1B = true;
    if constexpr (Intf::conv3ddwConfig.isSplitKernelHW) {
        out2L1Params.isFreeAL1 = true;
        out2L1Params.isFreeBL1 = true;
    } else {
        bool isLastNLoop = self->ctx.nIter_ == 1;
        bool isLastMLoop = self->ctx.mIter_ == 1;
        bool kIterCeilStepKaGreaterAl1Pbuffer = self->ctx.kIter_ > self->ctx.tiling_->stepKa * self->ctx.tiling_->al1Pbuffer;
        bool kIterCeilStepKbGreaterBl1Pbuffer = self->ctx.kIter_ > self->ctx.tiling_->stepKb * self->ctx.tiling_->bl1Pbuffer;

        out2L1Params.isFreeAL1 = kIterCeilStepKaGreaterAl1Pbuffer || self->ctx.singleShapeBatch_ > 1  || self->ctx.isSplitWo_ == 1 ||
            (self->ctx.tiling_->iterateOrder) || (!(self->ctx.tiling_->iterateOrder) && isLastNLoop);  // OrderM
        out2L1Params.isFreeBL1 = kIterCeilStepKbGreaterBl1Pbuffer || self->ctx.singleShapeBatch_ > 1  || self->ctx.isSplitWo_ == 1 ||
            (self->ctx.tiling_->iterateOrder && isLastMLoop) || (!(self->ctx.tiling_->iterateOrder));  // OrderN
    }
}

template <class Intf>
__aicore__ inline bool UpdateMNIdxOrderN(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    if (++self->ctx.curMIdx_ >= self->ctx.mIter_) {
        self->ctx.curMIdx_ = 0;
        if (++self->ctx.curNIdx_ >= self->ctx.nIter_) {
            self->ctx.curNIdx_ = 0;
            return false;
        }
        out2L1Params.isLoad2L1B = true; // OrderN, N轴循环结束，需要置换BL1
    }

    out2L1Params.isLoad2L1A = true; // OrderN, M轴循环结束，需要置换AL1
    out2L1Params.isFreeAL1 = true; // OrderN, M轴最后一次循环，需要释放AL1

    if (unlikely((self->ctx.curMIdx_ == self->ctx.mIter_ - 1) && (self->ctx.curNIdx_ == self->ctx.nIter_ - 1))) {
        out2L1Params.isFreeBL1 = true; // OrderN, N轴最后一次循环，需要释放BL1
    }
    return true;
}

template <class Intf>
__aicore__ inline bool UpdateMNIdxOrderM(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    if (++self->ctx.curNIdx_ >= self->ctx.nIter_) {
        self->ctx.curNIdx_ = 0;
        if (++self->ctx.curMIdx_ >= self->ctx.mIter_) {
            self->ctx.curMIdx_ = 0;
            return false;
        }
        out2L1Params.isLoad2L1A = true; // OrderM, M轴循环结束，需要置换AL1
    }

    out2L1Params.isLoad2L1B = true; // OrderM, N轴循环结束，需要置换BL1
    out2L1Params.isFreeBL1 = true; // OrderM, N轴最后一次循环，需要释放BL1

    if (unlikely((self->ctx.curNIdx_ == self->ctx.nIter_ - 1) && (self->ctx.curMIdx_ == self->ctx.mIter_ - 1))) {
        out2L1Params.isFreeAL1 = true; // OrderM, M轴最后一次循环，需要释放AL1
    }
    return true;
}

template <class Intf>
static __aicore__ inline void MmadLocal(
    Intf* self, const LocalTensor<typename Intf::SrcT>& l0a, const LocalTensor<typename Intf::SrcT>& l0b,
    LocalTensor<typename Intf::L0cT>& l0c)
{
    Mmad(l0c[self->ctx.dstL0cOffset_], l0a[self->ctx.srcL0aOffset_], l0b[self->ctx.srcL0bOffset_], self->ctx.mmad_);

    // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为10*256
    if (self->ctx.mmad_.m * self->ctx.mmad_.n < MMAD_THRESHOLD) {
        AscendC::PipeBarrier<PIPE_M>();
    }
}

} // namespace ConvolutionBackpropFunc

#endif
