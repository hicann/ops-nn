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
 * \file conv3d_bp_func_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_FUNC_ARCH51_H
#define CONV3D_BP_FUNC_ARCH51_H

#include "impl/conv_bp_input_sub_func_arch51.h"
#include "conv3d_bp_common_func_arch51.h"
#include "conv3d_bp_util_arch51.h"
#include "../../../inc/macro.h"

DECLARE_CHECK_IMPL(Init);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetWeight);
DECLARE_CHECK_IMPL(SetBias);
DECLARE_CHECK_IMPL(SetScale);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_IMPL(FreeBiasTensor);
DECLARE_CHECK_IMPL(FreeB1Tensor);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_IMPL(End);
namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void InitStepMParams(Intf* self)
{
    self->ctx.mIter_ = DivCeil(self->ctx.singleShapeM_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf* self)
{
    uint64_t tmpSingleCoreK = 0;
    uint32_t singleShapeCout1 = DivCeilC0<Intf>(self, self->ctx.singleShapeCout_);
    tmpSingleCoreK = singleShapeCout1 * self->ctx.singleShapeHWk_ * self->ctx.tiling_->c0;
    self->ctx.kIter_ = DivCeil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = tmpSingleCoreK - (self->ctx.kIter_ - 1) * self->ctx.tiling_->baseK;
    self->ctx.kIterStepKaTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKa_) - 1) * self->ctx.curStepKa_;
    self->ctx.kIterStepKbTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKb_) - 1) * self->ctx.curStepKb_;
    self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
    self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf* self)
{
    self->ctx.nIter_ = DivCeil(self->ctx.singleShapeCin_, self->ctx.tiling_->baseN);
    self->ctx.tailN_ = self->ctx.singleShapeCin_ - (self->ctx.nIter_ - 1) * self->ctx.tiling_->baseN;
}

template <class Intf>
__aicore__ inline void InitFullLoadFlag(Intf* self)
{
    bool isDoutDkFullLoad = self->ctx.tiling_->dk == 1 || self->ctx.tiling_->dout == 1;
    bool baseB1Cond = self->ctx.tiling_->baseN >= self->ctx.tiling_->singleCoreCin && self->ctx.tiling_->enlarge == 1;
    self->ctx.isA1FullLoadFlag_ = false;
    self->ctx.isB1FullLoadFlag_ = baseB1Cond &&
                                  self->ctx.tiling_->baseK * self->ctx.curStepKb_ >=
                                      self->ctx.tiling_->singleCoreCout * self->ctx.singleShapeHWk_ &&
                                  self->ctx.tiling_->bl1Pbuffer == 1 && self->ctx.tiling_->singleIterateDk == 1;
}

template <class Intf>
static __aicore__ inline void InitParamsForNormal(Intf* self)
{
    uint32_t mL1Size = self->ctx.tiling_->baseM;
    mL1Size = mL1Size > self->ctx.tiling_->singleCoreM ? self->ctx.tiling_->singleCoreM : mL1Size;
    self->ctx.curHoSize_ = CalFmapH<Intf>(self, mL1Size);
}

template <class Intf>
__aicore__ inline void InitSingleShapeParams(Intf* self)
{
    // 初始化为0，避免decache功能开启后全局变量不会初始化为0，出现随机值导致未知问题
    self->ctx.singleShapeDin_ = 0;
    self->ctx.singleShapeM_ = 0;
    self->ctx.singleShapeCin_ = 0;
    self->ctx.singleShapeCout_ = 0;
    self->ctx.singleShapeHWk_ = self->ctx.tiling_->hkWk;
}

template <class Intf>
__aicore__ inline void InitParamsPart2(Intf* self)
{
    self->ctx.mIter_ = 0;
    self->ctx.nIter_ = 0;
    self->ctx.kIter_ = 0;
    self->ctx.tailM_ = 0;
    self->ctx.tailN_ = 0;
    self->ctx.tailK_ = 0;
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.curLoadKbl1_ = 0;
    self->ctx.curDinIdx_ = 0;
    self->ctx.curDkIdx_ = 0;
    self->ctx.kIterStepKaTail = 0;
    self->ctx.kIterStepKbTail = 0;
    self->ctx.stepKaTail = 0;
    self->ctx.stepKbTail = 0;
    self->ctx.curHoIdx_ = 0;
    self->ctx.baseB1Offset_ = 0;
    self->ctx.blockBaseN_ = 0;
    self->ctx.baseUseM_ = 0;
    self->ctx.baseUseN_ = 0;
    self->ctx.baseUseK_ = 0;
    self->ctx.curDinStartIdx_ = 0;
    self->ctx.curHoStartIdx_ = 0;
    self->ctx.curCinStartIdx_ = 0;
    self->ctx.curMStartIdx_ = 0;
    self->ctx.curHkIdx_ = 0;
    self->ctx.curWkIdx_ = 0;
    self->ctx.isLoadA1_ = true;
    self->ctx.isLoadB1_ = true;
    self->ctx.isFreeA1_ = false;
    self->ctx.isFreeB1_ = false;
    self->ctx.needComputeFlag_ = true;
    self->ctx.computeBiasOnce_ = false;
}

template <class Intf>
__aicore__ inline void InitParams(Intf* self)
{
    InitSingleShapeParams<Intf>(self);
    self->ctx.isFirstIter_ = true;
    self->ctx.curStepKa_ = self->ctx.tiling_->stepKa;
    self->ctx.curStepKb_ = self->ctx.tiling_->stepKb;
    InitParamsForNormal<Intf>(self);
    self->ctx.hoWo_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.doHoWo_ = self->ctx.tiling_->dout * self->ctx.hoWo_;
    self->ctx.hiWi_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.diHiWi_ = self->ctx.tiling_->di * self->ctx.hiWi_;
    self->ctx.realWoSize_ = self->ctx.tiling_->wo;
    InitFullLoadFlag<Intf>(self);
    self->ctx.hoExpand_ = self->ctx.tiling_->hoExpand;
    self->ctx.l0PingPongFlag_ = 0;
    self->ctx.l0bPingPongAddr_ = DivDtypeByte<typename Intf::SrcBT>(TOTAL_L0B_SIZE >> 1);
    self->ctx.enableL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    self->ctx.load3d_.mStartPt = 0;
    self->ctx.l1ResPingPongFlag_ = 0;
    InitLoadToB2Params<Intf>(self);
    InitParamsPart2<Intf>(self);
}

template <class Intf>
__aicore__ inline void CalcMatrixByteSize(Intf* self, uint32_t& aMatrixByteSize, uint32_t& bMatrixByteSize)
{
    uint32_t hoSize = self->ctx.hoExpand_ < static_cast<uint64_t>(self->ctx.curHoSize_) ?
                          static_cast<uint32_t>(self->ctx.hoExpand_) :
                          self->ctx.curHoSize_;
    aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
                      DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) *
                      sizeof(typename Intf::SrcAT);
    bMatrixByteSize = self->ctx.tiling_->baseN * self->ctx.curStepKb_ * self->ctx.tiling_->baseK *
                      sizeof(typename Intf::SrcBT);
}

template <class Intf>
__aicore__ inline uint32_t CalcBiasL1Size(Intf* self)
{
    // 64: BT Buffer 需要64B对齐，加63后右移6位代替乘64
    uint32_t biasSize = self->ctx.tiling_->isBiasFullLoad ?
                            (DivCeil(self->ctx.tiling_->singleCoreCin * sizeof(typename Intf::BiasT), 64) << 6) :
                            (DivCeil(self->ctx.tiling_->baseN * sizeof(typename Intf::BiasT), 64) << 6);
    return AlignUp(biasSize, ONE_BLK_SIZE);
}

template <class Intf>
__aicore__ inline uint32_t CalcScaleBufSize(Intf* self)
{
    return AlignUp(self->ctx.tiling_->singleCoreCin * sizeof(typename Intf::ScaleT0), V516_FBUF_PRE_QUANT_BLOCK_BYTES);
}

template <class Intf>
__aicore__ inline void InitBiasTque(Intf* self)
{
    uint32_t biasSize = CalcBiasL1Size<Intf>(self);
    self->ctx.pipe_.InitBuffer(self->ctx.biasL1Que_, 1, biasSize);
    self->ctx.pipe_.InitBuffer(self->ctx.biasBTQue_, 1, biasSize);
}

template <class Intf>
__aicore__ inline void InitScaleTque(Intf* self)
{
    uint32_t scaleSize = CalcScaleBufSize<Intf>(self);
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            self->ctx.pipe_.InitBuffer(self->ctx.scale0L1Que_, 1, scaleSize);
        }
    }
}

template <class Intf>
__aicore__ inline void InitTque(Intf* self, const bool hasBias)
{
    uint32_t bMatrixByteSize = 0;
    uint32_t aMatrixByteSize = 0;
    CalcMatrixByteSize(self, aMatrixByteSize, bMatrixByteSize);

    self->ctx.pipe_.InitBuffer(self->ctx.inQueL1B_, self->ctx.tiling_->bl1Pbuffer, bMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.inQueL1A_, self->ctx.tiling_->al1Pbuffer, aMatrixByteSize);
    self->ctx.pipe_.InitBuffer(self->ctx.l0cBuf_, V516_L1_RES_TOTAL_BYTES);
    self->ctx.wholeL1ResTensor_ = self->ctx.l0cBuf_.template Get<typename Intf::DstT>();
    self->ctx.wholeCl0Tensor_ = self->ctx.l0cBuf_.template Get<typename Intf::L0cT>();
    uint32_t scaleFBSize = sizeof(typename Intf::ScaleT0) * self->ctx.tiling_->singleCoreCin;
    self->ctx.pipe_.InitBuffer(self->ctx.scaleFBTBuf_, AlignUp(scaleFBSize, V516_FBUF_PRE_QUANT_BLOCK_BYTES));
    self->ctx.scaleL1StagedCin_ = -1;
    self->ctx.scaleFbStagedCin_ = -1;
    self->ctx.fixmEventSeen_[0] = false;
    self->ctx.fixmEventSeen_[1] = false;
    self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
    if (unlikely(hasBias)) {
        InitBiasTque(self);
    }
    InitScaleTque(self);
}

template <class Intf>
static __aicore__ inline void FreeFullLoadL1Tensor(Intf* self)
{
    // 全载场景，如果Tensor已经被载入了，进行释放
    if (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_ && !self->ctx.isLoadB1_) {
        self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
        self->ctx.isLoadB1_ = true;
    }

    if (self->ctx.isA1FullLoadFlag_ && self->ctx.isFreeA1_ && !self->ctx.isLoadA1_) {
        self->ctx.inQueL1A_.FreeTensor(self->ctx.cacheA1Buf_);
        self->ctx.isLoadA1_ = true;
    }
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeForNoTilingHWk(Intf* self, LocalTensor<typename Intf::SrcAT>& l0a,
                                                    LocalTensor<typename Intf::SrcBT>& l0b,
                                                    LocalTensor<typename Intf::L0cT>& l0c, uint8_t& l0PingPongFlag)
{
    bool isFirstDk = (self->ctx.curDkIdx_ == 0);
    const uint32_t strideD = self->ctx.tiling_->strideD;
    const int64_t strideDMulDout = self->ctx.tiling_->dout * strideD;
    const bool strideDgtOne = strideD > 1;
    for (uint64_t curInnerKdIdx = self->ctx.curDkIdx_;
         curInnerKdIdx < self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk; curInnerKdIdx++) {
        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront -
                       curInnerKdIdx * self->ctx.tiling_->dilationD;
        if (dTmp < 0 || (strideDgtOne && (dTmp % strideD > 0)) || dTmp >= strideDMulDout) {
            continue;
        }
        int64_t curDoutIdx = strideDgtOne ? (dTmp / strideD) : dTmp;
        ComputeForKIter<Intf, hasBias>(self, l0a, l0b, l0c, curInnerKdIdx, curDoutIdx, isFirstDk, l0PingPongFlag);
        isFirstDk = false;
    }
    if (hasBias) {
        if (!self->ctx.computeBiasOnce_) {
            ComputeForBias<Intf, hasBias>(self, l0b, l0PingPongFlag);
        }
    }
}

template <class Intf>
static __aicore__ inline void ComputeStart(Intf* self, LocalTensor<typename Intf::L0cT>& l0c)
{
    UpdateLoadToB2ParamsN<Intf>(self);
    ConvParams<Intf>(self);
    LoadScalePPB<Intf>(self);
    l0c = self->ctx.wholeCl0Tensor_;
}

template <class Intf>
static __aicore__ inline void ComputeEnd(Intf* self, uint8_t l0PingPongFlag)
{
    self->ctx.l0PingPongFlag_ = l0PingPongFlag;
}

template <class Intf, bool hasBias>
static __aicore__ inline void NoNeedComputeHandle(Intf* self)
{
    if (!hasBias) {
        FreeFullLoadL1Tensor<Intf>(self);
        return;
    }

    LocalTensor<typename Intf::SrcAT> l0a;
    LocalTensor<typename Intf::SrcBT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    ComputeStart<Intf>(self, l0c);
    uint8_t l0PingPongFlag = self->ctx.l0PingPongFlag_;
    // 有bias场景 加bias再退出
    ComputeForBias<Intf, hasBias>(self, l0b, l0PingPongFlag);
    self->ctx.needComputeFlag_ = true;
    FreeFullLoadL1Tensor<Intf>(self);
    ComputeEnd<Intf>(self, l0PingPongFlag);
}

template <class Intf, bool hasBias>
static __aicore__ inline void Compute(Intf* self)
{
    if (!self->ctx.needComputeFlag_) {
        NoNeedComputeHandle<Intf, hasBias>(self);
        return;
    }

    LocalTensor<typename Intf::SrcAT> l0a;
    LocalTensor<typename Intf::SrcBT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    ComputeStart<Intf>(self, l0c);
    uint8_t l0PingPongFlag = self->ctx.l0PingPongFlag_;
    ComputeForNoTilingHWk<Intf, hasBias>(self, l0a, l0b, l0c, l0PingPongFlag);
    ComputeEnd<Intf>(self, l0PingPongFlag);
}

template <class Intf>
static __aicore__ inline void UpdateCurHoSize(Intf* self)
{
    // 要从当前核hoStartIdx开始算，普通场景可以跨行，要从GM起点开始计算
    uint32_t curWi = self->ctx.tiling_->wi;

    uint64_t curMIdx = self->ctx.curMStartIdx_ + self->ctx.curMIdx_ * self->ctx.tiling_->baseM;
    uint32_t curHiIdx;
    uint32_t curWiIdx;
    if (curMIdx == 0) {
        curHiIdx = 0;
        curWiIdx = 0;
    } else {
        curHiIdx = curMIdx / curWi;
        curWiIdx = curMIdx - curHiIdx * curWi;
    }
    self->ctx.load3d_.mStartPt = curWiIdx;
    uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + curWiIdx, curWi);
    uint32_t endHoIdx = 0;
    self->ctx.curHoIdx_ = static_cast<int32_t>(curHiIdx) - self->ctx.tiling_->backpropPadUp;
    endHoIdx = self->ctx.curHoIdx_ + (hiCal + (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH);
    UpdateCurHoSizeCore<Intf>(self, endHoIdx);
}

template <class Intf>
static __aicore__ inline void UpdateKComputeStatus(Intf* self)
{
    if (!self->ctx.needComputeFlag_) {
        // 其他方向要跳过计算，K可以不判断了
        return;
    }
    const uint32_t strideD = self->ctx.tiling_->strideD;
    const int64_t strideDThreshold = self->ctx.tiling_->dout * strideD;
    const bool strideDgtOne = strideD > 1;
    const int64_t DkThreshold = self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk;
    bool isKNeedCompute = false;
    for (uint64_t curKdIdx = self->ctx.curDkIdx_; curKdIdx < DkThreshold; curKdIdx++) {
        // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
        if (dTmp < 0 || (strideDgtOne && (dTmp % strideD > 0)) || dTmp >= strideDThreshold) {
            continue;
        }
        isKNeedCompute = true;
        break;
    }

    self->ctx.needComputeFlag_ = isKNeedCompute;
}

template <class Intf>
static __aicore__ inline void UpdateFullLoadL1Status(Intf* self)
{
    if (!self->ctx.isB1FullLoadFlag_ && !self->ctx.isA1FullLoadFlag_) {
        return;
    }
    bool isLastMNIter = self->ctx.curMIdx_ == self->ctx.mIter_ - 1 && self->ctx.curNIdx_ == self->ctx.nIter_ - 1;
    bool isLastDIter = (self->ctx.tiling_->dk == 1 &&
                        self->ctx.curDinIdx_ == self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_ - 1) ||
                       self->ctx.tiling_->dk > 1;
    bool suppressB1Free = self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1 && self->ctx.curEnableFullLoad_ &&
                          self->ctx.tiling_->group == 1;
    self->ctx.isFreeB1_ = isLastMNIter && isLastDIter && !suppressB1Free;
}

template <class Intf>
static __aicore__ inline void UpdateL1ComputeInfo(Intf* self)
{
    self->ctx.baseUseM_ = (self->ctx.curMIdx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
    uint32_t Hd = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
    uint32_t Wd = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
    // load3d指令的padList[3]固定为255
    uint32_t backpropPadDown = self->ctx.tiling_->backpropPadDown > 255 ? self->ctx.tiling_->backpropPadDown : 255;
    uint32_t Hp = self->ctx.hoExpand_ + self->ctx.tiling_->backpropPadUp + backpropPadDown;
    uint32_t Wp = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1 + self->ctx.tiling_->backpropPadLeft +
                  self->ctx.tiling_->backpropPadRight;
    uint64_t M = (Hp - Hd + 1) * (Wp - Wd + 1);
    if (unlikely(M < self->ctx.baseUseM_) && self->ctx.mIter_ == 1) {
        self->ctx.baseUseM_ = M;
    }
    self->ctx.baseUseN_ = (self->ctx.curNIdx_ + 1 == self->ctx.nIter_) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
    self->ctx.needComputeFlag_ = true;
    UpdateCurHoSize<Intf>(self);
    UpdateKComputeStatus<Intf>(self);
    UpdateFullLoadL1Status<Intf>(self);
}

template <class Intf>
static __aicore__ inline void SetDequantScale(Intf* self)
{
    bool hasVectorScale = false;
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        hasVectorScale = self->ctx.tiling_->quantMode0 ==
                         static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT);
    }
    if (hasVectorScale) {
        // 切换时先释放上一份（配对防池 id 泄漏）
        if (self->ctx.scaleL1StagedCin_ == self->ctx.curCinStartIdx_) {
            return;
        }
        if (self->ctx.scaleL1StagedCin_ != -1) {
            self->ctx.scale0L1Que_.FreeTensor(self->ctx.scale0L1Buf_);
        }
        self->ctx.scaleL1StagedCin_ = self->ctx.curCinStartIdx_;
        Convolution3DBackpropFunc::FullLoadToScaleL1<Intf>(self);
    }
}

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const Conv3DBackpropInputArch35TilingData& tiling,
                                       const bool hasBias)
    {
        self->ctx.hasBias_ = hasBias;
        // kernel侧通过ctx持有tiling指针，供后续全流程访问，避免向kernel内部逐层传递引用
        self->ctx.tiling_ = &(tiling);
        self->ctx.curEnableFullLoad_ = self->ctx.tiling_->enableFullLoad;
        InitParams<Intf>(self);
        InitTque<Intf>(self, hasBias);
    }
};

template <class Intf>
struct FreeBiasTensor {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
        self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
    }
};

// 源自 conv3d_bp_kernel_split_func.h 的 5162/NO_SPLIT 视图
template <class Intf>
struct FreeB1Tensor {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        // 整轮计算都未加载B1Tensor就不需要重复释放
        if (self->ctx.isLoadB1_ && !self->ctx.isFreeB1_) {
            return;
        }
        // c04场景和group>1（包括enlarge>1）场景不支持calRound间不释放B矩阵
        if (self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1 && self->ctx.curEnableFullLoad_ &&
            self->ctx.tiling_->group == 1) {
            self->ctx.isLoadB1_ = true;
            self->ctx.isFreeB1_ = false;
            self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
        }
    }
};

template <class Intf>
struct SetWeight {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcBT>& weight)
    {
        self->ctx.weightGlobal_ = weight;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcAT>& outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetBias {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::BiasT>& bias)
    {
        self->ctx.biasGlobal_ = bias;
    }
};

template <class Intf>
struct SetScale {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::ScaleT0>& scale)
    {
        self->ctx.scale0Global_ = scale;
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                       uint32_t singleShapeD)
    {
        if (self->ctx.singleShapeDin_ != singleShapeD) {
            self->ctx.singleShapeDin_ = singleShapeD;
        }

        if (self->ctx.singleShapeM_ != singleShapeM) {
            self->ctx.singleShapeM_ = singleShapeM;
            InitStepMParams<Intf>(self);
        }

        if (self->ctx.singleShapeCin_ != singleShapeN) {
            self->ctx.singleShapeCin_ = singleShapeN;
            InitStepNParams<Intf>(self);
        }

        if (self->ctx.singleShapeCout_ * self->ctx.singleShapeHWk_ != singleShapeK) {
            self->ctx.singleShapeCout_ = singleShapeK / self->ctx.singleShapeHWk_;
            InitStepKParams<Intf>(self);
        }
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint32_t curDinStartIdx, int64_t curMStartIdx,
                                       int32_t curCinStartIdx)
    {
        // 需要增加入参的startDinIdx判断，否则无法确定padding和补0问题
        // 增加入参，传参判断
        self->ctx.curDinStartIdx_ = curDinStartIdx;
        self->ctx.curMStartIdx_ = curMStartIdx;
        self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / self->ctx.tiling_->wi) -
                                   self->ctx.tiling_->backpropPadUp;
        self->ctx.curCinStartIdx_ = curCinStartIdx;
    }
};

template <class Intf>
static __aicore__ inline void InitFirstIterState(Intf* self)
{
    self->ctx.needComputeFlag_ = true;
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.curDinIdx_ = self->ctx.curDinStartIdx_;
    self->ctx.curDkIdx_ = 0;
    self->ctx.curHoIdx_ = self->ctx.curHoStartIdx_;
    self->ctx.isFirstIter_ = false;
    self->ctx.isLoadA1_ = true;
    self->ctx.isFreeA1_ = false;
    if (!self->ctx.isB1FullLoadFlag_ || self->ctx.tiling_->dk > 1 || !self->ctx.curEnableFullLoad_ ||
        self->ctx.tiling_->group != 1) {
        self->ctx.isLoadB1_ = true;
        self->ctx.isFreeB1_ = false;
    }
}

template <class Intf>
static __aicore__ inline bool AdvanceIterIndices(Intf* self)
{
    if (++self->ctx.curMIdx_ >= self->ctx.mIter_) {
        self->ctx.curMIdx_ = 0;
        if (++self->ctx.curNIdx_ >= self->ctx.nIter_) {
            self->ctx.curNIdx_ = 0;
            if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                return false;
            }
        }
    }
    return true;
}

template <class Intf, bool sync>
struct Iterate {
    // 一次iterate计算(baseM, baseN, baseD)，当前baseD=1
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline bool call(Intf* self, bool enPartialSum, bool hasBias)
    {
        (void)enPartialSum;
        if (unlikely(hasBias)) {
            self->ctx.computeBiasOnce_ = false;
        }
        if (unlikely(self->ctx.isFirstIter_)) {
            InitFirstIterState<Intf>(self);
        } else {
            if (!AdvanceIterIndices<Intf>(self)) {
                return false;
            }
        }

        UpdateL1ComputeInfo<Intf>(self);
        if (unlikely(hasBias)) {
            Compute<Intf, true>(self);
        } else {
            Compute<Intf, false>(self);
        }
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic,
                                       bool fullLoadBiasFlag_, bool freeBiasFlag_)
    {
        bool hasBias = self->ctx.hasBias_;
        if (unlikely(hasBias && self->ctx.tiling_->isBiasFullLoad)) {
            if (freeBiasFlag_) {
                self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
                self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
            }
            if (fullLoadBiasFlag_) {
                Convolution3DBackpropFunc::FullLoadBias<Intf>(self);
            }
        }
        SetDequantScale<Intf>(self);
        while (self->template Iterate<sync>(false, hasBias)) {
            self->template GetTensorC<sync>(output, enAtomic);
        }
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output);
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        self->ctx.inQueL1A_.FreeAllEvent();
        self->ctx.inQueL1B_.FreeAllEvent();

        if (unlikely(self->ctx.hasBias_)) {
            self->ctx.biasL1Que_.FreeAllEvent();
            self->ctx.biasBTQue_.FreeAllEvent();
        }

        if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                self->ctx.scale0L1Que_.FreeAllEvent();
            }
        }
    }
};
} // namespace Convolution3DBackpropFunc
#endif
