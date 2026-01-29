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
 * \file conv3d_bp_func.h
 * \brief
 */

#ifndef CONV3D_BP_FUNC_ADVANCE_H
#define CONV3D_BP_FUNC_ADVANCE_H

#include "conv3d_bp_common_func.h"
#include "conv3d_bp_large_attribute_func.h"
#include "conv3d_bp_kernel_split_func.h"

DECLARE_CHECK_IMPL(Init);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetWeight);
#if defined(__DAV_310R6__)
DECLARE_CHECK_IMPL(SetBias);
#endif
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_IMPL(SetFullLoadFlag);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_SYNC_IMPL(VecPreProcess);
DECLARE_CHECK_SYNC_IMPL(VecPostProcess);
DECLARE_CHECK_IMPL(End);
namespace Convolution3DBackpropFunc {

enum class IterateOrder : uint8_t {
    ORDER_M = 0,
    ORDER_N,
    ORDER_K,
    UNDEF,
};

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
    ascendc_assert((self->ctx.tiling_->ho > 0),
        "orignal ho is %d , which should be larger than 0", self->ctx.tiling_->ho);
    ascendc_assert((self->ctx.tiling_->wo > 0),
        "orignal wo is %d , which should be larger than 0", self->ctx.tiling_->wo);
    ascendc_assert((self->ctx.tiling_->hi > 0),
        "orignal hi is %d , which should be larger than 0", self->ctx.tiling_->hi);
    ascendc_assert((self->ctx.tiling_->wi > 0),
        "orignal wi is %d , which should be larger than 0", self->ctx.tiling_->wi);
    ascendc_assert((self->ctx.tiling_->hk > 0),
        "orignal hk is %d , which should be larger than 0", self->ctx.tiling_->hk);
    ascendc_assert((self->ctx.tiling_->wk > 0),
        "orignal wk is %d , which should be larger than 0", self->ctx.tiling_->wk);
    ascendc_assert((self->ctx.tiling_->stepM == 1 && self->ctx.tiling_->stepN == 1),
        "stepM or stepN is invalid.");
    ascendc_assert((self->ctx.tiling_->singleCoreBatch > 0),
        "singleCoreBatch is %d , which should be larger than 0", self->ctx.tiling_->singleCoreBatch);
    ascendc_assert((self->ctx.tiling_->singleCoreCout > 0),
        "singleCoreCout is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCout);
    ascendc_assert((self->ctx.tiling_->singleCoreCin > 0),
        "singleCoreCin is %d , which should be larger than 0", self->ctx.tiling_->singleCoreCin);
    ascendc_assert((self->ctx.tiling_->baseM > 0),
        "baseM is %d , which should be larger than 0", self->ctx.tiling_->baseM);
    ascendc_assert((self->ctx.tiling_->baseK > 0),
        "baseK is %d , which should be larger than 0", self->ctx.tiling_->baseK);
    ascendc_assert((self->ctx.tiling_->baseN > 0),
        "baseN is %d , which should be larger than 0", self->ctx.tiling_->baseN);
    ascendc_assert((self->ctx.tiling_->stepM > 0),
        "stepM is %d , which should be larger than 0", self->ctx.tiling_->stepM);
    ascendc_assert((self->ctx.tiling_->stepN > 0),
        "stepN is %d , which should be larger than 0", self->ctx.tiling_->stepN);
    ascendc_assert((self->ctx.tiling_->stepKa > 0),
        "stepKa is %d , which should be larger than 0", self->ctx.tiling_->stepKa);
    ascendc_assert((self->ctx.tiling_->stepKb > 0),
        "stepKb is %d , which should be larger than 0", self->ctx.tiling_->stepKb);
    ascendc_assert((self->ctx.tiling_->dk % self->ctx.tiling_->singleIterateDk == 0),
                "tiling_->dk should be divisible by tiling_->singleIterateDk");
#endif
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf *self)
{
    self->ctx.mIter_ = DivCeil(self->ctx.singleShapeM_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.mIter_ > 0),
        "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf *self)
{
    uint64_t tmpSingleCoreK = 0;
    // kernel拆分后，K方向去除无效计算
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        tmpSingleCoreK = static_cast<uint64_t>(self->ctx.tiling_->cout1) * self->ctx.splitHkWkC0List_[self->ctx.splitIndex_];
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            RecalcStepForKernelSplit<Intf>(self);   // 当pad大于0时，子kernel顺序会变化，stepka需要重新计算
        }
    } else {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            tmpSingleCoreK = self->ctx.singleShapeHWk_ * C04_COUT_SIZE;
        } else {
            uint32_t singleShapeCout1 = DivCeilC0<Intf>(self, self->ctx.singleShapeCout_);
            tmpSingleCoreK = singleShapeCout1 * self->ctx.singleShapeHWk_ * self->ctx.tiling_->c0;
        }
    }
    self->ctx.kIter_ = DivCeil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = tmpSingleCoreK - (self->ctx.kIter_ - 1) * self->ctx.tiling_->baseK;
    self->ctx.kIterStepKaTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKa_) - 1) * self->ctx.curStepKa_;
    self->ctx.kIterStepKbTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKb_) - 1) * self->ctx.curStepKb_;
    self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
    self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.kIter_ > 0),
        "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf *self)
{
    self->ctx.nIter_ = DivCeil(self->ctx.singleShapeCin_, self->ctx.tiling_->baseN);
    self->ctx.tailN_ = self->ctx.singleShapeCin_ - (self->ctx.nIter_ - 1) * self->ctx.tiling_->baseN;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.nIter_ > 0),
        "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_);
#endif
}

template <class Intf>
static __aicore__ inline void InitParamsForSplitDkSplit(Intf *self)
{
    self->ctx.enableSplitDk_ = self->ctx.tiling_->singleIterateDk != self->ctx.tiling_->dk;
#ifdef __CCE_KT_TEST__
    ascendc_assert((!self->ctx.enableSplitDk_ || (self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_K))),
                    "iterateOrder must be ORDER_K when splitDK.");
#endif
}

template <class Intf>
__aicore__ inline void InitFullLoadFlag(Intf *self)
{
    if constexpr (Intf::conv3dConfig.loadB1Condition != TPL_GM_TO_L1) {
        self->ctx.isB1FullLoadFlag_ = false;
        self->ctx.isA1FullLoadFlag_ = false;
        return;
    }

    self->ctx.isB1FullLoadFlag_ = (self->ctx.tiling_->stepN * self->ctx.tiling_->baseN >= self->ctx.tiling_->singleCoreCin) &&
        (self->ctx.tiling_->enlarge == 1);
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (!self->ctx.kSCoutFullLoad_) {
            self->ctx.isA1FullLoadFlag_ = false;
            self->ctx.isB1FullLoadFlag_ = false;
        } else {
            bool isDoutDkFullLoad = (self->ctx.tiling_->dk == 1 || self->ctx.tiling_->dout == 1);
            self->ctx.isA1FullLoadFlag_ = self->ctx.tiling_->al1Pbuffer == 1 && isDoutDkFullLoad;
            // pad为奇数(反向pad为偶数)的场景，每个sub kernel需要重载L1A
            // 当前仅支持各方向pad一致且kernel为4或2，简化条件
            if (self->ctx.tiling_->wk == 3 || ((self->ctx.tiling_->backpropPadLeft == 0 ||
                self->ctx.tiling_->backpropPadLeft == 2) && (self->ctx.tiling_->wk == 4 || self->ctx.tiling_->wk == 2))) {
                self->ctx.isA1FullLoadFlag_ = false;
            }
            self->ctx.isB1FullLoadFlag_ = self->ctx.isB1FullLoadFlag_ &&
                self->ctx.tiling_->bl1Pbuffer == 1 && isDoutDkFullLoad;
        }
    } else {
        self->ctx.isA1FullLoadFlag_ = false;
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            self->ctx.isB1FullLoadFlag_ = (self->ctx.tiling_->baseK * self->ctx.curStepKb_ >=
                self->ctx.tiling_->singleCoreCout * self->ctx.splitHkWkList_[0]) && self->ctx.isB1FullLoadFlag_ &&
                (self->ctx.tiling_->dk == 1 || self->ctx.tiling_->dout == 1);
        } else {
            self->ctx.isB1FullLoadFlag_ = (self->ctx.tiling_->baseK * self->ctx.curStepKb_ >=
                self->ctx.tiling_->singleCoreCout * self->ctx.singleShapeHWk_) && self->ctx.isB1FullLoadFlag_ &&
                self->ctx.tiling_->bl1Pbuffer == 1 && self->ctx.tiling_->singleIterateDk == 1;
        }
    }
}

template <class Intf>
__aicore__ inline void InitC04Params(Intf *self)
{
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        if ASCEND_IS_AIV {
            self->ctx.c04LoadToB1IterIdx_ = 0;
            uint32_t c04UbBufSize = (AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH -
                MASK_REG_WIDTH - AscendC::ONE_BLOCK_SIZE) >> 1;
            uint32_t c04UbPixNum = DivDtypeByte<typename Intf::SrcT>(c04UbBufSize);
            uint32_t tmpVal1 = self->ctx.dkHkWk_ << C04_SHIFT_SIZE;
            uint32_t tmpVal2 = AlignUp((self->ctx.hkWk_ << C04_SHIFT_SIZE), self->ctx.tiling_->c0);
            uint32_t tmpVal3 = tmpVal1 > tmpVal2 ? tmpVal1 : tmpVal2;
            self->ctx.vecBlockN_ = ((c04UbPixNum / tmpVal3) >> 4) << 4; // >> 4 等价于除BLOCK_CUBE，<< 4 等价于乘BLOCK_CUBE
        }
    }
}

template <class Intf>
__aicore__ inline void InitParamsPart3(Intf *self)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    self->ctx.dstB2Stride_ = 0;
    self->ctx.startAddrOffset_ = 0;
    self->ctx.headWi_ = 0;
    self->ctx.midHi_ = 0;
    self->ctx.tailWi_ = 0;
    self->ctx.curBackPropPadUp_ = 0;
#endif
    if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
        self->ctx.groupIterIdx_ = 0;
        self->ctx.curEnlarge = 0;
    }
}
template <class Intf>
__aicore__ inline void InitParamsPart2(Intf *self)
{
    self->ctx.mIter_ = 0;
    self->ctx.nIter_ = 0;
    self->ctx.kIter_ = 0;
    self->ctx.tailM_ = 0;
    self->ctx.tailN_ = 0;
    self->ctx.tailK_ = 0;
    self->ctx.curML0Idx_ = 0;
    self->ctx.curNL0Idx_ = 0;
    self->ctx.curStepM_ = 0;
    self->ctx.curStepN_ = 0;
    self->ctx.curLoadKbl1_ = 0;
    self->ctx.curNL1Idx_ = 0;
    self->ctx.curML1Idx_ = 0;
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
    self->ctx.curEnlargeCin1_ = 0;
    self->ctx.curEnlargeCout1_ = 0;
    self->ctx.curDinStartIdx_ = 0;
    self->ctx.curHoStartIdx_ = 0;
    self->ctx.curCinStartIdx_ = 0;
    self->ctx.curCoutStartIdx_ = 0;
    self->ctx.rearrangeHIndex_ = 0;
    self->ctx.rearrangeWIndex_ = 0;
    self->ctx.splitHIndex_ = 0;
    self->ctx.splitWIndex_ = 0;
    self->ctx.curMStartIdx_ = 0;
    self->ctx.curHkIdx_ = 0;
    self->ctx.curWkIdx_ = 0;
    self->ctx.curWoLeftIdx_ = 0;
    self->ctx.curWoRightIdx_ = 0;
    self->ctx.isLoadA1_ = true;
    self->ctx.isLoadB1_ = true;
    self->ctx.isFreeA1_ = false;
    self->ctx.isFreeB1_ = false;
    self->ctx.isLastDk_ = true;
    self->ctx.needComputeFlag_ = true;
}

template <class Intf>
static __aicore__ inline void InitParamsForNormal(Intf *self)
{
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        self->ctx.channelSize_ = C04_COUT_SIZE;
    } else {
        uint32_t curChannelSize = DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK);
        uint32_t alignedCout = AlignUpC0<Intf>(self, self->ctx.tiling_->singleCoreCout);
        self->ctx.channelSize_ = curChannelSize > alignedCout ? alignedCout : curChannelSize;
    }
    uint32_t mL1Size = self->ctx.tiling_->baseM * self->ctx.tiling_->stepM;
    mL1Size = mL1Size > self->ctx.tiling_->singleCoreM ? self->ctx.tiling_->singleCoreM : mL1Size;
    self->ctx.curHoSize_ = CalFmapH<Intf>(self, mL1Size);
}

template <class Intf>
__aicore__ inline void InitParams(Intf *self)
{
    // 初始化为0，避免decache功能开启后全局变量不会初始化为0，出现随机值导致未知问题
    self->ctx.singleShapeDin_ = 0;
    self->ctx.singleShapeM_ = 0;
    self->ctx.singleShapeCin_ = 0;
    self->ctx.singleShapeCout_ = 0;

    self->ctx.isFirstIter_ = true;
    self->ctx.hkWk_ = static_cast<uint64_t>(self->ctx.tiling_->hk) * self->ctx.tiling_->wk;
    self->ctx.singleShapeHWk_ = self->ctx.hkWk_;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK) {
        self->ctx.singleShapeHWk_ = self->ctx.tiling_->wk;
    } else if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        self->ctx.singleShapeHWk_ = 1;
    }
    self->ctx.curStepKa_ = self->ctx.tiling_->stepKa;
    self->ctx.curStepKb_ = self->ctx.tiling_->stepKb;
    InitParamsForSplitDkSplit<Intf>(self);

    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        InitParamsForKernelSplitHW<Intf>(self);
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        InitParamsForKernelSplitH<Intf>(self);
    } else  {
        InitParamsForNormal<Intf>(self);
    }

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    self->ctx.dkHkWk_ = static_cast<uint64_t>(self->ctx.tiling_->dk) * self->ctx.hkWk_;
    self->ctx.hoWo_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.doHoWo_ = self->ctx.tiling_->dout * self->ctx.hoWo_;
    self->ctx.hiWi_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.diHiWi_ = self->ctx.tiling_->di * self->ctx.hiWi_;
    self->ctx.realWoSize_ = self->ctx.tiling_->wo;
    InitFullLoadFlag<Intf>(self);
    InitC04Params<Intf>(self);
#endif
    self->ctx.hoExpand_ = (self->ctx.tiling_->ho - 1) * self->ctx.tiling_->strideH + 1;
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        self->ctx.hoExpand_ = self->ctx.tiling_->ho;
    }
    self->ctx.l0PingPongFlag_ = 0;
    self->ctx.l0aPingPongAddr_ = DivDtypeByte<typename Intf::SrcT>(TOTAL_L0A_SIZE >> 1);
    self->ctx.l0bPingPongAddr_ = DivDtypeByte<typename Intf::SrcT>(TOTAL_L0B_SIZE >> 1);
    self->ctx.enableL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    InitLoadToA2Params<Intf>(self);
    InitLoadToB2Params<Intf>(self);
    InitParamsPart2<Intf>(self);
    InitParamsPart3<Intf>(self);
}

template <class Intf>
__aicore__ inline void CalcMatrixByteSize(Intf *self, uint32_t &aMatrixByteSize, uint32_t &bMatrixByteSize)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        uint32_t hoSize = CalFmapHMaxForKernelSplit(self, self->ctx.tiling_->baseM);
        hoSize = self->ctx.tiling_->ho < hoSize ? self->ctx.tiling_->ho : hoSize;
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.channelSize_ * sizeof(typename Intf::SrcT);
        if (self->ctx.kSCoutFullLoad_) {
            bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN *
                self->ctx.channelSize_ * self->ctx.singleShapeHWk_ * sizeof(typename Intf::SrcT);
        } else {
            bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN *
                self->ctx.curStepKb_ * self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
        }
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        uint32_t hoSize = self->ctx.tiling_->ho < self->ctx.curHoSize_ ? self->ctx.tiling_->ho : self->ctx.curHoSize_;
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
            self->ctx.channelSize_ * sizeof(typename Intf::SrcT);
        bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN *
            self->ctx.curStepKb_ * self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    } else {
        uint32_t hoSize = self->ctx.hoExpand_ < static_cast<uint64_t>(self->ctx.curHoSize_) ?
                static_cast<uint32_t>(self->ctx.hoExpand_) : self->ctx.curHoSize_;
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
                C04_COUT_SIZE * sizeof(typename Intf::SrcT);
        } else if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            // 不加载L1不加载完整的hkwk时, wo无需搬一整行
            aMatrixByteSize = self->ctx.tiling_->baseM *
                DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) * sizeof(typename Intf::SrcT);
        } else {
            aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
                DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) *
                sizeof(typename Intf::SrcT);
        }
        bMatrixByteSize = self->ctx.tiling_->stepN * self->ctx.tiling_->baseN *
            self->ctx.curStepKb_ * self->ctx.tiling_->baseK * sizeof(typename Intf::SrcT);
    }
}

#if defined(__DAV_310R6__)
template <class Intf>
__aicore__ inline void InitBiasTque(Intf *self)
{
    if constexpr (Intf::Config::eType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        // 64: BT Buffer 需要64B对齐，加63后右移6位代替乘64
        uint32_t biasSize = self->ctx.tiling_->isBiasFullLoad
                                ? (DivCeil(self->ctx.tiling_->singleCoreCin * sizeof(typename Intf::BiasT), 64) << 6)
                                : (DivCeil(self->ctx.tiling_->baseN * sizeof(typename Intf::BiasT), 64) << 6);
        self->ctx.pipe_.InitBuffer(self->ctx.biasL1Que_, 1, biasSize);
        self->ctx.pipe_.InitBuffer(self->ctx.biasBTQue_, 1, biasSize);
    }
}
#endif

template <class Intf>
__aicore__ inline void InitTque(Intf *self)
{
    uint32_t bMatrixByteSize = 0;
    uint32_t aMatrixByteSize = 0;
    CalcMatrixByteSize(self, aMatrixByteSize, bMatrixByteSize);
#ifdef __CCE_KT_TEST__
    uint32_t usedBufferSize = aMatrixByteSize * self->ctx.tiling_->al1Pbuffer + bMatrixByteSize * self->ctx.tiling_->bl1Pbuffer;
    ascendc_assert((usedBufferSize <= TOTAL_L1_SIZE), "l1 size exceeds limit");
#endif

    if (IsL1bUseTQue<Intf>(self)) {
        if ASCEND_IS_AIC {
            self->ctx.pipe_.InitBuffer(self->ctx.inQueL1B_, self->ctx.tiling_->bl1Pbuffer, bMatrixByteSize);
        }
    } else {
        self->ctx.pipe_.InitBuffer(self->ctx.b1UbPing_, bMatrixByteSize);
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.pipe_.InitBuffer(self->ctx.b1UbPong_, bMatrixByteSize);
        }
    }

    if ASCEND_IS_AIC {
        self->ctx.pipe_.InitBuffer(self->ctx.inQueL1A_, self->ctx.tiling_->al1Pbuffer, aMatrixByteSize);

        uint32_t baseMN = self->ctx.tiling_->baseM * self->ctx.tiling_->baseN;
        self->ctx.pipe_.InitBuffer(self->ctx.outQueL0C_, self->ctx.tiling_->cl0Pbuffer, baseMN * sizeof(typename Intf::L0cT));
        self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
        self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
#if defined(__DAV_310R6__)
        InitBiasTque(self);
#endif
    }

    InitUbByteSize(self);
}

template <class Intf>
static __aicore__ inline void FreeFullLoadL1Tensor(Intf *self) {
    if ASCEND_IS_AIC {
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
}

template <class Intf>
static __aicore__ inline void ComputeForNoTilingHWk(Intf *self, LocalTensor<typename Intf::SrcT> &l0a,
    LocalTensor<typename Intf::SrcT> &l0b, LocalTensor<typename Intf::L0cT> &l0c, uint8_t &l0PingPongFlag)
{ 
    bool isFirstDk = true;
    for (uint64_t curInnerKdIdx = self->ctx.curDkIdx_; curInnerKdIdx < self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk; curInnerKdIdx++) {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            self->ctx.groupIterIdx_ = 0;
        }

        int64_t curDoutIdx = 0;
        if (!CalcCurDoutIdx<Intf>(self, curInnerKdIdx, curDoutIdx)) {
            continue;
        }
        ComputeForKIter<Intf>(self,l0a, l0b, l0c, curInnerKdIdx, curDoutIdx, isFirstDk, l0PingPongFlag);
        isFirstDk = false;
    }
}

template <class Intf>
static __aicore__ inline void ComputeStart(Intf *self, LocalTensor<typename Intf::L0cT> &l0c)
{
    if ASCEND_IS_AIC {
        InitMmadParams<Intf>(self);
        UpdateLoadToA2ParamsM<Intf>(self);
        UpdateLoadToB2ParamsN<Intf>(self);
        l0c = self->ctx.outQueL0C_.template AllocTensor<typename Intf::L0cT>();
    }

    CrossCoreSetHead<Intf>(self);
}

template <class Intf>
static __aicore__ inline void ComputeEnd(Intf *self, LocalTensor<typename Intf::L0cT> &l0c, uint8_t l0PingPongFlag)
{
    if ASCEND_IS_AIC {
        self->ctx.outQueL0C_.EnQue(l0c);
    }

    self->ctx.l0PingPongFlag_ = l0PingPongFlag;
    CrossCoreWaitTail<Intf>(self);
}

template <class Intf>
static __aicore__ inline void Compute(Intf *self)
{
    if (!self->ctx.needComputeFlag_) {
        FreeFullLoadL1Tensor<Intf>(self);
        return;
    }

    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    ComputeStart<Intf>(self, l0c);
    uint8_t l0PingPongFlag = self->ctx.l0PingPongFlag_;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1) {
        ComputeForNoTilingHWk<Intf>(self, l0a, l0b, l0c, l0PingPongFlag);
    } else {
        ComputeForTilingHkWk<Intf>(self, l0a, l0b, l0c, l0PingPongFlag);
    }
    ComputeEnd<Intf>(self, l0c, l0PingPongFlag);
}

template <class Intf>
static __aicore__ inline void UpdateIdxAndStep(Intf *self)
{
    self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
    self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
    self->ctx.curStepM_ = (self->ctx.mIter_ - self->ctx.curML0Idx_) > self->ctx.tiling_->stepM
                              ? self->ctx.tiling_->stepM
                              : (self->ctx.mIter_ - self->ctx.curML1Idx_);
    self->ctx.curStepN_ = (self->ctx.nIter_ - self->ctx.curNL0Idx_) > self->ctx.tiling_->stepN
                              ? self->ctx.tiling_->stepN
                              : (self->ctx.nIter_ - self->ctx.curNL1Idx_);
}

template <class Intf>
static __aicore__ inline void UpdateCurHoSize(Intf *self)
{
    // 要从当前核hoStartIdx开始算，普通场景可以跨行，要从GM起点开始计算
    uint32_t curWi = self->ctx.tiling_->wi;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        curWi = self->ctx.splitWi_;
    }

    uint64_t curMIdx = self->ctx.curMStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM;
    uint32_t curHiIdx = curMIdx / curWi;
    uint32_t curWiIdx = curMIdx - curHiIdx * curWi;
    self->ctx.load3d_.mStartPt = curWiIdx;
    uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + curWiIdx, curWi);
    uint32_t endHoIdx = 0;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
        self->ctx.curHoIdx_ = static_cast<int32_t>(curHiIdx) - self->ctx.tiling_->backpropPadUp;
        endHoIdx = self->ctx.curHoIdx_ + (hiCal + (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH);
    } else {
        self->ctx.curHoIdx_ = static_cast<int32_t>(curHiIdx) - self->ctx.curBackPropPadUp_;
        endHoIdx = self->ctx.curHoIdx_ + (hiCal + (self->ctx.splitHkList_[self->ctx.splitIndex_] - 1) *
            self->ctx.tiling_->dilationH);
    }
    UpdateCurHoSizeCore<Intf>(self, endHoIdx);
}

template <class Intf>
static __aicore__ inline void UpdateKComputeStatus(Intf *self)
{
    if (!self->ctx.needComputeFlag_) {
        // 其他方向要跳过计算，K可以不判断了
        return;
    }

    // 整个dk都在pad里面, 整轮计算跳过
    if (unlikely(self->ctx.tiling_->strideD > self->ctx.tiling_->dk)) {
        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront;
        if (dTmp % self->ctx.tiling_->strideD >= self->ctx.tiling_->dk ||
            dTmp / self->ctx.tiling_->strideD >= self->ctx.tiling_->dout) {
            self->ctx.needComputeFlag_ = false;
            return;
        }
    }

    bool isKNeedCompute = false;
    for (uint64_t curKdIdx = self->ctx.curDkIdx_; curKdIdx < self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk; curKdIdx++) {
        // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
        if (dTmp < 0 || (self->ctx.tiling_->strideD > 1 && (dTmp % self->ctx.tiling_->strideD > 0)) ||
            dTmp >= self->ctx.tiling_->dout * self->ctx.tiling_->strideD) {
            continue;
        }
        isKNeedCompute = true;
        break;
    }
    self->ctx.needComputeFlag_ = isKNeedCompute;
    if constexpr (Intf::conv3dConfig.loadB1Condition != TPL_GM_TO_L1) {
        UpdateHkComputeStatus<Intf>(self);
    }
}

template <class Intf>
static __aicore__ inline void UpdateFullLoadL1Status(Intf *self)
{
    if (!self->ctx.isB1FullLoadFlag_ && !self->ctx.isA1FullLoadFlag_) {
        return;
    }
    bool isLastMNIter = (self->ctx.curML0Idx_ == self->ctx.mIter_ - 1) &&
        self->ctx.curNL0Idx_ == self->ctx.nIter_ - 1;
    bool isLastDIter = (self->ctx.tiling_->dk == 1 && self->ctx.curDinIdx_ ==
        self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_ - 1) ||
        self->ctx.tiling_->dk > 1;

    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        self->ctx.isFreeB1_ = isLastMNIter && self->ctx.tiling_->dk > 1;
        bool isLastRearrangeHWIter = (self->ctx.rearrangeWIndex_ == self->ctx.tiling_->strideW - 1) &&
            (self->ctx.rearrangeHIndex_ == self->ctx.tiling_->strideH - 1);
        self->ctx.isFreeA1_ = isLastRearrangeHWIter;
        self->ctx.isFreeB1_ = self->ctx.isFreeB1_ && isLastRearrangeHWIter;
    } else {
        if (self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1) {
            // c04场景和group>1（包括enlarge>1）场景不支持calRound间不释放B矩阵
            if (self->ctx.enableFullLoad_ && !Intf::conv3dConfig.enableC04Flag && self->ctx.tiling_->group == 1) {
                isLastDIter = false;
            }
        }
        self->ctx.isFreeB1_ = isLastMNIter && isLastDIter;
    }
}

template <class Intf>
static __aicore__ inline void UpdateL1ComputeInfo(Intf *self)
{
    self->ctx.baseUseM_ =
        (self->ctx.curML0Idx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
        uint32_t Hd = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
        uint32_t Wd = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
        // load3d指令的padList[3]固定为255
        uint32_t Hp = self->ctx.hoExpand_ + self->ctx.tiling_->backpropPadUp + 255;
        uint32_t Wp = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1 
            + self->ctx.tiling_->backpropPadLeft + self->ctx.tiling_->backpropPadRight;
        uint64_t M = (Hp - Hd + 1) * (Wp - Wd + 1);
        if (unlikely(M < self->ctx.baseUseM_) && self->ctx.mIter_ == 1) {
            self->ctx.baseUseM_ = M;
        }
    }
    self->ctx.baseUseN_ =
        (self->ctx.curNL0Idx_ + 1 == self->ctx.nIter_) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
    self->ctx.needComputeFlag_ = true;
    UpdateCurHoSize<Intf>(self);
    UpdateKComputeStatus<Intf>(self);
    UpdateFullLoadL1Status<Intf>(self);
}

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const conv_bp_v2_kernel::TConv3DInputV2Tiling * tiling)
    {
        self->ctx.tiling_ = tiling;
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(true);
        }
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
        // kernel拆分、切Dk、C04场景，Init()先空下一个CrossCoreSetFlag()来保证End()里下的CrossCoreWaitFlag()可以成对
        CrossCoreSetHeadForMix<Intf>(self);
    }
};

template <class Intf>
struct SetWeight {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &weight)
    {
        self->ctx.weightGlobal_ = weight;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::SrcT> &outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

#if defined(__DAV_310R6__)
template <class Intf>
struct SetBias {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::BiasT> &bias)
    {
        self->ctx.biasGlobal_ = bias;
    }
};
#endif

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
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

        if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
            self->ctx.singleShapeCout_ = singleShapeK / self->ctx.singleShapeHWk_;
            InitStepKParams<Intf>(self);
        } else {
            if (self->ctx.singleShapeCout_ * self->ctx.singleShapeHWk_ != singleShapeK) {
                self->ctx.singleShapeCout_ = singleShapeK / self->ctx.singleShapeHWk_;
                InitStepKParams<Intf>(self);
            }
        }

        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            self->ctx.curEnlarge = self->ctx.singleShapeCout_ / (self->ctx.tiling_->cout / self->ctx.tiling_->oriGroup);
            self->ctx.curEnlargeCin1_ = DivCeil16(self->ctx.curEnlarge * (self->ctx.tiling_->cin / self->ctx.tiling_->oriGroup));
            self->ctx.curEnlargeCout1_ = DivCeilC0<Intf>(self, self->ctx.singleShapeCout_);
        }
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t curDinStartIdx, int64_t curMStartIdx,
        int32_t curCinStartIdx, int32_t curCoutStartIdx)
    {
        // 需要增加入参的startDinIdx判断，否则无法确定padding和补0问题
        // 增加入参，传参判断
        self->ctx.curDinStartIdx_ = curDinStartIdx;
        self->ctx.curMStartIdx_ = curMStartIdx;
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / self->ctx.tiling_->wi) - self->ctx.tiling_->backpropPadUp;
        } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            self->ctx.curBackPropPadUp_ = self->ctx.tiling_->backpropPadUp >> 1;
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / (self->ctx.splitWi_)) - self->ctx.curBackPropPadUp_;
        } else {
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / self->ctx.tiling_->wi) - self->ctx.curBackPropPadUp_;
        }
        self->ctx.curCinStartIdx_ = curCinStartIdx;
        self->ctx.curCoutStartIdx_ = curCoutStartIdx;
    }
};

template <class Intf>
struct SetFullLoadFlag {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, bool enableFullLoad)
    {
        // 设置全载模板标志位
        self->ctx.enableFullLoad_ = enableFullLoad;
    }
};

template <class Intf, bool sync>
struct Iterate {
    // 一次iterate计算(baseM, baseN, baseD)，当前baseD=1
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline bool call(Intf *self, bool enPartialSum)
    {
        /*
        |   <---------singleShapeM------->        |
        |  <---L1A_ping--->  |  <---L1A_pong--->  |
        |_L0A1_|_L0A2_|_L0A3_|_L0A4_|_L0A5_|_L0A6_|
        ↑  <--curStepM_-->    |                    ↑
        curML0Idx_            ↑                  mIter_
        curML1Idx_        next_curML1Idx

        |   <---------singleShapeN------->        |
        |  <---L1B_ping--->  |  <---L1B_pong--->  |
        |_L0B1_|_L0B2_|_L0B3_|_L0B4_|_L0B5_|_L0B6_|
        ↑  <--curStepN_-->    |                    ↑
        curNL0Idx_            ↑                   nIter_
        curNL1Idx_       next_curNL1Idx

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong *
        L1B_pong L0上也是驻留B，循环A order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 …………
        L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6 order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 …………
        L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN
        if (unlikely(self->ctx.isFirstIter_)) {
            if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
                self->ctx.groupIterIdx_ = 0;
            }
            self->ctx.needComputeFlag_ = true;
            self->ctx.curML0Idx_ = 0;
            self->ctx.curNL0Idx_ = 0;
            self->ctx.curML1Idx_ = 0;
            self->ctx.curNL1Idx_ = 0;
            self->ctx.curDinIdx_ = self->ctx.curDinStartIdx_;
            self->ctx.curDkIdx_ = 0;
            self->ctx.curHoIdx_ = self->ctx.curHoStartIdx_;
            self->ctx.isFirstIter_ = false;
            self->ctx.isLoadA1_ = true;
            self->ctx.isFreeA1_ = false;
            self->ctx.curStepM_ = (self->ctx.mIter_ - self->ctx.curML0Idx_) > self->ctx.tiling_->stepM
                                      ? self->ctx.tiling_->stepM
                                      : (self->ctx.mIter_ - self->ctx.curML1Idx_);
            self->ctx.curStepN_ = (self->ctx.nIter_ - self->ctx.curNL0Idx_) > self->ctx.tiling_->stepN
                                      ? self->ctx.tiling_->stepN
                                      : (self->ctx.nIter_ - self->ctx.curNL1Idx_);
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                self->ctx.rearrangeHIndex_ = 0;
                self->ctx.rearrangeWIndex_ = 0;
                self->ctx.splitHIndex_ = self->ctx.splitHStartIndex_;
                self->ctx.splitWIndex_ = self->ctx.splitWStartIndex_;
                if (self->ctx.tiling_->dk > 1) {    // dk等于1时，全载时无需重复加载B矩阵
                    self->ctx.isLoadB1_ = true;
                    self->ctx.isFreeB1_ = false;
                }
            } else {
                if (!self->ctx.isB1FullLoadFlag_ || self->ctx.tiling_->dk > 1 ||
                Intf::conv3dConfig.enableC04Flag || !self->ctx.enableFullLoad_ || self->ctx.tiling_->group != 1) {
                    self->ctx.isLoadB1_ = true;
                    self->ctx.isFreeB1_ = false;
                }
            }
        } else if (likely(self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                if (IterateForKernelSplit<Intf>(self)) {
                    UpdateFullLoadL1Status<Intf>(self);
                    Compute<Intf>(self);
                    return true;
                }
            }
            if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
                if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                    self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    if (self->ctx.curNL0Idx_ >= self->ctx.nIter_ && self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ = 0;
                        if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                            return false;
                        }
                    }
                    if (self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        } else if (unlikely(self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_K))) {
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                if (IterateForKernelSplit<Intf>(self)) {
                    UpdateFullLoadL1Status<Intf>(self);
                    Compute<Intf>(self);
                    return true;
                }
            }
            if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                self->ctx.curML0Idx_ = self->ctx.curML1Idx_;
                if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                    self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    if (self->ctx.curNL0Idx_ >= self->ctx.nIter_ && self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ = 0;
                        if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                            self->ctx.curDinIdx_ = self->ctx.curDinStartIdx_;
                            self->ctx.curDkIdx_ += self->ctx.tiling_->singleIterateDk;
                            if (self->ctx.curDkIdx_ >= self->ctx.tiling_->dk) {
                                return false;
                            }
                            self->ctx.isLastDk_ = (self->ctx.curDkIdx_ == self->ctx.tiling_->dk - 1);
                        }
                    }
                    if (self->ctx.curML1Idx_ >= self->ctx.mIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        } else {  // order_M
            if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
                if (IterateForKernelSplit<Intf>(self)) {
                    UpdateFullLoadL1Status<Intf>(self);
                    Compute<Intf>(self);
                    return true;
                }
            }
            if (++self->ctx.curNL0Idx_ >= self->ctx.curNL1Idx_ + self->ctx.curStepN_) {
                self->ctx.curNL0Idx_ = self->ctx.curNL1Idx_;
                if (++self->ctx.curML0Idx_ >= self->ctx.curML1Idx_ + self->ctx.curStepM_) {
                    self->ctx.curNL1Idx_ += self->ctx.curStepN_;
                    if (self->ctx.curML0Idx_ >= self->ctx.mIter_ && self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        self->ctx.curML1Idx_ = 0;
                        self->ctx.curNL1Idx_ = 0;
                        if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                            return false;
                        }
                    }
                    if (self->ctx.curNL1Idx_ >= self->ctx.nIter_) {
                        self->ctx.curNL1Idx_ = 0;
                        self->ctx.curML1Idx_ += self->ctx.curStepM_;
                    }
                    UpdateIdxAndStep<Intf>(self);
                }
            }
        }

        UpdateL1ComputeInfo<Intf>(self);
        Compute<Intf>(self);
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic)
    {
#if defined(__DAV_310R6__)
        if (self->ctx.tiling_->isBiasFullLoad && Intf::Config::eType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            Convolution3DBackpropFunc::FullLoadBias<Intf>(self);
        }
#endif
        if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
            while (self->template Iterate<sync>()) {
                self->template VecPreProcess<sync>(output, enAtomic);
                self->template GetTensorC<sync>(output, enAtomic);
                self->template VecPostProcess<sync>(output, enAtomic);
            }
        } else {
            self->template IterateAllForKernelSplit<sync>(output, enAtomic);
        }
#if defined(__DAV_310R6__)
        if (self->ctx.tiling_->isBiasFullLoad && Intf::Config::eType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if ASCEND_IS_AIC {
                self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
                self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
            }
        }
#endif
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf, bool sync>
struct VecPreProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if (unlikely(self->ctx.enableSplitDk_)) {
            if ASCEND_IS_AIC {
                // 反向event(实际只需要保证vector核内的时序,不加反向event时cube到vector之间的event有可能会下超过16次,导致报错)
                if (self->ctx.isLastDk_) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_FIXP_ID);
                }
            }
        }
    }
};

template <class Intf, bool sync>
struct VecPostProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            Rearrange2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
        }
        if (unlikely(self->ctx.enableSplitDk_)) {
            if ASCEND_IS_AIC {
                if (self->ctx.isLastDk_) {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MTE2_VEC_ID);
                }
            }
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() != 0) {
                    return;
                }
                if (self->ctx.isLastDk_) {
                    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(FLAG_MTE2_VEC_ID);
                    CastToDstType<Intf>(self, output, enAtomic, enSequentialWrite);
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_FIXP_ID);
                }
            }
        }
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self)
    {
        CrossCoreWaitTailForMix<Intf>(self);

        self->ctx.inQueL1A_.FreeAllEvent();
        self->ctx.inQueL1B_.FreeAllEvent();
        self->ctx.outQueL0C_.FreeAllEvent();
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};
}  // namespace Convolution3DBackpropFunc
#endif
