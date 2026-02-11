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
 * \file conv3d_bp_kernel_split_func.h
 * \brief
 */

#ifndef CONV3D_BP_KERNEL_SPLIT_FUNC_ADVANCE_H
#define CONV3D_BP_KERNEL_SPLIT_FUNC_ADVANCE_H

#include "conv3d_bp_common_func.h"

DECLARE_CHECK_IMPL(SetKernelSplitParams);
DECLARE_CHECK_IMPL(SetSingleShapeParams);
DECLARE_CHECK_IMPL(FreeB1Tensor);
DECLARE_CHECK_SYNC_IMPL(IterateAllForKernelSplit);
namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void RecalcStepForKernelSplit(Intf *self)
{
    self->ctx.curStepKa_ = DivCeil(static_cast<uint64_t>(self->ctx.channelSize_ *
        self->ctx.splitHkWkList_[self->ctx.splitIndex_]),
        static_cast<uint64_t>(self->ctx.tiling_->baseK));
    self->ctx.curStepKb_ = DivCeil(static_cast<uint64_t>(self->ctx.coutChannelSize_ *
        self->ctx.splitHkWkList_[self->ctx.splitIndex_]), static_cast<uint64_t>(self->ctx.tiling_->baseK));
    if (self->ctx.tiling_->stepKb % self->ctx.curStepKb_ == 0) {
        self->ctx.curStepKb_ = self->ctx.tiling_->stepKb;   // 此时，子hk之间可以整除，因此可以增加stepkb
    }
}

template <class Intf>
__aicore__ inline uint32_t CalFmapHForKernelSplitInner(Intf *self, uint32_t mL1Size, uint32_t hk)
{
    uint32_t hiCal;
    constexpr uint32_t fMapHNum = 2; // 保证不整除时加载足够的ho
    if (mL1Size % self->ctx.splitWi_ == 0) {
        hiCal = mL1Size / self->ctx.splitWi_;
    } else if (mL1Size > self->ctx.splitWi_) {
        hiCal = mL1Size / self->ctx.splitWi_ + fMapHNum;
    } else {
        hiCal = fMapHNum;
    }
    uint32_t khDilation = (hk - 1) * self->ctx.tiling_->dilationH + 1;
    return (hiCal - 1) + khDilation;
}

template <class Intf>
__aicore__ inline uint32_t CalFmapHForKernelSplit(Intf *self, uint32_t mL1Size)
{
    return CalFmapHForKernelSplitInner(self, mL1Size, self->ctx.splitHkList_[self->ctx.splitIndex_]);
}

template <class Intf>
__aicore__ inline uint32_t CalFmapHMaxForKernelSplit(Intf *self, uint32_t mL1Size)
{
    return CalFmapHForKernelSplitInner(self, mL1Size, self->ctx.splitHkList_[0]); // 0: max sub kernel
}

static __aicore__ inline int32_t CalcSubKernelBackpadBegin(int32_t kernelSize,
    int32_t stride, int32_t backPad, int32_t idx)
{
    int32_t subKernelSize = (kernelSize + stride - idx - 1) >> 1;
    int32_t subKernelBackPadMax = subKernelSize - 1;
    int32_t missPointNum = (kernelSize - idx - backPad) >> 1;
    int32_t subKernelBackPad = subKernelBackPadMax - missPointNum;
    return subKernelBackPad;
}

static __aicore__ inline int32_t CalcSubKernelBackpadEnd(int32_t kernelSize,
    int32_t stride, int32_t backPad, int32_t idx)
{
    int32_t subKernelSize = (kernelSize + stride - idx - 1) >> 1;
    int32_t subKernelBackPadMax = subKernelSize - 1;
    int32_t idxReverse = kernelSize - ((subKernelSize - 1) * stride + 1 + idx);
    int32_t missPointNum = (kernelSize - idxReverse - backPad) >> 1;
    int32_t subKernelBackPad = subKernelBackPadMax - missPointNum;
    return subKernelBackPad;
}

template <class Intf>
static __aicore__ inline void CalcSubKernelPadListForKernelSplit(Intf *self)
{
    uint32_t subKernel02PadRight = 0;
    uint32_t subKernel13PadRight = 0;
    // wi奇数且需要交换子kernel顺序的场景下，右侧pad要+1
    if (((self->ctx.tiling_->backpropPadLeft == 0 || self->ctx.tiling_->backpropPadLeft == 2) &&
        (self->ctx.tiling_->wk == 4 || self->ctx.tiling_->wk == 2)) ||
        (self->ctx.tiling_->backpropPadLeft == 1 && self->ctx.tiling_->wk == 3)) {
        if ((self->ctx.tiling_->wi & 0x1) == 1) {
            subKernel02PadRight = 1;
            subKernel13PadRight = 0;
        }
    } else {
        if ((self->ctx.tiling_->wi & 0x1) == 1) {
            subKernel02PadRight = 0;
            subKernel13PadRight = 1;
        }
    }

    // 当前仅支持stride=2，简化计算公式
    self->ctx.subPadLeftList_[0] = CalcSubKernelBackpadBegin(self->ctx.tiling_->wk,
        self->ctx.tiling_->strideW, self->ctx.tiling_->backpropPadLeft, 0);
    self->ctx.subPadLeftList_[1] = CalcSubKernelBackpadBegin(self->ctx.tiling_->wk,
        self->ctx.tiling_->strideW, self->ctx.tiling_->backpropPadLeft, 1);
    self->ctx.subPadLeftList_[2] = self->ctx.subPadLeftList_[0];
    self->ctx.subPadLeftList_[3] = self->ctx.subPadLeftList_[1];
    self->ctx.subPadRightList_[0] = CalcSubKernelBackpadEnd(self->ctx.tiling_->wk,
        self->ctx.tiling_->strideW, self->ctx.tiling_->backpropPadRight, 0) + subKernel02PadRight;
    self->ctx.subPadRightList_[1] = CalcSubKernelBackpadEnd(self->ctx.tiling_->wk,
        self->ctx.tiling_->strideW, self->ctx.tiling_->backpropPadRight, 1) + subKernel13PadRight;
    self->ctx.subPadRightList_[2] = self->ctx.subPadRightList_[0];
    self->ctx.subPadRightList_[3] = self->ctx.subPadRightList_[1];
    self->ctx.subPadUpList_[0] = CalcSubKernelBackpadBegin(self->ctx.tiling_->hk,
        self->ctx.tiling_->strideH, self->ctx.tiling_->backpropPadUp, 0);
    self->ctx.subPadUpList_[1] = self->ctx.subPadUpList_[0];
    self->ctx.subPadUpList_[2] = CalcSubKernelBackpadBegin(self->ctx.tiling_->hk,
        self->ctx.tiling_->strideH, self->ctx.tiling_->backpropPadUp, 1);
    self->ctx.subPadUpList_[3] = self->ctx.subPadUpList_[2];
    self->ctx.subPadDownList_[0] = CalcSubKernelBackpadEnd(self->ctx.tiling_->hk,
        self->ctx.tiling_->strideH, self->ctx.tiling_->backpropPadDown, 0);
    self->ctx.subPadDownList_[1] = self->ctx.subPadDownList_[0];
    self->ctx.subPadDownList_[2] = CalcSubKernelBackpadEnd(self->ctx.tiling_->hk,
        self->ctx.tiling_->strideH, self->ctx.tiling_->backpropPadDown, 1);
    self->ctx.subPadDownList_[3] = self->ctx.subPadDownList_[2];
}

template <class Intf>
static __aicore__ inline void InitParamsForKernelSplitHW(Intf *self)
{
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.tiling_->hk == self->ctx.tiling_->strideH),
                    "kernelH should be EQ strideH");
    ascendc_assert((self->ctx.tiling_->wk == self->ctx.tiling_->strideW),
                    "kernelW should be EQ strideW");
#endif
    self->ctx.splitWStartIndex_ = self->ctx.tiling_->padLeft % self->ctx.tiling_->strideW;
    self->ctx.splitHStartIndex_ = self->ctx.tiling_->padUp % self->ctx.tiling_->strideH;
    self->ctx.splitIndex_ = self->ctx.splitHStartIndex_ * self->ctx.tiling_->strideW + self->ctx.splitWStartIndex_;

    // 0,1,2,3 sub kernel index, kernel4*4：4个子kernel均为2*2；kernel3*3：2*2,2*1,1*2,1*1；kernel2*2：4个子kernel均为1*1
    self->ctx.splitWkList_[0] = DivCeil(self->ctx.tiling_->wk, self->ctx.tiling_->strideW);
    self->ctx.splitWkList_[1] = self->ctx.tiling_->wk - self->ctx.splitWkList_[0];
    self->ctx.splitWkList_[2] = self->ctx.splitWkList_[0];
    self->ctx.splitWkList_[3] = self->ctx.splitWkList_[1];
    self->ctx.splitHkList_[0] = DivCeil(self->ctx.tiling_->hk, self->ctx.tiling_->strideH);
    self->ctx.splitHkList_[1] = self->ctx.splitHkList_[0];
    self->ctx.splitHkList_[2] = self->ctx.tiling_->hk - self->ctx.splitHkList_[0];
    self->ctx.splitHkList_[3] = self->ctx.splitHkList_[2];

    for (uint32_t i = 0; i < SUB_KERNEL_NUM; i++) {
        self->ctx.splitHkWkList_[i] = self->ctx.splitWkList_[i] * self->ctx.splitHkList_[i];
        self->ctx.splitHkWkC0List_[i] = self->ctx.splitHkWkList_[i] * self->ctx.tiling_->c0;
    }

    self->ctx.splitWi_ = DivCeil(self->ctx.tiling_->wi, self->ctx.tiling_->strideW);
    // hi不对齐strideh时，subkernel Msize 比 first msize hi少1
    self->ctx.subKernelM_ = static_cast<uint64_t>(self->ctx.tiling_->hi) / self->ctx.tiling_->strideH *
        self->ctx.splitWi_;

    uint32_t alignedCout = AlignUpC0<Intf>(self, self->ctx.tiling_->singleCoreCout);
    if (self->ctx.kSCoutFullLoad_) {
        self->ctx.channelSize_ = alignedCout;
    } else {
        uint32_t curChannelSize = (self->ctx.curStepKa_ * self->ctx.tiling_->baseK) / self->ctx.splitHkWkList_[0]; // 3: min sub_kernel size
        self->ctx.channelSize_ = curChannelSize > alignedCout ? alignedCout : curChannelSize;
    }
    self->ctx.coutChannelSize_ = self->ctx.curStepKb_ * self->ctx.tiling_->baseK / self->ctx.splitHkWkList_[0];
    self->ctx.curHoSize_ = CalFmapHForKernelSplit(self, self->ctx.tiling_->baseM);
    CalcSubKernelPadListForKernelSplit<Intf>(self);
}

template <class Intf>
static __aicore__ inline void InitParamsForKernelSplitH(Intf *self)
{
    self->ctx.splitIndex_ = 0;
    self->ctx.splitWi_ = self->ctx.tiling_->wi;
    self->ctx.splitWkList_[0] = self->ctx.tiling_->wk;
    self->ctx.splitHkList_[0] = DivCeil(self->ctx.tiling_->hk, self->ctx.tiling_->strideH);
    self->ctx.splitHkWkList_[0] = self->ctx.splitHkList_[0] * self->ctx.tiling_->wk;
    uint32_t curChannelSize = self->ctx.curStepKa_ * self->ctx.tiling_->baseK / self->ctx.splitHkWkList_[0];
    uint32_t alignedCout = AlignUpC0<Intf>(self, self->ctx.tiling_->singleCoreCout);
    self->ctx.channelSize_ = curChannelSize > alignedCout ? alignedCout : curChannelSize;
    self->ctx.coutChannelSize_ = self->ctx.curStepKb_ * self->ctx.tiling_->baseK / self->ctx.splitHkWkList_[0];
    uint32_t mL1Size = self->ctx.tiling_->baseM * self->ctx.tiling_->stepM;
    mL1Size = mL1Size > self->ctx.tiling_->singleCoreM ? self->ctx.tiling_->singleCoreM : mL1Size;
    self->ctx.curHoSize_ = CalFmapH<Intf>(self, mL1Size);
}

template <class Intf>
static __aicore__ inline void UpdateHoSizeForKernelSplit(Intf *self)
{
    // hoidx和 baseUseM或子kernel发生变化，需要重新计算 curHoSize
    uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.splitWi_);
    uint32_t endHoIdx = self->ctx.curHoIdx_ + (hiCal + (self->ctx.splitHkList_[self->ctx.splitIndex_] - 1) *
        self->ctx.tiling_->dilationH);
    UpdateCurHoSizeCore<Intf>(self, endHoIdx);
}

template <class Intf>
static __aicore__ inline void UpdateHoIdxForKernelSplit(Intf *self)
{
    if (self->ctx.tiling_->backpropPadUp == 2) {    // 2 : 反向pad=2，需要更新hoidx
        if (self->ctx.rearrangeHIndex_ > 0 && self->ctx.rearrangeWIndex_ == 0) {
            ++self->ctx.curHoIdx_;
            UpdateHoSizeForKernelSplit<Intf>(self);
        }
    }

    if (self->ctx.tiling_->hk == 2 || self->ctx.tiling_->hk == 4) {   // 2 4 : hk等于2或者4时子kernel大小一致
        return;
    }

    UpdateHoSizeForKernelSplit<Intf>(self);
}

template <class Intf>
static __aicore__ inline void RecalcBaseUseMForKernelSplit(Intf *self)
{
    if (self->ctx.rearrangeHIndex_ == 0 || self->ctx.rearrangeWIndex_ > 0) {
        return; // 仅需在rearrangeHIndex_=1, rearrangeWIndex_=0时进入
    }

    // hi不对齐strideH时，会出现两个子kenrel计算得Msize不一样大得场景，需要重新计算baseUseN, 避免多搬
    uint64_t curMIdx = self->ctx.curMStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM;
    if (curMIdx > self->ctx.subKernelM_) {
        self->ctx.needComputeFlag_ = false;
        self->ctx.curHoSize_ = 0;
    } else if (curMIdx + self->ctx.baseUseM_ > self->ctx.subKernelM_) {
        self->ctx.baseUseM_ = self->ctx.subKernelM_ - curMIdx;
        self->ctx.mmad_.m = ((self->ctx.baseUseM_ + BLOCK_CUBE - 1) >> 4) << 4;
    }
}

template <class Intf>
static __aicore__ inline bool IterateForKernelSplit(Intf *self)
{
    bool res = true;
    if (++self->ctx.splitWIndex_ >= self->ctx.tiling_->strideW) {
        self->ctx.splitWIndex_ = 0;
    }
    if (++self->ctx.rearrangeWIndex_ >= self->ctx.tiling_->strideW) {
        self->ctx.rearrangeWIndex_ = 0;
        if (++self->ctx.splitHIndex_ >= self->ctx.tiling_->strideH) {
            self->ctx.splitHIndex_ = 0;
        }
        if (++self->ctx.rearrangeHIndex_ >= self->ctx.tiling_->strideH) {
            self->ctx.rearrangeHIndex_ = 0;
            res = false;
        }
    }

    // 切换子kernel的时候刷新不同子kernel的参数
    self->ctx.splitIndex_ = self->ctx.splitHIndex_ * self->ctx.tiling_->strideW + self->ctx.splitWIndex_;
    RecalcBaseUseMForKernelSplit<Intf>(self);
    // 不同的sub kernel当反向pad up不同时，HoIdx也不同
    UpdateHoIdxForKernelSplit<Intf>(self);
    RecalcStepForKernelSplit<Intf>(self);

    uint32_t tmpSingleCoreK = self->ctx.tiling_->cout1 * self->ctx.splitHkWkC0List_[self->ctx.splitIndex_];
    self->ctx.kIter_ = DivCeil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = tmpSingleCoreK - (self->ctx.kIter_ - 1) * self->ctx.tiling_->baseK;
    self->ctx.kIterStepKaTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKa_) - 1) * self->ctx.curStepKa_;
    self->ctx.kIterStepKbTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKb_) - 1) * self->ctx.curStepKb_;
    self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
    self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
    self->ctx.load3d_.filterW = self->ctx.splitWkList_[self->ctx.splitIndex_];
    self->ctx.load3d_.filterH = self->ctx.splitHkList_[self->ctx.splitIndex_];
    self->ctx.load3d_.filterSizeW = (self->ctx.splitWkList_[self->ctx.splitIndex_] >> 8) & 255;
    self->ctx.load3d_.filterSizeH = (self->ctx.splitHkList_[self->ctx.splitIndex_] >> 8) & 255;

    return res;
}

template <class Intf>
struct SetKernelSplitParams {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t kSCoutFullLoad, uint32_t kSUseWorkSpace)
    {
        self->ctx.kSCoutFullLoad_ = kSCoutFullLoad;
        self->ctx.kSUseWorkSpace_ = kSUseWorkSpace;
    }
};

template <class Intf>
struct SetSingleShapeParams {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, uint32_t curSplitHk, int32_t curBackpropPadUp)
    {
        self->ctx.curBackPropPadUp_ = curBackpropPadUp;
        self->ctx.splitHkList_[self->ctx.splitIndex_] = curSplitHk;
        self->ctx.splitHkWkList_[self->ctx.splitIndex_] = self->ctx.splitHkList_[self->ctx.splitIndex_] *
            self->ctx.splitWkList_[self->ctx.splitIndex_];
        self->ctx.splitHkWkC0List_[self->ctx.splitIndex_] = self->ctx.splitHkWkList_[self->ctx.splitIndex_] *
            self->ctx.tiling_->c0;
        RecalcStepForKernelSplit<Intf>(self);

        self->ctx.load3d_.filterW = self->ctx.splitWkList_[self->ctx.splitIndex_];
        self->ctx.load3d_.filterH = self->ctx.splitHkList_[self->ctx.splitIndex_];
        self->ctx.load3d_.filterSizeW = (self->ctx.splitWkList_[self->ctx.splitIndex_] >> 8) & 255;
        self->ctx.load3d_.filterSizeH = (self->ctx.splitHkList_[self->ctx.splitIndex_] >> 8) & 255;
    }
};

template <class Intf>
struct FreeB1Tensor {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self)
    {
        // 整轮计算都未加载B1Tensor就不需要重复释放
        if (self->ctx.isLoadB1_ && !self->ctx.isFreeB1_) {
            return;
        }
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            if (self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1) {
                self->ctx.isLoadB1_ = true;
                self->ctx.isFreeB1_ = false;
                self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
            }
        } else if (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
            // c04场景和group>1（包括enlarge>1）场景不支持calRound间不释放B矩阵
            if (self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1 &&
            !Intf::conv3dConfig.enableC04Flag && self->ctx.enableFullLoad_ && self->ctx.tiling_->group == 1) {
                self->ctx.isLoadB1_ = true;
                self->ctx.isFreeB1_ = false;
                self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
            }
        }
    }
};


template <class Intf>
__aicore__ inline void CrossCoreCSeitVForKS(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if (self->ctx.needComputeFlag_) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(FLAG_MTE2_VEC_ID);
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreCWaitVForKS(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if (self->ctx.needComputeFlag_) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_FIXP_ID);
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreVSeitCForKS(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if (self->ctx.needComputeFlag_) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_FIXP_ID);
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreVWaitCForKS(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if (self->ctx.needComputeFlag_) {
        if (self->ctx.kSUseWorkSpace_) {
        // workspace方案会存在从gm往ub写的过程，因此vec首指令为MTE2
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE2>(FLAG_MTE2_VEC_ID);
        } else {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MTE2_VEC_ID);
        }
    }
#endif
}


template <class Intf, bool sync>
struct IterateAllForKernelSplit {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf *self, const GlobalTensor<typename Intf::DstT> &output, uint8_t enAtomic)
    {
        while (self->template Iterate<sync>()) {
            if ASCEND_IS_AIC {
                if (self->ctx.rearrangeWIndex_ == 0) {
                    CrossCoreCWaitVForKS<Intf>(self);
                }
                self->template GetTensorC<sync>(output, enAtomic);
                if (self->ctx.rearrangeWIndex_ == self->ctx.tiling_->strideW - 1) {
                    CrossCoreCSeitVForKS<Intf>(self);
                }
            }
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() != 0) {
                    continue;
                }
                if (self->ctx.rearrangeWIndex_ == self->ctx.tiling_->strideW - 1) {
                    CrossCoreVWaitCForKS<Intf>(self);
                    if (self->ctx.needComputeFlag_) {
                        self->template VecPostProcess<sync>(output, enAtomic);
                    }
                    CrossCoreVSeitCForKS<Intf>(self);
                }
            }
        }
    }
};
}  // namespace Convolution3DBackpropFunc
#endif