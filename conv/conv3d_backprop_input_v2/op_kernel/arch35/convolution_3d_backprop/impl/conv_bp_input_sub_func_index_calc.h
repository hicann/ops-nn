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
 * \file conv_bp_input_sub_func_index_calc.h
 * \brief Index and size calculation utilities for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_INDEX_CALC_H
#define CONV3D_BP_INPUT_SUB_FUNC_INDEX_CALC_H

#include "conv_bp_input_sub_func_utils.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline uint32_t CalcCurCinSizeB1(Intf* self, uint32_t curCinIdx)
{
    uint32_t curCinSize = self->ctx.tiling_->baseN;
    // consider tail
    uint32_t curCinRemain = self->ctx.singleShapeCin_ - curCinIdx;
    curCinSize = curCinSize < curCinRemain ? curCinSize : curCinRemain;
    return curCinSize;
}

template <class Intf, bool ksCoutFullLoad>
__aicore__ inline void CalcCoutIndexAndSizeB1(Intf* self, uint64_t kIdx, uint32_t& curCoutIdx, uint32_t& curCoutSize)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (ksCoutFullLoad) { // cout全载场景
            curCoutSize = self->ctx.tiling_->singleCoreCout;
            return;
        }
    }

    // 考虑到Preload的场景，L1的载入量要根据传入的KIdx确定，不能使用全局变量
    uint32_t kbL1Size = 0;
    if (unlikely(kIdx >= self->ctx.kIterStepKbTail)) {
        kbL1Size = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    } else {
        kbL1Size = self->ctx.curStepKb_ * self->ctx.tiling_->baseK;
    }

    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        curCoutIdx = kIdx * self->ctx.tiling_->baseK / self->ctx.splitHkWkList_[self->ctx.splitIndex_];
    } else {
        curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    }
    curCoutSize = DivHkWk<Intf>(self, kbL1Size);

    uint32_t curCoutRemain = self->ctx.singleShapeCout_ - curCoutIdx;
    curCoutSize = curCoutSize < curCoutRemain ? curCoutSize : curCoutRemain;
    if (self->ctx.enableSplitK_) {
        curCoutIdx += self->ctx.curCoutStartIdx_;
    }
}

template <class Intf>
static __aicore__ inline void CalcCutInWIndex(Intf* self, const uint32_t crossBlockNum)
{
    uint32_t doubleBaseUseM = self->ctx.baseUseM_ << crossBlockNum;
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi,
                              self->ctx.tiling_->strideW); // 奇数场景当前tiling限制wi<512，对其之后不会溢出
    uint32_t mSize = self->ctx.curMIdx_ * (self->ctx.tiling_->baseM << crossBlockNum);
    uint32_t curWiPos = wiUsed - (mSize % wiUsed); // 上一轮baseM搬完后一行Wi还剩下未处理的长度
    if (curWiPos > doubleBaseUseM || curWiPos == wiUsed) {
        // 未处理的长度大于baseUseM 或 等于整行wi，即上一次搬运不涉及尾块
        // 则无需处理首块
        self->ctx.headWi_ = 0;
    } else {
        self->ctx.headWi_ = curWiPos;
    }
    uint32_t leftBaseUseM = doubleBaseUseM - self->ctx.headWi_;
    self->ctx.midHi_ = leftBaseUseM / wiUsed;
    self->ctx.tailWi_ = leftBaseUseM - self->ctx.midHi_ * wiUsed;
}

} // namespace Convolution3DBackpropFunc

#endif
