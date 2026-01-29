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
 * \file conv3d_bp_large_attribute_func.h
 * \brief
 */

#ifndef CONV3D_BP_LARGE_ATTRIBUTE_FUNC_ADVANCE_H
#define CONV3D_BP_LARGE_ATTRIBUTE_FUNC_ADVANCE_H

#include "conv3d_bp_common_func.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline bool ComputeForWkLoop(Intf *self, uint32_t kwDilation,
    uint32_t woExpand, int32_t woStartIdx, int64_t curWkIdx)
{
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK) {
        return true;
    }

    self->ctx.load3d_.mStartPt = 0; // 切hwk时，会精确到加载wo的第一个点，load3d无需跳过
    self->ctx.curWkIdx_ = curWkIdx;
    self->ctx.curWoLeftIdx_ = kwDilation - curWkIdx * self->ctx.tiling_->dilationW -
        1 - self->ctx.tiling_->backpropPadLeft + woStartIdx; // 计算当前wk 对应wo开始位置
    uint32_t skipLeftWoSize = self->ctx.curWoLeftIdx_ > 0 ? self->ctx.curWoLeftIdx_ : 0;
    int32_t woRightIdx = self->ctx.curWoLeftIdx_ + self->ctx.baseUseM_;  // 计算当前wk 对应wo结束位置
    if (self->ctx.curWoLeftIdx_ >= static_cast<int32_t>(woExpand) || woRightIdx <= 0) {
        return false;
    }
    self->ctx.curWoRightIdx_ = woRightIdx < woExpand ? woRightIdx - woExpand :
        self->ctx.baseUseM_ - woExpand + skipLeftWoSize; // 计算准确的反向padright大小
    uint32_t skipRightWoSize = self->ctx.curWoRightIdx_ < 0 ? abs(self->ctx.curWoRightIdx_) : 0;
    self->ctx.realWoSize_ = self->ctx.tiling_->wo -  DivCeil(skipLeftWoSize, self->ctx.tiling_->strideW) -
        DivCeil(skipRightWoSize, self->ctx.tiling_->strideW);
    if (self->ctx.realWoSize_ <= 0){
        return false;
    }
    return true;
}

template <class Intf>
static __aicore__ inline void ComputeForTilingHkWk(Intf *self, LocalTensor<typename Intf::SrcT> &l0a,
    LocalTensor<typename Intf::SrcT> &l0b, LocalTensor<typename Intf::L0cT> &l0c, uint8_t &l0PingPongFlag)
{
    bool isFirstDHWk = true;
    int32_t curHoIdx = self->ctx.curHoIdx_;
    uint32_t khDilation = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
    uint32_t kwDilation = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
    uint32_t woExpand = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1;
    int32_t woStartIdx = (self->ctx.curMStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM) % self->ctx.tiling_->wi; // M循环的起始Wi位置
    uint32_t curMStartPt = self->ctx.load3d_.mStartPt;
    for (uint64_t curInnerKdIdx = self->ctx.curDkIdx_; curInnerKdIdx < self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk; curInnerKdIdx++) {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            self->ctx.groupIterIdx_ = 0;
        }
        int64_t curDoutIdx = 0;
        if (!CalcCurDoutIdx<Intf>(self, curInnerKdIdx, curDoutIdx)) {
            continue;
        }

        self->ctx.curHoIdx_ += khDilation - 1; //计算，每一行hk对应的curHoIdx
        for (uint64_t curHkIdx = 0; curHkIdx < self->ctx.tiling_->hk; curHkIdx++) { // dilation时跳过空洞部分
            self->ctx.curHkIdx_ = curHkIdx;
            int32_t endHoIdx = self->ctx.curHoIdx_ + DivCeil(self->ctx.baseUseM_ + curMStartPt, self->ctx.tiling_->wi);
            if (self->ctx.curHoIdx_ < static_cast<int32_t>(self->ctx.hoExpand_) && endHoIdx > 0) {
                uint32_t realStartHo = self->ctx.curHoIdx_ < 0 ? 0 : self->ctx.curHoIdx_;
                uint32_t realEndHoIdx = endHoIdx > self->ctx.hoExpand_ ? self->ctx.hoExpand_ : endHoIdx;
                self->ctx.curHoSize_ = realEndHoIdx - realStartHo;
                if (realStartHo % self->ctx.tiling_->strideH != 0 && (realStartHo % self->ctx.tiling_->strideH + self->ctx.curHoSize_ - 1 < self->ctx.tiling_->strideH)) {
                    self->ctx.curHoIdx_ -= self->ctx.tiling_->dilationH; // 前放大补零部分不计算，需更新curHoIdx，且需要跳过空洞部分
                    continue;
                }
            } else {
                self->ctx.curHoIdx_ -= self->ctx.tiling_->dilationH; // 不计算时仍需更新curHoIdx，且需要跳过空洞部分
                continue;
            }

            int64_t wkIter = self->ctx.tiling_->wk;
            if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK) {
                wkIter = 1;
            }

            for (int64_t curWkIdx = 0; curWkIdx < wkIter; curWkIdx++) {
                if (!ComputeForWkLoop<Intf>(self, kwDilation, woExpand, woStartIdx, curWkIdx)) {
                    continue;
                }
                ComputeForKIter<Intf>(self, l0a, l0b, l0c, curInnerKdIdx, curDoutIdx, isFirstDHWk, l0PingPongFlag);
                isFirstDHWk = false;
            }
            self->ctx.curHoIdx_ -= self->ctx.tiling_->dilationH; //计算一行Hk后需更新curHoIdx，且需要跳过空洞部分
        }
        self->ctx.curHoIdx_ = curHoIdx;
    }
}

template <class Intf>
static __aicore__ inline void UpdateWkComputeStatus(Intf *self)
{
    if (!self->ctx.needComputeFlag_) {
        // 其他方向要跳过计算，K可以不判断了
        return;
    }

    uint32_t kwDilation = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
    uint32_t woExpand = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1;
    int32_t woStartIdx = (self->ctx.curMStartIdx_ + self->ctx.curML0Idx_ * self->ctx.tiling_->baseM) %
        self->ctx.tiling_->wi; // M循环的起始Wi位置
    bool isKNeedCompute = false;
    for (int64_t curWkIdx = 0; curWkIdx < self->ctx.tiling_->wk; curWkIdx++) {      
        int32_t curWoLeftIdx = kwDilation - curWkIdx * self->ctx.tiling_->dilationW -
            1 - self->ctx.tiling_->backpropPadLeft + woStartIdx; // 计算当前wk 对应wo开始位置
        int32_t woRightIdx = curWoLeftIdx + self->ctx.baseUseM_;  // 计算当前wk 对应wo结束位置
        if (curWoLeftIdx >= static_cast<int32_t>(woExpand) || woRightIdx <= 0) {
            continue;
        }
        if (curWoLeftIdx > 0 && curWoLeftIdx % self->ctx.tiling_->strideW != 0 &&
            DivCeil(curWoLeftIdx, self->ctx.tiling_->strideW) == DivCeil(woRightIdx, self->ctx.tiling_->strideW)) {
            continue;
        }
        isKNeedCompute = true;
        break;
    }

    self->ctx.needComputeFlag_ = isKNeedCompute;
}

template <class Intf>
static __aicore__ inline void UpdateHkComputeStatus(Intf *self)
{
    if (!self->ctx.needComputeFlag_) {
        // 其他方向要跳过计算，K可以不判断了
        return;
    }

    bool isKNeedCompute = false;
    uint32_t khDilation = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
    int32_t curHoIdx = self->ctx.curHoIdx_ + khDilation - 1;
    for (uint64_t curHkIdx = 0; curHkIdx < self->ctx.tiling_->hk; curHkIdx++) { // dilation时跳过空洞部分
        int32_t endHoIdx = curHoIdx + DivCeil(self->ctx.baseUseM_ + self->ctx.load3d_.mStartPt, self->ctx.tiling_->wi);
        if (curHoIdx >= static_cast<int32_t>(self->ctx.hoExpand_) || endHoIdx <= 0) {
            curHoIdx -= self->ctx.tiling_->dilationH; // 不计算时仍需更新curHoIdx，且需要跳过空洞部分
            continue;
        }
        if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            if (curHoIdx > 0 && curHoIdx % self->ctx.tiling_->strideH != 0) {
                curHoIdx -= self->ctx.tiling_->dilationH; // 不计算时仍需更新curHoIdx，且需要跳过空洞部分
                continue;
            }
        } else {
            if (curHoIdx > 0 && curHoIdx % self->ctx.tiling_->strideH != 0 &&
                DivCeil(curHoIdx, self->ctx.tiling_->strideH) == DivCeil(endHoIdx, self->ctx.tiling_->strideH)) {
                curHoIdx -= self->ctx.tiling_->dilationH; // 不计算时仍需更新curHoIdx，且需要跳过空洞部分
                continue;
            }
        }
        isKNeedCompute = true;
        break;
    }
    self->ctx.needComputeFlag_ = isKNeedCompute;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        UpdateWkComputeStatus<Intf>(self);
    }
}
}  // namespace Convolution3DBackpropFunc
#endif