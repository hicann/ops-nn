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
 * \file conv_bp_input_sub_func_load_l1_to_l0_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_L1_TO_L0_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_L1_TO_L0_ARCH51_H

#include "conv_bp_input_sub_func_utils_arch51.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void InitLoadToB2Params(Intf* self)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.mStep = 1;
    self->ctx.load2dv2_.dstStride = 1;
    uint8_t cond = CalcLoadB2Condition<Intf>(self);
    self->ctx.load2dv2_.ifTranspose = (cond == B2_TRANSPOSE_AND_REVERSE || cond == B2_TRANSPOSE_ONLY) ? 1 : 0;
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsN(Intf* self)
{
    self->ctx.blockBaseN_ = DivCeil16(self->ctx.baseUseN_);
    uint32_t blockSize = self->ctx.tiling_->c0 << 4;
    uint8_t cond = CalcLoadB2Condition<Intf>(self);
    if (cond != B2_REVERSE_ONLY) {
        self->ctx.load2dv2_.kStep = self->ctx.blockBaseN_;
        self->ctx.baseB1Offset_ = 0;
        self->ctx.dstB2Stride_ = self->ctx.blockBaseN_ * blockSize;
    } else {
        self->ctx.load2dv2_.srcStride = -static_cast<int32_t>(self->ctx.blockBaseN_); // 逆序
        self->ctx.dstB2Stride_ = 0;
        self->ctx.load2dv2_.mStep = self->ctx.blockBaseN_;
        self->ctx.load2dv2_.dstStride = self->ctx.blockBaseN_;
    }
}

template <class Intf>
static __aicore__ inline void UpdateLoadToB2ParamsK(Intf* self)
{
    uint8_t cond = CalcLoadB2Condition<Intf>(self);
    if (cond != B2_REVERSE_ONLY) {
        self->ctx.load2dv2_.srcStride = self->ctx.curLoadKbl1_ >> self->ctx.tiling_->c0BitsB;
    }
}

template <class Intf>
static __aicore__ inline void LoadToB2ReverseOnlyCommon(Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix,
                                                        uint32_t blockSize, uint32_t srcB1Offset, uint32_t dstB2Offset,
                                                        const uint32_t kPos, LocalTensor<typename Intf::SrcBT>& l0b)
{
    uint32_t curHkWkSize = self->ctx.singleShapeHWk_;
    uint32_t kStartPos = kPos * (self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB);
    uint32_t kEndPos = kStartPos + ((self->ctx.baseUseK_ + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB);
    uint32_t hwKStartIdx = DivHkWk<Intf>(self, kStartPos);
    uint32_t hwKEndIdx = DivCeilHkWk<Intf>(self, kEndPos);
    uint32_t curKRepeat = hwKEndIdx - hwKStartIdx;
    for (uint32_t i = 0; i < curKRepeat; i++) {
        uint32_t curHWkStart = (hwKStartIdx + i) * curHkWkSize;
        uint32_t curHWkEnd = curHWkStart + curHkWkSize;
        uint32_t kStepStart = curHWkStart < kStartPos ? kStartPos : curHWkStart;
        uint32_t kStepEnd = curHWkEnd > kEndPos ? kEndPos : curHWkEnd;
        self->ctx.load2dv2_.kStep = kStepEnd - kStepStart;
        self->ctx.load2dv2_.kStartPosition = curHWkStart;
        self->ctx.load2dv2_.mStartPosition = (curHWkEnd - 1 - kStepStart) * self->ctx.blockBaseN_;
        LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
        dstB2Offset += (kStepEnd - kStepStart) * self->ctx.blockBaseN_ * blockSize;
    }
}

template <class Intf>
static __aicore__ inline void LoadToB2NoTransposeNoReverse(Intf* self,
                                                           const LocalTensor<typename Intf::SrcBT>& l1B1Matrix,
                                                           uint32_t srcB1Offset, uint32_t kPos,
                                                           LocalTensor<typename Intf::SrcBT>& l0b)
{
    self->ctx.load2dv2_.mStartPosition = 0;
    self->ctx.load2dv2_.kStartPosition = 0;
    self->ctx.load2dv2_.kStep = 1;
    self->ctx.load2dv2_.mStep = self->ctx.blockBaseN_ * DivCeilC0<Intf>(self, self->ctx.baseUseK_);
    srcB1Offset = kPos * AlignUp16(self->ctx.baseUseN_) * self->ctx.tiling_->baseK;
    LoadData(l0b, l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
}

template <class Intf>
static __aicore__ inline void LoadToB2(Intf* self, const LocalTensor<typename Intf::SrcBT>& l1B1Matrix, uint32_t kPos,
                                       LocalTensor<typename Intf::SrcBT>& l0b)
{
    uint32_t blockSize = self->ctx.tiling_->c0 << 4;
    uint32_t srcB1Offset = 0;
    uint32_t dstB2Offset = 0;
    uint8_t cond = CalcLoadB2Condition<Intf>(self);
    if (unlikely(cond == B2_TRANSPOSE_ONLY)) {
        uint32_t kBlockC0Num = kPos * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB;
        uint32_t curKRepeat = self->ctx.baseUseK_ >> self->ctx.tiling_->c0BitsB;
        for (uint32_t i = 0; i < curKRepeat; i++) {
            uint32_t idxC1out = kBlockC0Num + i;
            dstB2Offset = i * self->ctx.blockBaseN_ * blockSize;
            srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
        }
    } else if (unlikely(cond == B2_REVERSE_ONLY)) {
        if (self->ctx.tiling_->dkHkWk == 1) {
            LoadToB2NoTransposeNoReverse<Intf>(self, l1B1Matrix, srcB1Offset, kPos, l0b);
        } else {
            LoadToB2ReverseOnlyCommon<Intf>(self, l1B1Matrix, blockSize, srcB1Offset, dstB2Offset, kPos, l0b);
        }
    } else {
        // TRANSPOSE_AND_REVERSE
        uint32_t kBlockC0Num = kPos * self->ctx.tiling_->baseK >> self->ctx.tiling_->c0BitsB;
        uint32_t curL1Cout = DivHkWk<Intf>(self, self->ctx.curLoadKbl1_) << self->ctx.tiling_->c0BitsB;
        uint32_t curKRepeat = self->ctx.baseUseK_ >> self->ctx.tiling_->c0BitsB;
        for (uint32_t i = 0; i < curKRepeat; i++) {
            uint32_t iRev = kBlockC0Num + i;
            uint32_t idxC1out = DivHkWk<Intf>(self, iRev);
            srcB1Offset = self->ctx.baseB1Offset_ + idxC1out * blockSize +
                          (self->ctx.singleShapeHWk_ - 1 + idxC1out * self->ctx.singleShapeHWk_ - iRev) * curL1Cout;
            LoadData(l0b[dstB2Offset], l1B1Matrix[srcB1Offset], self->ctx.load2dv2_);
            dstB2Offset += self->ctx.dstB2Stride_;
        }
    }
}

} // namespace Convolution3DBackpropFunc

#endif // CONV3D_BP_INPUT_SUB_FUNC_LOAD_L1_TO_L0_ARCH51_H
