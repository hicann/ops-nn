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
constexpr uint8_t FLAG_FIXP_ID = 5;
constexpr uint8_t FLAG_MTE3_ID = 6;
constexpr uint16_t MMAD_THRESHOLD = 2560;

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
    // W
    self->ctx.load3d_.l1W = self->ctx.tiling_->wi;
    // H
    self->ctx.load3d_.l1H = 1;
    self->ctx.load3d_.padList[0] = self->ctx.tiling_->padLeft;
    self->ctx.load3d_.padList[1] = self->ctx.tiling_->padRight;
    // padUp now is set in Load BL1
    self->ctx.load3d_.padList[3] = 255; // 255: set max pad_bottom for compatibility
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf* self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
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
