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
 * \file conv_bp_filter_sub_func_store_l0c_dispatch.h
 * \brief L0c to Gm dispatch and entry functions for conv3d backprop filter
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_STORE_L0C_DISPATCH_H
#define CONV3D_BP_FILTER_SUB_FUNC_STORE_L0C_DISPATCH_H

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatNCDHW(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                            LocalTensor<typename Intf::L0cT>& l0c,
                                                            bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if constexpr (Intf::conv3ddwConfig.groupEnlarge) {
            LoadL0c2UbForGroup(self, l0c);
        } else if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOne(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndivided(self, output, l0c);
        } else {
            LoadL0c2GmNormal(self, output, l0c);
        }
    } else {
        MovOutL0cForDeterministicRefactor(self, l0c, output);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatNDHWC(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                            LocalTensor<typename Intf::L0cT>& l0c,
                                                            bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOne(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndividedNz2Nd(self, output, l0c);
        } else {
            LoadL0c2GmNormalNz2Nd(self, output, l0c);
        }
    } else {
        MovOutL0cForDeterministicRefactor(self, l0c, output);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormatDHWCN(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                            LocalTensor<typename Intf::L0cT>& l0c,
                                                            bool enSequentialWrite = false)
{
    if (!enSequentialWrite) {
        if (self->ctx.dhwK_ == 1) {
            LoadL0c2GmDkhkwkEqOneNz2DHWCN(self, output, l0c);
        } else if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4:用来替换除法运算
            LoadL0c2GmBaseNUndividedNz2DHWCN(self, output, l0c);
        } else {
            LoadL0c2GmNormalNz2DHWCN(self, output, l0c);
        }
    } else {
        MovOutL0cForDeterministicRefactor(self, l0c, output);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GmForTransFormat(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                       LocalTensor<typename Intf::L0cT>& l0c,
                                                       bool enSequentialWrite = false)
{
    if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        LoadL0c2GmForTransFormatNCDHW(self, output, l0c, enSequentialWrite);
    } else if constexpr (Intf::Config::dType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        LoadL0c2GmForTransFormatNDHWC(self, output, l0c, enSequentialWrite);
    } else { // DHWCN
        LoadL0c2GmForTransFormatDHWCN(self, output, l0c, enSequentialWrite);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                         uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    LocalTensor<typename Intf::L0cT> l0c;
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template DeQue<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template DeQue<typename Intf::L0cT>();
    }

    if (enAtomic == 1) {
        SetAtomicAdd<typename Intf::DstT>();
    }

    LoadL0c2GmForTransFormat(self, output, l0c, enSequentialWrite);

    if (enAtomic == 1) {
        SetAtomicNone();
    }

    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.FreeTensor(l0c);
    } else {
        self->ctx.l0cPong_.FreeTensor(l0c);
    }
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        self->ctx.l0cPingPongFlag_ = !self->ctx.l0cPingPongFlag_;
    }
}

} // namespace ConvolutionBackpropFunc

#endif
