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
 * \file conv_bp_input_sub_func_store_l0c_dispatch.h
 * \brief L0c to Gm dispatch and entry functions for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_DISPATCH_H
#define CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_DISPATCH_H

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void FreeTensorC1Buf(Intf* self, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    if ASCEND_IS_AIC_SCALAR {
        // l0c不搬运数据，需释放l0c的空间，否则会等待卡死
        if (self->ctx.l0cPingPongFlag_) {
            self->ctx.l0cPing_.FreeTensor(useC1Buf);
        } else {
            self->ctx.l0cPong_.FreeTensor(useC1Buf);
        }
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPingPongFlag_ = !self->ctx.l0cPingPongFlag_;
        }
    }
}

template <class Intf>
static __aicore__ inline void L0CDeQue(Intf* self, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    if ASCEND_IS_AIV_SHOULD_RETURN {
        return;
    }

    if (self->ctx.l0cPingPongFlag_) {
        useC1Buf = self->ctx.l0cPing_.template DeQue<typename Intf::L0cT>();
    } else {
        useC1Buf = self->ctx.l0cPong_.template DeQue<typename Intf::L0cT>();
    }
}

template <class Intf>
static __aicore__ inline void SetEnAtomic(Intf* self, uint8_t enAtomic)
{
    if constexpr (std::is_same<typename Intf::DstT, float>::value) {
        if (enAtomic == 1) {
            SetAtomicAdd<typename Intf::DstT>();
        }
    } else if constexpr (
        std::is_same<typename Intf::DstT, bfloat16_t>::value || std::is_same<typename Intf::DstT, half>::value ||
        std::is_same<typename Intf::DstT, hifloat8_t>::value ||
        std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        if (self->ctx.enableSplitDk_) {
            // 满足此条件时当前Din不是第一次计算，开启累加
            enAtomic = (self->ctx.curDkIdx_ > 0) &&
                       (self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - self->ctx.tiling_->dout + 1 !=
                        self->ctx.curDkIdx_);
            if (enAtomic == 1) {
                SetAtomicAdd<float>();
            }
        }
        if (self->ctx.enableSplitK_ && self->ctx.useUbAccumForSplitK_) {
            if (enAtomic == 1) {
                SetAtomicAdd<float>();
            }
        }
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2OutForNz2Dn(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        LoadL0c2OutForKernelSplitHW<Intf>(self, useC1Buf);
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        LoadL0c2GmDnForKernelSplitH<Intf>(self, output, useC1Buf);
    } else {
        LoadL0c2GmForNz2Dn<Intf>(self, output, useC1Buf);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2OutForNz2Nd(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        LoadL0c2GmNdForKernelSplitH<Intf>(self, output, useC1Buf);
    } else {
        LoadL0c2GmForNz2Nd<Intf>(self, output, useC1Buf);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(
    Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
        if ASCEND_IS_AIV_SHOULD_RETURN {
            return;
        }
    }

    // 只有当参与了mmad运算时，才进行L0C的出队和拷贝
    if (!self->ctx.needComputeFlag_) {
        return;
    }

    LocalTensor<typename Intf::L0cT> useC1Buf;
    L0CDeQue<Intf>(self, useC1Buf);
    SetEnAtomic<Intf>(self, enAtomic);
    if (self->ctx.tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE2_FIX));
        SetFlag<HardEvent::MTE2_FIX>(eventId);
        WaitFlag<HardEvent::MTE2_FIX>(eventId);
    }
    if constexpr (Intf::Config::dType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        LoadL0c2OutForNz2Dn<Intf>(self, output, useC1Buf);
    } else {
#if (__NPU_ARCH__ != 5102)
        LoadL0c2OutForNz2Nd<Intf>(self, output, useC1Buf);
#endif
    }
    if constexpr (
        std::is_same<typename Intf::DstT, float>::value || std::is_same<typename Intf::DstT, bfloat16_t>::value ||
        std::is_same<typename Intf::DstT, half>::value) {
        if (enAtomic == 1) {
            SetAtomicNone();
        }
    }
    FreeTensorC1Buf(self, useC1Buf);
}

} // namespace Convolution3DBackpropFunc

#endif
