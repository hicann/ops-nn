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
 * \file conv_bp_input_sub_func_store_l0c_fixpipe.h
 * \brief Fixpipe wrappers and quantization parameter helpers for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_FIXPIPE_H
#define CONV3D_BP_INPUT_SUB_FUNC_STORE_L0C_FIXPIPE_H

#include "conv_bp_input_sub_func_utils.h"

using AscendC::Fixpipe;
using AscendC::FixpipeParamsArch3510;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {

// 选当前输出对应的 scalar scale GM：第二路输出(y1)用 scale1Global_，第一路(y0)用 scale0Global_。
// scale1Global_ 仅在双输出编译(DTYPE_Y1)下存在，而 IsSecondOutput=true 的 Intf 也只在 DTYPE_Y1 下实例化，
// 故第二路分支由 if constexpr 编译期隔离，单输出编译不会引用到不存在的成员。
template <class Intf>
static __aicore__ inline auto& GetScaleGlobal(Intf* self)
{
    if constexpr (Intf::IsSecondOutput) {
#ifdef DTYPE_Y1
        return self->ctx.scale1Global_;
#else
        // 单输出编译不会实例化 IsSecondOutput=true 的 Intf，此分支不可达。
        return self->ctx.scale0Global_;
#endif
    } else {
        return self->ctx.scale0Global_;
    }
}

// 选当前输出对应的 vector scale L1 buffer：第二路用 scale1L1Buf_，第一路用 scale0L1Buf_。
template <class Intf>
static __aicore__ inline auto& GetScaleL1Buf(Intf* self)
{
    if constexpr (Intf::IsSecondOutput) {
#ifdef DTYPE_Y1
        return self->ctx.scale1L1Buf_;
#else
        return self->ctx.scale0L1Buf_;
#endif
    } else {
        return self->ctx.scale0L1Buf_;
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2GMFixPipe(Intf* self, const int64_t srcOffset, const int64_t dstOffset,
                                                const GlobalTensor<typename Intf::DstT>& output,
                                                const LocalTensor<typename Intf::L0cT>& useC1Buf,
                                                FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR>& fixPipeParams)
{
    if (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        Convolution3DBackprop::GetOutputQuantMode<Intf>(self) ==
            static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        uint64_t scaleAddr = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR>(
            output[dstOffset], useC1Buf[srcOffset], GetScaleL1Buf<Intf>(self)[scaleAddr], fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR>(output[dstOffset], useC1Buf[srcOffset],
                                                                            fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2UbFixPipe(Intf* self, const int64_t srcOffset, const int64_t dstOffset,
                                                const LocalTensor<typename Intf::DstT>& vecOutBuf,
                                                const LocalTensor<typename Intf::L0cT>& useC1Buf,
                                                FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR>& fixPipeParams)
{
    if (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        Convolution3DBackprop::GetOutputQuantMode<Intf>(self) ==
            static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        uint64_t scaleAddr = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_UB>(
            vecOutBuf[dstOffset], useC1Buf[srcOffset], GetScaleL1Buf<Intf>(self)[scaleAddr], fixPipeParams);
    } else {
        Fixpipe<typename Intf::DstT, typename Intf::L0cT, CFG_COLUMN_MAJOR_UB>(vecOutBuf[dstOffset],
                                                                               useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf, CO2Layout layout = CO2Layout::COLUMN_MAJOR>
static __aicore__ inline void SetQuantInt32ToHalf(Intf* self, FixpipeParamsArch3510<layout>& fixPipeParams)
{
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        const uint8_t quantMode = Convolution3DBackprop::GetOutputQuantMode<Intf>(self);
        if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            fixPipeParams.quantPre = QuantMode_t::VDEQF16; // int32 -> fp16 tensor quant
        } else if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT)) {
            fixPipeParams.quantPre = QuantMode_t::DEQF16; // int32 -> fp16 scalar quant
            fixPipeParams.deqScalar = GetScaleGlobal<Intf>(self).GetValue(0);
        } else {
            // The formal scale slot can retain a supported compile-time format even when it is not connected.
            // NO_QUANT must not read that absent GM input; fixed-point fp16 writeback uses the unit coefficient.
            fixPipeParams.quantPre = QuantMode_t::DEQF16;
            fixPipeParams.deqScalar = DQ_SCALAR_QF_ONE;
        }
    } else {
        fixPipeParams.quantPre = QuantMode_t::DEQF16; // int32 -> fp16 scalar quant
        fixPipeParams.deqScalar = DQ_SCALAR_QF_ONE;
    }
}

template <class Intf, CO2Layout layout = CO2Layout::COLUMN_MAJOR>
static __aicore__ inline void SetQuantInt8(Intf* self, FixpipeParamsArch3510<layout>& fixPipeParams)
{
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        const uint8_t quantMode = Convolution3DBackprop::GetOutputQuantMode<Intf>(self);
        if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            fixPipeParams.quantPre = QuantMode_t::VREQ8;
        } else if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT)) {
            fixPipeParams.quantPre = QuantMode_t::REQ8;
            fixPipeParams.deqScalar = GetScaleGlobal<Intf>(self).GetValue(0);
        } else {
            fixPipeParams.quantPre = QuantMode_t::REQ8;
            fixPipeParams.deqScalar = DQ_SCALAR_QF_ONE;
        }
    } else {
        fixPipeParams.quantPre = QuantMode_t::REQ8;
        fixPipeParams.deqScalar = DQ_SCALAR_QF_ONE;
    }
}

template <class Intf, CO2Layout layout = CO2Layout::COLUMN_MAJOR>
static __aicore__ inline void SetFixPipeQuantVal(Intf* self, FixpipeParamsArch3510<layout>& fixPipeParams)
{
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    if constexpr (std::is_same<typename Intf::SrcAT, half>::value && std::is_same<typename Intf::SrcBT, half>::value) {
        fixPipeParams.fixShiftVal = SHIFT_VALUE_LEN - static_cast<uint8_t>(self->ctx.tiling_->fixedShiftVal);
    }
#endif
    if constexpr (std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322BF16;
    } else if constexpr ((std::is_same<typename Intf::L0cT, int32_t>::value) &&
                         (std::is_same<typename Intf::DstT, half>::value)) {
        SetQuantInt32ToHalf<Intf, layout>(self, fixPipeParams);
    } else if constexpr (std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322F16;
    } else if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr (std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322FP8_PRE;
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr (std::is_same<typename Intf::DstT, int8_t>::value) {
        SetQuantInt8<Intf, layout>(self, fixPipeParams);
    }
}

} // namespace Convolution3DBackpropFunc

#endif
