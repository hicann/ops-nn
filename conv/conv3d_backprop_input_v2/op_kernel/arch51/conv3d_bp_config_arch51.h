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
 * \file conv3d_bp_config_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_CONFIG_ARCH51_H
#define CONV3D_BP_CONFIG_ARCH51_H

#include "conv3d_bp_util_arch51.h"
#include "../../../inc/macro.h"

using namespace AscendC;

namespace Convolution3DBackprop {

enum class CubeFormat : uint8_t {
    NDC1HWC0,
    NDHWC,
    NCDHW,
    DHWCN,
    FRACTALZ_3D,
    ND,
    UNSUPPORT,
    FRACTALZ,
};

enum class QuantMode : std::uint8_t {
    NO_QUANT = 0,
    SCALAR_QUANT, // perTensor量化模式，Scale shape为(1)
    VECTOR_QUANT, // perChannel量化模式，Scale shape和结果的channel一致
    UNDEFINED
};

template <typename T>
struct GetDstType {
    using Type = T;
};

template <>
struct GetDstType<float> {
    using Type = float;
};

template <>
struct GetDstType<half> {
    using Type = int32_t;
};

template <>
struct GetDstType<int8_t> {
    using Type = int32_t;
};

template <>
struct GetDstType<hifloat8_t> {
    using Type = float;
};

template <>
struct GetDstType<fp8_e4m3fn_t> {
    using Type = float;
};

template <>
struct GetDstType<bfloat16_t> {
    using Type = float;
};

// ConvType，定义卷积输入输出对象的属性
template <TPosition POSITION, CubeFormat FORMAT, typename T>
struct ConvType {
    constexpr static TPosition pos = POSITION;   // Convolution输入或输出时的scope
    constexpr static CubeFormat format = FORMAT; // Convolution输入或者输出的format
    using Type = T;                              // Convolution输入或输出的数据类型
};

// 打包字段，内部实现的上下文，包含了用户构造的ConvBpParam
template <class A, class B, class C, class D, class E, class F>
struct ConvBpContext {
    using xType = A;
    using cType = C;
    using dType = D;
    using eType = E;
    using fType = F;
    using SrcT = typename A::Type;
    using SrcAT = typename C::Type; // dedy Type
    using SrcBT = typename A::Type; // filter Type
    using DstT = typename D::Type;
    using BiasT = typename E::Type;
    using ScaleT0 = typename F::Type;
    using L0cT = typename GetDstType<SrcAT>::Type;

    constexpr static auto formatA = A::format;
    constexpr static auto formatB = B::format;
    constexpr static auto formatC = C::format;
    constexpr static auto formatD = D::format;
    constexpr static auto formatE = E::format;
    constexpr static auto formatF = F::format;

    constexpr static auto posA = A::pos;
    constexpr static auto posB = B::pos;
    constexpr static auto posC = C::pos;
    constexpr static auto posD = D::pos;
    constexpr static auto posE = E::pos;
    constexpr static auto posF = F::pos;

    using ContextData = struct _ {
        __aicore__ inline _() {}
    };
};

template <class A, class B, class C, class D, class E, class F>
struct Conv3DBpInputCfg : public ConvBpContext<A, B, C, D, E, F> {
public:
    __aicore__ inline Conv3DBpInputCfg() {}

    using ContextData = struct _ : public ConvBpContext<A, B, C, D, E, F>::ContextData {
        __aicore__ inline _() {}
    };
};

} // namespace Convolution3DBackprop
#endif
