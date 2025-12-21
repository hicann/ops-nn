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
 * \file conv3d_bp_config_base.h
 * \brief
 */

#ifndef CONV3D_BP_CONFIG_ADVANCE_H
#define CONV3D_BP_CONFIG_ADVANCE_H

#include "conv3d_bp_util.h"
#include "../../conv3d_backprop_input_v2_arch35_tiling_key.h"

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
    using Type = float;
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
    constexpr static TPosition pos = POSITION;    // Convolution输入或输出时的scope
    constexpr static CubeFormat format = FORMAT;  // Convolution输入或者输出的format
    using Type = T;                               // Convolution输入或输出的数据类型
};

struct Conv3dConfig {
    uint8_t loadB2Condition = TPL_TRANSPOSE_AND_REVERSE;
    uint8_t kernelSplitMode = TPL_NO_SPLIT_KERNEL;
    uint8_t groupMode = TPL_GROUP_MODE_ORIGIN;
    uint8_t loadB1Condition = TPL_GM_TO_L1;
    bool enableC04Flag = false;
};

// 该函数提供默认的模板参数，完成最基本的算子功能场景，尽量泛化
__aicore__ constexpr Conv3dConfig GetDefaultConfig()
{
    return {
        .loadB2Condition = TPL_TRANSPOSE_AND_REVERSE,
        .kernelSplitMode = TPL_NO_SPLIT_KERNEL,
        .groupMode = TPL_GROUP_MODE_ORIGIN,
        .loadB1Condition = TPL_GM_TO_L1,
        .enableC04Flag = false,
    };
}

constexpr Conv3dConfig CONV3D_CFG_DEFAULT = GetDefaultConfig();

// 打包字段，内部实现的上下文，包含了用户构造的ConvBpParam
template <class A, class B, class C, class D, class E>
struct ConvBpContext {
    using xType = A;
    using cType = C;
    using dType = D;
    using eType = E;
    using SrcT = typename A::Type;
    using SrcAT = typename A::Type;
    using SrcBT = typename C::Type;
    using DstT = typename D::Type;
    using BiasT = typename E::Type;
    using L0cT = typename GetDstType<SrcT>::Type;

    constexpr static auto formatA = A::format;
    constexpr static auto formatB = B::format;
    constexpr static auto formatC = C::format;
    constexpr static auto formatD = D::format;
    constexpr static auto formatE = E::format;

    constexpr static auto posA = A::pos;
    constexpr static auto posB = B::pos;
    constexpr static auto posC = C::pos;
    constexpr static auto posD = D::pos;
    constexpr static auto posE = E::pos;

    using ContextData = struct _ {
        __aicore__ inline _() {}
    };
};
}  // namespace Convolution3DBackprop
#endif
