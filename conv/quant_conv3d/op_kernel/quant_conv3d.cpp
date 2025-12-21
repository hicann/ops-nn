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
 * \file quant_conv3d.cpp
 * \brief
 */

#include "../conv3d_v2_apt/arch35/conv3dv2.h"
#include "../conv3d_v2_apt/arch35/group_conv3dv2.h"
#include "../conv3d_v2_apt/arch35/conv3dv2_tilingkey.h"

using namespace AscendC;

constexpr ConvFormat fmapFormat = ConvFormat::NCDHW;
constexpr ConvFormat weightFormat = ConvFormat::NCDHW;
constexpr ConvFormat outputFormat = ConvFormat::NCDHW;
constexpr ConvFormat biasFormat = ConvFormat::ND;

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType>
__global__ __aicore__ void quant_conv3d(GM_ADDR x, GM_ADDR filter, GM_ADDR scale, GM_ADDR bias, GM_ADDR offset,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)

    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>;  // only for compile
#endif

    ExtendParams extendParams(scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv3dV2Base<fmapType, weightType, outputType, biasType, Conv3DV2Param<FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    } else {
        GroupConv3dV2<fmapType, weightType, outputType, biasType, Conv3DV2Param<FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> groupConv3d;
        groupConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    }

#endif

    KERNEL_TASK_TYPE(2058, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2106, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2314, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2362, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2570, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2618, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2826, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2874, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2122, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2170, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2378, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2426, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2634, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2682, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2890, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2938, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2186, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2234, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2442, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2490, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2698, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2746, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2954, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3002, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2250, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2298, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2506, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2554, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2762, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2810, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3018, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3066, KERNEL_TYPE_MIX_AIC_1_2);
}