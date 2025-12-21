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
 * \file extend_conv2d.cpp
 * \brief
 */

#include "../conv2d_v2/arch35/conv2d_v2.h"
#include "../conv2d_v2/arch35/conv2d_v2_group.h"
#include "../conv2d_v2/arch35/conv2d_v2_tilingkey.h"

using namespace AscendC;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCHW && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::NCHW;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_HWCN && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::HWCN;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && \
    defined(FORMAT_Y0) && FORMAT_Y0 == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#endif
constexpr ConvFormat biasFormat = ConvFormat::ND;

#if defined(FORMAT_FILTER) && (FORMAT_FILTER == FORMAT_FRACTAL_Z || FORMAT_FILTER == FORMAT_FRACTAL_Z_C04)
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_AIC_ONLY)
#else
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_MIX_AIC_1_2)
#endif

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode,
         int8_t InnerBatch>
__global__ __aicore__ void extend_conv2d(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w,
    GM_ADDR scale0, GM_ADDR relu_weight0, GM_ADDR clip_value0, GM_ADDR scale1, GM_ADDR relu_weight1,
    GM_ADDR clip_value1, GM_ADDR y0, GM_ADDR y1, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y0)

    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y0>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>;  // only for compile
#endif
#if defined(DTYPE_Y1)
    using output1Type = DTYPE_Y1;
#else
    using output1Type = half;  // only for compile
#endif

    ExtendParams extendParams(scale0, relu_weight0, clip_value0, scale1, relu_weight1, clip_value1, y1);

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv2dBase<fmapType, weightType, outputType, biasType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch,
                          IsExtendConv2D::TRUE, output1Type>> baseConv2d;
        baseConv2d.RunConv2dKernel(x, filter, bias, y0, tilingData, &extendParams);
    } else {
        GroupConv2d<fmapType, weightType, outputType, biasType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch,
                          IsExtendConv2D::TRUE, output1Type>> groupConv2d;
        groupConv2d.RunConv2dKernel(x, filter, bias, y0, tilingData, &extendParams);
    }

#endif

    SET_KERNEL_TASK_TYPE(2058);
    SET_KERNEL_TASK_TYPE(2106);
    SET_KERNEL_TASK_TYPE(2314);
    SET_KERNEL_TASK_TYPE(2362);
    SET_KERNEL_TASK_TYPE(2570);
    SET_KERNEL_TASK_TYPE(2618);
    SET_KERNEL_TASK_TYPE(2826);
    SET_KERNEL_TASK_TYPE(2874);
    SET_KERNEL_TASK_TYPE(2122);
    SET_KERNEL_TASK_TYPE(2170);
    SET_KERNEL_TASK_TYPE(2378);
    SET_KERNEL_TASK_TYPE(2426);
    SET_KERNEL_TASK_TYPE(2634);
    SET_KERNEL_TASK_TYPE(2682);
    SET_KERNEL_TASK_TYPE(2890);
    SET_KERNEL_TASK_TYPE(2938);
    SET_KERNEL_TASK_TYPE(2186);
    SET_KERNEL_TASK_TYPE(2234);
    SET_KERNEL_TASK_TYPE(2442);
    SET_KERNEL_TASK_TYPE(2490);
    SET_KERNEL_TASK_TYPE(2698);
    SET_KERNEL_TASK_TYPE(2746);
    SET_KERNEL_TASK_TYPE(2954);
    SET_KERNEL_TASK_TYPE(3002);
    SET_KERNEL_TASK_TYPE(2250);
    SET_KERNEL_TASK_TYPE(2298);
    SET_KERNEL_TASK_TYPE(2506);
    SET_KERNEL_TASK_TYPE(2554);
    SET_KERNEL_TASK_TYPE(2762);
    SET_KERNEL_TASK_TYPE(2810);
    SET_KERNEL_TASK_TYPE(3018);
    SET_KERNEL_TASK_TYPE(3066);
    SET_KERNEL_TASK_TYPE(16384);
    SET_KERNEL_TASK_TYPE(16448);
    SET_KERNEL_TASK_TYPE(16450);
    SET_KERNEL_TASK_TYPE(16458);
    SET_KERNEL_TASK_TYPE(16576);
    SET_KERNEL_TASK_TYPE(16578);
    SET_KERNEL_TASK_TYPE(16586);
    SET_KERNEL_TASK_TYPE(17024);
    SET_KERNEL_TASK_TYPE(17032);
    SET_KERNEL_TASK_TYPE(17034);
    SET_KERNEL_TASK_TYPE(17088);
    SET_KERNEL_TASK_TYPE(17096);
    SET_KERNEL_TASK_TYPE(17098);
}