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
 * \file conv2dv2_apt.cpp
 * \brief
 */

#include "arch35/conv2d_v2.h"
#include "arch35/conv2d_v2_group.h"
#include "arch35/conv2d_v2_tilingkey.h"

using namespace AscendC;
using namespace Conv2DV2Key;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::NCHW;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_HWCN && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::HWCN;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NCHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCHW
constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
#elif defined(FORMAT_X) && FORMAT_X == FORMAT_NHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_FRACTAL_Z_C04 && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NHWC
constexpr ConvFormat fmapFormat = ConvFormat::NHWC;
constexpr ConvFormat weightFormat = ConvFormat::FRACTAL_Z_C04;
constexpr ConvFormat outputFormat = ConvFormat::NHWC;
#endif
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;

#if defined(FORMAT_FILTER) && (FORMAT_FILTER == FORMAT_FRACTAL_Z || FORMAT_FILTER == FORMAT_FRACTAL_Z_C04)
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_AIC_ONLY)
#else
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_MIX_AIC_1_2)
#endif

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode,
         int8_t InnerBatch, int8_t DisContinuous>
__global__ __aicore__ void conv2dv2(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR user = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)
    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>;  // only for compile
#endif
    using scaleType = ConvType<TPosition::GM, scaleFormat, uint64_t>;

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv2dBase<fmapType, weightType, outputType, biasType, scaleType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch, DisContinuous>> baseConv2d;
        baseConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
    } else {
        GroupConv2d<fmapType, weightType, outputType, biasType, scaleType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch, DisContinuous>> groupConv2d;
        groupConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
    }

#endif
}