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

#if defined(FORMAT_FILTER) && (FORMAT_FILTER == FORMAT_FRACTAL_Z || FORMAT_FILTER == FORMAT_FRACTAL_Z_C04)
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_AIC_ONLY)
#else
#define SET_KERNEL_TASK_TYPE(key) KERNEL_TASK_TYPE(key, KERNEL_TYPE_MIX_AIC_1_2)
#endif

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode,
         int8_t InnerBatch>
__global__ __aicore__ void conv2dv2(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR user = GetUserWorkspace(workspace);
 
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

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv2dBase<fmapType, weightType, outputType, biasType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch>> baseConv2d;
        baseConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
    } else {
        GroupConv2d<fmapType, weightType, outputType, biasType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType,
                          EnableSmallChannel, WeightUbTrans, FmapCopyMode, InnerBatch>> groupConv2d;
        groupConv2d.RunConv2dKernel(x, filter, bias, y, tilingData);
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
    SET_KERNEL_TASK_TYPE(4096);
    SET_KERNEL_TASK_TYPE(4352);
    SET_KERNEL_TASK_TYPE(4160);
    SET_KERNEL_TASK_TYPE(4162);
    SET_KERNEL_TASK_TYPE(4178);
    SET_KERNEL_TASK_TYPE(4416);
    SET_KERNEL_TASK_TYPE(4418);
    SET_KERNEL_TASK_TYPE(4434);
    SET_KERNEL_TASK_TYPE(4170);
    SET_KERNEL_TASK_TYPE(4186);
    SET_KERNEL_TASK_TYPE(4202);
    SET_KERNEL_TASK_TYPE(4218);
    SET_KERNEL_TASK_TYPE(4426);
    SET_KERNEL_TASK_TYPE(4442);
    SET_KERNEL_TASK_TYPE(4458);
    SET_KERNEL_TASK_TYPE(4474);
    SET_KERNEL_TASK_TYPE(4736);
    SET_KERNEL_TASK_TYPE(4992);
    SET_KERNEL_TASK_TYPE(4744);
    SET_KERNEL_TASK_TYPE(4776);
    SET_KERNEL_TASK_TYPE(4746);
    SET_KERNEL_TASK_TYPE(4762);
    SET_KERNEL_TASK_TYPE(4778);
    SET_KERNEL_TASK_TYPE(4794);
    SET_KERNEL_TASK_TYPE(5000);
    SET_KERNEL_TASK_TYPE(5032);
    SET_KERNEL_TASK_TYPE(5002);
    SET_KERNEL_TASK_TYPE(5018);
    SET_KERNEL_TASK_TYPE(5034);
    SET_KERNEL_TASK_TYPE(5050);
    SET_KERNEL_TASK_TYPE(4288);
    SET_KERNEL_TASK_TYPE(4290);
    SET_KERNEL_TASK_TYPE(4306);
    SET_KERNEL_TASK_TYPE(4544);
    SET_KERNEL_TASK_TYPE(4546);
    SET_KERNEL_TASK_TYPE(4562);
    SET_KERNEL_TASK_TYPE(4549);
    SET_KERNEL_TASK_TYPE(4565);
    SET_KERNEL_TASK_TYPE(4581);
    SET_KERNEL_TASK_TYPE(4597);
    SET_KERNEL_TASK_TYPE(4298);
    SET_KERNEL_TASK_TYPE(4314);
    SET_KERNEL_TASK_TYPE(4330);
    SET_KERNEL_TASK_TYPE(4346);
    SET_KERNEL_TASK_TYPE(4554);
    SET_KERNEL_TASK_TYPE(4570);
    SET_KERNEL_TASK_TYPE(4586);
    SET_KERNEL_TASK_TYPE(4602);
    SET_KERNEL_TASK_TYPE(4800);
    SET_KERNEL_TASK_TYPE(5056);
    SET_KERNEL_TASK_TYPE(5061);
    SET_KERNEL_TASK_TYPE(5077);
    SET_KERNEL_TASK_TYPE(5093);
    SET_KERNEL_TASK_TYPE(5109);
    SET_KERNEL_TASK_TYPE(4808);
    SET_KERNEL_TASK_TYPE(4840);
    SET_KERNEL_TASK_TYPE(4810);
    SET_KERNEL_TASK_TYPE(4826);
    SET_KERNEL_TASK_TYPE(4842);
    SET_KERNEL_TASK_TYPE(4858);
    SET_KERNEL_TASK_TYPE(5064);
    SET_KERNEL_TASK_TYPE(5096);
    SET_KERNEL_TASK_TYPE(5066);
    SET_KERNEL_TASK_TYPE(5082);
    SET_KERNEL_TASK_TYPE(5098);
    SET_KERNEL_TASK_TYPE(5114);
    SET_KERNEL_TASK_TYPE(8298);
    SET_KERNEL_TASK_TYPE(8314);
    SET_KERNEL_TASK_TYPE(8554);
    SET_KERNEL_TASK_TYPE(8570);
    SET_KERNEL_TASK_TYPE(8872);
    SET_KERNEL_TASK_TYPE(8874);
    SET_KERNEL_TASK_TYPE(8890);
    SET_KERNEL_TASK_TYPE(9128);
    SET_KERNEL_TASK_TYPE(9130);
    SET_KERNEL_TASK_TYPE(9146);
    SET_KERNEL_TASK_TYPE(8677);
    SET_KERNEL_TASK_TYPE(8693);
    SET_KERNEL_TASK_TYPE(8426);
    SET_KERNEL_TASK_TYPE(8442);
    SET_KERNEL_TASK_TYPE(8682);
    SET_KERNEL_TASK_TYPE(8698);
    SET_KERNEL_TASK_TYPE(9189);
    SET_KERNEL_TASK_TYPE(9205);
    SET_KERNEL_TASK_TYPE(8936);
    SET_KERNEL_TASK_TYPE(8938);
    SET_KERNEL_TASK_TYPE(8954);
    SET_KERNEL_TASK_TYPE(9192);
    SET_KERNEL_TASK_TYPE(9194);
    SET_KERNEL_TASK_TYPE(9210);
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
    SET_KERNEL_TASK_TYPE(139626);
    SET_KERNEL_TASK_TYPE(139642);
    SET_KERNEL_TASK_TYPE(140200);
    SET_KERNEL_TASK_TYPE(140202);
    SET_KERNEL_TASK_TYPE(140218);
    SET_KERNEL_TASK_TYPE(139749);
    SET_KERNEL_TASK_TYPE(139765);
    SET_KERNEL_TASK_TYPE(139754);
    SET_KERNEL_TASK_TYPE(139770);
    SET_KERNEL_TASK_TYPE(140261);
    SET_KERNEL_TASK_TYPE(140277);
    SET_KERNEL_TASK_TYPE(140264);
    SET_KERNEL_TASK_TYPE(140266);
    SET_KERNEL_TASK_TYPE(140282);
}