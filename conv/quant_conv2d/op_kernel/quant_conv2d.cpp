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
 * \file quant_conv2d.cpp
 * \brief
 */

#include "../inc/macro.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && !defined(KERNELUT)
#include "../conv2d_v2/conv2d_v2_common_tilingdata.h"
using Conv2DTilingData = Ops::NN::Conv2dV2::Conv2DTilingData;
#endif
#include "../conv2d_v2/arch35/conv2d_v2.h"
#include "../conv2d_v2/arch35/conv2d_v2_group.h"
#include "../conv2d_v2/arch35/conv2d_v2_tilingkey.h"

using namespace AscendC;

constexpr ConvFormat fmapFormat = ConvFormat::NCHW;
constexpr ConvFormat weightFormat = ConvFormat::NCHW;
constexpr ConvFormat outputFormat = ConvFormat::NCHW;
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;

template <int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
          int8_t GroupType, int8_t EnableSmallChannel, int8_t WeightUbTrans, int8_t FmapCopyMode, int8_t InnerBatch,
          int8_t BatchOne, int8_t NoPad, int8_t SmallWeight, int8_t SmallKernel, int8_t IsASWT, int8_t ConvToL1 = 0>
__global__ __aicore__ void quant_conv2d(GM_ADDR x, GM_ADDR filter, GM_ADDR scale, GM_ADDR bias, GM_ADDR offset,
                                        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) && !defined(KERNELUT)
    REGISTER_TILING_DEFAULT(Ops::NN::Conv2dV2::Conv2DTilingData);
#endif
    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)

    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>; // only for compile
#endif
    using scaleType = ConvType<TPosition::GM, scaleFormat, uint64_t>;

    ExtendParams extendParams(scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv2dBase<
            fmapType, weightType, outputType, biasType, scaleType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, GroupType, EnableSmallChannel,
                          WeightUbTrans, FmapCopyMode, InnerBatch, BatchOne, NoPad, SmallWeight>>
            baseConv2d;
        baseConv2d.RunConv2dKernel(x, filter, bias, y, tilingData, &extendParams);
    } else {
        GroupConv2d<
            fmapType, weightType, outputType, biasType, scaleType,
            Conv2DV1Param<FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, GroupType, EnableSmallChannel,
                          WeightUbTrans, FmapCopyMode, InnerBatch, BatchOne, NoPad, SmallWeight>>
            groupConv2d;
        groupConv2d.RunConv2dKernel(x, filter, bias, y, tilingData, &extendParams);
    }

#endif
}
