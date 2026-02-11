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

#include "../conv3d_v2/arch35/conv3d_v2.h"   
#include "../conv3d_v2/arch35/conv3d_v2_group.h" 
#include "../conv3d_v2/arch35/conv3d_v2_tilingkey.h"
using namespace AscendC;

constexpr ConvFormat fmapFormat = ConvFormat::NCDHW;
constexpr ConvFormat weightFormat = ConvFormat::NCDHW;
constexpr ConvFormat outputFormat = ConvFormat::NCDHW;
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;

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
    REGISTER_TILING_DEFAULT(Ops::NN::Conv3dV2::Conv3DV2TilingData);
    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)

    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, weightFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, int32_t>;  // only for compile
#endif
    using scaleType = ConvType<TPosition::GM, scaleFormat, uint64_t>;

    ExtendParams extendParams(scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv3dV2Base<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    } else {
        GroupConv3dV2<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType>> groupConv3d;
        groupConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    }

#endif
}