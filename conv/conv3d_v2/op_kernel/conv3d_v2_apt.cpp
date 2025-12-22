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
 * \file conv3d_v2_apt.cpp
 * \brief
 */

#include "arch35/conv3d_v2.h"    
#include "arch35/conv3d_v2_group.h"  
#include "arch35/conv3d_v2_tilingkey.h"

using namespace AscendC;
using namespace Conv3DV2Key;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat fmapFormat = ConvFormat::NCDHW;
constexpr ConvFormat filterFormat = ConvFormat::NCDHW;
constexpr ConvFormat outputFormat = ConvFormat::NCDHW;
#endif

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NDHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_DHWCN && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NDHWC
constexpr ConvFormat fmapFormat = ConvFormat::NDHWC;
constexpr ConvFormat filterFormat = ConvFormat::DHWCN;
constexpr ConvFormat outputFormat = ConvFormat::NDHWC;
#endif

constexpr ConvFormat biasFormat = ConvFormat::ND;

template<int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder, int8_t GroupType>
__global__ __aicore__ void conv3dv2(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR user = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

#if defined(DTYPE_X) && defined(DTYPE_FILTER) && defined(DTYPE_Y)
    using fmapType = ConvType<TPosition::GM, fmapFormat, DTYPE_X>;
    using weightType = ConvType<TPosition::GM, filterFormat, DTYPE_FILTER>;
    using outputType = ConvType<TPosition::GM, outputFormat, DTYPE_Y>;
#if defined(DTYPE_BIAS)
    using biasType = ConvType<TPosition::GM, biasFormat, DTYPE_BIAS>;
#else
    using biasType = ConvType<TPosition::GM, biasFormat, half>;  // only for compile
#endif

    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv3dV2Base<fmapType, weightType, outputType, biasType, Conv3DV2Param<FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData);
    } else {
        GroupConv3dV2<fmapType, weightType, outputType, biasType, Conv3DV2Param<FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> groupConv3d;
        groupConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData);
    }

#endif
}
