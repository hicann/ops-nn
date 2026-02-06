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
 * \file conv3d_v2.cpp
 * \brief
 */

#if (__CCE_AICORE__ > 300)
#include "arch35/conv3d_v2.h"
#include "arch35/conv3d_v2_group.h"
#include "arch35/conv3d_v2_tilingkey.h"
#else
#include "conv3dv2.h"
#include "lib/matmul_intf.h"
#include "conv3dv2_tiling_key.h"
#include "conv3d_v2_kernel_dispatch.h"
#endif

using namespace AscendC;
#if (__CCE_AICORE__ > 300)
using namespace Conv3DV2Key;
#endif

#if (__CCE_AICORE__ > 300)

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW
constexpr ConvFormat fmapFormat = ConvFormat::NCDHW;
constexpr ConvFormat filterFormat = ConvFormat::NCDHW;

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat outputFormat = ConvFormat::NCDHW;
#elif defined(FORMAT_Y) && FORMAT_Y == FORMAT_NDHWC
constexpr ConvFormat outputFormat = ConvFormat::NDHWC;
#endif

#endif

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NDHWC && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_DHWCN && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NDHWC
constexpr ConvFormat fmapFormat = ConvFormat::NDHWC;
constexpr ConvFormat filterFormat = ConvFormat::DHWCN;
constexpr ConvFormat outputFormat = ConvFormat::NDHWC;
#endif

#else

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat aFormat = ConvFormat::NCDHW;
constexpr ConvFormat bFormat = ConvFormat::NCDHW;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_OFF;
constexpr QuantType quantType = QuantType::NO_QUANT;
#elif defined(ORIG_DTYPE_X) && ORIG_DTYPE_X != DT_INT8
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NDC1HWC0;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
constexpr QuantType quantType = QuantType::NO_QUANT;
#else
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
constexpr QuantType quantType = QuantType::PER_CHANNEL_NO_OFFSET;
#endif

#endif

#if (__CCE_AICORE__ > 300)
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;
#endif

#if (__CCE_AICORE__ > 300)
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
    REGISTER_TILING_DEFAULT(Ops::NN::Conv3dV2::Conv3DV2TilingData);
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

#if defined(ORIG_DTYPE_X) && ORIG_DTYPE_X == DT_INT8 && defined(DTYPE_SCALE)
    using scaleType = ConvType<TPosition::GM, scaleFormat, DTYPE_SCALE>;
    ExtendParams extendParams(scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv3dV2Base<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    } else {
        GroupConv3dV2<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> groupConv3d;
        groupConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData, &extendParams);
    }
#else
    using scaleType = ConvType<TPosition::GM, scaleFormat, uint64_t>;
    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {
        Conv3dV2Base<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong, L0PingPong, OutputOrder, IterOrder, GroupType>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData);
    } else {
        GroupConv3dV2<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<
            FmapTiling, WeightTiling, L1PingPong,
            L0PingPong, OutputOrder, IterOrder, GroupType>> groupConv3d;
        groupConv3d.RunConv3dV2Kernel(x, filter, bias, y, tilingData);
    }
#endif

#endif
}
#else
template <uint8_t ConvL0PingPongT, uint8_t BL1ByPassFlagT, uint8_t GroupConvTypeT, uint8_t OutputOrderT>
__global__ __aicore__ void conv3dv2(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset, GM_ADDR offset_w, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    ASC_OP_LOGD("[Conv3dv2] Begin to process conv3dv2.\n");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR user_workspace = GetUserWorkspace(workspace);
    ASC_OP_LOGD("[Conv3dv2] Get workspace success.\n");
    REGISTER_TILING_DEFAULT(Ops::NN::Conv3dV2::Conv3DV2TilingData);
    GET_TILING_DATA(tilingData, tiling);
    ASC_OP_LOGD("[Conv3dv2] Get tiling data success.\n");
    ASC_OP_LOGD(
        "[Conv3dv2] mUB %u, nUB %u, scaleandbiastype %u, quantType %u.\n", tilingData.conv3dApiTiling.mUB,
        tilingData.conv3dApiTiling.nUB, tilingData.conv3dApiTiling.scaleAndBiasLoadType,
        tilingData.conv3dApiTiling.quantType);

    using A_TYPE = ConvType<TPosition::GM, aFormat, DTYPE_X>;
    using B_TYPE = ConvType<TPosition::GM, bFormat, DTYPE_FILTER>;
    using C_TYPE = ConvType<TPosition::GM, cFormat, DTYPE_Y>;

    // When bias dtype is not specified, use float as default type
#if defined(DTYPE_BIAS)
    using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, DTYPE_BIAS>;
#else
    using BIAS_TYPE = ConvType<TPosition::GM, ConvFormat::ND, float>;
#endif

    DispatchConv3DV2Kernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
                           static_cast<ConvL0PingPong>(ConvL0PingPongT),
                           static_cast<ConvBL1ByPass>(BL1ByPassFlagT),
                           static_cast<GroupConvType>(GroupConvTypeT),
                           static_cast<OutputOrder>(OutputOrderT),
                           quantType>(
        x, filter, bias, scale, offset, y, user_workspace, &tilingData);
}
#endif