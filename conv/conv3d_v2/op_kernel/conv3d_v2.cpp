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

#include "conv3dv2_tiling_key.h"
#include "conv3dv2_with_groups.h"
#include "conv3dv2_pointwise.h"
#include "conv3dv2_hw_mode.h"
#include "conv3dv2.h"

using namespace AscendC;

#if defined(FORMAT_X) && FORMAT_X == FORMAT_NCDHW && defined(FORMAT_FILTER) && FORMAT_FILTER == FORMAT_NCDHW && \
    defined(FORMAT_Y) && FORMAT_Y == FORMAT_NCDHW
constexpr ConvFormat aFormat = ConvFormat::NCDHW;
constexpr ConvFormat bFormat = ConvFormat::NCDHW;
constexpr ConvFormat cFormat = ConvFormat::NCDHW;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_OFF;
#else
constexpr ConvFormat aFormat = ConvFormat::NDC1HWC0;
constexpr ConvFormat bFormat = ConvFormat::FRACTAL_Z_3D;
constexpr ConvFormat cFormat = ConvFormat::NDC1HWC0;
constexpr ConvBL1ByPass bL1ByPassFlag = ConvBL1ByPass::BYPASS_ON;
#endif

#if defined(ORIG_DTYPE_X) && ORIG_DTYPE_X != DT_INT8
constexpr QuantType quantType = QuantType::NO_QUANT;
#else
constexpr QuantType quantType = QuantType::PER_CHANNEL_NO_OFFSET;
#endif

template <class KernelType, class... InitArgs>
__aicore__ inline void DispatchConvKernel(KernelType&& op, InitArgs... args) {
    op.Init(args...);
    ASC_OP_LOGD("[Conv3dv2] Op init success.\n");
    op.Process();
    ASC_OP_LOGD("[Conv3dv2] Op process success.\n");
}

template <uint8_t ConvL0PingPong, uint8_t BL1ByPassFlag, uint8_t GroupConvType, uint8_t OutputOrder>
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

    using ParamType = Conv3DV2Param<ConvL0PingPong, BL1ByPassFlag, GroupConvType, OutputOrder, quantType>;
    if constexpr (aFormat == ConvFormat::NCDHW) {
        // PointWise
        if constexpr (
            BL1ByPassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
            GroupConvType == static_cast<uint8_t>(GroupConvType::NoGroup_Conv) &&
            OutputOrder == static_cast<uint8_t>(OutputOrder::M_Mode)) {
            KernelConv3DV2PointWise<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
            DispatchConvKernel(op, x, filter, bias, scale, offset, y, user_workspace, &tilingData);
        }

    } else if constexpr (GroupConvType == static_cast<uint8_t>(GroupConvType::GroupConv_Weight_Gfz)) {
        // Group Convolution (Only if aFormat != NCDHW)
        KernelConv3DV2WithGroups<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, user_workspace, &tilingData);

    } else if constexpr (OutputOrder == static_cast<uint8_t>(OutputOrder::HW_Mode)) {
        // HW mode (The conditions above are false)
        KernelConv3DV2HwMode<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, user_workspace, &tilingData);

    } else {
        // base (Default)
        KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, user_workspace, &tilingData);
    }
}