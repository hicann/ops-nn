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
 * \file conv3d_v2_kernel_dispatch.h
 * \brief Public interface for dispatching Conv3D V2 kernel to appropriate template implementations
 */

#ifndef CONV3D_V2_KERNEL_DISPATCH_H
#define CONV3D_V2_KERNEL_DISPATCH_H

#include "conv3dv2_with_groups.h"
#include "conv3dv2_pointwise.h"
#include "conv3dv2_hw_mode.h"
#include "conv3d/conv3d_config.h"

using namespace AscendC;

template <conv3d::ConvL0PingPong pingPong_T, conv3d::ConvBL1ByPass bl1BypassFlag_T,
          conv3d::GroupConvType groupConvType_T,
          conv3d::OutputOrder outputOrder_T, conv3d::QuantType quantType_T>
struct Conv3DV2Param : public ConvParam {
    __aicore__ inline Conv3DV2Param() = default;
    constexpr static int8_t outputOrder = static_cast<int8_t>(outputOrder_T);
    constexpr static int8_t l0pingpong = static_cast<int8_t>(pingPong_T);
    constexpr static int8_t bl1bypass = static_cast<int8_t>(bl1BypassFlag_T);
    constexpr static int8_t groupConvType = static_cast<int8_t>(groupConvType_T);
    constexpr static int8_t quantType = static_cast<int8_t>(quantType_T);
};

template <class KernelType, class... InitArgs>
__aicore__ inline void DispatchConvKernel(KernelType&& op, InitArgs... args) {
    op.Init(args...);
    ASC_OP_LOGD("[Conv3dv2] Op init success.\n");
    op.Process();
    ASC_OP_LOGD("[Conv3dv2] Op process success.\n");
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
          conv3d::ConvL0PingPong ConvL0PingPongT, conv3d::ConvBL1ByPass BL1ByPassFlagT,
          conv3d::GroupConvType GroupConvTypeT, conv3d::OutputOrder OutputOrderT,
          conv3d::QuantType QuantTypeT>
__aicore__ inline void DispatchConv3DV2Kernel(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR y, GM_ADDR workspace,
    const Ops::NN::Conv3dV2::Conv3DV2TilingData* tilingData)
{
    using ParamType = Conv3DV2Param<ConvL0PingPongT, BL1ByPassFlagT, GroupConvTypeT, OutputOrderT, QuantTypeT>;

    if constexpr (A_TYPE::format == ConvFormat::NCDHW) {
        // PointWise
        if constexpr (
            BL1ByPassFlagT == conv3d::ConvBL1ByPass::BYPASS_OFF &&
            GroupConvTypeT == conv3d::GroupConvType::NoGroup_Conv &&
            OutputOrderT == conv3d::OutputOrder::M_Mode) {
            KernelConv3DV2PointWise<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
            DispatchConvKernel(op, x, filter, bias, scale, offset, y, workspace, tilingData);
        }

    } else if constexpr (GroupConvTypeT == conv3d::GroupConvType::GroupConv_Weight_Gfz) {
        // Group Convolution (Only if A_TYPE::format != NCDHW)
        KernelConv3DV2WithGroups<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, workspace, tilingData);

    } else if constexpr (OutputOrderT == conv3d::OutputOrder::HW_Mode) {
        // HW mode (The conditions above are false)
        KernelConv3DV2HwMode<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, workspace, tilingData);

    } else {
        // base (Default)
        KernelConv3DV2<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, ParamType> op;
        DispatchConvKernel(op, x, filter, bias, scale, offset, y, workspace, tilingData);
    }
}

#endif // CONV3D_V2_KERNEL_DISPATCH_H
