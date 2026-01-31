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
 * \file conv3d_v2_template.h
 * \brief
 */
 
#ifndef CONV3D_V2_TEMPLATE_H
#define CONV3D_V2_TEMPLATE_H
#if defined(__NPU_ARCH__) &&((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
#include "arch35/conv3d_v2.h"
#endif
#include "kernel_operator.h"
#include "conv3d_v2_tiling_data.h"
#include "../common/arch35/conv_config.h"
using namespace AscendC;
using namespace conv;
constexpr ConvFormat fmapFormat = ConvFormat::NCDHW;
constexpr ConvFormat filterFormat = ConvFormat::NCDHW;
constexpr ConvFormat outputFormat = ConvFormat::NCDHW;
constexpr ConvFormat biasFormat = ConvFormat::ND;
constexpr ConvFormat scaleFormat = ConvFormat::ND;
 
template<typename inputT, typename filterT, typename biasT, typename scaleT, typename outputT, 
         int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong, int8_t OutputOrder,
         int8_t IterOrder>
__global__ __aicore__ void conv3dv2_template(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, Ops::NN::Conv3dV2::Conv3DV2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using fmapType = ConvType<TPosition::GM, fmapFormat, inputT>;
    using weightType = ConvType<TPosition::GM, filterFormat, filterT>;
    using outputType = ConvType<TPosition::GM, outputFormat, outputT>;
    using biasType = ConvType<TPosition::GM, biasFormat, biasT>;
    using scaleType = ConvType<TPosition::GM, scaleFormat, scaleT>;
 
    Conv3dV2Base<fmapType, weightType, outputType, biasType, scaleType, Conv3DV2Param<FmapTiling, WeightTiling, L1PingPong,
        L0PingPong, OutputOrder, IterOrder, 0>> baseConv3d;
        baseConv3d.RunConv3dV2Kernel(x, filter, bias, y, tiling);
    return;
}
 
#endif // CONV3D_V2_TEMPLATE_H