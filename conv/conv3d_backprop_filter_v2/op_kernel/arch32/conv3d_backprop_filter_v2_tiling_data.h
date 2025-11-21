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
 * \file conv3d_backprop_filter_v2_tiling_data.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H
#define CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H
#include <cstdint>

namespace AscendC{
struct TConv3DDwTiling {
    uint32_t batch = 0;
    uint32_t cin = 0;
    uint32_t cout = 0;
    uint32_t cin1G = 0;
    uint32_t cout1G = 0;
    uint32_t dout = 0;
    uint32_t ho = 0;
    uint32_t wo = 0;
    uint32_t di = 0;
    uint32_t hi = 0;
    uint32_t wi = 0;
    uint32_t dk = 0;
    uint32_t hk = 0;
    uint32_t wk = 0;
    uint32_t group = 0;
    uint32_t strideD = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t padFront = 0;
    uint32_t padBack = 0;
    uint32_t padUp = 0;
    uint32_t padDown = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t dilationD = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t channelSize = 0;
    uint32_t al0Pbuffer = 0;
    uint32_t bl0Pbuffer = 0;
    uint32_t cl0Pbuffer = 0;
    uint32_t al1Pbuffer = 0;
    uint32_t bl1Pbuffer = 0;
    uint32_t baseM = 0;
    uint32_t baseK = 0;
    uint32_t baseN = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t stepM = 0;
    uint32_t stepN = 0;
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
    uint32_t iterateOrder = 0;
    uint32_t bl1Bound = 0;
    uint32_t hf32Flag = 0;
    uint32_t singleCoreDk = 0;
    uint32_t singleCoreGroup = 0;
    uint32_t singleCoreCout = 0;
    uint32_t singleCoreHo = 0;
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreCin = 0;
};
struct Conv3DBackpropFilterV2Params {
    uint64_t batchDim = 0;
    uint32_t groupDim = 0;
    uint32_t mDim = 0;
    uint32_t kDim = 0;
    uint32_t nDim = 0;
    uint32_t dkDim = 0;
    uint32_t totalL1Size = 0;
};
struct TConv3DDwBasicBlockTiling {
    uint32_t coreBindOrder = 0;
    uint32_t usedCoreNum = 0;
    uint32_t singleCoreM = 0;
    uint32_t singleCoreN = 0;
    uint64_t singleCoreK = 0;
};
struct Conv3DBackpropFilterV2TilingData {
    Conv3DBackpropFilterV2Params params;
    TConv3DDwTiling dwTiling;
    TConv3DDwBasicBlockTiling basicBlockTiling;
};
}  // namespace conv_bp_v2_kernel
#endif  // CONV3D_BACKPROP_FILTER_V2_TILING_DATA_H