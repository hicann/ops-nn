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
 * \file conv3d_backprop_input_v2_tiling_data.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ADVANCE_H

namespace conv_bp_v2_kernel {
struct TConv3DInputV2Tiling {
    uint8_t al0Pbuffer = 1;
    uint8_t bl0Pbuffer = 1;
    uint8_t cl0Pbuffer = 1;
    uint8_t al1Pbuffer = 1;
    uint8_t bl1Pbuffer = 1;
    uint8_t iterateOrder = 1;
    uint8_t c0 = 1;
    uint8_t c0Bits = 1;
    uint8_t enlarge = 1;
    uint8_t hf32Flag = 1;
    uint8_t initOutputFlag = 1;
    uint8_t isBiasFullLoad = 1;
    uint8_t enableVecTrans = 1;
    uint32_t batch = 1;
    uint32_t cin = 1;
    uint32_t cout = 1;
    uint32_t cinG = 1;
    uint32_t coutG = 1;
    uint32_t cout1 = 1;
    uint32_t cin1 = 1;
    uint32_t cout1G = 1;
    uint32_t cin1G = 1;
    uint32_t dout = 1;
    uint32_t ho = 1;
    uint32_t wo = 1;
    uint32_t di = 1;
    uint32_t hi = 1;
    uint32_t wi = 1;
    uint32_t dk = 1;
    uint32_t hk = 1;
    uint32_t wk = 1;
    uint32_t group = 1;
    uint32_t oriGroup = 1;
    uint32_t strideD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padFront = 1;
    uint32_t padBack = 1;
    uint32_t padUp = 1;
    uint32_t padDown = 1;
    uint32_t padLeft = 1;
    uint32_t padRight = 1;
    uint32_t backpropPadTail = 1;
    uint32_t backpropPadUp = 1;
    uint32_t backpropPadDown = 1;
    uint32_t backpropPadLeft = 1;
    uint32_t backpropPadRight = 1;
    uint32_t dilationD = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t singleCoreGroup = 1;
    uint32_t singleCoreCout = 1;
    uint32_t singleCoreCin = 1;
    uint32_t singleCoreDin = 1;
    uint32_t baseM = 1;
    uint32_t baseK = 1;
    uint32_t baseN = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint32_t singleIterateDk = 1;
    uint64_t singleCoreBatch = 1;
    uint64_t singleCoreM = 1;
    uint64_t enRelu = 0;
};

struct Conv3DBackpropInputV2Params {
    uint32_t batchDim = 1;
    uint32_t groupDim = 1;
    uint32_t mDim = 1;
    uint32_t kDim = 1;
    uint32_t nDim = 1;
    uint32_t dDim = 1;
    uint64_t coreNum = 1;
};

struct TConv3DInputV2KSTiling {
    uint32_t kSCoutFullLoad = 0;
    uint32_t kSUseWorkSpace = 0;
};

struct Conv3DBackpropInputV2TilingData{
    struct Conv3DBackpropInputV2Params params;
    struct TConv3DInputV2Tiling conv3DDxTiling;
    struct TConv3DInputV2KSTiling conv3DDxKSTiling;
};

}  // namespace conv_bp_v2_kernel
#endif  // CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ADVANCE_H