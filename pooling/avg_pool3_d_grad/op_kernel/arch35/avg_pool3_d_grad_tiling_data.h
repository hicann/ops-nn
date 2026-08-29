/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_tiling_data.h
 * \brief 3D average pooling backward tiling data structs (arch35).
 *        Modeled on avg_pool_v2_grad_tiling_data.h, extended to D/H/W.
 */

#ifndef AVG_POOL3_D_GRAD_TILING_DATA_H
#define AVG_POOL3_D_GRAD_TILING_DATA_H

namespace AvgPool3DGrad {

constexpr int32_t FORMAT_NCDHW = 0;
constexpr int32_t FORMAT_NDHWC = 1;

class AvgPool3DGradNCDHWTilingData {
public:
    int64_t ncTotal = 0;
    int64_t dGrad = 0;
    int64_t hGrad = 0;
    int64_t wGrad = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t dKernel = 1;
    int64_t hKernel = 1;
    int64_t wKernel = 1;
    int64_t dStride = 1;
    int64_t hStride = 1;
    int64_t wStride = 1;
    int64_t padFrontD = 0;
    int64_t padBackD = 0;
    int64_t padTopH = 0;
    int64_t padDownH = 0;
    int64_t padLeftW = 0;
    int64_t padRightW = 0;
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t outputBufferSize = 0;
    int64_t gradBufferSize = 0;
    int64_t dProBatchSize = 0;
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
    int64_t divisorOverride = 1;
    int64_t countIncludePad = 0;
};

class AvgPool3DGradNDHWCTilingData {
public:
    int64_t nTotal = 0;
    int64_t dGrad = 0;
    int64_t hGrad = 0;
    int64_t wGrad = 0;
    int64_t cOutput = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t dKernel = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t dStride = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padFront = 0;
    int64_t padBack = 0;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    int64_t nOutputInner = 0;
    int64_t nOutputTail = 0;
    int64_t nOutputOuter = 0;
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t cOutputInner = 0;
    int64_t cOutputTail = 0;
    int64_t cOutputOuter = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;
    int64_t outputBufferSize = 0;
    int64_t inputGradBufferSize = 0;
    int64_t dProBatchSize = 0;
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
    int64_t divisorOverride = 0;
    int64_t countIncludePad = 0;
    int64_t tilingKey = 0;
};

class AvgPool3DGradSimtTilingData {
public:
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t dInDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t dOutDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
    int64_t kSizeD = 0;
    int64_t kSizeH = 0;
    int64_t kSizeW = 0;
    int64_t stridesD = 0;
    int64_t stridesH = 0;
    int64_t stridesW = 0;
    int64_t padDLeft = 0;
    int64_t padDRight = 0;
    int64_t padHLeft = 0;
    int64_t padHRight = 0;
    int64_t padWLeft = 0;
    int64_t padWRight = 0;
    int64_t countIncludePad = 0;
    int64_t divisorOverride = 0;
};

class AvgPool3DGradKsizeOneTilingData {
public:
    int64_t ubBufferSize = 0;
    int64_t elementsPerCore = 0;
    int64_t tailCoreElements = 0;
    int64_t totalElements = 0;
    int64_t divisor = 1;
    int64_t usedCoreNum = 0;
};

} // namespace AvgPool3DGrad
#endif // AVG_POOL3_D_GRAD_TILING_DATA_H
