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
 * \file avg_pool_v2_grad_tiling_data.h
 * \brief
 */

#ifndef AVG_POOL_V2_GRAD_TILING_DATA_H
#define AVG_POOL_V2_GRAD_TILING_DATA_H

namespace AvgPoolV2Grad {
class AvgPoolV2GradTilingData {
public:
    int64_t nDim;
    int64_t cDim;
    int64_t hInDim;
    int64_t wInDim;
    int64_t hOutDim;
    int64_t wOutDim;
    int64_t kH; // kernel
    int64_t kW;
    int64_t sH; // stride
    int64_t sW;
    int64_t tPad;
    int64_t bPad;
    int64_t lPad;
    int64_t rPad;
    int64_t divisorOverride;
    int64_t countIncludePad;
};
class AvgPoolV2GradSimtTilingData {
public:
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t hPooledDim = 0;
    int64_t wPooledDim = 0;
    int64_t kSizeH = 0;
    int64_t kSizeW = 0;
    int64_t stridesH = 0;
    int64_t stridesW = 0;
    int64_t padHLeft = 0;
    int64_t padHRight = 0;
    int64_t padWLeft = 0;
    int64_t padWRight = 0;
    int64_t countIncludePad = 0;
    int64_t divisorOverride = 0;
};

class AvgPoolV2GradNCHWTilingData {
public:
    int64_t hGrad = 0;
    int64_t wGrad = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t hKernel = 1;
    int64_t wKernel = 1;
    int64_t hStride = 1;
    int64_t wStride = 1;
    int64_t padTopH = 0;
    int64_t padDownH = 0;
    int64_t padLeftW = 0;
    int64_t padRightW = 0;
    int64_t highAxisInner = 0;
    int64_t highAxisTail = 0;
    int64_t highAxisOuter = 0;
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
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
    int64_t divisorOverride = 1;
};
} // namespace AvgPoolV2Grad
#endif // AVG_POOL_V2_GRAD_STRUCT_H