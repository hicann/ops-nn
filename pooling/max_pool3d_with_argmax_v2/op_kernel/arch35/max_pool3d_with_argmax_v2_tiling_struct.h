/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file max_pool3d_with_argmax_v2_tiling_struct.h
 * \brief max_pool3d_with_argmax_v2  tiling data
 */

#ifndef MAX_POOL_3D_WITH_ARGMAX_V2_TILING_DATA_H
#define MAX_POOL_3D_WITH_ARGMAX_V2_TILING_DATA_H

namespace MaxPool3DWithArgmaxV2Tiling{
class MaxPool3DWithArgmaxV2GatherTilingData {
public:
    int64_t dInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
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
    int64_t padLeft = 0;
    int64_t padTop = 0;
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
    int64_t inputBufferSize = 0;
    int64_t maxValueBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t isPad = 0;
    int64_t dDilation = 0;
    int64_t hDilation = 0;
    int64_t wDilation = 0;
};

class MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData {
public:
    int64_t dInDim = 1;
    int64_t hInDim = 1;
    int64_t wInDim = 1;
    int64_t dOutDim = 1;
    int64_t hOutDim = 1;
    int64_t wOutDim = 1;
    int64_t kD = 1;
    int64_t kH = 1;
    int64_t kW = 1;
    int64_t sD = 1;
    int64_t sH = 1;
    int64_t sW = 1;
    int64_t pD = 1;
    int64_t pH = 1;
    int64_t pW = 1;
    int64_t dD = 1;
    int64_t dH = 1;
    int64_t dW = 1;
    int64_t blockFactor = 1;
    int64_t blockTail = 1;
    int64_t totalIdx = 1;
    int64_t coreNums = 1;
    int64_t maxCount = 1;
    int64_t isSigOut = 0;
};

class MaxPool3DWithArgmaxV2SimtTilingData {
public:
    int64_t threadNums = 1;
    int64_t blockNums = 1;
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
    int64_t padD = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationD = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t ceilMode = 0;
};
}

#endif