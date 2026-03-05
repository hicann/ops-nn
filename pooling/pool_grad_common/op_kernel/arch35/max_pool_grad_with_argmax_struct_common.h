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
 * \file max_pool_grad_with_argmax_struct_common.h
 * \brief tiling base data
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_STRUCT_H
#define MAX_POOL_GRAD_WITH_ARGMAX_STRUCT_H

#include <cstdint>

namespace MaxPoolGradWithArgmaxNHWCNameSpace {
class MaxPoolGradWithArgmaxNHWCTilingCommonData {
public:
    int64_t hArgmax = 0;
    int64_t wArgmax = 0;
    int64_t cOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t nOutputInner = 0;
    int64_t nOutputTail = 0;
    int64_t nOutputOuter = 0;
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
    int64_t gradBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
    int64_t tilingKey = 0;
};

class MaxPoolGradWithArgmaxSimtTilingCommonData {
public:
    int64_t nDim = 0;
    int64_t cDim = 0;
    int64_t hInDim = 0;
    int64_t wInDim = 0;
    int64_t hOutDim = 0;
    int64_t wOutDim = 0;
    int64_t kSizeH = 0;
    int64_t kSizeW = 0;
    int64_t stridesH = 0;
    int64_t stridesW = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t ceilMode = 0;
};

class MaxPoolGradWithArgmaxSizeOneTilingCommonData {
public:
    int64_t usedCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t coreLoop = 0;
    int64_t tailCoreLoop = 0;
    int64_t ubFactor = 0;
    int64_t tailUbFactor = 0;
    int64_t tailCoreTailUbFactor = 0;
    int64_t tilingKey = 0;
};


class MaxPoolGradWithArgmaxNCHWScalarTilingCommonData {
public:
    int64_t hArgmax = 0;
    int64_t wArgmax = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
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
    int64_t argmaxBufferSize = 0;
    int64_t argmaxNcInner = 0;
    int64_t argmaxNcOuter = 0;
    int64_t argmaxNcTail = 0;
    int64_t argmaxHInner = 0;
    int64_t argmaxHOuter = 0;
    int64_t argmaxHTail = 0;
    int64_t argmaxWInner = 0;
    int64_t argmaxWOuter = 0;
    int64_t argmaxWTail = 0;
    int64_t argmaxInnerLoop = 0;
    int64_t argmaxNcInnerTail = 0;
    int64_t argmaxNcOuterTail = 0;
    int64_t argmaxNcTailTail = 0;
    int64_t argmaxHInnerTail = 0;
    int64_t argmaxHOuterTail = 0;
    int64_t argmaxHTailTail = 0;
    int64_t argmaxWInnerTail = 0;
    int64_t argmaxWOuterTail = 0;
    int64_t argmaxWTailTail = 0;
    int64_t argmaxInnerLoopTail = 0;
};

class MaxPoolGradWithArgmaxNCHWTilingCommonData {
public:
    int64_t hArgmax = 0;
    int64_t wArgmax = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
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
    int64_t argmaxBufferSize = 0;
    int64_t hProBatchSize = 0;
    int64_t wProBatchSize = 0;
    int64_t tilingKey = 0;
};
}
#endif //MAX_POOL_GRAD_WITH_ARGMAX_STRUCT_H