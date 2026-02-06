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
 * \file max_pool_grad_with_argmax_tiling.h
 * \brief
 */

#ifndef _GE_MAX_POOL_GRAD_WITH_ARGMAX_TILING_DEF_H_
#define _GE_MAX_POOL_GRAD_WITH_ARGMAX_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)

struct MaxPoolGradWithArgmaxNHWCTilingCommonData {
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

#pragma pack()

#pragma pack(1)

struct MaxPoolGradWithArgmaxSimtTilingCommonData {
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
};
#pragma pack()

inline void InitTilingData(uint8_t* tiling, MaxPoolGradWithArgmaxNHWCTilingCommonData* constData)
{
    memcpy_s(constData, sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData), tiling, sizeof(MaxPoolGradWithArgmaxNHWCTilingCommonData));
}

inline void InitTilingData(uint8_t* tiling, MaxPoolGradWithArgmaxSimtTilingCommonData* constData)
{
    memcpy_s(constData, sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData), tiling, sizeof(MaxPoolGradWithArgmaxSimtTilingCommonData));
}

#endif // MAX_POOL_GRAD_WITH_ARGMAX_TILING_DEF_H_