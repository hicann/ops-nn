/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AVG_POOL3_D_GRAD_TILING_DEF_H
#define AVG_POOL3_D_GRAD_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#define DTYPE_GRADS half
#define __CCE_UT_TEST__
#define __CCE_AICORE__ 220

#pragma pack(1)

struct AvgPool3dGradTilingAttrParam {
    uint64_t N = 0;
    uint64_t C = 0;
    uint64_t outN = 0;
    uint64_t outD = 0;
    uint64_t outH = 0;
    uint64_t outW = 0;
    uint64_t inN = 0;
    uint64_t inD = 0;
    uint64_t inH = 0;
    uint64_t inW = 0;
    uint64_t kD = 0;
    uint64_t kH = 0;
    uint64_t kW = 0;
    uint64_t dD = 0;
    uint64_t dH = 0;
    uint64_t dW = 0;
    uint64_t padD = 0;
    uint64_t padH = 0;
    uint64_t padW = 0;
    uint64_t countIncludePad = 0;
    uint64_t divisorOverride = 0;
    uint64_t isOverLap = 0;
    uint64_t isDetermine = 0;
};

struct AvgPool3dGradTilingCParam {
    uint64_t normalCoreCNum = 0;
    uint64_t lastCoreCNum = 0;
    uint64_t cTotal = 0;
    uint64_t cCount = 0;
    uint64_t cNum = 0;
    uint64_t cLine = 0;
    uint64_t cTail = 0;
    uint64_t cAlign = 0;
};

struct AvgPool3dGradTilingCastCopy {
    uint64_t maxDataNumInUb = 0;
    uint64_t normalCoreNum = 0;
    uint64_t tailCoreNum = 0;
    uint64_t normalCoreDataNum = 0;
    uint64_t tailCoreDataNum = 0;
    uint64_t normalCoreFormerCopyTime = 0;
    uint64_t normalCoreTailCopyTime = 0;
    uint64_t normalCoreFormerDataNum = 0;
    uint64_t normalCoreTailDataNum = 0;
    uint64_t tailCoreFormerCopyTime = 0;
    uint64_t tailCoreTailCopyTime = 0;
    uint64_t tailCoreFormerDataNum = 0;
    uint64_t tailCoreTailDataNum = 0;
};

struct AvgPool3dGradTilingHWParam {
    uint64_t normalCoreHWNum = 0;
    uint64_t lastCoreHWNum = 0;
    uint64_t hwTotal = 0;
    uint64_t hwCount = 0;
    uint64_t hwNum = 0;
    uint64_t nLine = 0;
    uint64_t hwTail = 0;
    uint64_t hwAlign = 0;
};

struct AvgPool3dGradTilingBlockParam {
    uint64_t singleCoreNc = 0;
    uint64_t singleCoreDo = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    uint64_t ncCnt = 0;
    uint64_t doCnt = 0;
    uint64_t hoCnt = 0;
    uint64_t woCnt = 0;
    uint64_t totalCnt = 0;
    uint64_t ncTailData = 0;
    uint64_t doTailData = 0;
    uint64_t hoTailData = 0;
    uint64_t woTailData = 0;
    uint64_t baseDo = 0;
    uint64_t baseHo = 0;
    uint64_t baseWo = 0;
    uint64_t baseDi = 0;
    uint64_t baseHi = 0;
    uint64_t baseWi = 0;
    uint64_t ubSize = 0;
    uint64_t isScatter = 0;
};

struct AvgPool3dGradTilingParam {
    AvgPool3dGradTilingAttrParam attrParam;
    AvgPool3dGradTilingCParam cParam;
    AvgPool3dGradTilingCastCopy castCopyParam;
    AvgPool3dGradTilingHWParam hwParam;
    AvgPool3dGradTilingBlockParam blockParam;
};

#pragma pack()

inline void InitAvgPool3dGradTilingParam(uint8_t* tiling, AvgPool3dGradTilingParam* const_data)
{
    memcpy(const_data, tiling, sizeof(AvgPool3dGradTilingParam));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AvgPool3dGradTilingParam tiling_data;        \
    InitAvgPool3dGradTilingParam(tiling_arg, &tiling_data)
#endif
