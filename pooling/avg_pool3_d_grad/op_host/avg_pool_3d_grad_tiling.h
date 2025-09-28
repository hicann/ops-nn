/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool_3d_grad_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (27) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */

#ifndef AVG_POOL_3D_GRAD_TILING_H
#define AVG_POOL_3D_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingAttrParam)
TILING_DATA_FIELD_DEF(uint64_t, N)
TILING_DATA_FIELD_DEF(uint64_t, C)
TILING_DATA_FIELD_DEF(uint64_t, outN)
TILING_DATA_FIELD_DEF(uint64_t, outD)
TILING_DATA_FIELD_DEF(uint64_t, outH)
TILING_DATA_FIELD_DEF(uint64_t, outW)
TILING_DATA_FIELD_DEF(uint64_t, inN)
TILING_DATA_FIELD_DEF(uint64_t, inD)
TILING_DATA_FIELD_DEF(uint64_t, inH)
TILING_DATA_FIELD_DEF(uint64_t, inW)
TILING_DATA_FIELD_DEF(uint64_t, kD)
TILING_DATA_FIELD_DEF(uint64_t, kH)
TILING_DATA_FIELD_DEF(uint64_t, kW)
TILING_DATA_FIELD_DEF(uint64_t, dD)
TILING_DATA_FIELD_DEF(uint64_t, dH)
TILING_DATA_FIELD_DEF(uint64_t, dW)
TILING_DATA_FIELD_DEF(uint64_t, padD)
TILING_DATA_FIELD_DEF(uint64_t, padH)
TILING_DATA_FIELD_DEF(uint64_t, padW)
TILING_DATA_FIELD_DEF(uint64_t, countIncludePad)
TILING_DATA_FIELD_DEF(int64_t, divisorOverride)
TILING_DATA_FIELD_DEF(uint64_t, isOverLap)
TILING_DATA_FIELD_DEF(uint64_t, isDetermine)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingAttrParamOp, AvgPool3dGradTilingAttrParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingCParam)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreCNum)
TILING_DATA_FIELD_DEF(uint64_t, lastCoreCNum)
TILING_DATA_FIELD_DEF(uint64_t, cAlign)
TILING_DATA_FIELD_DEF(uint64_t, cTotal)
TILING_DATA_FIELD_DEF(uint64_t, cCount)
TILING_DATA_FIELD_DEF(uint64_t, cNum)
TILING_DATA_FIELD_DEF(uint64_t, cLine)
TILING_DATA_FIELD_DEF(uint64_t, cTail)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingCParamOp, AvgPool3dGradTilingCParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingCastCopy)
TILING_DATA_FIELD_DEF(uint64_t, maxDataNumInUb)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDataNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreFormerCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreTailCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreTailDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailCopyTime)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreFormerDataNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreTailDataNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingCastCopyOp, AvgPool3dGradTilingCastCopy)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingHWParam)
TILING_DATA_FIELD_DEF(uint64_t, normalCoreHWNum)
TILING_DATA_FIELD_DEF(uint64_t, lastCoreHWNum)
TILING_DATA_FIELD_DEF(uint64_t, hwAlign)
TILING_DATA_FIELD_DEF(uint64_t, hwTotal)
TILING_DATA_FIELD_DEF(uint64_t, hwCount)
TILING_DATA_FIELD_DEF(uint64_t, hwNum)
TILING_DATA_FIELD_DEF(uint64_t, nLine)
TILING_DATA_FIELD_DEF(uint64_t, hwTail)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingHWParamOp, AvgPool3dGradTilingHWParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingBlockParam)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreNc)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo)
TILING_DATA_FIELD_DEF(uint64_t, ncCnt)
TILING_DATA_FIELD_DEF(uint64_t, doCnt)
TILING_DATA_FIELD_DEF(uint64_t, hoCnt)
TILING_DATA_FIELD_DEF(uint64_t, woCnt)
TILING_DATA_FIELD_DEF(uint64_t, totalCnt)
TILING_DATA_FIELD_DEF(uint64_t, ncTailData)
TILING_DATA_FIELD_DEF(uint64_t, doTailData)
TILING_DATA_FIELD_DEF(uint64_t, hoTailData)
TILING_DATA_FIELD_DEF(uint64_t, woTailData)
TILING_DATA_FIELD_DEF(uint64_t, baseDo)
TILING_DATA_FIELD_DEF(uint64_t, baseHo)
TILING_DATA_FIELD_DEF(uint64_t, baseWo)
TILING_DATA_FIELD_DEF(uint64_t, baseDi)
TILING_DATA_FIELD_DEF(uint64_t, baseHi)
TILING_DATA_FIELD_DEF(uint64_t, baseWi)
TILING_DATA_FIELD_DEF(uint64_t, ubSize)
TILING_DATA_FIELD_DEF(uint64_t, isScatter)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3dGradTilingBlockParamOp, AvgPool3dGradTilingBlockParam)

BEGIN_TILING_DATA_DEF(AvgPool3dGradTilingParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingAttrParam, attrParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingCParam, cParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingCastCopy, castCopyParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingHWParam, hwParam)
TILING_DATA_FIELD_DEF_STRUCT(AvgPool3dGradTilingBlockParam, blockParam)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(AvgPool3DGrad, AvgPool3dGradTilingParam)

struct AvgPool3dGradCompileInfo {
    uint32_t totalCoreNum = 48;
    uint64_t ubSizePlatForm = 0;
    bool isAscendC = false;
};
} // namespace optiling
#endif // AVG_POOL_3D_GRAD_TILING_H