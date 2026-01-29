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
 * \file conv3d_v2_base_tiling_cache.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {
bool Conv3dBaseTilingV2::CheckSupportCacheTiling()
{
    if (paramInfo_.nodeType == "QuantConv3D") {
        return false;
    }

    return true;    
}

bool Conv3dBaseTilingV2::GetTilingFromCache()
{
    if (!CheckSupportCacheTiling()) {
        return false;
    }

    Conv3dTilingCache& tilingCache = Conv3dTilingCache::GetInstance();
    OP_LOGD(context_->GetNodeName(), "%s AscendC: current cache size is %zu.",
            paramInfo_.nodeType.c_str(), tilingCache.GetCacheSize());
    GetCacheTilingInputArgs();
    if (tilingCache.GetCachedTiling(cacheInputArgs_, cachedTilingData_)) {
        TranslateCachedTilingData();
        return true;
    }
    return false;
}

bool Conv3dBaseTilingV2::AddTilingToCache()
{
    if (!CheckSupportCacheTiling()) {
        return false;
    }

    Conv3dTilingCache& tilingCache = Conv3dTilingCache::GetInstance();
    if (tilingCache.GetCacheSize() == MAX_CACHE_SIZE) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: current cache size is 4096, do not add tiling to cache.",
                paramInfo_.nodeType.c_str());
        return false;
    }
    GetCachedTilingData();

    return tilingCache.AddCachedTiling(cacheInputArgs_, cachedTilingData_);
}

void Conv3dBaseTilingV2::GetCacheTilingInputArgs()
{
    cacheInputArgs_.inputDtype = descInfo_.fMapDtype;
    cacheInputArgs_.weightDtype = descInfo_.weightDtype;
    cacheInputArgs_.outputDtype = descInfo_.outDtype;
    cacheInputArgs_.biasDtype = descInfo_.biasDtype;
    cacheInputArgs_.inputShapeN = shapeInfo_.batch;
    cacheInputArgs_.inputShapeH = shapeInfo_.hi;
    cacheInputArgs_.inputShapeD = shapeInfo_.di;
    cacheInputArgs_.inputShapeW = shapeInfo_.wi;
    cacheInputArgs_.weightShapeN = shapeInfo_.co;
    cacheInputArgs_.weightShapeC = shapeInfo_.ci / attrInfo_.groups;
    cacheInputArgs_.weightShapeD = shapeInfo_.kd;
    cacheInputArgs_.weightShapeH = shapeInfo_.kh;
    cacheInputArgs_.weightShapeW = shapeInfo_.kw;
    cacheInputArgs_.outputShapeD = shapeInfo_.dout;;
    cacheInputArgs_.outputShapeH = shapeInfo_.ho;
    cacheInputArgs_.outputShapeW = shapeInfo_.wo;
    cacheInputArgs_.inputFormat = descInfo_.fMapFormat;
    cacheInputArgs_.weightFormat = descInfo_.weightFormat;
    cacheInputArgs_.outputFormat = descInfo_.outFormat;
    cacheInputArgs_.groups = attrInfo_.groups;
    cacheInputArgs_.strideD = attrInfo_.strideD;
    cacheInputArgs_.strideH = attrInfo_.strideH;
    cacheInputArgs_.strideW = attrInfo_.strideW;
    cacheInputArgs_.dilationD = attrInfo_.dilationD;
    cacheInputArgs_.dilationH = attrInfo_.dilationH;
    cacheInputArgs_.dilationW = attrInfo_.dilationW;
    cacheInputArgs_.padHead = attrInfo_.padHead;
    cacheInputArgs_.padTail = attrInfo_.padTail;
    cacheInputArgs_.padTop = attrInfo_.padTop;
    cacheInputArgs_.padBottom = attrInfo_.padBottom;
    cacheInputArgs_.padLeft = attrInfo_.padLeft;
    cacheInputArgs_.padRight = attrInfo_.padRight;
    cacheInputArgs_.biasFlag = flagInfo_.hasBias;
    cacheInputArgs_.hf32Flag = (attrInfo_.hf32Mode == 1);
    cacheInputArgs_.dual_output = 0;
    cacheInputArgs_.quantMode0 = 0;
    cacheInputArgs_.reluMode0 = 0;
    cacheInputArgs_.clipMode0 = 0;
    cacheInputArgs_.scaleFlag0 = 0;
    cacheInputArgs_.quantMode1 = 0;
    cacheInputArgs_.reluMode1 = 0;
    cacheInputArgs_.clipMode1 = 0;
    cacheInputArgs_.scaleFlag1 = 0;
}


void Conv3dBaseTilingV2::GetCachedTilingData() {
    cachedTilingData_ = tilingData_;
}

void Conv3dBaseTilingV2::TranslateCachedTilingData()
{
    tilingData_ = cachedTilingData_;
    tilingData_.conv3dApiTiling.hasScale = static_cast<uint8_t>(flagInfo_.quantFlag);
    tilingData_.conv3dApiTiling.offsetx = attrInfo_.offsetx;
    tilingData_.conv3dApiTiling.roundMode = attrInfo_.roundMode;
    flagInfo_.mSplitModeFlag = cachedTilingData_.conv3dApiTiling.outputOrder == 0 ? true : false;
}
}
}