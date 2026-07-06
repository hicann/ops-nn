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
 * \file conv2d_v2_base_tiling_cache.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {
bool Conv2dBaseTiling::CheckSupportCacheTiling()
{
    if (paramInfo_.nodeType == "QuantConv2D") {
        return false;
    }

    return true;
}

bool Conv2dBaseTiling::GetTilingFromCache()
{
    if (!CheckSupportCacheTiling()) {
        return false;
    }

    Conv2dTilingCache& tilingCache = Conv2dTilingCache::GetInstance();
    OP_LOGD(context_->GetNodeName(), "%s AscendC: current cache size is %zu.", paramInfo_.nodeType.c_str(),
            tilingCache.GetCacheSize());
    GetCacheTilingInputArgs();
    if (tilingCache.GetCachedTiling(cacheInputArgs_, cachedTilingData_)) {
        TranslateCachedTilingData();
        return true;
    }

    return false;
}

bool Conv2dBaseTiling::AddTilingToCache()
{
    if (!CheckSupportCacheTiling()) {
        return false;
    }

    Conv2dTilingCache& tilingCache = Conv2dTilingCache::GetInstance();
    if (tilingCache.GetCacheSize() == MAX_CACHE_SIZE) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: current cache size is 4096, do not add tiling to cache.",
                paramInfo_.nodeType.c_str());
        return false;
    }

    GetCachedTilingData();

    return tilingCache.AddCachedTiling(cacheInputArgs_, cachedTilingData_);
}

void Conv2dBaseTiling::GetCachedTilingDataAux()
{
    cachedTilingData_.fmapKStride = tilingData_.get_fmapKStride();
    cachedTilingData_.weightKStride = tilingData_.get_weightKStride();
    cachedTilingData_.cinOffsetBlockInGM = tilingData_.get_cinOffsetBlockInGM();
    cachedTilingData_.coutOffsetBlock = tilingData_.get_coutOffsetBlock();
    cachedTilingData_.nL1DivBlockSize = tilingData_.get_nL1DivBlockSize();
    cachedTilingData_.iterateMNOrder = tilingData_.get_iterateMNOrder();
    cachedTilingData_.biasFullLoadFlag = tilingData_.get_biasFullLoadFlag();
    cachedTilingData_.fixpParamsFullLoadFlag = tilingData_.get_fixpParamsFullLoadFlag();
    cachedTilingData_.hf32Enable = tilingData_.get_hf32Enable();
    cachedTilingData_.hf32TransMode = tilingData_.get_hf32TransMode();
    cachedTilingData_.batchDim = tilingData_.get_batchDim();
    cachedTilingData_.groupDim = tilingData_.get_groupDim();
    cachedTilingData_.nDim = tilingData_.get_nDim();
    cachedTilingData_.hoDim = tilingData_.get_hoDim();
    cachedTilingData_.woDim = tilingData_.get_woDim();
    cachedTilingData_.cinOpt = tilingData_.get_cinOpt();
    cachedTilingData_.coutOpt = tilingData_.get_coutOpt();
    cachedTilingData_.groupOpt = tilingData_.get_groupOpt();
    cachedTilingData_.enableC04Flag = flagInfo_.enableC04Flag;
    cachedTilingData_.mSplitModeFlag = flagInfo_.mSplitModeFlag;
    cachedTilingData_.unionDataXt = tilingData_.get_unionDataXt();
}

void Conv2dBaseTiling::GetCachedTilingData()
{
    cachedTilingData_.singleCoreBatch = tilingData_.get_singleCoreBatch();
    cachedTilingData_.singleCoreHo = tilingData_.get_singleCoreHo();
    cachedTilingData_.singleCoreWo = tilingData_.get_singleCoreWo();
    cachedTilingData_.singleCoreCi = tilingData_.get_singleCoreCi();
    cachedTilingData_.singleCoreCo = tilingData_.get_singleCoreCo();
    cachedTilingData_.hoL1 = tilingData_.get_hoL1();
    cachedTilingData_.woL1 = tilingData_.get_woL1();
    cachedTilingData_.kAL1 = tilingData_.get_kAL1();
    cachedTilingData_.kBL1 = tilingData_.get_kBL1();
    cachedTilingData_.khL1 = tilingData_.get_khL1();
    cachedTilingData_.kwL1 = tilingData_.get_kwL1();
    cachedTilingData_.nBL1 = tilingData_.get_nBL1();
    cachedTilingData_.hoL0 = tilingData_.get_hoL0();
    cachedTilingData_.woL0 = tilingData_.get_woL0();
    cachedTilingData_.kL0 = tilingData_.get_kL0();
    cachedTilingData_.nL0 = tilingData_.get_nL0();
    cachedTilingData_.pBufferFlag = tilingData_.get_pBufferFlag();
    cachedTilingData_.enlarge = tilingData_.get_enlarge();
    cachedTilingData_.singleCoreGroups = tilingData_.get_singleCoreGroups();
    cachedTilingData_.singleCoreGroupOpt = tilingData_.get_singleCoreGroupOpt();
    cachedTilingData_.bUbNStep = tilingData_.get_bUbNStep();
    cachedTilingData_.bUbKStep = tilingData_.get_bUbKStep();
    cachedTilingData_.khUb = tilingData_.get_khUb();
    cachedTilingData_.kwUb = tilingData_.get_kwUb();
    cachedTilingData_.orgHixWi = tilingData_.get_orgHixWi();
    cachedTilingData_.kernelHxkernelW = tilingData_.get_kernelHxkernelW();
    cachedTilingData_.kernelHxkernelWxkernelD = tilingData_.get_kernelHxkernelWxkernelD();
    cachedTilingData_.aL1SpaceSize = tilingData_.get_aL1SpaceSize();
    cachedTilingData_.multiNBL1 = tilingData_.get_multiNBL1();
    cachedTilingData_.cinAInCore = tilingData_.get_cinAInCore();
    cachedTilingData_.cinATailInCore = tilingData_.get_cinATailInCore();
    cachedTilingData_.cinBInCore = tilingData_.get_cinBInCore();
    cachedTilingData_.cinBTailInCore = tilingData_.get_cinBTailInCore();
    cachedTilingData_.mStep = tilingData_.get_mStep();
    cachedTilingData_.kStep = tilingData_.get_kStep();
    cachedTilingData_.nStep = tilingData_.get_nStep();
    cachedTilingData_.innerBatch = tilingData_.get_innerBatch();
    GetCachedTilingDataAux();
}

void Conv2dBaseTiling::GetCacheTilingInputArgsExtend()
{
    cacheInputArgs_.dual_output = attrInfo_.dualOutput;
    cacheInputArgs_.quantMode0 = fixpipeInfo_.quantMode0;
    cacheInputArgs_.reluMode0 = fixpipeInfo_.reluMode0;
    cacheInputArgs_.clipMode0 = fixpipeInfo_.clipMode0;
    cacheInputArgs_.scaleFlag0 = 0;
    cacheInputArgs_.quantMode1 = fixpipeInfo_.quantMode1;
    cacheInputArgs_.reluMode1 = fixpipeInfo_.reluMode1;
    cacheInputArgs_.clipMode1 = fixpipeInfo_.clipMode1;
    cacheInputArgs_.scaleFlag1 = 0;
    cacheInputArgs_.output1Format = descInfo_.out1Format;
    cacheInputArgs_.output1Dtype = descInfo_.out1Dtype;
    cacheInputArgs_.output1ShapeN = oriShapeAttrInfo_.oriOutput1N;
    cacheInputArgs_.output1ShapeC = oriShapeAttrInfo_.oriOutput1C;
    cacheInputArgs_.output1ShapeH = oriShapeAttrInfo_.oriOutput1H;
    cacheInputArgs_.output1ShapeW = oriShapeAttrInfo_.oriOutput1W;
}

void Conv2dBaseTiling::GetCacheTilingInputArgs()
{
    cacheInputArgs_.inputDtype = descInfo_.fMapDtype;
    cacheInputArgs_.weightDtype = descInfo_.weightDtype;
    cacheInputArgs_.outputDtype = descInfo_.outDtype;
    cacheInputArgs_.biasDtype = descInfo_.biasDtype;
    cacheInputArgs_.inputShapeN = shapeInfo_.batch;
    cacheInputArgs_.inputShapeH = shapeInfo_.hi;
    cacheInputArgs_.inputShapeD = 1;
    cacheInputArgs_.inputShapeW = shapeInfo_.wi;
    cacheInputArgs_.weightShapeN = shapeInfo_.co;
    cacheInputArgs_.weightShapeC = shapeInfo_.ci / attrInfo_.groups;
    cacheInputArgs_.weightShapeD = 1;
    cacheInputArgs_.weightShapeH = shapeInfo_.kh;
    cacheInputArgs_.weightShapeW = shapeInfo_.kw;
    cacheInputArgs_.outputShapeD = 1;
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
    cacheInputArgs_.padHead = 0;
    cacheInputArgs_.padTail = 0;
    cacheInputArgs_.padTop = attrInfo_.padTop;
    cacheInputArgs_.padBottom = attrInfo_.padBottom;
    cacheInputArgs_.padLeft = attrInfo_.padLeft;
    cacheInputArgs_.padRight = attrInfo_.padRight;
    cacheInputArgs_.biasFlag = flagInfo_.hasBias;
    cacheInputArgs_.hf32Flag = (attrInfo_.hf32Mode == 1);
    GetCacheTilingInputArgsExtend();
}

void Conv2dBaseTiling::TranslateCachedRunInfo()
{
    tilingData_.set_batch(cacheInputArgs_.inputShapeN);
    tilingData_.set_hin(cacheInputArgs_.inputShapeH);
    tilingData_.set_win(cacheInputArgs_.inputShapeW);
    tilingData_.set_batchDim(cachedTilingData_.batchDim);
    tilingData_.set_hoDim(cachedTilingData_.hoDim);
    tilingData_.set_woDim(cachedTilingData_.woDim);
    tilingData_.set_nDim(cachedTilingData_.nDim);
    tilingData_.set_cin(shapeInfo_.ci);
    tilingData_.set_cout(cacheInputArgs_.weightShapeN);
    tilingData_.set_kh(cacheInputArgs_.weightShapeH);
    tilingData_.set_kw(cacheInputArgs_.weightShapeW);
    tilingData_.set_hout(cacheInputArgs_.outputShapeH);
    tilingData_.set_wout(cacheInputArgs_.outputShapeW);
    tilingData_.set_cinOpt(cachedTilingData_.cinOpt);
    tilingData_.set_coutOpt(cachedTilingData_.coutOpt);
    tilingData_.set_groupOpt(cachedTilingData_.groupOpt);
    tilingData_.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.set_groupDim(cachedTilingData_.groupDim);
}

void Conv2dBaseTiling::TranslateCachedApiTilingPartTwo()
{
    tilingData_.set_strideH(cacheInputArgs_.strideH);
    tilingData_.set_strideW(cacheInputArgs_.strideW);
    tilingData_.set_dilationH(cacheInputArgs_.dilationH);
    tilingData_.set_dilationW(cacheInputArgs_.dilationW);
    tilingData_.set_padTop(cacheInputArgs_.padTop);
    tilingData_.set_padBottom(cacheInputArgs_.padBottom);
    tilingData_.set_padLeft(cacheInputArgs_.padLeft);
    tilingData_.set_padRight(cacheInputArgs_.padRight);
    tilingData_.set_iterateMNOrder(cachedTilingData_.iterateMNOrder);
    tilingData_.set_biasFullLoadFlag(cachedTilingData_.biasFullLoadFlag);
    tilingData_.set_fixpParamsFullLoadFlag(cachedTilingData_.fixpParamsFullLoadFlag);
    tilingData_.set_hf32Enable(cachedTilingData_.hf32Enable);
    tilingData_.set_hf32TransMode(cachedTilingData_.hf32TransMode);
    tilingData_.set_hasBias(cacheInputArgs_.biasFlag);
    tilingData_.set_hasScale(static_cast<uint8_t>(flagInfo_.quantFlag || flagInfo_.extendConvFlag));
    tilingData_.set_offsetx(attrInfo_.offsetx);
    tilingData_.set_roundMode(attrInfo_.roundMode);
    tilingData_.set_dualOutput(fixpipeInfo_.dualOutput);
    tilingData_.set_quantMode0(fixpipeInfo_.quantMode0);
    tilingData_.set_reluMode0(fixpipeInfo_.reluMode0);
    tilingData_.set_clipMode0(fixpipeInfo_.clipMode0);
    tilingData_.set_quantMode1(fixpipeInfo_.quantMode1);
    tilingData_.set_reluMode1(fixpipeInfo_.reluMode1);
    tilingData_.set_clipMode1(fixpipeInfo_.clipMode1);
    tilingData_.set_innerBatch(cachedTilingData_.innerBatch);
    tilingData_.set_unionDataXt(cachedTilingData_.unionDataXt);
}

void Conv2dBaseTiling::TranslateCachedApiTilingPartOne()
{
    tilingData_.set_orgHi(cacheInputArgs_.inputShapeH);
    tilingData_.set_orgWi(cacheInputArgs_.inputShapeW);
    tilingData_.set_orgHo(cacheInputArgs_.outputShapeH);
    tilingData_.set_orgWo(cacheInputArgs_.outputShapeW);
    tilingData_.set_singleCoreBatch(cachedTilingData_.singleCoreBatch);
    tilingData_.set_singleCoreHo(cachedTilingData_.singleCoreHo);
    tilingData_.set_singleCoreWo(cachedTilingData_.singleCoreWo);
    tilingData_.set_orgCi(shapeInfo_.ci);
    tilingData_.set_orgCo(cacheInputArgs_.weightShapeN);
    tilingData_.set_singleCoreCi(cachedTilingData_.singleCoreCi);
    tilingData_.set_singleCoreCo(cachedTilingData_.singleCoreCo);
    tilingData_.set_hoL1(cachedTilingData_.hoL1);
    tilingData_.set_woL1(cachedTilingData_.woL1);
    tilingData_.set_kAL1(cachedTilingData_.kAL1);
    tilingData_.set_kBL1(cachedTilingData_.kBL1);
    tilingData_.set_khL1(cachedTilingData_.khL1);
    tilingData_.set_kwL1(cachedTilingData_.kwL1);
    tilingData_.set_nBL1(cachedTilingData_.nBL1);
    tilingData_.set_hoL0(cachedTilingData_.hoL0);
    tilingData_.set_woL0(cachedTilingData_.woL0);
    tilingData_.set_kL0(cachedTilingData_.kL0);
    tilingData_.set_nL0(cachedTilingData_.nL0);
    tilingData_.set_pBufferFlag(cachedTilingData_.pBufferFlag);
    tilingData_.set_groups(cacheInputArgs_.groups);
    tilingData_.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.set_singleCoreGroups(cachedTilingData_.singleCoreGroups);
    tilingData_.set_singleCoreGroupOpt(cachedTilingData_.singleCoreGroupOpt);
    tilingData_.set_bUbNStep(cachedTilingData_.bUbNStep);
    tilingData_.set_bUbKStep(cachedTilingData_.bUbKStep);
    tilingData_.set_khUb(cachedTilingData_.khUb);
    tilingData_.set_kwUb(cachedTilingData_.kwUb);
    tilingData_.set_orgHixWi(cachedTilingData_.orgHixWi);
    tilingData_.set_kernelHxkernelW(cachedTilingData_.kernelHxkernelW);
    tilingData_.set_kernelHxkernelWxkernelD(cachedTilingData_.kernelHxkernelWxkernelD);
    tilingData_.set_aL1SpaceSize(cachedTilingData_.aL1SpaceSize);
    tilingData_.set_multiNBL1(cachedTilingData_.multiNBL1);
    tilingData_.set_cinAInCore(cachedTilingData_.cinAInCore);
    tilingData_.set_cinATailInCore(cachedTilingData_.cinATailInCore);
    tilingData_.set_cinBInCore(cachedTilingData_.cinBInCore);
    tilingData_.set_cinBTailInCore(cachedTilingData_.cinBTailInCore);
    tilingData_.set_mStep(cachedTilingData_.mStep);
    tilingData_.set_kStep(cachedTilingData_.kStep);
    tilingData_.set_nStep(cachedTilingData_.nStep);
    tilingData_.set_fmapKStride(cachedTilingData_.fmapKStride);
    tilingData_.set_weightKStride(cachedTilingData_.weightKStride);
    tilingData_.set_cinOffsetBlockInGM(cachedTilingData_.cinOffsetBlockInGM);
    tilingData_.set_coutOffsetBlock(cachedTilingData_.coutOffsetBlock);
    tilingData_.set_nL1DivBlockSize(cachedTilingData_.nL1DivBlockSize);
    tilingData_.set_kernelH(cacheInputArgs_.weightShapeH);
    tilingData_.set_kernelW(cacheInputArgs_.weightShapeW);
}

void Conv2dBaseTiling::TranslateCachedTilingData()
{
    TranslateCachedRunInfo();
    TranslateCachedApiTilingPartOne();
    TranslateCachedApiTilingPartTwo();
    flagInfo_.enableC04Flag = cachedTilingData_.enableC04Flag;
    flagInfo_.mSplitModeFlag = cachedTilingData_.mSplitModeFlag;
}

} // namespace conv_ops_tiling
} // namespace optiling