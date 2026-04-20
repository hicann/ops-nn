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
    OP_LOGD(context_->GetNodeName(), "%s AscendC: current cache size is %zu.",
            paramInfo_.nodeType.c_str(), tilingCache.GetCacheSize());
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
    cachedTilingData_.fmapKStride = tilingData_.convApiTiling.get_fmapKStride();
    cachedTilingData_.weightKStride = tilingData_.convApiTiling.get_weightKStride();
    cachedTilingData_.cinOffsetBlockInGM = tilingData_.convApiTiling.get_cinOffsetBlockInGM();
    cachedTilingData_.coutOffsetBlock = tilingData_.convApiTiling.get_coutOffsetBlock();
    cachedTilingData_.nL1DivBlockSize = tilingData_.convApiTiling.get_nL1DivBlockSize();
    cachedTilingData_.iterateMNOrder = tilingData_.convApiTiling.get_iterateMNOrder();
    cachedTilingData_.biasFullLoadFlag = tilingData_.convApiTiling.get_biasFullLoadFlag();
    cachedTilingData_.fixpParamsFullLoadFlag = tilingData_.convApiTiling.get_fixpParamsFullLoadFlag();
    cachedTilingData_.hf32Enable = tilingData_.convApiTiling.get_hf32Enable();
    cachedTilingData_.hf32TransMode = tilingData_.convApiTiling.get_hf32TransMode();
    cachedTilingData_.batchDim = tilingData_.convRunInfo.get_batchDim();
    cachedTilingData_.groupDim = tilingData_.convRunInfo.get_groupDim();
    cachedTilingData_.nDim = tilingData_.convRunInfo.get_nDim();
    cachedTilingData_.hoDim = tilingData_.convRunInfo.get_hoDim();
    cachedTilingData_.woDim = tilingData_.convRunInfo.get_woDim();
    cachedTilingData_.cinOpt = tilingData_.convRunInfo.get_cinOpt();
    cachedTilingData_.coutOpt = tilingData_.convRunInfo.get_coutOpt();
    cachedTilingData_.groupOpt = tilingData_.convRunInfo.get_groupOpt();
    cachedTilingData_.enableC04Flag = flagInfo_.enableC04Flag;
    cachedTilingData_.mSplitModeFlag = flagInfo_.mSplitModeFlag;
    cachedTilingData_.unionDataXt = tilingData_.convApiTiling.get_unionDataXt();
}

void Conv2dBaseTiling::GetCachedTilingData()
{
    cachedTilingData_.singleCoreBatch = tilingData_.convApiTiling.get_singleCoreBatch();
    cachedTilingData_.singleCoreHo = tilingData_.convApiTiling.get_singleCoreHo();
    cachedTilingData_.singleCoreWo = tilingData_.convApiTiling.get_singleCoreWo();
    cachedTilingData_.singleCoreCi = tilingData_.convApiTiling.get_singleCoreCi();
    cachedTilingData_.singleCoreCo = tilingData_.convApiTiling.get_singleCoreCo();
    cachedTilingData_.hoL1 = tilingData_.convApiTiling.get_hoL1();
    cachedTilingData_.woL1 = tilingData_.convApiTiling.get_woL1();
    cachedTilingData_.kAL1 = tilingData_.convApiTiling.get_kAL1();
    cachedTilingData_.kBL1 = tilingData_.convApiTiling.get_kBL1();
    cachedTilingData_.khL1 = tilingData_.convApiTiling.get_khL1();
    cachedTilingData_.kwL1 = tilingData_.convApiTiling.get_kwL1();
    cachedTilingData_.nBL1 = tilingData_.convApiTiling.get_nBL1();
    cachedTilingData_.hoL0 = tilingData_.convApiTiling.get_hoL0();
    cachedTilingData_.woL0 = tilingData_.convApiTiling.get_woL0();
    cachedTilingData_.kL0 = tilingData_.convApiTiling.get_kL0();
    cachedTilingData_.nL0 = tilingData_.convApiTiling.get_nL0();
    cachedTilingData_.pBufferFlag = tilingData_.convApiTiling.get_pBufferFlag();
    cachedTilingData_.enlarge = tilingData_.convApiTiling.get_enlarge();
    cachedTilingData_.singleCoreGroups = tilingData_.convApiTiling.get_singleCoreGroups();
    cachedTilingData_.singleCoreGroupOpt = tilingData_.convApiTiling.get_singleCoreGroupOpt();
    cachedTilingData_.bUbNStep = tilingData_.convApiTiling.get_bUbNStep();
    cachedTilingData_.bUbKStep = tilingData_.convApiTiling.get_bUbKStep();
    cachedTilingData_.khUb = tilingData_.convApiTiling.get_khUb();
    cachedTilingData_.kwUb = tilingData_.convApiTiling.get_kwUb();
    cachedTilingData_.orgHixWi = tilingData_.convApiTiling.get_orgHixWi();
    cachedTilingData_.kernelHxkernelW = tilingData_.convApiTiling.get_kernelHxkernelW();
    cachedTilingData_.kernelHxkernelWxkernelD = tilingData_.convApiTiling.get_kernelHxkernelWxkernelD();
    cachedTilingData_.aL1SpaceSize = tilingData_.convApiTiling.get_aL1SpaceSize();
    cachedTilingData_.multiNBL1 = tilingData_.convApiTiling.get_multiNBL1();
    cachedTilingData_.cinAInCore = tilingData_.convApiTiling.get_cinAInCore();
    cachedTilingData_.cinATailInCore = tilingData_.convApiTiling.get_cinATailInCore();
    cachedTilingData_.cinBInCore = tilingData_.convApiTiling.get_cinBInCore();
    cachedTilingData_.cinBTailInCore = tilingData_.convApiTiling.get_cinBTailInCore();
    cachedTilingData_.mStep = tilingData_.convApiTiling.get_mStep();
    cachedTilingData_.kStep = tilingData_.convApiTiling.get_kStep();
    cachedTilingData_.nStep = tilingData_.convApiTiling.get_nStep();
    cachedTilingData_.innerBatch = tilingData_.convApiTiling.get_innerBatch();
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
	tilingData_.convRunInfo.set_batch(cacheInputArgs_.inputShapeN);
    tilingData_.convRunInfo.set_hin(cacheInputArgs_.inputShapeH);
    tilingData_.convRunInfo.set_win(cacheInputArgs_.inputShapeW);
    tilingData_.convRunInfo.set_batchDim(cachedTilingData_.batchDim);
    tilingData_.convRunInfo.set_hoDim(cachedTilingData_.hoDim);
    tilingData_.convRunInfo.set_woDim(cachedTilingData_.woDim);
    tilingData_.convRunInfo.set_nDim(cachedTilingData_.nDim);
    tilingData_.convRunInfo.set_cin(shapeInfo_.ci);
    tilingData_.convRunInfo.set_cout(cacheInputArgs_.weightShapeN);
    tilingData_.convRunInfo.set_kh(cacheInputArgs_.weightShapeH);
    tilingData_.convRunInfo.set_kw(cacheInputArgs_.weightShapeW);
    tilingData_.convRunInfo.set_hout(cacheInputArgs_.outputShapeH);
    tilingData_.convRunInfo.set_wout(cacheInputArgs_.outputShapeW);
    tilingData_.convRunInfo.set_strideH(cacheInputArgs_.strideH);
    tilingData_.convRunInfo.set_strideW(cacheInputArgs_.strideW);
    tilingData_.convRunInfo.set_dilationH(cacheInputArgs_.dilationH);
    tilingData_.convRunInfo.set_dilationW(cacheInputArgs_.dilationW);
    tilingData_.convRunInfo.set_padTop(cacheInputArgs_.padTop);
    tilingData_.convRunInfo.set_padLeft(cacheInputArgs_.padLeft);
    tilingData_.convRunInfo.set_hasBias(cacheInputArgs_.biasFlag);
    tilingData_.convRunInfo.set_groups(cacheInputArgs_.groups);
    tilingData_.convRunInfo.set_cinOpt(cachedTilingData_.cinOpt);
    tilingData_.convRunInfo.set_coutOpt(cachedTilingData_.coutOpt);
    tilingData_.convRunInfo.set_groupOpt(cachedTilingData_.groupOpt);
    tilingData_.convRunInfo.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.convRunInfo.set_groupDim(cachedTilingData_.groupDim);
}

void Conv2dBaseTiling::TranslateCachedApiTilingPartTwo()
{
    tilingData_.convApiTiling.set_strideH(cacheInputArgs_.strideH);
    tilingData_.convApiTiling.set_strideW(cacheInputArgs_.strideW);
    tilingData_.convApiTiling.set_dilationH(cacheInputArgs_.dilationH);
    tilingData_.convApiTiling.set_dilationW(cacheInputArgs_.dilationW);
    tilingData_.convApiTiling.set_padTop(cacheInputArgs_.padTop);
    tilingData_.convApiTiling.set_padBottom(cacheInputArgs_.padBottom);
    tilingData_.convApiTiling.set_padLeft(cacheInputArgs_.padLeft);
    tilingData_.convApiTiling.set_padRight(cacheInputArgs_.padRight);
    tilingData_.convApiTiling.set_iterateMNOrder(cachedTilingData_.iterateMNOrder);
    tilingData_.convApiTiling.set_biasFullLoadFlag(cachedTilingData_.biasFullLoadFlag);
    tilingData_.convApiTiling.set_fixpParamsFullLoadFlag(cachedTilingData_.fixpParamsFullLoadFlag);
    tilingData_.convApiTiling.set_hf32Enable(cachedTilingData_.hf32Enable);
    tilingData_.convApiTiling.set_hf32TransMode(cachedTilingData_.hf32TransMode);
    tilingData_.convApiTiling.set_hasBias(cacheInputArgs_.biasFlag);
    tilingData_.convApiTiling.set_hasScale(static_cast<uint8_t>(flagInfo_.quantFlag || flagInfo_.extendConvFlag));
    tilingData_.convApiTiling.set_offsetx(attrInfo_.offsetx);
    tilingData_.convApiTiling.set_roundMode(attrInfo_.roundMode);
    tilingData_.convApiTiling.set_dualOutput(fixpipeInfo_.dualOutput);
    tilingData_.convApiTiling.set_quantMode0(fixpipeInfo_.quantMode0);
    tilingData_.convApiTiling.set_reluMode0(fixpipeInfo_.reluMode0);
    tilingData_.convApiTiling.set_clipMode0(fixpipeInfo_.clipMode0);
    tilingData_.convApiTiling.set_quantMode1(fixpipeInfo_.quantMode1);
    tilingData_.convApiTiling.set_reluMode1(fixpipeInfo_.reluMode1);
    tilingData_.convApiTiling.set_clipMode1(fixpipeInfo_.clipMode1);
    tilingData_.convApiTiling.set_innerBatch(cachedTilingData_.innerBatch);
    tilingData_.convApiTiling.set_unionDataXt(cachedTilingData_.unionDataXt);
}

void Conv2dBaseTiling::TranslateCachedApiTilingPartOne()
{
    tilingData_.convApiTiling.set_orgHi(cacheInputArgs_.inputShapeH);
    tilingData_.convApiTiling.set_orgWi(cacheInputArgs_.inputShapeW);
    tilingData_.convApiTiling.set_orgHo(cacheInputArgs_.outputShapeH);
    tilingData_.convApiTiling.set_orgWo(cacheInputArgs_.outputShapeW);
    tilingData_.convApiTiling.set_singleCoreBatch(cachedTilingData_.singleCoreBatch);
    tilingData_.convApiTiling.set_singleCoreHo(cachedTilingData_.singleCoreHo);
    tilingData_.convApiTiling.set_singleCoreWo(cachedTilingData_.singleCoreWo);
    tilingData_.convApiTiling.set_orgCi(shapeInfo_.ci);
    tilingData_.convApiTiling.set_orgCo(cacheInputArgs_.weightShapeN);
    tilingData_.convApiTiling.set_singleCoreCi(cachedTilingData_.singleCoreCi);
    tilingData_.convApiTiling.set_singleCoreCo(cachedTilingData_.singleCoreCo);
    tilingData_.convApiTiling.set_hoL1(cachedTilingData_.hoL1);
    tilingData_.convApiTiling.set_woL1(cachedTilingData_.woL1);
    tilingData_.convApiTiling.set_kAL1(cachedTilingData_.kAL1);
    tilingData_.convApiTiling.set_kBL1(cachedTilingData_.kBL1);
    tilingData_.convApiTiling.set_khL1(cachedTilingData_.khL1);
    tilingData_.convApiTiling.set_kwL1(cachedTilingData_.kwL1);
    tilingData_.convApiTiling.set_nBL1(cachedTilingData_.nBL1);
    tilingData_.convApiTiling.set_hoL0(cachedTilingData_.hoL0);
    tilingData_.convApiTiling.set_woL0(cachedTilingData_.woL0);
    tilingData_.convApiTiling.set_kL0(cachedTilingData_.kL0);
    tilingData_.convApiTiling.set_nL0(cachedTilingData_.nL0);
    tilingData_.convApiTiling.set_pBufferFlag(cachedTilingData_.pBufferFlag);
    tilingData_.convApiTiling.set_groups(cacheInputArgs_.groups);
    tilingData_.convApiTiling.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.convApiTiling.set_singleCoreGroups(cachedTilingData_.singleCoreGroups);
    tilingData_.convApiTiling.set_singleCoreGroupOpt(cachedTilingData_.singleCoreGroupOpt);
    tilingData_.convApiTiling.set_bUbNStep(cachedTilingData_.bUbNStep);
    tilingData_.convApiTiling.set_bUbKStep(cachedTilingData_.bUbKStep);
    tilingData_.convApiTiling.set_khUb(cachedTilingData_.khUb);
    tilingData_.convApiTiling.set_kwUb(cachedTilingData_.kwUb);
    tilingData_.convApiTiling.set_orgHixWi(cachedTilingData_.orgHixWi);
    tilingData_.convApiTiling.set_kernelHxkernelW(cachedTilingData_.kernelHxkernelW);
    tilingData_.convApiTiling.set_kernelHxkernelWxkernelD(cachedTilingData_.kernelHxkernelWxkernelD);
    tilingData_.convApiTiling.set_aL1SpaceSize(cachedTilingData_.aL1SpaceSize);
    tilingData_.convApiTiling.set_multiNBL1(cachedTilingData_.multiNBL1);
    tilingData_.convApiTiling.set_cinAInCore(cachedTilingData_.cinAInCore);
    tilingData_.convApiTiling.set_cinATailInCore(cachedTilingData_.cinATailInCore);
    tilingData_.convApiTiling.set_cinBInCore(cachedTilingData_.cinBInCore);
    tilingData_.convApiTiling.set_cinBTailInCore(cachedTilingData_.cinBTailInCore);
    tilingData_.convApiTiling.set_mStep(cachedTilingData_.mStep);
    tilingData_.convApiTiling.set_kStep(cachedTilingData_.kStep);
    tilingData_.convApiTiling.set_nStep(cachedTilingData_.nStep);
    tilingData_.convApiTiling.set_fmapKStride(cachedTilingData_.fmapKStride);
    tilingData_.convApiTiling.set_weightKStride(cachedTilingData_.weightKStride);
    tilingData_.convApiTiling.set_cinOffsetBlockInGM(cachedTilingData_.cinOffsetBlockInGM);
    tilingData_.convApiTiling.set_coutOffsetBlock(cachedTilingData_.coutOffsetBlock);
    tilingData_.convApiTiling.set_nL1DivBlockSize(cachedTilingData_.nL1DivBlockSize);
    tilingData_.convApiTiling.set_kernelH(cacheInputArgs_.weightShapeH);
    tilingData_.convApiTiling.set_kernelW(cacheInputArgs_.weightShapeW);
}

void Conv2dBaseTiling::TranslateCachedTilingData()
{
    TranslateCachedRunInfo();
    TranslateCachedApiTilingPartOne();
    TranslateCachedApiTilingPartTwo();
    flagInfo_.enableC04Flag = cachedTilingData_.enableC04Flag;
    flagInfo_.mSplitModeFlag = cachedTilingData_.mSplitModeFlag;
}

}
}