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
    if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
        return false;
    }

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

    GetCachedTilingDataPartOne();
    GetCachedTilingDataPartTwo();
    GetCachedTilingDataPartThree();

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

void Conv3dBaseTilingV2::GetCachedTilingDataPartOne()
{
    cachedTilingData_.orgDo = tilingData_.conv3dApiTiling.get_orgDo();
    cachedTilingData_.orgHo = tilingData_.conv3dApiTiling.get_orgHo();
    cachedTilingData_.orgWo = tilingData_.conv3dApiTiling.get_orgWo();
    cachedTilingData_.orgDi = tilingData_.conv3dApiTiling.get_orgDi();
    cachedTilingData_.orgHi = tilingData_.conv3dApiTiling.get_orgHi();
    cachedTilingData_.orgWi = tilingData_.conv3dApiTiling.get_orgWi();
    cachedTilingData_.singleCoreBatch = tilingData_.conv3dApiTiling.get_singleCoreBatch();
    cachedTilingData_.singleCoreDo = tilingData_.conv3dApiTiling.get_singleCoreDo();
    cachedTilingData_.singleCoreM = tilingData_.conv3dApiTiling.get_singleCoreM();
    cachedTilingData_.singleCoreWo = tilingData_.conv3dApiTiling.get_singleCoreWo();
    cachedTilingData_.singleCoreHo = tilingData_.conv3dApiTiling.get_singleCoreHo();
    cachedTilingData_.kL0xorgCoAlignN0 = tilingData_.conv3dApiTiling.get_kL0xorgCoAlignN0();
    cachedTilingData_.kernelHxkernelW = tilingData_.conv3dApiTiling.get_kernelHxkernelW();
    cachedTilingData_.cin1xOriHixOriWixk0 = tilingData_.conv3dApiTiling.get_cin1xOriHixOriWixk0();
    cachedTilingData_.oriHixOriWixk0 = tilingData_.conv3dApiTiling.get_oriHixOriWixk0();
    cachedTilingData_.oriWixk0 = tilingData_.conv3dApiTiling.get_oriWixk0();
    cachedTilingData_.orgHixWi = tilingData_.conv3dApiTiling.get_orgHixWi();
    cachedTilingData_.orgHoxWo = tilingData_.conv3dApiTiling.get_orgHoxWo();
    cachedTilingData_.orgCi = tilingData_.conv3dApiTiling.get_orgCi();
    cachedTilingData_.kernelD = tilingData_.conv3dApiTiling.get_kernelD();
    cachedTilingData_.kernelH = tilingData_.conv3dApiTiling.get_kernelH();
    cachedTilingData_.kernelW = tilingData_.conv3dApiTiling.get_kernelW();
    cachedTilingData_.singleCoreCo = tilingData_.conv3dApiTiling.get_singleCoreCo();
    cachedTilingData_.orgCo = tilingData_.conv3dApiTiling.get_orgCo();
    cachedTilingData_.singleCoreCi = tilingData_.conv3dApiTiling.get_singleCoreCi();
    cachedTilingData_.singleCoreGroups = tilingData_.conv3dApiTiling.get_singleCoreGroups();
    cachedTilingData_.singleCoreGroupOpt = tilingData_.conv3dApiTiling.get_singleCoreGroupOpt();
    cachedTilingData_.groups = tilingData_.conv3dApiTiling.get_groups();
    cachedTilingData_.enlarge = tilingData_.conv3dApiTiling.get_enlarge();
    cachedTilingData_.strideH = tilingData_.conv3dApiTiling.get_strideH();
    cachedTilingData_.strideW = tilingData_.conv3dApiTiling.get_strideW();
    cachedTilingData_.strideD = tilingData_.conv3dApiTiling.get_strideD();
    cachedTilingData_.dilationH = tilingData_.conv3dApiTiling.get_dilationH();
    cachedTilingData_.dilationW = tilingData_.conv3dApiTiling.get_dilationW();
    cachedTilingData_.dilationD = tilingData_.conv3dApiTiling.get_dilationD();
    cachedTilingData_.padHead = tilingData_.conv3dApiTiling.get_padHead();
    cachedTilingData_.padTail = tilingData_.conv3dApiTiling.get_padTail();
    cachedTilingData_.padTop = tilingData_.conv3dApiTiling.get_padTop();
    cachedTilingData_.padBottom = tilingData_.conv3dApiTiling.get_padBottom();
    cachedTilingData_.padLeft = tilingData_.conv3dApiTiling.get_padLeft();
    cachedTilingData_.padRight = tilingData_.conv3dApiTiling.get_padRight();
}

void Conv3dBaseTilingV2::GetCachedTilingDataPartTwo()
{
    cachedTilingData_.mL0 = tilingData_.conv3dApiTiling.get_mL0();
    cachedTilingData_.woL0 = tilingData_.conv3dApiTiling.get_woL0();
    cachedTilingData_.kL0 = tilingData_.conv3dApiTiling.get_kL0();
    cachedTilingData_.nL0 = tilingData_.conv3dApiTiling.get_nL0();
    cachedTilingData_.kAL1 = tilingData_.conv3dApiTiling.get_kAL1();
    cachedTilingData_.kAL1Tail = tilingData_.conv3dApiTiling.get_kAL1Tail();
    cachedTilingData_.kBL1 = tilingData_.conv3dApiTiling.get_kBL1();
    cachedTilingData_.kBL1Tail = tilingData_.conv3dApiTiling.get_kBL1Tail();
    cachedTilingData_.nBL1 = tilingData_.conv3dApiTiling.get_nBL1();
    cachedTilingData_.mAL1 = tilingData_.conv3dApiTiling.get_mAL1();
    cachedTilingData_.woL1 = tilingData_.conv3dApiTiling.get_woL1();
    cachedTilingData_.hoL0 = tilingData_.conv3dApiTiling.get_hoL0();
    cachedTilingData_.hoL1 = tilingData_.conv3dApiTiling.get_hoL1();
    cachedTilingData_.KBL1Divk0 = tilingData_.conv3dApiTiling.get_KBL1Divk0();
    cachedTilingData_.KBL1TailDivk0 = tilingData_.conv3dApiTiling.get_KBL1TailDivk0();
    cachedTilingData_.nBL1DivnL0 = tilingData_.conv3dApiTiling.get_nBL1DivnL0();
    cachedTilingData_.mAL1DivmL0 = tilingData_.conv3dApiTiling.get_mAL1DivmL0();
    cachedTilingData_.fmapKStride = tilingData_.conv3dApiTiling.get_fmapKStride();
    cachedTilingData_.weightKStride = tilingData_.conv3dApiTiling.get_weightKStride();
    cachedTilingData_.cinOffsetBlockInGM = tilingData_.conv3dApiTiling.get_cinOffsetBlockInGM();
    cachedTilingData_.coutOffsetBlock = tilingData_.conv3dApiTiling.get_coutOffsetBlock();
    cachedTilingData_.nL1DivBlockSize = tilingData_.conv3dApiTiling.get_nL1DivBlockSize();
    cachedTilingData_.cin1InAL1 = tilingData_.conv3dApiTiling.get_cin1InAL1();
    cachedTilingData_.cin1InAL1Tail = tilingData_.conv3dApiTiling.get_cin1InAL1Tail();
    cachedTilingData_.cinBInCore = tilingData_.conv3dApiTiling.get_cinBInCore();
    cachedTilingData_.cinBTailInCore = tilingData_.conv3dApiTiling.get_cinBTailInCore();
    cachedTilingData_.cinAInCore = tilingData_.conv3dApiTiling.get_cinAInCore();
    cachedTilingData_.cinATailInCore = tilingData_.conv3dApiTiling.get_cinATailInCore();
    cachedTilingData_.nL0xk0 = tilingData_.conv3dApiTiling.get_nL0xk0();
    cachedTilingData_.mStep = tilingData_.conv3dApiTiling.get_mStep();
    cachedTilingData_.kStep = tilingData_.conv3dApiTiling.get_kStep();
    cachedTilingData_.nStep = tilingData_.conv3dApiTiling.get_nStep();
    cachedTilingData_.aL1SpaceSize = tilingData_.conv3dApiTiling.get_aL1SpaceSize();
    cachedTilingData_.multiNBL1 = tilingData_.conv3dApiTiling.get_multiNBL1();
    cachedTilingData_.pBufferFlag = tilingData_.conv3dApiTiling.get_pBufferFlag();
}

void Conv3dBaseTilingV2::GetCachedTilingDataPartThree()
{
    cachedTilingData_.mUB = tilingData_.conv3dApiTiling.get_mUB();
    cachedTilingData_.nUB = tilingData_.conv3dApiTiling.get_nUB();
    cachedTilingData_.scaleAndBiasLoadType = tilingData_.conv3dApiTiling.get_scaleAndBiasLoadType();
    cachedTilingData_.workspaceSize = tilingData_.conv3dApiTiling.get_workspaceSize();
    cachedTilingData_.kernelHxkernelWxkernelD = tilingData_.conv3dApiTiling.get_kernelHxkernelWxkernelD();
    cachedTilingData_.offsetx = tilingData_.conv3dApiTiling.get_offsetx();
    cachedTilingData_.roundMode = tilingData_.conv3dApiTiling.get_roundMode();
    cachedTilingData_.hasBias = tilingData_.conv3dApiTiling.get_hasBias();
    cachedTilingData_.hasScale = tilingData_.conv3dApiTiling.get_hasScale();
    cachedTilingData_.bl1FullLoad = tilingData_.conv3dApiTiling.get_bl1FullLoad();
    cachedTilingData_.al1FullLoad = tilingData_.conv3dApiTiling.get_al1FullLoad();
    cachedTilingData_.bl1BypassFlag = tilingData_.conv3dApiTiling.get_bl1BypassFlag();
    cachedTilingData_.iterateMNOrder = tilingData_.conv3dApiTiling.get_iterateMNOrder();
    cachedTilingData_.biasFullLoadFlag = tilingData_.conv3dApiTiling.get_biasFullLoadFlag();
    cachedTilingData_.fixpParamsFullLoadFlag = tilingData_.conv3dApiTiling.get_fixpParamsFullLoadFlag();
    cachedTilingData_.hf32Enable = tilingData_.conv3dApiTiling.get_hf32Enable();
    cachedTilingData_.hf32TransMode = tilingData_.conv3dApiTiling.get_hf32TransMode();
    cachedTilingData_.quantType = tilingData_.conv3dApiTiling.get_quantType();
    cachedTilingData_.batch = tilingData_.conv3dRunInfo.get_batch();
    cachedTilingData_.batchDim = tilingData_.conv3dRunInfo.get_batchDim();
    cachedTilingData_.doDim = tilingData_.conv3dRunInfo.get_doDim();
    cachedTilingData_.mDim = tilingData_.conv3dRunInfo.get_mDim();
    cachedTilingData_.wDim = tilingData_.conv3dRunInfo.get_wDim();
    cachedTilingData_.nDim = tilingData_.conv3dRunInfo.get_nDim();
    cachedTilingData_.groupDim = tilingData_.conv3dRunInfo.get_groupDim();
    cachedTilingData_.hoDim = tilingData_.conv3dRunInfo.get_hoDim();
    cachedTilingData_.groupOpt = tilingData_.conv3dRunInfo.get_groupOpt();
    cachedTilingData_.cinOpt = tilingData_.conv3dRunInfo.get_cinOpt();
    cachedTilingData_.coutOpt = tilingData_.conv3dRunInfo.get_coutOpt();
    cachedTilingData_.mSplitModeFlag = flagInfo_.mSplitModeFlag;
}

void Conv3dBaseTilingV2::TranslateCachedRunInfo()
{
    tilingData_.conv3dRunInfo.set_batch(cachedTilingData_.batch);
    tilingData_.conv3dRunInfo.set_cin(cachedTilingData_.orgCi);
    tilingData_.conv3dRunInfo.set_din(cachedTilingData_.orgDi);
    tilingData_.conv3dRunInfo.set_hin(cachedTilingData_.orgHi);
    tilingData_.conv3dRunInfo.set_win(cachedTilingData_.orgWi);
    tilingData_.conv3dRunInfo.set_cout(cachedTilingData_.orgCo);
    tilingData_.conv3dRunInfo.set_kd(cachedTilingData_.kernelD);
    tilingData_.conv3dRunInfo.set_kh(cachedTilingData_.kernelH);
    tilingData_.conv3dRunInfo.set_kw(cachedTilingData_.kernelW);
    tilingData_.conv3dRunInfo.set_dout(cachedTilingData_.orgDo);
    tilingData_.conv3dRunInfo.set_hout(cachedTilingData_.orgHo);
    tilingData_.conv3dRunInfo.set_wout(cachedTilingData_.orgWo);
    tilingData_.conv3dRunInfo.set_batchDim(cachedTilingData_.batchDim);
    tilingData_.conv3dRunInfo.set_doDim(cachedTilingData_.doDim);
    tilingData_.conv3dRunInfo.set_mDim(cachedTilingData_.mDim);
    tilingData_.conv3dRunInfo.set_wDim(cachedTilingData_.wDim);
    tilingData_.conv3dRunInfo.set_nDim(cachedTilingData_.nDim);
    tilingData_.conv3dRunInfo.set_groupDim(cachedTilingData_.groupDim);
    tilingData_.conv3dRunInfo.set_hoDim(cachedTilingData_.hoDim);
    tilingData_.conv3dRunInfo.set_strideH(cachedTilingData_.strideH);
    tilingData_.conv3dRunInfo.set_strideW(cachedTilingData_.strideW);
    tilingData_.conv3dRunInfo.set_strideD(cachedTilingData_.strideD);
    tilingData_.conv3dRunInfo.set_dilationH(cachedTilingData_.dilationH);
    tilingData_.conv3dRunInfo.set_dilationW(cachedTilingData_.dilationW);
    tilingData_.conv3dRunInfo.set_dilationD(cachedTilingData_.dilationD);
    tilingData_.conv3dRunInfo.set_padHead(cachedTilingData_.padHead);
    tilingData_.conv3dRunInfo.set_padTail(cachedTilingData_.padTail);
    tilingData_.conv3dRunInfo.set_padTop(cachedTilingData_.padTop);
    tilingData_.conv3dRunInfo.set_padBottom(cachedTilingData_.padBottom);
    tilingData_.conv3dRunInfo.set_padLeft(cachedTilingData_.padLeft);
    tilingData_.conv3dRunInfo.set_padRight(cachedTilingData_.padRight);
    tilingData_.conv3dRunInfo.set_groups(cachedTilingData_.groups);
    tilingData_.conv3dRunInfo.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.conv3dRunInfo.set_cinOpt(cachedTilingData_.cinOpt);
    tilingData_.conv3dRunInfo.set_coutOpt(cachedTilingData_.coutOpt);
    tilingData_.conv3dRunInfo.set_groupOpt(cachedTilingData_.groupOpt);
    tilingData_.conv3dRunInfo.set_hasBias(cachedTilingData_.hasBias);
}

void Conv3dBaseTilingV2::TranslateCachedApiTilingPartOne()
{
    tilingData_.conv3dApiTiling.set_orgDo(cachedTilingData_.orgDo);
    tilingData_.conv3dApiTiling.set_orgHo(cachedTilingData_.orgHo);
    tilingData_.conv3dApiTiling.set_orgWo(cachedTilingData_.orgWo);
    tilingData_.conv3dApiTiling.set_orgDi(cachedTilingData_.orgDi);
    tilingData_.conv3dApiTiling.set_orgHi(cachedTilingData_.orgHi);
    tilingData_.conv3dApiTiling.set_orgWi(cachedTilingData_.orgWi);
    tilingData_.conv3dApiTiling.set_singleCoreBatch(cachedTilingData_.singleCoreBatch);
    tilingData_.conv3dApiTiling.set_singleCoreDo(cachedTilingData_.singleCoreDo);
    tilingData_.conv3dApiTiling.set_singleCoreM(cachedTilingData_.singleCoreM);
    tilingData_.conv3dApiTiling.set_singleCoreWo(cachedTilingData_.singleCoreWo);
    tilingData_.conv3dApiTiling.set_singleCoreHo(cachedTilingData_.singleCoreHo);
    tilingData_.conv3dApiTiling.set_kL0xorgCoAlignN0(cachedTilingData_.kL0xorgCoAlignN0);
    tilingData_.conv3dApiTiling.set_kernelHxkernelW(cachedTilingData_.kernelHxkernelW);
    tilingData_.conv3dApiTiling.set_cin1xOriHixOriWixk0(cachedTilingData_.cin1xOriHixOriWixk0);
    tilingData_.conv3dApiTiling.set_oriHixOriWixk0(cachedTilingData_.oriHixOriWixk0);
    tilingData_.conv3dApiTiling.set_oriWixk0(cachedTilingData_.oriWixk0);
    tilingData_.conv3dApiTiling.set_orgHixWi(cachedTilingData_.orgHixWi);
    tilingData_.conv3dApiTiling.set_orgHoxWo(cachedTilingData_.orgHoxWo);
    tilingData_.conv3dApiTiling.set_orgCi(cachedTilingData_.orgCi);
    tilingData_.conv3dApiTiling.set_kernelD(cachedTilingData_.kernelD);
    tilingData_.conv3dApiTiling.set_kernelH(cachedTilingData_.kernelH);
    tilingData_.conv3dApiTiling.set_kernelW(cachedTilingData_.kernelW);
    tilingData_.conv3dApiTiling.set_singleCoreCo(cachedTilingData_.singleCoreCo);
    tilingData_.conv3dApiTiling.set_orgCo(cachedTilingData_.orgCo);
    tilingData_.conv3dApiTiling.set_singleCoreCi(cachedTilingData_.singleCoreCi);
    tilingData_.conv3dApiTiling.set_singleCoreGroups(cachedTilingData_.singleCoreGroups);
    tilingData_.conv3dApiTiling.set_singleCoreGroupOpt(cachedTilingData_.singleCoreGroupOpt);
    tilingData_.conv3dApiTiling.set_groups(cachedTilingData_.groups);
    tilingData_.conv3dApiTiling.set_enlarge(cachedTilingData_.enlarge);
    tilingData_.conv3dApiTiling.set_strideH(cachedTilingData_.strideH);
    tilingData_.conv3dApiTiling.set_strideW(cachedTilingData_.strideW);
    tilingData_.conv3dApiTiling.set_strideD(cachedTilingData_.strideD);
    tilingData_.conv3dApiTiling.set_dilationH(cachedTilingData_.dilationH);
    tilingData_.conv3dApiTiling.set_dilationW(cachedTilingData_.dilationW);
    tilingData_.conv3dApiTiling.set_dilationD(cachedTilingData_.dilationD);
    tilingData_.conv3dApiTiling.set_padHead(cachedTilingData_.padHead);
    tilingData_.conv3dApiTiling.set_padTail(cachedTilingData_.padTail);
    tilingData_.conv3dApiTiling.set_padTop(cachedTilingData_.padTop);
    tilingData_.conv3dApiTiling.set_padBottom(cachedTilingData_.padBottom);
    tilingData_.conv3dApiTiling.set_padLeft(cachedTilingData_.padLeft);
    tilingData_.conv3dApiTiling.set_padRight(cachedTilingData_.padRight);
}

void Conv3dBaseTilingV2::TranslateCachedApiTilingPartTwo()
{
    tilingData_.conv3dApiTiling.set_mL0(cachedTilingData_.mL0);
    tilingData_.conv3dApiTiling.set_woL0(cachedTilingData_.woL0);
    tilingData_.conv3dApiTiling.set_kL0(cachedTilingData_.kL0);
    tilingData_.conv3dApiTiling.set_nL0(cachedTilingData_.nL0);
    tilingData_.conv3dApiTiling.set_kAL1(cachedTilingData_.kAL1);
    tilingData_.conv3dApiTiling.set_kAL1Tail(cachedTilingData_.kAL1Tail);
    tilingData_.conv3dApiTiling.set_kBL1(cachedTilingData_.kBL1);
    tilingData_.conv3dApiTiling.set_kBL1Tail(cachedTilingData_.kBL1Tail);
    tilingData_.conv3dApiTiling.set_nBL1(cachedTilingData_.nBL1);
    tilingData_.conv3dApiTiling.set_mAL1(cachedTilingData_.mAL1);
    tilingData_.conv3dApiTiling.set_woL1(cachedTilingData_.woL1);
    tilingData_.conv3dApiTiling.set_hoL0(cachedTilingData_.hoL0);
    tilingData_.conv3dApiTiling.set_hoL1(cachedTilingData_.hoL1);
    tilingData_.conv3dApiTiling.set_KBL1Divk0(cachedTilingData_.KBL1Divk0);
    tilingData_.conv3dApiTiling.set_KBL1TailDivk0(cachedTilingData_.KBL1TailDivk0);
    tilingData_.conv3dApiTiling.set_nBL1DivnL0(cachedTilingData_.nBL1DivnL0);
    tilingData_.conv3dApiTiling.set_mAL1DivmL0(cachedTilingData_.mAL1DivmL0);
    tilingData_.conv3dApiTiling.set_fmapKStride(cachedTilingData_.fmapKStride);
    tilingData_.conv3dApiTiling.set_weightKStride(cachedTilingData_.weightKStride);
    tilingData_.conv3dApiTiling.set_cinOffsetBlockInGM(cachedTilingData_.cinOffsetBlockInGM);
    tilingData_.conv3dApiTiling.set_coutOffsetBlock(cachedTilingData_.coutOffsetBlock);
    tilingData_.conv3dApiTiling.set_nL1DivBlockSize(cachedTilingData_.nL1DivBlockSize);
    tilingData_.conv3dApiTiling.set_cin1InAL1(cachedTilingData_.cin1InAL1);
    tilingData_.conv3dApiTiling.set_cin1InAL1Tail(cachedTilingData_.cin1InAL1Tail);
    tilingData_.conv3dApiTiling.set_cinBInCore(cachedTilingData_.cinBInCore);
    tilingData_.conv3dApiTiling.set_cinBTailInCore(cachedTilingData_.cinBTailInCore);
    tilingData_.conv3dApiTiling.set_cinAInCore(cachedTilingData_.cinAInCore);
    tilingData_.conv3dApiTiling.set_cinATailInCore(cachedTilingData_.cinATailInCore);
    tilingData_.conv3dApiTiling.set_nL0xk0(cachedTilingData_.nL0xk0);
    tilingData_.conv3dApiTiling.set_mStep(cachedTilingData_.mStep);
    tilingData_.conv3dApiTiling.set_kStep(cachedTilingData_.kStep);
    tilingData_.conv3dApiTiling.set_nStep(cachedTilingData_.nStep);
    tilingData_.conv3dApiTiling.set_aL1SpaceSize(cachedTilingData_.aL1SpaceSize);
    tilingData_.conv3dApiTiling.set_multiNBL1(cachedTilingData_.multiNBL1);
    tilingData_.conv3dApiTiling.set_pBufferFlag(cachedTilingData_.pBufferFlag);
    tilingData_.conv3dApiTiling.set_groupOpt(cachedTilingData_.groupOpt);
    tilingData_.conv3dApiTiling.set_cinOpt(cachedTilingData_.cinOpt);
    tilingData_.conv3dApiTiling.set_coutOpt(cachedTilingData_.coutOpt);
}

void Conv3dBaseTilingV2::TranslateCachedApiTilingPartThree()
{
    tilingData_.conv3dApiTiling.set_mUB(cachedTilingData_.mUB);
    tilingData_.conv3dApiTiling.set_nUB(cachedTilingData_.nUB);
    tilingData_.conv3dApiTiling.set_scaleAndBiasLoadType(cachedTilingData_.scaleAndBiasLoadType);
    tilingData_.conv3dApiTiling.set_workspaceSize(cachedTilingData_.workspaceSize);
    tilingData_.conv3dApiTiling.set_kernelHxkernelWxkernelD(cachedTilingData_.kernelHxkernelWxkernelD);
    tilingData_.conv3dApiTiling.set_offsetx(cachedTilingData_.offsetx);
    tilingData_.conv3dApiTiling.set_roundMode(cachedTilingData_.roundMode);
    tilingData_.conv3dApiTiling.set_hasBias(cachedTilingData_.hasBias);
    tilingData_.conv3dApiTiling.set_hasScale(cachedTilingData_.hasScale);
    tilingData_.conv3dApiTiling.set_bl1FullLoad(cachedTilingData_.bl1FullLoad);
    tilingData_.conv3dApiTiling.set_al1FullLoad(cachedTilingData_.al1FullLoad);
    tilingData_.conv3dApiTiling.set_bl1BypassFlag(cachedTilingData_.bl1BypassFlag);
    tilingData_.conv3dApiTiling.set_iterateMNOrder(cachedTilingData_.iterateMNOrder);
    tilingData_.conv3dApiTiling.set_biasFullLoadFlag(cachedTilingData_.biasFullLoadFlag);
    tilingData_.conv3dApiTiling.set_fixpParamsFullLoadFlag(cachedTilingData_.fixpParamsFullLoadFlag);
    tilingData_.conv3dApiTiling.set_hf32Enable(cachedTilingData_.hf32Enable);
    tilingData_.conv3dApiTiling.set_hf32TransMode(cachedTilingData_.hf32TransMode);
    tilingData_.conv3dApiTiling.set_quantType(cachedTilingData_.quantType);
}

void Conv3dBaseTilingV2::TranslateCachedTilingData()
{
    TranslateCachedRunInfo();
    TranslateCachedApiTilingPartOne();
    TranslateCachedApiTilingPartTwo();
    TranslateCachedApiTilingPartThree();

    tilingData_.conv3dApiTiling.set_hasScale(static_cast<uint8_t>(flagInfo_.quantFlag));
    tilingData_.conv3dApiTiling.set_offsetx(attrInfo_.offsetx);
    tilingData_.conv3dApiTiling.set_roundMode(attrInfo_.roundMode);
    flagInfo_.mSplitModeFlag = cachedTilingData_.mSplitModeFlag;
}

}
}