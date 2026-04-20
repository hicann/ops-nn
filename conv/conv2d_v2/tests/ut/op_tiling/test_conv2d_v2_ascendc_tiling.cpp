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
 * \file test_conv2d_v2_ascendc_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace conv_tiling_utils;

namespace {
struct TilingParam {
    // api tilingdata
    uint64_t orgHi;
    uint64_t orgWi;
    uint64_t orgHo;
    uint64_t orgWo;
    uint64_t oriHinxWin;
    uint64_t singleCoreBatch;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;
    uint64_t UnionDataXt;
    uint32_t orgCi;
    uint32_t orgCo;
    uint32_t singleCoreCi;
    uint32_t singleCoreCo;
    uint32_t hoL1;
    uint32_t woL1;
    uint32_t kAL1;
    uint32_t kBL1;
    uint32_t khL1;
    uint32_t kwL1;
    uint32_t nBL1;
    uint32_t hoL0;
    uint32_t woL0;
    uint32_t kL0;
    uint32_t nL0;
    uint32_t pBufferFlag;
    uint32_t groups_api;
    uint32_t enlarge_api;
    uint32_t singleCoreGroups;
    uint32_t singleCoreGroupOpt;
    uint32_t bUbNStep;
    uint32_t bUbKStep;
    uint32_t khUb;
    uint32_t kwUb;
    uint32_t kernelHxkernelW;
    uint32_t kernelHxkernelWxkernelD;
    uint32_t aL1SpaceSize;
    uint32_t multiNBL1;
    uint32_t cinAInCore;
    uint32_t cinATailInCore;
    uint32_t cinBInCore;
    uint32_t cinBTailInCore;
    uint32_t mStep;
    uint32_t kStep;
    uint32_t nStep;
    uint32_t fmapKStride;
    uint32_t weightKStride;
    uint32_t cinOffsetBlockInGM;
    uint32_t coutOffsetBlock;
    uint32_t nL1DivBlockSize;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH_api;
    uint32_t strideW_api;
    uint32_t dilationH_api;
    uint32_t dilationW_api;
    uint32_t padTop_api;
    uint32_t padBottom;
    uint32_t padLeft_api;
    uint32_t padRight;
    uint32_t innerBatch;
    uint8_t iterateMNOrder;
    uint8_t biasFullLoadFlag;
    uint8_t fixpParamsFullLoadFlag;
    uint8_t hf32Enable;
    uint8_t hf32TransMode;
    uint8_t hasBias_api;
    uint8_t hasScale;
    uint8_t dualOutput;
    uint8_t quantMode0;
    uint8_t reluMode0;
    uint8_t clipMode0;
    uint8_t quantMode1;
    uint8_t reluMode1;
    uint8_t clipMode1;
    int8_t offsetx;
    int8_t roundMode;
    // ops tilingdata
    uint64_t hin;
    uint64_t win;
    uint64_t hout;
    uint64_t wout;
    uint32_t batch;
    uint32_t cin;
    uint32_t cout;
    uint32_t kh;
    uint32_t kw;
    uint32_t batchDim;
    uint32_t groupDim;
    uint32_t nDim;
    uint32_t hoDim;
    uint32_t woDim;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t padTop;
    uint32_t padLeft;
    uint32_t groups;
    uint32_t enlarge;
    uint32_t cinOpt;
    uint32_t coutOpt;
    uint32_t groupOpt;
    uint8_t hasBias;
};

struct DtypeSize {
    uint32_t fMapDtypeSize;
    uint32_t weightDtypeSize;
    uint32_t biasDtypeSize;
    uint32_t scaleDtypeSize;
};

struct KParmas {
    uint32_t kAL1min;
    uint32_t kBL1min;
};

struct BasicParams {
    int64_t availableL1Size;
    uint64_t basicBlockM;
    uint64_t basicBlockN;
    DtypeSize dtypeSize;
};

struct PadModeParams {
    const string padMode;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    int64_t batch;
    int64_t cin;
    int64_t hi;
    int64_t wi;
    int64_t cout;
    int64_t kH;
    int64_t kW;
};

bool isConv1dFlag(TilingParam &tilingData, bool isC04Mode)
{
    if (isC04Mode) {
        return false;
    }
    if (tilingData.orgHi == 1 && tilingData.kernelH == 1 && tilingData.strideH == 1 && tilingData.dilationH == 1 &&
        tilingData.padTop == 0 && tilingData.padBottom == 0) {
        return true;
    }
    return false;
}

uint64_t CalcMinUsedL1SizeInMsplitMode(TilingParam &tilingData, KParmas kParmas,
   DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t kAL1min = kParmas.kAL1min;
    uint32_t kBL1min = kParmas.kBL1min;
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    uint32_t biasDtypeSize = dtypeSizes.biasDtypeSize;
    uint32_t weightDtypeSize = dtypeSizes.weightDtypeSize;
    uint32_t scaleDtypeSize = dtypeSizes.scaleDtypeSize;
    uint64_t nBL1min = CUBE_N0;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * biasDtypeSize, CUBE_C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * scaleDtypeSize, CUBE_C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, CUBE_C0_SIZE);
    uint64_t hoAL1min = std::min(CUBE_M0 / tilingData.orgWo + NUM_2, tilingData.orgHo);
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH,
                                    tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * tilingData.orgWi * kAL1min * fMapDtypeSize, CUBE_C0_SIZE);
    uint64_t minL1LoadSize = biasUsedL1Size + fmapUsedL1Size + weightUsedL1Size + scaleUsedL1Size;
    return minL1LoadSize;
}

uint64_t CalcMinUsedL1SizeInHWsplitMode(TilingParam &tilingData, KParmas kParmas, uint32_t wiAL1min,
   DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t kAL1min = kParmas.kAL1min;
    uint32_t kBL1min = kParmas.kBL1min;
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    uint32_t biasDtypeSize = dtypeSizes.biasDtypeSize;
    uint32_t weightDtypeSize = dtypeSizes.weightDtypeSize;
    uint32_t scaleDtypeSize = dtypeSizes.scaleDtypeSize;
    uint64_t nBL1min = CUBE_N0;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * biasDtypeSize, CUBE_C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * scaleDtypeSize, CUBE_C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, CUBE_C0_SIZE);
    uint64_t hoAL1min = tilingData.orgWo < CUBE_M0 ? ConvCeilDiv(CUBE_M0, tilingData.orgWo) : 1;
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH,
                                    tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, CUBE_C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    return minL1LoadSize;
}

bool CheckL1SizeLimitsInHWsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    // require hiL1 * wiL1 >= CUBE_M0
    uint64_t woAL1min = CUBE_M0;
    uint32_t k0 = CUBE_C0_SIZE / fMapDtypeSize;
    uint64_t wiAL1min = InferWiL1(woAL1min, tilingData.orgWi, tilingData.kernelW,
        tilingData.dilationW, tilingData.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(tilingData, KParmas{k0, tilingData.kernelH * tilingData.kernelW * k0}, wiAL1min, dtypeSizes, hasScale);
    if (usdL1SizeUnderMinHWtiling > MEM_SIZE_512K) {
        return false;
    }
    return true;
}
 
bool CheckL1SizeLimitsInMsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    uint32_t k0 = CUBE_C0_SIZE / fMapDtypeSize;
    KParmas kParmas = {k0, tilingData.kernelH * tilingData.kernelW * k0};
    uint64_t usdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, kParmas, dtypeSizes, hasScale);
    if (usdL1SizeUnderMinMtiling > MEM_SIZE_512K) {
        return false;
    }
    return true;
}

bool CheckC04L1SizeLimitsInHWsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    // c04 require wi fulload L1
    uint32_t k0 = CUBE_C0_SIZE / fMapDtypeSize;
    KParmas kParmas = {CUBE_C04_SIZE, ConvAlignB(CUBE_C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0)};
    uint64_t usdL1SizeUnderMinHWtiling =
        CalcMinUsedL1SizeInHWsplitMode(tilingData, kParmas, tilingData.orgWi, dtypeSizes, hasScale);
    if (usdL1SizeUnderMinHWtiling > MEM_SIZE_512K) {
        return false;
    }
    return true;
}
 
bool CheckC04L1SizeLimitsInMsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t fMapDtypeSize = dtypeSizes.fMapDtypeSize;
    uint32_t k0 = CUBE_C0_SIZE / fMapDtypeSize;
    uint64_t c04UsdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, KParmas{CUBE_C04_SIZE, ConvAlignB(CUBE_C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0)}, dtypeSizes, hasScale);
    if (c04UsdL1SizeUnderMinMtiling > MEM_SIZE_512K) {
        return false;
    }
    return true;
}

// M split mode return 1, HW split mode retun 0, M and HW split mode both fail return -1
int32_t GetSplitMode(TilingParam &tilingData, uint32_t featuremapDtyeSize, uint32_t weightDtypeSize,
                     bool hasScale, bool isC04Mode)
{
    uint32_t biasDtyeSize = featuremapDtyeSize;
    uint32_t scaleDtyeSize = 0;
    if (tilingData.hasBias && hasScale) {
        biasDtyeSize = DTYPESIZE_4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
    }
    if (hasScale) {
        scaleDtyeSize = DTYPESIZE_8; // scaleDtye is int64/uint64 for int8/hif8/fp8
    }
    bool MsplitModeL1LimitCheckRes = false;
    bool HWsplitModeL1LimitCheckRes = false;
    DtypeSize dtypeSize = {featuremapDtyeSize, weightDtypeSize, biasDtyeSize, scaleDtyeSize};
    if (isC04Mode) {
        MsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInMsplitMode(tilingData, dtypeSize, hasScale);
        HWsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInHWsplitMode(tilingData, dtypeSize, hasScale);
    } else {
        MsplitModeL1LimitCheckRes = CheckL1SizeLimitsInMsplitMode(tilingData, dtypeSize, hasScale);
        HWsplitModeL1LimitCheckRes = CheckL1SizeLimitsInHWsplitMode(tilingData, dtypeSize, hasScale);
    }
    bool isConv1d = isConv1dFlag(tilingData, false);
    if (isConv1d) {
        MsplitModeL1LimitCheckRes = false; // only hw split mode in conv1d
    }

    if(MsplitModeL1LimitCheckRes && HWsplitModeL1LimitCheckRes) {
        return 1;
    }
    if(!MsplitModeL1LimitCheckRes && HWsplitModeL1LimitCheckRes) {
        return 0;
    }
    if(MsplitModeL1LimitCheckRes && !HWsplitModeL1LimitCheckRes) {
        return 1;
    }
    if(!MsplitModeL1LimitCheckRes && !HWsplitModeL1LimitCheckRes) {
        return -1;
    }
    return 0;
}

uint64_t CalcUsdL1SizeMode(TilingParam &tilingData,
                           DtypeSize dtypeSize,
                           int8_t pbAL1,
                           int8_t pbBL1)
{
    uint64_t mL1Max = tilingData.hoL1 < tilingData.singleCoreHo ? tilingData.hoL1 : tilingData.singleCoreHo;
    uint64_t hoAL1Tmp = min(mL1Max / tilingData.orgWo + NUM_2, tilingData.orgHo);
    uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.strideH + (tilingData.kernelH - 1) / tilingData.dilationH + 1,
        tilingData.orgHi);
    uint64_t al1Size = hiL1Tmp * tilingData.orgWi * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) *
        (pbAL1 + 1) * dtypeSize.fMapDtypeSize;
    uint64_t bl1Size = tilingData.nBL1 * tilingData.kBL1 * (pbBL1 + 1) * dtypeSize.weightDtypeSize;
    return al1Size + bl1Size;
}

uint64_t CalcUsdL1SizeHWode(TilingParam &tilingData,
                           DtypeSize dtypeSize,
                           int8_t pbAL1,
                           bool isC04Flag)
{
    uint64_t hiL1 = InferHiL1(tilingData.hoL1, tilingData.orgHi, tilingData.kernelH,
        tilingData.dilationH, tilingData.strideH);
    uint64_t wiL1 = InferWiL1(tilingData.woL1, tilingData.orgWi, tilingData.kernelW,
        tilingData.dilationW, tilingData.strideW);
    uint64_t al1Size = hiL1 * wiL1 * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) *
        (pbAL1 + 1) * dtypeSize.fMapDtypeSize;
    if (isC04Flag) {
        al1Size = ConvCeilDiv(hiL1 * wiL1 * CUBE_C04_SIZE * dtypeSize.fMapDtypeSize, CUBE_C0_SIZE) *
            CUBE_C0_SIZE * (pbAL1 + 1);
    }
    uint64_t bl1Size = tilingData.nBL1 * tilingData.kBL1 * dtypeSize.weightDtypeSize;
    return al1Size + bl1Size;
}

uint64_t CalcUsdL1Size(TilingParam &tilingData,
                       DtypeSize dtypeSize,
                       int64_t padCompensationValue,
                       bool hasQuantScale,
                       bool isC04Flag)
{
    uint32_t featuremapDtyeSize = dtypeSize.fMapDtypeSize;
    uint32_t weightDtyeSize = dtypeSize.weightDtypeSize;
    uint32_t outputOrder = (tilingData.hoL1 > 0 && tilingData.woL1 == 0) ? 1 : 0;
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    int8_t pbBL1 = (pBuffer & 0x10) >> NUM_4;
    bool hasBias = tilingData.hasBias;
    uint32_t biasDtyeSize = featuremapDtyeSize;
    uint32_t scaleDtyeSize = 0;
    if(hasBias & hasQuantScale) {
        biasDtyeSize = DTYPESIZE_4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
    }
    if(hasQuantScale) {
        scaleDtyeSize = DTYPESIZE_8; // scaleDtye is int64/uint64 for int8/hif8/fp8
    }
    uint64_t curl1Size = 0;
    uint64_t al1Bl1Size = 0;
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;
    uint64_t fixpSize = 0;
    if (outputOrder == 1) { // Mmode
        fixpSize = tilingData.nL0;
        al1Bl1Size = CalcUsdL1SizeMode(tilingData, dtypeSize, pbAL1, pbBL1);
    }
    if (outputOrder == 0) { // HWmode
        fixpSize = tilingData.nBL1;
        al1Bl1Size = CalcUsdL1SizeHWode(tilingData, dtypeSize, pbAL1, isC04Flag);
    }
    if (hasBias) {
        if (tilingData.biasFullLoadFlag) {
            biasL1Size = ConvCeilDiv(tilingData.singleCoreCo, CUBE_N0) * CUBE_N0 * biasDtyeSize;
        } else {
            biasL1Size = fixpSize * biasDtyeSize;
        }
    }
    if (hasQuantScale) {
        if (tilingData.fixpParamsFullLoadFlag) {
            scaleL1Size = ConvCeilDiv(tilingData.singleCoreCo, CUBE_N0) * CUBE_N0 * scaleDtyeSize;
        } else {
            scaleL1Size = fixpSize * scaleDtyeSize;
        }
    }
    curl1Size = al1Bl1Size + biasL1Size + scaleL1Size;

    return curl1Size;
}

uint64_t CalcUsdL0ASize(TilingParam &tilingData,
                        uint32_t outputOrder,
                        uint32_t featuremapDtyeSize,
                        int8_t pbAL0)
{
    uint64_t curl0aSize = 0;
    if (outputOrder == 0) {
        curl0aSize = tilingData.hoL0 * tilingData.woL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.hoL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcUsdL0BSize(TilingParam &tilingData,
                        uint32_t weightDtyeSize,
                        int8_t pbBL0)
{
    return tilingData.nL0 * tilingData.kL0 * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcUsdL0CSize(TilingParam &tilingData,
                        uint32_t outputOrder,
                        int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    uint32_t mmadDtypeSize = DTYPESIZE_4;
    if (outputOrder == 0) {
        curl0cSize = tilingData.hoL0 * tilingData.woL0 * tilingData.nL0 * (pbCL0 + 1) * mmadDtypeSize;
    } else {
        curl0cSize = tilingData.hoL0 * tilingData.nL0 * (pbCL0 + 1) * mmadDtypeSize;
    }
    return curl0cSize;
}

void GetInitBasicBlockMN(TilingParam &tilingData, uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    uint64_t howo = tilingData.hout * tilingData.wout;
    if (tilingData.cout <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_1024;
        basicBlockN = BASICBLOCK_INIT_VALUE_64;
    } else if (tilingData.cout > BASICBLOCK_BOUNDARY_VALUE_64
        && tilingData.cout <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_512;
        basicBlockN = BASICBLOCK_INIT_VALUE_128;
    } else if (howo <= BASICBLOCK_BOUNDARY_VALUE_64) {
        basicBlockM = BASICBLOCK_INIT_VALUE_64;
        basicBlockN = BASICBLOCK_INIT_VALUE_1024;
    } else if (howo > BASICBLOCK_BOUNDARY_VALUE_64
        && howo <= BASICBLOCK_BOUNDARY_VALUE_128) {
        basicBlockM = BASICBLOCK_INIT_VALUE_128;
        basicBlockN = BASICBLOCK_INIT_VALUE_512;
    } else {
        basicBlockM = BASICBLOCK_INIT_VALUE_256;
        basicBlockN = BASICBLOCK_INIT_VALUE_256;
    }
}

bool CmpFirstAdjustMnTile(TilingParam &tilingData, int64_t& mTile, int64_t& nTile,
                          BasicParams basicParams, vector<int64_t> pads)
{
    DtypeSize dtypeSize = basicParams.dtypeSize;
    int64_t fMapDtypeSize = dtypeSize.fMapDtypeSize;
    int64_t weightDtypeSize = dtypeSize.weightDtypeSize;
    int64_t availableL1Size = basicParams.availableL1Size;
    uint64_t basicBlockM = basicParams.basicBlockM;
    uint64_t basicBlockN = basicParams.basicBlockN;
    int64_t k0 = CUBE_C0_SIZE / fMapDtypeSize;
    int64_t maxHiWiL1 = availableL1Size / fMapDtypeSize / k0 / NUM_2;
    int64_t padTop = pads[0];
    int64_t padBottom = pads[1];
    if (maxHiWiL1 <= 0) {
        return false;
    }
    int64_t maxhiL1 = maxHiWiL1 / static_cast<int64_t>(tilingData.win);
    if (maxhiL1 <= NUM_2) {
        return false;
    }
    int64_t hoMax = 0;
    int64_t padCompensationValue = 0;
    if (basicBlockM >= tilingData.wout * tilingData.hout) { // L1 can full load M direction
        padCompensationValue = padTop + padBottom;
        hoMax = (maxhiL1 + padCompensationValue -
            static_cast<int64_t>(tilingData.dilationH) * (static_cast<int64_t>(tilingData.kh) - 1) - 1) /
            static_cast<int64_t>(tilingData.strideH) + 1;
    } else {
        padCompensationValue = max(padTop, padBottom);
        hoMax = (maxhiL1 + padCompensationValue - static_cast<int64_t>(tilingData.dilationH) *
            (static_cast<int64_t>(tilingData.kh) - 1) - 1) / static_cast<int64_t>(tilingData.strideH) + 1;
    }
    if (hoMax <= 0) {
        return false;
    }
    int64_t maxHoWoL1 = hoMax * static_cast<int64_t>(tilingData.wout);
    int64_t cmpM = tilingData.hout * tilingData.wout;
    int64_t cmpN = availableL1Size / weightDtypeSize / NUM_2 / k0 / tilingData.kh / tilingData.kw;
    mTile = min(min(cmpM, maxHoWoL1), static_cast<int64_t>(basicBlockM));
    nTile = min(min(static_cast<int64_t>(tilingData.cout), cmpN), static_cast<int64_t>(basicBlockN));
    if (tilingData.groupOpt == 0) {
        nTile = ConvCeilDiv(nTile, tilingData.groups);
    } else if (tilingData.groupOpt != 0) {
        nTile = ConvCeilDiv(nTile, tilingData.groups) * tilingData.enlarge;
    }
    mTile = mTile / CUBE_M0 * CUBE_M0;
    nTile = nTile / CUBE_N0 * CUBE_N0;
    if (mTile < CUBE_M0 || nTile < CUBE_N0) {
        return false;
    }
    return true;
}

void SelectMmodeAlgorithm(TilingParam &tilingData, bool& mBasicBlockModeFlag,
                          DtypeSize dtypeSize, vector<int64_t> pads, bool hasScale)
{
    uint32_t featuremapDtyeSize = dtypeSize.fMapDtypeSize;
    bool hasBias = tilingData.hasBias;
    mBasicBlockModeFlag = false;
    uint64_t basicBlockM = 0;
    uint64_t basicBlockN = 0;
    GetInitBasicBlockMN(tilingData, basicBlockM, basicBlockN);
    uint64_t mCut = ConvCeilDiv(tilingData.wout * tilingData.hout, basicBlockM);
    uint64_t nCut = ConvCeilDiv(tilingData.cout, basicBlockN);
    uint64_t group = tilingData.groups;
    if (tilingData.groupOpt != 0) {
        group = tilingData.groupOpt;
    }
    if (mCut * nCut * tilingData.batch * group <= AIC_NUM) {
        return;
    }
    int64_t biasSize = 0;
    int64_t scaleSize = 0;
    if (hasBias) {
        biasSize = featuremapDtyeSize; // for fp32/hf32/fp16/bf16
        if (hasScale) {
            biasSize = DTYPESIZE_4; // bias dtype is fp32 for int8/fp8/hif8
        }
    }
    if (hasScale) {
        scaleSize = DTYPESIZE_8; // scale dtype is uint64/int64 for int8/fp8/hif8
    }
    int64_t availableL1Size = MEM_SIZE_512K - biasSize - scaleSize;
    int64_t mTile = 0;
    int64_t nTile = 0;
    BasicParams basicParams;
    basicParams.availableL1Size = availableL1Size;
    basicParams.basicBlockM = basicBlockM;
    basicParams.basicBlockN = basicBlockN;
    basicParams.dtypeSize = dtypeSize;
    if (!CmpFirstAdjustMnTile(tilingData, mTile, nTile, basicParams, pads)) {
        return;
    }
    mBasicBlockModeFlag = true;
}

void CheckHWModeForConv2dPartOne(TilingParam &tilingData, uint64_t k0, bool dma_flag)
{
    // K direction check
    EXPECT_GE(tilingData.kL0, k0);
    EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));
    EXPECT_EQ(tilingData.kL0 % k0, 0);
    
    // N direction check
    EXPECT_GE(tilingData.nL0, CUBE_N0);
    EXPECT_GE(tilingData.nBL1, tilingData.nL0);
    EXPECT_LE(tilingData.nBL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, CUBE_N0) * CUBE_N0, tilingData.nL0) * tilingData.nL0);
    EXPECT_EQ(tilingData.nL0 % CUBE_N0, 0);
    uint32_t nBL1DivCheck = 0;
    if (tilingData.nBL1 % tilingData.nL0 == 0 ||
        tilingData.nBL1 == ConvCeilDiv(tilingData.singleCoreCo, CUBE_N0) * CUBE_N0) {
        nBL1DivCheck = 1;
    }
    EXPECT_EQ(nBL1DivCheck, 1);
    
    // W direction check
    EXPECT_GE(tilingData.woL0, CUBE_M0);
    EXPECT_GE(tilingData.woL1, tilingData.woL0);
    EXPECT_LE(tilingData.woL1, 
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreWo, CUBE_M0) * CUBE_M0, tilingData.woL0) * tilingData.woL0);
    if (tilingData.woL0 != tilingData.woL1) {
        EXPECT_EQ(tilingData.woL0 % CUBE_M0, 0);
    }
    if (tilingData.woL0 < tilingData.orgWo) {
        // woL0 does not reach the upper limit, thus hoL0 must be 1.
        EXPECT_EQ(tilingData.hoL0, 1);
    }
    if (tilingData.hoL0 > 1 && !(dma_flag && tilingData.groups > 1)) {
        EXPECT_EQ(tilingData.woL0, tilingData.orgWo);
        EXPECT_EQ(tilingData.woL1, tilingData.orgWo);
    }

    // H direction check
    uint32_t hoL1DivCheck = 0;
    EXPECT_GE(tilingData.hoL0, 1);
    EXPECT_GE(tilingData.hoL1, tilingData.hoL0);
    EXPECT_LE(tilingData.hoL1, tilingData.singleCoreHo);
    uint32_t hoL1Check = 0;
    if (tilingData.hoL1 % tilingData.hoL0 == 0 || tilingData.hoL1 == tilingData.singleCoreHo) {
        hoL1Check = 1;
    }
    EXPECT_EQ(hoL1Check, 1);
}

void CheckHWModeTilingDataValidForConv2d(TilingParam &tilingData, uint64_t k0, bool isC04Flag, bool dma_flag, bool isOptPreloadFlag)
{
    CheckHWModeForConv2dPartOne(tilingData, k0, dma_flag);

    if (isC04Flag) {
        EXPECT_EQ(tilingData.kAL1, ConvCeilDiv(CUBE_C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0) * k0);
        EXPECT_EQ(tilingData.kBL1, ConvCeilDiv(CUBE_C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0) * k0);
        if (tilingData.orgHi > 1) {
            EXPECT_EQ(tilingData.woL1, ConvCeilDiv(tilingData.orgWo, CUBE_M0) * CUBE_M0);
        }
    } else if (dma_flag && tilingData.groups > 1) {
        EXPECT_EQ(tilingData.kAL1 % (k0 * tilingData.khL1 * tilingData.kwL1), 0);
        EXPECT_EQ(tilingData.kBL1 % (k0 * tilingData.khL1 * tilingData.kwL1), 0);
    } else {
        EXPECT_LE(tilingData.kAL1,
                ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW);
        EXPECT_LE(tilingData.kBL1, ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelHxkernelWxkernelD);
        EXPECT_EQ(tilingData.kAL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
        EXPECT_EQ(tilingData.kBL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
        uint32_t kAL1DivCheck = 0;
        if (tilingData.kAL1 % tilingData.kL0 == 0 ||
            tilingData.kAL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
            kAL1DivCheck = 1;
        }
        EXPECT_EQ(kAL1DivCheck, 1);
        uint32_t kBL1DivCheck = false;
        if (tilingData.kBL1 % tilingData.kL0 == 0 ||
            tilingData.kBL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW ||
            tilingData.kBL1 == ConvCeilDiv(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelHxkernelWxkernelD) {
            kBL1DivCheck = true;
        }
        EXPECT_EQ(kBL1DivCheck, 1);
    }
    if (isOptPreloadFlag) {
        EXPECT_EQ(tilingData.hoL1, tilingData.hoL0);
        EXPECT_EQ(tilingData.woL1, tilingData.woL0);
        EXPECT_EQ(tilingData.nBL1, tilingData.nL0);
    }
}

void CheckGroupsTiling(TilingParam &tilingData, uint64_t tilingKey)
{
    int32_t groupMode = tilingData.enlarge > 0 ? NUM_2 : 1;
    uint64_t realCo = 0;
    if (groupMode == 1) {
        realCo = tilingData.cout / tilingData.groups;
    } else if (groupMode == NUM_2) {
        realCo = tilingData.coutOpt;
    }
    EXPECT_LE(tilingData.nDim, ConvCeilDiv(realCo, CUBE_N0));
}

bool CheckValidTilingDataPartOne(TilingParam &tilingData,
                                 DtypeSize dtypeSize,
                                 std::vector<int64_t> pads,
                                 bool hasScale,
                                 uint64_t tilingKey)
{
    uint64_t k0 = CUBE_C0_SIZE / dtypeSize.fMapDtypeSize;
    bool hasBias = tilingData.hasBias;
    uint32_t weightDtyeSize = dtypeSize.weightDtypeSize;
    uint32_t featuremapDtyeSize = dtypeSize.fMapDtypeSize;
    bool isC04Flag = (tilingData.bUbNStep > 0 && tilingData.bUbKStep == 0) ? true : false;
    bool dma_flag = (tilingKey & 0x4000) >> NUM_14;
    if (tilingData.groups > 1) {
        CheckGroupsTiling(tilingData, tilingKey);
        if (dma_flag) {
            EXPECT_TRUE((tilingKey & 0x400) >> NUM_10);
        }
    }
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    uint64_t nBL1 = tilingData.multiNBL1 * tilingData.nL0;
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    bool mBasicBlockModeFlag = false;
    int64_t padCompensationValue = 0;
    int64_t padTop = pads[0];
    int64_t padBottom = pads[1];
    int32_t splitModeFromCmp = GetSplitMode(tilingData, featuremapDtyeSize, weightDtyeSize, hasScale, isC04Flag);
    if (outputOrder == 1) {
        SelectMmodeAlgorithm(tilingData, mBasicBlockModeFlag, dtypeSize, pads, hasScale);
    }
    if (mBasicBlockModeFlag) {
        if (tilingData.hoL0 >= tilingData.wout * tilingData.hout) {
            padCompensationValue = padTop + padBottom;
        } else {
            padCompensationValue = max(padTop, padBottom);
        }
    }

    EXPECT_LE(CalcUsdL1Size(tilingData, dtypeSize, padCompensationValue, hasScale, isC04Flag), MEM_SIZE_512K);
    EXPECT_LE(CalcUsdL0ASize(tilingData, outputOrder, featuremapDtyeSize, pbAL0), MEM_SIZE_64K);
    EXPECT_LE(CalcUsdL0BSize(tilingData, weightDtyeSize, pbBL0), MEM_SIZE_64K);
    EXPECT_LE(CalcUsdL0CSize(tilingData, outputOrder, pbCL0), MEM_SIZE_256K);
    return mBasicBlockModeFlag;
}

void CheckMModeTilingDataValidForConv2d(TilingParam &tilingData, DtypeSize dtypeSize, bool mBasicBlockModeFlag, bool isOptPreloadFlag)
{
    uint64_t k0 = CUBE_C0_SIZE / dtypeSize.fMapDtypeSize;
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    EXPECT_GT(tilingData.kAL1, 0);
    EXPECT_GT(tilingData.kBL1, 0);
    EXPECT_GT(tilingData.hoL1, 0);
    EXPECT_GT(tilingData.multiNBL1 * tilingData.nL0, 0);
    EXPECT_GT(tilingData.hoL0, 0);
    EXPECT_GT(tilingData.kL0, 0);
    EXPECT_GT(tilingData.nL0, 0);

    EXPECT_EQ(tilingData.kAL1 % k0, 0);
    EXPECT_EQ(tilingData.nBL1 % CUBE_N0, 0);
    EXPECT_EQ(tilingData.nL0 % CUBE_N0, 0);
    EXPECT_EQ(tilingData.kL0 % k0, 0);
    if (!mBasicBlockModeFlag) { // only check m-mode/hw-mode formulation algorithm
        EXPECT_EQ(tilingData.kAL1 % tilingData.kL0, 0);
        EXPECT_EQ(tilingData.kBL1 % tilingData.kL0, 0);
    }
    uint32_t mmadDtypeSize = DTYPESIZE_4;
    EXPECT_LE(tilingData.nBL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, CUBE_N0) * CUBE_N0, tilingData.nL0) * tilingData.nL0);
    EXPECT_LE(tilingData.hoL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreHo, CUBE_M0) * CUBE_M0, tilingData.hoL0) * tilingData.hoL0);
    EXPECT_LE(tilingData.kAL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
    EXPECT_LE(tilingData.kBL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
    EXPECT_LE(tilingData.hoL0,
        ConvCeilDiv(std::min(static_cast<uint64_t>(MEM_SIZE_64K) / (k0 * (pbAL0 + 1) * dtypeSize.fMapDtypeSize),
        static_cast<uint64_t>(MEM_SIZE_256K) / (CUBE_N0 * (pbCL0 + 1) * mmadDtypeSize)), CUBE_M0) * CUBE_M0);
    EXPECT_LE(tilingData.nL0,
        ConvCeilDiv(std::min(static_cast<uint64_t>(MEM_SIZE_64K) / (k0 * (pbBL0 + 1) * dtypeSize.weightDtypeSize),
        static_cast<uint64_t>(MEM_SIZE_256K) / (CUBE_M0 * (pbCL0 + 1) * mmadDtypeSize)), CUBE_N0) * CUBE_N0);
    if (!mBasicBlockModeFlag) { // only check m-mode formulation algorithm
        EXPECT_LE(tilingData.kL0, ConvGcd(ConvCeilDiv(tilingData.kAL1, k0), ConvCeilDiv(tilingData.kBL1, k0)) * k0);
    }
    EXPECT_EQ(tilingData.woL1, 0);
    EXPECT_EQ(tilingData.woL0, 0);
    EXPECT_EQ(tilingData.singleCoreWo, 0);
    EXPECT_EQ(tilingData.hoL1 % CUBE_M0, 0);
    if (isOptPreloadFlag) {
        EXPECT_EQ(tilingData.hoL1, tilingData.hoL0);
        EXPECT_EQ(tilingData.nBL1, tilingData.nL0);
    }
}

void CheckValidTilingData(TilingParam &tilingData,
                          DtypeSize dtypeSize,
                          std::vector<int64_t> pads,
                          bool hasScale,
                          uint64_t tilingKey)
{
    bool mBasicBlockModeFlag = CheckValidTilingDataPartOne(tilingData, dtypeSize, pads, hasScale, tilingKey);
    uint64_t k0 = CUBE_C0_SIZE / dtypeSize.fMapDtypeSize;
    bool isC04Flag = (tilingData.bUbNStep > 0 && tilingData.bUbKStep == 0) ? true : false;
    bool dma_flag = (tilingKey & 0x4000) >> NUM_14;
    bool isOptPreloadFlag = ((tilingKey >> NUM_2) & 0x3) == 0 &&  // WeightTiling == 0 (FULLLOAD_BL1)
                            ((tilingKey >> NUM_4) & 0x3) == 3 &&  // L1PingPong == 3 (ALL_OPEN)
                            ((tilingKey >> NUM_9) & 0x1) == 0 &&  // IterOrder == 0 (MITER_FIRST)
                            ((tilingKey >> NUM_10) & 0x3) == 2;   // GroupType == 2 (OPT_GROUP_CONV)
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    if (outputOrder == 1) {
        CheckMModeTilingDataValidForConv2d(tilingData, dtypeSize, mBasicBlockModeFlag, isOptPreloadFlag);
    }

    if (outputOrder == 0) {
        CheckHWModeTilingDataValidForConv2d(tilingData, k0, isC04Flag, dma_flag, isOptPreloadFlag);
    }
}

void GetOriPadFromPadModeConv2D(PadModeParams padModeParams, uint32_t& padu, uint32_t& padd,
   uint32_t& padl, uint32_t& padr)
{
    string padMode = padModeParams.padMode;
    uint32_t strideH = padModeParams.strideH;
    uint32_t strideW = padModeParams.strideW;
    uint32_t dilationH = padModeParams.dilationH;
    uint32_t dilationW = padModeParams.dilationW;
    int64_t batch = padModeParams.batch;
    int64_t cin = padModeParams.cin;
    int64_t hi = padModeParams.hi;
    int64_t wi = padModeParams.wi;
    int64_t cout = padModeParams.cout;
    int64_t kH = padModeParams.kH;
    int64_t kW = padModeParams.kW;
    if (padMode == "SPECIFIC") {
        return;
    }

    if (padMode == "VALID") {
        padu = 0;
        padd = 0;
        padl = 0;
        padr = 0;
        return;
    }
    auto padH = (ConvCeilDiv(hi, strideH) - 1) * strideH + dilationH * (kH - 1) - hi + 1;
    auto padW = (ConvCeilDiv(wi, strideW) - 1) * strideW + dilationW * (kW - 1) - wi + 1;
    if (padMode == "SAME" || padMode == "SAME_UPPER") {
        padd = ConvCeilDiv(padH, NUM_2);
        padu = padH - padd;
        padr = ConvCeilDiv(padW, NUM_2);
        padl = padW - padr;
    } else {
        // padMode is "SAME_LOWER"
        padu = ConvCeilDiv(padH, NUM_2);
        padd = padH - padu;
        padl = ConvCeilDiv(padW, NUM_2);
        padr = padW - padl;
    }
}

struct Conv2DParams {
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    ge::DataType dtype;
    uint32_t isHasBias;
    uint32_t isHasScale;
    bool enableHf32Mode;
    uint32_t groups;
    std::string padMode;
    bool isErrorCaseFlag;
    std::string format;
};

void Conv2DTestCase(const Conv2DParams& params)
{
    bool hasBias = params.isHasBias == 1;
    bool hasScale = params.isHasScale == 1;
    // Extract padding, strides, and dilations
    uint32_t padu = params.pads[0], padd = params.pads[1], padl = params.pads[2], padr = params.pads[3];
    uint32_t strideH = params.strides[0], strideW = params.strides[1];
    uint32_t dilationH = params.dilations[0], dilationW = params.dilations[1];

    // Extract shapes
    int64_t cout = params.weightShape[0], kH = params.weightShape[1], kW = params.weightShape[2];
    int64_t batch = params.fmShape[0], cin = params.fmShape[1], hi = params.fmShape[2], wi = params.fmShape[3];

    // Adjust padding based on pad mode
    PadModeParams padModeParams =
        {params.padMode, strideH, strideW, dilationH, dilationW, batch, cin, hi, wi, cout, kH, kW};
    GetOriPadFromPadModeConv2D(padModeParams, padu, padd, padl, padr);

    // Infer output dimensions
    ConvShape convShapeH = {hi, kH, padu, padd, dilationH, strideH};
    ConvShape convShapeW = {wi, kW, padl, padr, dilationW, strideW};
    int64_t ho = InferOut(convShapeH);
    int64_t wo = InferOut(convShapeW);
    EXPECT_GE(ho, 1);
    EXPECT_GE(wo, 1);

    // Set formats
    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    // Set shapes
    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / params.groups, kH, kW}, {cout, cin / params.groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape quantScale = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    // Adjust for NHWC format
    if (params.format == "NHWC") {
        fmapFormat = ge::FORMAT_NHWC;
        weightFormat = ge::FORMAT_HWCN;
        outputFormat = ge::FORMAT_NHWC;
        featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
        weight = {{kH, kW, cin / params.groups, cout}, {kH, kW, cin / params.groups, cout}};
        output = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
    }

    // Prepare input and output shapes
    std::vector<void*> input_shape_ref = hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
        std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    // Set strides, pads, and dilations
    std::vector<int64_t> stridesVec = params.format == "NHWC" ?
        std::vector<int64_t>{1, strideH, strideW, 1} : std::vector<int64_t>{1, 1, strideH, strideW};
    std::vector<int64_t> padsVec = {padu, padd, padl, padr};
    std::vector<int64_t> dilationsVec = params.format == "NHWC" ?
        std::vector<int64_t>{1, dilationH, dilationW, 1} : std::vector<int64_t>{1, 1, dilationH, dilationW};

    // Set op type and tiling function
    std::string op_type = "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // Set compile info
    string compile_info_string = R"({"hardware_info": 
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});

    // Initialize platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

    // Create tiling data and workspace
    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    // Set data type sizes
    std::map<ge::DataType, uint32_t> dtypesizeMap = {
        {ge::DT_FLOAT16, DTYPESIZE_2}, {ge::DT_FLOAT, DTYPESIZE_4}, {ge::DT_INT8, 1},
        {ge::DT_BF16, DTYPESIZE_2}, {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}
    };

    uint32_t weightDtyeSize = dtypesizeMap.at(params.dtype);
    uint32_t featuremapDtyeSize = dtypesizeMap.at(params.dtype);
    uint64_t k0 = CUBE_C0_SIZE / featuremapDtyeSize;
    ge::DataType bias_dtype = (!hasScale && params.dtype == ge::DT_HIFLOAT8) ? ge::DT_FLOAT : params.dtype;

    // Build tiling context
    auto holder = gert::TilingContextFaker()
        .SetOpType(op_type)
        .NodeIoNum(NUM_4, 1)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes(input_shape_ref)
        .OutputShapes(output_shapes_ref)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
        .NodeInputTd(0, params.dtype, fmapFormat, fmapFormat)
        .NodeInputTd(1, params.dtype, weightFormat, weightFormat)
        .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, params.dtype, outputFormat, outputFormat)
        .NodeAttrs({
            {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
            {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
            {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
            {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(params.format)},
            {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(params.padMode)},
            {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(params.enableHf32Mode)}
        })
        .TilingData(tilingDataPtr.get())
        .Workspace(ws_size)
        .Build();

    // Set platform info
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    
    // Run tiling function and check results
    if (params.isErrorCaseFlag) {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
        return;
    }

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto buf = reinterpret_cast<TilingParam*>(tiling_context->GetRawTilingData()->GetData());
    TilingParam tilingParam = *buf;
    uint64_t tilingKey = tiling_context->GetTilingKey();
    EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, AIC_NUM);
    EXPECT_GE(tilingParam.batchDim, 1);
    EXPECT_GE(tilingParam.hoDim, 1);
    EXPECT_GE(tilingParam.nDim, 1);
    EXPECT_GE(tilingParam.groupDim, 1);
    DtypeSize dtypeSize;
    dtypeSize.fMapDtypeSize = featuremapDtyeSize;
    dtypeSize.weightDtypeSize = weightDtyeSize;
    if (tilingParam.batchDim > 0 && tilingParam.hoDim > 0 && tilingParam.nDim > 0 && tilingParam.groupDim > 0) {
        CheckValidTilingData(tilingParam, dtypeSize, padsVec, hasScale, tilingKey);
    }
}
} // namespace

class Conv2dv2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};
TEST_F(Conv2dv2Tiling, run_conv2d_case_cache_1) {
    for (int i = 0; i < NUM_10; i++) {
        Conv2DTestCase(Conv2DParams{{1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"});
    }
}

class Conv2dParameterizedTest : public ::testing::TestWithParam<Conv2DParams> {};
TEST_P(Conv2dParameterizedTest, RunConv2DCase) {
    Conv2DTestCase(GetParam());
}
INSTANTIATE_TEST_CASE_P(NewConv2dCases, Conv2dParameterizedTest, ::testing::Values(
    Conv2DParams{{2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,72,56}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,144,112}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,52,76}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,52,76}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,144,112}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,104,152}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,80,192}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,36,28}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,112,144}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,144,112}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,152,104}, {4,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,26,38}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1152,896}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,897,1153}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,104,152}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1153,897}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,2560,20,48}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,112,144}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1025,1025}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,152,104}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,960,80,192}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,576,448}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,56,72}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,152,104}, {320,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,544,480}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,40,96}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,20,48}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,208,304}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,833,1217}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,76,52}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,609,417}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,76,52}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,896,1152}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,288,224}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,288,224}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1024,1024}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,608,416}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,112,144}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,416,608}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,417,609}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,40,96}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,448,576}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,112,144}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,449,577}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1217,833}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1920,40,96}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,144,112}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,80,192}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,152,104}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,144,112}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,3,1216,832}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,832,1216}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,544,480}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,128,128}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1088,960}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,448,576}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,104,152}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,304,208}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,576,448}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,289,225}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,209,305}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,112,144}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,112,144}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,305,209}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,545,481}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,1089,961}, {128,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,128,128}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,128,128}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,136,120}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,208,304}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,136,120}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,224,288}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,72,56}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,152,104}, {8,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,104,152}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,76,52}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,136,120}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,52,76}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,152,104}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,144,112}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,56,72}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,104,152}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,272,240}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,1280,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,136,120}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,112,144}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,104,152}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,72,56}, {640,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,56,72}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,136,120}, {8,3,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,72,56}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,64,64}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,4,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,34,30}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,112,144}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,1280,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,28,36}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,152,104}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,36,28}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,68,60}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,76,52}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,52,76}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,68,60}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,144,112}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,64,64}, {640,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,38,26}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,28,36}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,224,288}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,608,416}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,512,512}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,152,104}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,225,289}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,513,513}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,577,449}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,128,416,608}, {256,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,256,256}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,144,112}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,288,224}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,128,128}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,304,208}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,8,104,152}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,256,272,240}, {512,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,72,56}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,257,257}, {512,1,1}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,512,273,241}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,52,76}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,56,72}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,640,38,26}, {1280,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,68,60}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,320,76,52}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,8,1,100}, {2,1,12}, {0,0,3,4}, {1,63}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{3,3,114,376}, {8,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,6,234,216}, {9,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{5,9,456,234}, {10,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{6,12,246,342}, {11,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{10,21,53,923}, {14,1,3}, {0,0,0,0}, {6,7}, {1,255}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,24,57,821}, {15,1,2}, {0,0,0,0}, {8,9}, {1,255}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,27,24,714}, {16,1,1}, {0,0,0,0}, {10,11}, {1,255}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,30,42,523}, {17,1,2}, {0,0,0,0}, {12,13}, {1,255}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{19,39,553,1}, {20,255,1}, {255,18,0,0}, {18,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{20,42,341,1}, {21,255,1}, {19,255,0,0}, {20,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{21,45,443,1}, {22,255,1}, {21,22,0,0}, {22,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{22,48,323,1}, {23,255,1}, {255,24,0,0}, {24,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{28,66,9,12766}, {29,5,43}, {37,38,39,40}, {7,8}, {11,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{134217712,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,134217712,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,1,134217712}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{134217712,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,134217712,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,1,134217712}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{67108832,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,67108832,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,1,67108832}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{67108832,1,1,1}, {1,1,1}, {1,1,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,67108832,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1,1,67108832}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{15,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,10,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{10,9,32,45}, {5440,1,11}, {3,6,9,2}, {2,2}, {2,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,22,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,61,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,125,46,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,222,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{5,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{19,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{45,3,54,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{84,8,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,8,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,8,94,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,9,95,97}, {13,32,6}, {1,2,3,4}, {12,13}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{6,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,10,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{9,22,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{10,61,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,125,46,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,222,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{24,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{52,3,54,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{72,8,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,8,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,12,86,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,14,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {2,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,13,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,6}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,12,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,5}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,8,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{6,11,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,10,57,69}, {889,4,3}, {2,3,5,7}, {4,9}, {5,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{9,22,59,88}, {13,11,9}, {4,5,3,2}, {7,8}, {3,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{10,61,45,76}, {11,6,7}, {5,11,11,15}, {6,4}, {2,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,125,48,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,222,68,79}, {15,4,3}, {2,3,6,7}, {4,8}, {6,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,889,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {4,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,4097,46,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{24,6,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {3,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{52,3,58,86}, {12,7,5}, {3,6,1,2}, {3,8}, {2,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{72,8,59,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,4,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,8,86,88}, {11,2,2}, {4,8,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,12,86,96}, {12,5,22}, {1,2,3,4}, {10,9}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{3,1,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,2,99,94}, {30,10,7}, {6,7,8,9}, {8,7}, {5,7}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{5,3,65,57}, {60,5,2}, {1,0,11,12}, {5,3}, {2,6}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{6,4,44,75}, {87,3,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,5,76,65}, {246,8,6}, {3,2,7,6}, {7,6}, {1,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,6,58,69}, {889,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{9,7,32,45}, {5440,1,11}, {3,6,9,2}, {2,2}, {2,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{10,24,55,88}, {13,11,9}, {4,5,3,2}, {8,8}, {4,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{11,65,43,76}, {11,6,7}, {5,11,12,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,128,49,77}, {13,3,5}, {5,11,8,15}, {6,4}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{13,220,68,79}, {15,4,3}, {2,3,5,7}, {4,9}, {6,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{14,899,78,59}, {6,4,3}, {2,3,5,7}, {2,4}, {5,4}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,4097,56,35}, {9,7,2}, {3,2,2,4}, {4,3}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{19,7,55,88}, {13,11,9}, {4,5,3,2}, {12,8}, {4,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{45,6,58,86}, {12,7,5}, {3,4,1,2}, {3,8}, {2,3}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{81,9,53,84}, {11,3,1}, {2,3,0,2}, {2,5}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{15,5,99,99}, {13,3,2}, {4,5,3,2}, {1,1}, {18,17}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,7,86,83}, {11,2,2}, {4,9,8,2}, {1,2}, {34,35}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{7,12,96,96}, {12,5,22}, {1,2,3,4}, {10,11}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,11,97,97}, {13,32,6}, {1,2,3,4}, {12,13}, {3,2}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,512,18,18}, {2048,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,15,15,16}, {1,2,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,3,4,4}, {1000,2,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,768,16,16}, {768,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,291,480,480}, {291,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{32,512,105,105}, {1024,1,1}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {64,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,64}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {255,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {256,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {1,255}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,64,30,30}, {1,1,1}, {0,0,0,0}, {1,1}, {1,256}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,1280,36,28}, {134217712,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{1,1280,32,32}, {67108832,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_INT8, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{2,1280,36,28}, {268435448,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", true, "NCHW"},
    Conv2DParams{{16,16,448,576}, {16,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,4480,5760}, {16,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1089,961}, {16,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1089,961}, {16,30,30}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1089,961}, {16,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1089,961}, {16,30,30}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,1,16}, {16,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, true, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,12,1,1}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,15,10,10}, {6,2,2}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 3, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,11,10}, {12,3,4}, {0,0,1,1}, {1,2}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 4, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,15,10,11}, {20,4,3}, {1,1,0,0}, {1,1}, {1,2}, ge::DT_FLOAT16, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,12,11,11}, {30,6,6}, {1,2,3,4}, {2,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 6, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,7,11,20}, {42,6,6}, {2,1,4,3}, {1,1}, {2,1}, ge::DT_FLOAT16, 1, 0, false, 7, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,1024,1024}, {64,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,32,32}, {64,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,512,512}, {128,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,8,8}, {1024,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,8,16}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,16,16}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,32,32}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,64,64}, {MEM_SIZE_512K,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,3,3}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,32,32}, {3,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,64,64}, {261952,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,64,64}, {32,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,64,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,8,26,92}, {85,10,9}, {9,9,8,8}, {13,5}, {1,9}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,32,64,64}, {32,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,192,16,16}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,512,18,18}, {2048,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1280,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,320,128,128}, {320,1,1}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1280,64,64}, {640,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,128,32,32}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,256,33,33}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,256,16,16}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,128,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,256,512,512}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,256,513,513}, {256,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,256,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,512,256,256}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,512,257,257}, {512,3,3}, {0,0,0,0}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,512,128,128}, {512,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,320,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,320,128,128}, {320,3,3}, {1,1,1,1}, {2,2}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,640,128,128}, {640,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,960,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,640,128,128}, {320,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,640,26,38}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,256,18,18}, {2048,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1280,32,32}, {1280,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,256,18,18}, {2048,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{6,8,231,221}, {119,12,11}, {11,11,2,2}, {2,6}, {5,5}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{12,12,211,132}, {87,11,14}, {0,0,8,8}, {3,3}, {7,3}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{8,224,38,38}, {1344,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 0, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{36,32,16,16}, {256,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 4, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,1,1}, {1,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,8,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,8}, {2,4,1}, {2,2,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{16,16,1,1000000}, {16,1,3}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{2,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {1,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,200}, {2,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,8}, {2,1,1}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,16,1,1000000}, {16,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "VALID", false, "NCHW"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME", false, "NCHW"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_UPPER", false, "NCHW"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_LOWER", false, "NCHW"},
    Conv2DParams{{1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_BF16, 1, 0, false, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, false, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT, 1, 0, true, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{1,2,3,16}, {16,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,64,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,3,64,64}, {32,3,3}, {1,1,1,1}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{1,20,3,1000}, {20,3,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", true, "NHWC"},
    Conv2DParams{{1,20,1,1000000}, {20,1,30}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", true, "NHWC"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "VALID", false, "NHWC"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME", false, "NHWC"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_UPPER", false, "NHWC"},
    Conv2DParams{{1,1,256,256}, {1,3,4}, {9999,9999,9999,9999}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SAME_LOWER", false, "NHWC"},
    Conv2DParams{{1,2,3,180000}, {160000,2,7}, {0,0,0,0}, {1,1}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 1, "SPECIFIC", true, "NHWC"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 5, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 4, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 8, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {65,65,65,65}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 8, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT, 1, 0, false, 5, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_BF16, 1, 0, false, 5, "SPECIFIC", false, "NHWC"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 4, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 8, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,32,224,224}, {768,32,79}, {65,65,65,65}, {32,32}, {1,1}, ge::DT_FLOAT16, 1, 0, false, 8, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_FLOAT, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{4,35,224,224}, {765,32,79}, {3,3,3,3}, {32,32}, {1,1}, ge::DT_BF16, 1, 0, false, 5, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,6,142,132}, {6,3,3}, {0,0,0,0}, {56,56}, {1,1}, ge::DT_HIFLOAT8, 0, 0, false, 6, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{1,40,29,595}, {80,11,11}, {4,4,5,5}, {43,15}, {1,44}, ge::DT_BF16, 0, 0, false, 2, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{64,1024,1,1376}, {256,1,255}, {0,0,134,0}, {1,11}, {1,2}, ge::DT_FLOAT16, 0, 0, false, 1, "SPECIFIC", false, "NCHW"},
    Conv2DParams{{256,128,267,9}, {256,105,6}, {47,49,2,4}, {41,7}, {1,1}, ge::DT_HIFLOAT8, 1, 0, false, 1, "SPECIFIC", false, "NCHW"}
)
);