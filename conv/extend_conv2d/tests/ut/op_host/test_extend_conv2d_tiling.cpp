/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv2d_tiling.cpp
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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h" 
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
uint64_t L1_SIZE = 524288;
uint64_t L0A_SIZE = 65536;
uint64_t L0B_SIZE = 65536;
uint64_t L0C_SIZE = 262144;
uint64_t SIZE_4096 = 4096;
uint64_t AICORE_NUM = 32;
uint64_t N0_SIZE = 16;
uint64_t M0_SIZE = 16;
uint64_t C0_SIZE = 32;
uint64_t C04_SIZE = 4;
uint64_t DTYPESIZE_2 = 2;
uint64_t DTYPESIZE_4 = 4;
uint64_t DTYPESIZE_8 = 8;
uint64_t NUM_2 = 2;
uint64_t NUM_3 = 3;
uint64_t NUM_4 = 4;
uint64_t NUM_5 = 5;
uint64_t NUM_9 = 9;
uint64_t NUM_10 = 10;
uint64_t NUM_14 = 14;
uint64_t MEM_SIZE_64K = 65536;
uint64_t MEM_SIZE_256K = 262144;
uint64_t MEM_SIZE_512K = 524288;
uint64_t DIM_2 = 2;
uint64_t DIM_3 = 3;
uint64_t DIM_4 = 4;

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

struct KParmas {
    uint32_t kAL1min;
    uint32_t kBL1min;
};

struct DtypeSize {
    uint32_t fMapDtypeSize;
    uint32_t weightDtypeSize;
    uint32_t biasDtypeSize;
    uint32_t scaleDtypeSize;
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

struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

struct BasicParams {
    int64_t availableL1Size;
    uint64_t basicBlockM;
    uint64_t basicBlockN;
    DtypeSize dtypeSize;
};

uint64_t InferHo(ConvShape convShape)
{
    uint64_t inputHi = convShape.inputV;
    uint64_t kH = convShape.kernelV;
    uint64_t padTop = convShape.padone;
    uint64_t padBottom = convShape.padtwo;
    uint64_t dilationH = convShape.dilationV;
    uint64_t strideH = convShape.strideV;
    if (strideH == 0) {
        return 0;
    }
    return (inputHi + padTop + padBottom - dilationH * (kH - 1) - 1) / strideH + 1;
}

uint64_t InferWo(ConvShape convShape)
{
    uint64_t inputWi = convShape.inputV;
    uint64_t kW = convShape.kernelV;
    uint64_t padLeft = convShape.padone;
    uint64_t padRight = convShape.padtwo;
    uint64_t dilationW = convShape.dilationV;
    uint64_t strideW = convShape.strideV;
    if (strideW == 0) {
        return 0;
    }
    return (inputWi + padLeft + padRight - dilationW * (kW - 1) - 1) / strideW + 1;
}

int64_t InferHiL1(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH)
{
    int64_t khDilated = (singlekH - 1) * dilationH + 1;
    int64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

int64_t InferWiL1(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW)
{
    int64_t kwDilated = (singlekW - 1) * dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

uint64_t ConvCeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t ConvGcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

uint64_t ConvAlignB(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return 0;
    }
    return ((a + b - 1) / b) * b;
}

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

uint64_t CalcMinUsedL1SizeInMsplitMode(TilingParam &tilingData, uint32_t kAL1min, uint32_t kBL1min, DtypeSize dtypeSizes, bool hasScale)
{
    uint64_t nBL1min = N0_SIZE;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * dtypeSizes.biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * dtypeSizes.scaleDtypeSize, C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * dtypeSizes.weightDtypeSize, C0_SIZE);
    uint64_t hoAL1min = std::min(M0_SIZE / tilingData.orgWo + NUM_2, tilingData.orgHo);
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * tilingData.orgWi * kAL1min * dtypeSizes.fMapDtypeSize, C0_SIZE);
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
    uint64_t nBL1min = N0_SIZE;
    uint64_t biasUsedL1Size = tilingData.hasBias_api ? ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = hasScale ? ConvAlignB(nBL1min * scaleDtypeSize, C0_SIZE) : 0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);
    uint64_t hoAL1min = tilingData.orgWo < M0_SIZE ? ConvCeilDiv(M0_SIZE, tilingData.orgWo) : 1;
    uint64_t hiAL1min = InferHiL1(hoAL1min, tilingData.orgHi, tilingData.kernelH,
                                    tilingData.dilationH_api, tilingData.strideH_api);
    uint64_t fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    return minL1LoadSize;
}

bool CheckL1SizeLimitsInHWsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    // require hiL1 * wiL1 >= m0
    uint64_t woAL1min = M0_SIZE;
    uint32_t k0 = C0_SIZE / dtypeSizes.fMapDtypeSize;
    KParmas kParmas = {k0, tilingData.kernelH * tilingData.kernelW * k0};
    uint64_t wiAL1min = InferWiL1(woAL1min, tilingData.orgWi, tilingData.kernelW, tilingData.dilationW, tilingData.strideW);
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(tilingData, kParmas, wiAL1min, dtypeSizes, hasScale);
    if (usdL1SizeUnderMinHWtiling > L1_SIZE) {
        return false;
    }
    return true;
}
 
bool CheckL1SizeLimitsInMsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t k0 = C0_SIZE / dtypeSizes.fMapDtypeSize;
    uint64_t usdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, k0, tilingData.kernelH * tilingData.kernelW * k0,
      dtypeSizes, hasScale);
    if (usdL1SizeUnderMinMtiling > L1_SIZE) {
        return false;
    }
    return true;
}

bool CheckC04L1SizeLimitsInHWsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    // c04 require wi fulload L1
    uint32_t k0 = C0_SIZE / dtypeSizes.fMapDtypeSize;
    KParmas kParmas = {C04_SIZE, ConvAlignB(C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0)};
    uint64_t usdL1SizeUnderMinHWtiling = CalcMinUsedL1SizeInHWsplitMode(tilingData, kParmas, tilingData.orgWi,
            dtypeSizes, hasScale);
    if (usdL1SizeUnderMinHWtiling > L1_SIZE) {
        return false;
    }
    return true;
}
 
bool CheckC04L1SizeLimitsInMsplitMode(TilingParam &tilingData, DtypeSize dtypeSizes, bool hasScale)
{
    uint32_t k0 = C0_SIZE / dtypeSizes.fMapDtypeSize;
    uint64_t c04UsdL1SizeUnderMinMtiling = CalcMinUsedL1SizeInMsplitMode(tilingData, C04_SIZE, ConvAlignB(C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0),
      dtypeSizes, hasScale);
    if (c04UsdL1SizeUnderMinMtiling > L1_SIZE) {
        return false;
    }
    return true;
}

// M split mode return 1, HW split mode retun 0, M and HW split mode both fail return -1
int32_t GetSplitMode(TilingParam &tilingData, uint32_t featuremapDtyeSize,
    uint32_t weightDtypeSize, bool hasScale, bool isC04Mode)
{
    uint32_t biasDtyeSize = 0;
    uint32_t scaleDtyeSize = 0;
    if (tilingData.hasBias) {
        if (hasScale) {
            biasDtyeSize = DTYPESIZE_4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
        }
        else {
            biasDtyeSize = featuremapDtyeSize; // biasdtype is same as fmdtype for fp32/hf32/fp16/bf16
        }
    }
    if (hasScale) {
        scaleDtyeSize = DTYPESIZE_8; // scaleDtye is int64/uint64 for int8/hif8/fp8
    }
    bool MsplitModeL1LimitCheckRes = false;
    bool HWsplitModeL1LimitCheckRes = false;
    DtypeSize dtypeSizes = {featuremapDtyeSize, biasDtyeSize, weightDtypeSize, scaleDtyeSize};
    if (isC04Mode) {
        MsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInMsplitMode(tilingData, dtypeSizes, hasScale);
        HWsplitModeL1LimitCheckRes = CheckC04L1SizeLimitsInHWsplitMode(tilingData, dtypeSizes, hasScale);
    } else {
        MsplitModeL1LimitCheckRes = CheckL1SizeLimitsInMsplitMode(tilingData, dtypeSizes, hasScale);
        HWsplitModeL1LimitCheckRes = CheckL1SizeLimitsInHWsplitMode(tilingData, dtypeSizes, hasScale);
    }
    if (isConv1dFlag(tilingData, false)) {
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
        al1Size = ConvCeilDiv(hiL1 * wiL1 * C04_SIZE * dtypeSize.fMapDtypeSize, C0_SIZE) *
            C0_SIZE * (pbAL1 + 1);
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
            biasL1Size = ConvCeilDiv(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * biasDtyeSize;
        } else {
            biasL1Size = fixpSize * biasDtyeSize;
        }
    }
    if (hasQuantScale) {
        if (tilingData.fixpParamsFullLoadFlag) {
            scaleL1Size = ConvCeilDiv(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * scaleDtyeSize;
        } else {
            scaleL1Size = fixpSize * scaleDtyeSize;
        }
    }
    curl1Size = al1Bl1Size + biasL1Size + scaleL1Size;

    return curl1Size;
}

uint64_t CalcUsdL0ASize(TilingParam &tilingData, uint32_t outputOrder, uint32_t featuremapDtyeSize, int8_t pbAL0)
{
    uint64_t curl0aSize = 0;
    if (outputOrder == 0) {
        curl0aSize = tilingData.hoL0 * tilingData.woL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.hoL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcUsdL0BSize(TilingParam &tilingData, uint32_t weightDtyeSize, int8_t pbBL0)
{
    return tilingData.nL0 * tilingData.kL0 * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcUsdL0CSize(TilingParam &tilingData, uint32_t outputOrder, int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    if (outputOrder == 0) {
        curl0cSize = tilingData.hoL0 * tilingData.woL0 * tilingData.nL0 * (pbCL0 + 1) * DTYPESIZE_4;
    } else {
        curl0cSize = tilingData.hoL0 * tilingData.nL0 * (pbCL0 + 1) * DTYPESIZE_4;
    }
    return curl0cSize;
}

void GetInitBasicBlockMN(TilingParam &tilingData, uint64_t& basicBlockM, uint64_t& basicBlockN)
{
    constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_64 = 64;
    constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_128 = 128;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_64 = 64;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_128 = 128;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_256 = 256;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_512 = 512;
    constexpr uint32_t BASICBLOCK_INIT_VALUE_1024 = 1024;
    
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
    int64_t k0 = C0_SIZE / fMapDtypeSize;
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
    mTile = mTile / M0_SIZE * M0_SIZE;
    nTile = nTile / N0_SIZE * N0_SIZE;
    if (mTile < M0_SIZE || nTile < N0_SIZE) {
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
    if (mCut * nCut * tilingData.batch * group <= AICORE_NUM) {
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

void CheckHWModeForConv2dPartOne(TilingParam &tilingData, uint64_t k0)
{
    // K direction check
    EXPECT_GE(tilingData.kL0, k0);
    EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));
    EXPECT_EQ(tilingData.kL0 % k0, 0);
    
    // N direction check
    EXPECT_GE(tilingData.nL0, N0_SIZE);
    EXPECT_GE(tilingData.nBL1, tilingData.nL0);
    EXPECT_LE(tilingData.nBL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE, tilingData.nL0) * tilingData.nL0);
    EXPECT_EQ(tilingData.nL0 % N0_SIZE, 0);
    uint32_t nBL1DivCheck = 0;
    if (tilingData.nBL1 % tilingData.nL0 == 0 ||
        tilingData.nBL1 == ConvCeilDiv(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE) {
        nBL1DivCheck = 1;
    }
    EXPECT_EQ(nBL1DivCheck, 1);
    
    // W direction check
    EXPECT_GE(tilingData.woL0, M0_SIZE);
    EXPECT_GE(tilingData.woL1, tilingData.woL0);
    EXPECT_LE(tilingData.woL1, 
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreWo, M0_SIZE) * M0_SIZE, tilingData.woL0) * tilingData.woL0);
    if (tilingData.woL0 != tilingData.woL1) {
        EXPECT_EQ(tilingData.woL0 % M0_SIZE, 0);
    }
    if (tilingData.woL0 < tilingData.orgWo) {
        // woL0 does not reach the upper limit, thus hoL0 must be 1.
        EXPECT_EQ(tilingData.hoL0, 1);
    }
    if (tilingData.hoL0 > 1 && !(tilingData.groups > 1)) {
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

void CheckHWModeTilingDataValidForConv2d(TilingParam &tilingData, uint64_t k0, bool isC04Flag, bool isOptPreloadFlag)
{
    CheckHWModeForConv2dPartOne(tilingData, k0);

    if (isC04Flag) {
        EXPECT_EQ(tilingData.kAL1, ConvCeilDiv(C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0) * k0);
        EXPECT_EQ(tilingData.kBL1, ConvCeilDiv(C04_SIZE * tilingData.kernelH * tilingData.kernelW, k0) * k0);
        if (tilingData.orgHi > 1) {
            EXPECT_EQ(tilingData.woL1, ConvCeilDiv(tilingData.orgWo, M0_SIZE) * M0_SIZE);
        }
    } else if (tilingData.groups > 1) {
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
    EXPECT_LE(tilingData.nDim, ConvCeilDiv(realCo, N0_SIZE));
}

bool CheckValidTilingDataPartOne(TilingParam &tilingData,
                                 DtypeSize dtypeSize,
                                 std::vector<int64_t> pads,
                                 bool hasScale,
                                 uint64_t tilingKey)
{
    uint64_t k0 = C0_SIZE / dtypeSize.fMapDtypeSize;
    bool hasBias = tilingData.hasBias;
    uint32_t weightDtyeSize = dtypeSize.weightDtypeSize;
    uint32_t featuremapDtyeSize = dtypeSize.fMapDtypeSize;
    bool isC04Flag = (tilingData.bUbNStep > 0 && tilingData.bUbKStep == 0) ? true : false;
    if (tilingData.groups > 1) {
        CheckGroupsTiling(tilingData, tilingKey);
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

    EXPECT_LE(CalcUsdL1Size(tilingData, dtypeSize, padCompensationValue, hasScale, isC04Flag), L1_SIZE);
    EXPECT_LE(CalcUsdL0ASize(tilingData, outputOrder, featuremapDtyeSize, pbAL0), L0A_SIZE);
    EXPECT_LE(CalcUsdL0BSize(tilingData, weightDtyeSize, pbBL0), L0B_SIZE);
    EXPECT_LE(CalcUsdL0CSize(tilingData, outputOrder, pbCL0), L0C_SIZE);
    return mBasicBlockModeFlag;
}

void CheckMModeTilingDataValidForConv2d(TilingParam &tilingData, DtypeSize dtypeSize, bool mBasicBlockModeFlag, bool isOptPreloadFlag)
{
    uint64_t k0 = C0_SIZE / dtypeSize.fMapDtypeSize;
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
    EXPECT_EQ(tilingData.nBL1 % N0_SIZE, 0);
    EXPECT_EQ(tilingData.nL0 % N0_SIZE, 0);
    EXPECT_EQ(tilingData.kL0 % k0, 0);
    if (!mBasicBlockModeFlag) { // only check m-mode/hw-mode formulation algorithm
        EXPECT_EQ(tilingData.kAL1 % tilingData.kL0, 0);
        EXPECT_EQ(tilingData.kBL1 % tilingData.kL0, 0);
    }
    uint32_t mmadDtypeSize = DTYPESIZE_4;
    EXPECT_LE(tilingData.nBL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE, tilingData.nL0) * tilingData.nL0);
    EXPECT_LE(tilingData.hoL1,
        ConvCeilDiv(ConvCeilDiv(tilingData.singleCoreHo, M0_SIZE) * M0_SIZE, tilingData.hoL0) * tilingData.hoL0);
    EXPECT_LE(tilingData.kAL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
    EXPECT_LE(tilingData.kBL1, tilingData.kernelH * tilingData.kernelW * ConvCeilDiv(tilingData.cin, k0) * k0);
    EXPECT_LE(tilingData.hoL0,
        ConvCeilDiv(std::min(MEM_SIZE_64K / (k0 * (pbAL0 + 1) * dtypeSize.fMapDtypeSize),
        MEM_SIZE_256K / (N0_SIZE * (pbCL0 + 1) * mmadDtypeSize)), M0_SIZE) * M0_SIZE);
    EXPECT_LE(tilingData.nL0,
        ConvCeilDiv(std::min(MEM_SIZE_64K / (k0 * (pbBL0 + 1) * dtypeSize.weightDtypeSize),
        MEM_SIZE_256K / (M0_SIZE * (pbCL0 + 1) * mmadDtypeSize)), N0_SIZE) * N0_SIZE);
    if (!mBasicBlockModeFlag) { // only check m-mode formulation algorithm
        EXPECT_LE(tilingData.kL0, ConvGcd(ConvCeilDiv(tilingData.kAL1, k0), ConvCeilDiv(tilingData.kBL1, k0)) * k0);
    }
    EXPECT_EQ(tilingData.woL1, 0);
    EXPECT_EQ(tilingData.woL0, 0);
    EXPECT_EQ(tilingData.singleCoreWo, 0);
    EXPECT_EQ(tilingData.hoL1 % M0_SIZE, 0);
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
    uint64_t k0 = C0_SIZE / dtypeSize.fMapDtypeSize;
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
        CheckHWModeTilingDataValidForConv2d(tilingData, k0, isC04Flag, isOptPreloadFlag);
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

void ExtendConv2DTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape,
    vector<uint32_t> pads, vector<uint32_t> strides, vector<uint32_t> dilations,
    ge::DataType inDataType, ge::DataType out0DataType, ge::DataType out1DataType,
    bool isHasBias = true, bool isHasScale0 = false, bool isHasScale1 = false,
    bool isHasReluWeight0 = false, bool isHasReluWeight1 = false,
    bool isHasClipValue0 = false, bool isHasClipValue1 = false,
    bool enableRelu0 = false, bool enableRelu1 = false, bool dualOutput = false,
    bool enableHf32Mode = false, uint32_t groups = 1,
    string padMode = "SPECIFIC", uint8_t errorCaseStatus = 0, string format = "NCHW", string round_mode = "rint")
{
    bool hasBias = isHasBias == 1;
    bool hasScale = isHasScale0 == 1;
    bool isErrorCaseFlag = errorCaseStatus == 0 ? false : true;

    uint32_t padu = pads[0];
    uint32_t padd = pads[1];
    uint32_t padl = pads[DIM_2];
    uint32_t padr = pads[DIM_3];
    uint32_t strideH = strides[0];
    uint32_t strideW = strides[1];
    uint32_t dilationH = dilations[0];
    uint32_t dilationW = dilations[1];
    int64_t cout = weightShape[0];
    int64_t kH = weightShape[1];
    int64_t kW = weightShape[DIM_2];
    int64_t batch = fmShape[0];
    int64_t cin = fmShape[1];
    int64_t hi = fmShape[DIM_2];
    int64_t wi = fmShape[DIM_3];
    PadModeParams padModeParams =
        {padMode, strideH, strideW, dilationH, dilationW, batch, cin, hi, wi, cout, kH, kW};
    GetOriPadFromPadModeConv2D(padModeParams, padu, padd, padl, padr);
    ConvShape convShapeH = {hi, kH, padu, padd, dilationH, strideH};
    ConvShape convShapeW = {wi, kW, padl, padr, dilationW, strideW};
    int64_t ho = InferHo(convShapeH);
    int64_t wo = InferWo(convShapeW);
    EXPECT_GE(ho, 1);
    EXPECT_GE(wo, 1);

    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / groups, kH, kW}, {cout, cin / groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape scale0 = {{cout}, {cout}};
    gert::StorageShape scale1 = {{cout}, {cout}};
    gert::StorageShape reluWeight0 = {{cout}, {cout}};
    gert::StorageShape reluWeight1 = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output0 = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};
    gert::StorageShape output1 = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    if (format == "NHWC") {
      fmapFormat = ge::FORMAT_NHWC;
      weightFormat = ge::FORMAT_HWCN;
      outputFormat = ge::FORMAT_NHWC;

      featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
      weight = {{kH, kW, cin / groups, cout}, {kH, kW, cin / groups, cout}};
      output0 = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
      output1 = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
    }

    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;
   if (hasBias) {
      if (isHasScale1) {
         input_shape_ref = {&featuremap, &weight, &bias, &scale0, &scale1};
      } else {
         input_shape_ref = {&featuremap, &weight, &bias, &scale0};
      }
   } else {
      if (isHasScale1) {
         input_shape_ref = {&featuremap, &weight, nullptr, nullptr, &scale0, nullptr, nullptr, &scale1};
      } else {
         input_shape_ref = {&featuremap, &weight, nullptr, nullptr, &scale0};
      }
   }
    std::vector<void*> output_shapes_ref = {&output0, &output1};
    ge::DataType fmapDataType = inDataType;
    ge::DataType weightDataType = inDataType;
    ge::DataType biasDataType = (inDataType == ge::DT_INT8) ? ge::DT_INT32 : (inDataType == ge::DT_HIFLOAT8 || inDataType == ge::DT_FLOAT8_E4M3FN) ? ge::DT_FLOAT : inDataType;
    ge::DataType offsetWDataType = ge::DT_INT8; // format is ND
    ge::DataType scale0DType = ge::DT_INT64;
    ge::DataType scale1DType = ge::DT_INT64;
    ge::DataType reluWeight0DType = ge::DT_FLOAT;
    ge::DataType reluWeight1DType = ge::DT_FLOAT;
    ge::DataType clipValue0DType = ge::DT_FLOAT16;
    ge::DataType outputDataType = out0DataType;

    std::vector<int64_t> strides = {1, 1, strideH, strideW};
    std::vector<int64_t> pads = {padu, padd, padl, padr};
    std::vector<int64_t> dilations = {1, 1, dilationH, dilationW};

    if (format == "NHWC") {
      strides = {1, strideH, strideW, 1};
      dilations = {1, dilationH, dilationW, 1};
    }
    std::string op_type = "ExtendConv2D";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    string compile_info_string = R"({"hardware_info": 
      {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
       "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
       "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
       "core_type_list": "CubeCore,VectorCore"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

    auto tilingDataPtr = gert::TilingData::CreateCap(SIZE_4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(SIZE_4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    std::map<ge::DataType, uint32_t> dtypesizeMap = {
        {ge::DT_FLOAT16, DTYPESIZE_2}, {ge::DT_FLOAT, DTYPESIZE_4},
        {ge::DT_INT8, 1}, {ge::DT_BF16, DTYPESIZE_2}, {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}};

    uint32_t weightDtyeSize = dtypesizeMap.at(fmapDataType);
    uint32_t featuremapDtyeSize = dtypesizeMap.at(fmapDataType);
    uint64_t k0 = C0_SIZE / featuremapDtyeSize;
    ge::DataType bias_dtype = (!hasScale && fmapDataType == ge::DT_HIFLOAT8) ? ge::DT_FLOAT : fmapDataType;

    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // generate TilingParseContext to do TilingPrepareForConv2DV2
    auto kernel_holder = gert::KernelRunContextFaker()
            .KernelIONum(NUM_2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    // 新增
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    //新增
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    fe::PlatFormInfos platform_info1;
    platform_info1.Init();
    auto holder = isHasScale1 ?
                     gert::TilingContextFaker().SetOpType(op_type)
                                             .NodeIoNum(NUM_5, NUM_2)
                                             .IrInstanceNum({1, 1, 1, 0, 1, 0, 0, 1}) // 控制算子原型对应位置的可选输入，是否存在
                                             .InputShapes(input_shape_ref)
                                             .OutputShapes(output_shapes_ref)
                                             .CompileInfo(&compile_info)
                                             .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                             .NodeInputTd(0, fmapDataType, fmapFormat, fmapFormat)
                                             .NodeInputTd(1, weightDataType, weightFormat, weightFormat)
                                             .NodeInputTd(DIM_2, biasDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(DIM_3, scale0DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(DIM_4, scale1DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeOutputTd(0, out0DataType, outputFormat, outputFormat)
                                             .NodeOutputTd(1, out1DataType, outputFormat, outputFormat)
                                             .NodeAttrs({
                                             {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                             {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                             {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                             {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                             {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                             {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(round_mode)},
                                             {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32Mode)},
                                             {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu0)},
                                             {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu1)},
                                             {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(dualOutput)},
                                             {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                             {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}
                                             })
                                             .TilingData(tilingDataPtr.get())
                                             .Workspace(ws_size)
                                             .Build() :
                     gert::TilingContextFaker().SetOpType(op_type)
                                             .NodeIoNum(NUM_4, NUM_2)
                                             .IrInstanceNum({1, 1, 1, 0, 1}) // 控制算子原型对应位置的可选输入，是否存在
                                             .InputShapes(input_shape_ref)
                                             .OutputShapes(output_shapes_ref)
                                             .CompileInfo(&compile_info)
                                             .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                             .NodeInputTd(0, fmapDataType, fmapFormat, fmapFormat)
                                             .NodeInputTd(1, weightDataType, weightFormat, weightFormat)
                                             .NodeInputTd(DIM_2, biasDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeInputTd(DIM_3, scale0DType, ge::FORMAT_ND, ge::FORMAT_ND)
                                             .NodeOutputTd(0, outputDataType, outputFormat, outputFormat)
                                             .NodeOutputTd(1, outputDataType, outputFormat, outputFormat)
                                             .NodeAttrs({
                                             {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                             {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                             {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                             {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                             {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                             {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(round_mode)},
                                             {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32Mode)},
                                             {"enable_relu0", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu0)},
                                             {"enable_relu1", Ops::NN::AnyValue::CreateFrom<bool>(enableRelu1)},
                                             {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(dualOutput)},
                                             {"dtype0", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                             {"dtype1", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}
                                             })
                                             .TilingData(tilingDataPtr.get())
                                             .Workspace(ws_size)
                                             .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (isErrorCaseFlag) {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
        return;
    }
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto buf = (TilingParam*)tiling_context->GetRawTilingData()->GetData();
    TilingParam tilingParam = *buf;
    uint64_t tilingKey = tiling_context->GetTilingKey();
    EXPECT_LE(tilingParam.batchDim * tilingParam.hoDim * tilingParam.nDim * tilingParam.groupDim, AICORE_NUM);
    EXPECT_GE(tilingParam.batchDim, 1);
    EXPECT_GE(tilingParam.hoDim, 1);
    EXPECT_GE(tilingParam.nDim, 1);
    EXPECT_GE(tilingParam.groupDim, 1);
    if (tilingParam.batchDim > 0 && tilingParam.hoDim > 0 && tilingParam.nDim > 0 && tilingParam.groupDim > 0) {
        DtypeSize dtypeSize = {featuremapDtyeSize, weightDtyeSize, 0, 0};
        CheckValidTilingData(tilingParam, dtypeSize, pads, hasScale, tilingKey);
    }
}
} // namespace

class ExtendConv2dTiling : public testing::Test {
    protected:
      static void SetUpTestCase() {
          std::cout << "Conv2d ascendc ops tiling testParam setup" << std::endl;
      }
      static void TearDownTestCase() {
          std::cout << "Conv2d ascendc ops tiling testParam tearDown" << std::endl;
      }
};

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_001) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_002) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      true, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_003) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_004) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_005) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_006) { //extendconv2d cache tiling
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_007) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      true, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_008) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, true, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_009) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_010) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_011) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_012) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "round"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_013) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_014) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_015) {
  ExtendConv2DTestCase({1,256,64,64}, {256,3,3},{0,0,0,0}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_016) {
  ExtendConv2DTestCase({4,23,1894,16}, {253,1,2},{0,0,0,0}, {48,7}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 23, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_singleCoreCo_fix_case1) {
  ExtendConv2DTestCase({4,4,43765,5}, {1024,165,4},{146,10,1,1}, {13,2}, {4,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_BF16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 4, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW", "rint"); // errorCaseStatus, format, round_mode
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_fp16_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_int8_fp16_out_case_cache_tiling) {
   for (int i = 0; i < 2; i++) {
      ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                           ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                           true, true, true, //isHasBias, isHasScale0, isHasScale1,
                           false, false, // isHasReluWeight0, isHasReluWeight1
                           false, false, //isHasClipValue0, isHasClipValue1
                           false, false, true, // enableRelu0, enableRelu1, dualOutput
                           false, 1, // enableHf32Mode, groups
                           "SPECIFIC",
                           0, "NHWC"); // errorCaseStatus, format
   }
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_fp8_in_int8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_fp8_in_fp8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_hif8_in_hif8_out_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      1, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_int8_out_groups2_case) {
  ExtendConv2DTestCase({1,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, false, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 2, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
 
TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nhwc_int8_in_fp16_fp16_out_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, true, true, //isHasBias, isHasScale0, isHasScale1,
                      false, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      false, false, true, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NHWC"); // errorCaseStatus, format
}

TEST_F(ExtendConv2dTiling, run_ExtendConv2D_case_nchw_conv_leakyrelu_case) {
  ExtendConv2DTestCase({2,16,256,256}, {32,3,3},{2,2,2,2}, {1,1}, {1,1},
                      ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, // inDataType, out0DataType, out1DataType
                      true, false, false, //isHasBias, isHasScale0, isHasScale1,
                      true, false, // isHasReluWeight0, isHasReluWeight1
                      false, false, //isHasClipValue0, isHasClipValue1
                      true, false, false, // enableRelu0, enableRelu1, dualOutput
                      false, 1, // enableHf32Mode, groups
                      "SPECIFIC",
                      0, "NCHW"); // errorCaseStatus, format
}
