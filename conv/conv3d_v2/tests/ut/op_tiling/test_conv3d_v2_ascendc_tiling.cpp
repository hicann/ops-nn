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
 * \file test_conv3d_v2_ascendc_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tests/ut/common/kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "conv/conv3d_v2/op_host/op_tiling/conv3d_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"

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
uint64_t C04_SIZE = 8;
uint64_t DTYPESIZE_2 = 2;
uint64_t DTYPESIZE_4 = 4;
uint64_t DTYPESIZE_8 = 8;
uint64_t NUM_2 = 2;
uint64_t NUM_3 = 3;
uint64_t NUM_4 = 4;
uint64_t NUM_5 = 5;
uint64_t DIM_2 = 2;
uint64_t DIM_3 = 3;
uint64_t DIM_4 = 4;
uint64_t DIM_5 = 5;
uint64_t NUM_10 = 10;

struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

struct DtypeSize {
    uint32_t fMapDtypeSize;
    uint32_t weightDtypeSize;
    uint32_t biasDtypeSize;
    uint32_t scaleDtypeSize;
};

struct TilingParam {
    // api tilingdata
    uint64_t orgDo;
    uint64_t orgHo;
    uint64_t orgWo;
    uint64_t orgDi;
    uint64_t orgHi;
    uint64_t orgWi;
    uint64_t singleCoreBatch;
    uint64_t singleCoreDo;
    uint64_t singleCoreM;
    uint64_t singleCoreWo;
    uint64_t singleCoreHo;
    uint64_t kL0xorgCoAlignN0;
    uint64_t kernelHxkernelW;
    uint64_t cin1xOriHixOriWixk0;
    uint64_t oriHixOriWixk0;
    uint64_t oriWixk0;
    uint64_t orgHixWi;
    uint64_t orgHoxWo;
    uint64_t unionDataXt;
    uint32_t orgCi;
    uint32_t kernelD;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t singleCoreCo;
    uint32_t orgCo;
    uint32_t singleCoreCi;
    uint32_t singleCoreGroups;
    uint32_t singleCoreGroupOpt;
    uint32_t groups_api;
    uint32_t enlarge_api;
    uint32_t strideH_api;
    uint32_t strideW_api;
    uint32_t strideD_api;
    uint32_t dilationH_api;
    uint32_t dilationW_api;
    uint32_t dilationD_api;
    uint32_t padHead_api;
    uint32_t padTail_api;
    uint32_t padTop_api;
    uint32_t padBottom_api;
    uint32_t padLeft_api;
    uint32_t padRight_api;
    uint32_t mL0;
    uint32_t woL0;
    uint32_t kL0;
    uint32_t nL0;
    uint32_t kAL1;
    uint32_t kAL1Tail;
    uint32_t kBL1;
    uint32_t kBL1Tail;
    uint32_t nBL1;
    uint32_t mAL1;
    uint32_t woL1;
    uint32_t hoL0;
    uint32_t hoL1;
    uint32_t KBL1Divk0;
    uint32_t KBL1TailDivk0;
    uint32_t nBL1DivnL0;
    uint32_t mAL1DivmL0;
    uint32_t fmapKStride;
    uint32_t weightKStride;
    uint32_t cinOffsetBlockInGM;
    uint32_t coutOffsetBlock;
    uint32_t nL1DivBlockSize;
    uint32_t cin1InAL1;
    uint32_t cin1InAL1Tail;
    uint32_t cinBInCore;
    uint32_t cinBTailInCore;
    uint32_t cinAInCore;
    uint32_t cinATailInCore;
    uint32_t nL0xk0;
    uint32_t mStep;
    uint32_t kStep;
    uint32_t nStep;
    uint32_t aL1SpaceSize;
    uint32_t multiNBL1;
    uint32_t pBufferFlag;
    uint32_t groupOpt_api;
    uint32_t cinOpt_api;
    uint32_t coutOpt_api;
    uint32_t mUB;
    uint32_t nUB;
    uint32_t scaleAndBiasLoadType;
    uint32_t workspaceSize;
    uint32_t kernelHxkernelWxkernelD;
    int8_t offsetx;
    int8_t roundMode;
    uint8_t hasBias_api;
    uint8_t hasScale;
    uint8_t bl1FullLoad;
    uint8_t al1FullLoad;
    uint8_t bl1BypassFlag;
    uint8_t iterateMNOrder;
    uint8_t biasFullLoadFlag;
    uint8_t fixpParamsFullLoadFlag;
    uint8_t hf32Enable;
    uint8_t hf32TransMode;
    uint8_t quantType;
    uint8_t resvered1;
    uint8_t resvered2;
    uint8_t resvered3;
    // ops tilingdata
    uint32_t batch;
    uint32_t cin;
    uint32_t din;
    uint64_t hin;
    uint64_t win;
    uint32_t cout;
    uint32_t kd;
    uint32_t kh;
    uint32_t kw;
    uint32_t dout;
    uint64_t hout;
    uint64_t wout;
    uint32_t batchDim;
    uint32_t doDim;
    uint32_t mDim;
    uint32_t wDim;
    uint32_t nDim;
    uint32_t groupDim;
    uint32_t hoDim;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t strideD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t dilationD;
    uint32_t padHead;
    uint32_t padTail;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t groups;
    uint32_t enlarge;
    uint32_t cinOpt;
    uint32_t coutOpt;
    uint32_t groupOpt;
    uint8_t hasBias;
};

struct PadModeParams {
    const string padMode;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationD;
    uint32_t dilationH;
    uint32_t dilationW;
    int64_t batch;
    int64_t cin;
    int64_t di;
    int64_t hi;
    int64_t wi;
    int64_t cout;
    int64_t kD;
    int64_t kH;
    int64_t kW;
};

struct PadRes {
    uint32_t& padh;
    uint32_t& padt;
    uint32_t& padu;
    uint32_t& padd;
    uint32_t& padl;
    uint32_t& padr;
};

struct Conv3dTestParams {
    vector<int64_t> fmShape;
    vector<int64_t> weightShape;
    vector<uint32_t> pads;
    vector<uint32_t> strides;
    vector<uint32_t> dilations;
    ge::DataType dtype;
    uint32_t isHasBias = 1;
    uint32_t groups = 1;
    int64_t fixBatcho = 0;
    int64_t fixDo = 0;
    int64_t fixHo = 0;
    int64_t fixWo = 0;
    bool isErrorCaseFlag = false;
    string padMode = "SPECIFIC";
    bool enableHf32 = false;
    ge::Format format = ge::Format::FORMAT_NCDHW;
    ge::Format outformat = ge::Format::FORMAT_NCDHW;
    bool isConv3dDequant=false;
    bool hasScale = false;
    ge::DataType biasDtypeIn = ge::DT_FLOAT16;
    ge::DataType outputDtypeIn = ge::DT_FLOAT16;
    ge::DataType scaleDtypeIn = ge::DT_FLOAT16;
};

uint64_t InferOut(ConvShape convShape)
{
    if (convShape.strideV == 0) {
        return 0;
    }
    return (convShape.inputV + convShape.padone + convShape.padtwo - convShape.dilationV * (convShape.kernelV - 1) - 1) / convShape.strideV + 1;
}

int64_t InferHiL1ForConv3dV2(uint64_t inputHoL1, int64_t hi, uint64_t singlekH, uint64_t dilationH, uint64_t strideH)
{
    int64_t khDilated = (singlekH - 1) * dilationH + 1;
    int64_t tmpHiL1 = (inputHoL1 - 1) * strideH + khDilated;
    if (tmpHiL1 > hi) {
        tmpHiL1 = hi;
    }

    return tmpHiL1;
}

int64_t InferWiL1ForConv3dV2(uint64_t inputWoL1, int64_t wi, uint64_t singlekW, uint64_t dilationW, uint64_t strideW)
{
    int64_t kwDilated = (singlekW - 1) * dilationW + 1;
    int64_t tmpWiL1 = (inputWoL1 - 1) * strideW + kwDilated;
    if (tmpWiL1 > wi) {
        tmpWiL1 = wi;
    }

    return tmpWiL1;
}

uint64_t CeilDivForConv3dV2(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t gcd(uint64_t a, uint64_t b) { // Get the greatest common divisor of a and b
    while (b != 0) {
        uint64_t temp = a % b;
        a = b;
        b = temp;
    }
    return a;
}

uint64_t CalcConv3dMmodeUsdL1Size(TilingParam &tilingData, DtypeSize dtypeSize, bool hasBias, bool hasQuantScale)
{
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    int8_t pbBL1 = (pBuffer & 0x10) >> NUM_4;
    uint64_t hoAL1Tmp = tilingData.hoL1 / tilingData.orgWo + NUM_2;
    uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.strideH + (tilingData.kernelH - 1) / tilingData.dilationH + 1, tilingData.orgHi);
    uint64_t al1Size = hiL1Tmp * tilingData.orgWi * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * dtypeSize.fMapDtypeSize;
    uint64_t bl1Size = tilingData.nBL1 * tilingData.kBL1 * (pbBL1 + 1) * dtypeSize.weightDtypeSize;
    if (hasBias) {
        if (tilingData.biasFullLoadFlag == 1) {
            biasL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * dtypeSize.biasDtypeSize;
        } else { // tilingData.biasFullLoadFlag == 0
            biasL1Size = tilingData.nL0 * dtypeSize.biasDtypeSize;
        }
    }
    if (hasQuantScale) {
        if (tilingData.fixpParamsFullLoadFlag) {
            scaleL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * dtypeSize.scaleDtypeSize;
        } else {
            scaleL1Size = tilingData.nL0 * dtypeSize.scaleDtypeSize;
        }
    }
    return al1Size + bl1Size + biasL1Size + scaleL1Size;
}

uint64_t CalcConv3dHWmodeUsdL1Size(TilingParam &tilingData, DtypeSize dtypeSize, bool hasBias, bool hasQuantScale)
{
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    uint64_t hiL1 = InferWiL1ForConv3dV2(tilingData.hoL1, tilingData.orgHi, tilingData.kernelH, tilingData.dilationH, tilingData.strideH);
    uint64_t wiL1 = InferWiL1ForConv3dV2(tilingData.woL1, tilingData.orgWi, tilingData.kernelW, tilingData.dilationW_api, tilingData.strideW_api);
    uint64_t al1Size = hiL1 * wiL1 * (tilingData.kAL1 / (tilingData.kernelH * tilingData.kernelW)) * (pbAL1 + 1) * dtypeSize.fMapDtypeSize;
    uint64_t bl1Size = tilingData.nBL1 * tilingData.kBL1 * dtypeSize.weightDtypeSize;
    if (hasBias) {
        if (tilingData.biasFullLoadFlag) {
            biasL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * dtypeSize.biasDtypeSize;
        } else {
            biasL1Size = tilingData.nBL1 * dtypeSize.biasDtypeSize;
        }
    }
    if (hasQuantScale) {
        if (tilingData.fixpParamsFullLoadFlag) {
            scaleL1Size = CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE * dtypeSize.scaleDtypeSize;
        } else {
            scaleL1Size = tilingData.nBL1 * dtypeSize.scaleDtypeSize;
        }
    }
    return al1Size + bl1Size + biasL1Size + scaleL1Size;
}

uint64_t CalcConv3dUsdL1Size(TilingParam &tilingData, DtypeSize dtypeSize, bool hasBias, bool hasQuantScale)
{
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    uint64_t curl1Size = 0;
    if (outputOrder == 1) { // Mmode
        curl1Size = CalcConv3dMmodeUsdL1Size(tilingData, dtypeSize, hasBias, hasQuantScale);
    } else { // HWmode
        curl1Size = CalcConv3dHWmodeUsdL1Size(tilingData, dtypeSize, hasBias, hasQuantScale);
    }

    return curl1Size;
}

uint64_t CalcConv3dUsdL0ASize(TilingParam &tilingData,
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

uint64_t CalcConv3dUsdL0BSize(TilingParam &tilingData,
                              uint32_t weightDtyeSize,
                              int8_t pbBL0)
{
    return tilingData.nL0 * tilingData.kL0 * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcConv3dUsdL0CSize(TilingParam &tilingData, uint32_t outputOrder, int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    if (outputOrder == 0) {
        curl0cSize = tilingData.hoL0 * tilingData.woL0 * tilingData.nL0 * (pbCL0 + 1) * DTYPESIZE_4;
    } else {
        curl0cSize = tilingData.hoL0 * tilingData.nL0 * (pbCL0 + 1) * DTYPESIZE_4;
    }
    return curl0cSize;
}

void CheckHWModeTilingDataValidForConv3dV2(TilingParam &tilingData, uint64_t k0)
{
    // K direction check
    EXPECT_GE(tilingData.kL0, k0);
    EXPECT_LE(tilingData.kL0, std::min(tilingData.kAL1, tilingData.kBL1));
    EXPECT_EQ(tilingData.kL0 % k0, 0);

    // N direction check
    EXPECT_GE(tilingData.nL0, N0_SIZE);
    EXPECT_GE(tilingData.nBL1, tilingData.nL0);
    EXPECT_LE(tilingData.nBL1, CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE, tilingData.nL0) * tilingData.nL0);
    EXPECT_EQ(tilingData.nL0 % N0_SIZE, 0);
    uint32_t nBL1DivCheck = 0;
    if (tilingData.nBL1 % tilingData.nL0 == 0 ||
        tilingData.nBL1 == CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE) {
        nBL1DivCheck = 1;
    }
    EXPECT_EQ(nBL1DivCheck, 1);

    // W direction check
    EXPECT_GE(tilingData.woL0, M0_SIZE);
    EXPECT_GE(tilingData.woL1, tilingData.woL0);
    EXPECT_LE(tilingData.woL1,
        CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreWo, M0_SIZE) * M0_SIZE, tilingData.woL0) * tilingData.woL0);
    EXPECT_EQ(tilingData.woL0 % M0_SIZE, 0);
    if (tilingData.woL0 < CeilDivForConv3dV2(tilingData.orgWo, M0_SIZE) * M0_SIZE) {
        // woL0 does not reach the upper limit, thus hoL0 must be 1.
        EXPECT_EQ(tilingData.hoL0, 1);
    }
    if (tilingData.hoL0 > 1) {
        EXPECT_EQ(tilingData.woL0, CeilDivForConv3dV2(tilingData.orgWo, M0_SIZE) * M0_SIZE);
        EXPECT_EQ(tilingData.woL1, CeilDivForConv3dV2(tilingData.orgWo, M0_SIZE) * M0_SIZE);
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

    EXPECT_LE(tilingData.kAL1, CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW * tilingData.kernelD);
    EXPECT_LE(tilingData.kBL1, CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW * tilingData.kernelD);
    EXPECT_EQ(tilingData.kAL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
    EXPECT_EQ(tilingData.kBL1 % (k0 * tilingData.kernelH * tilingData.kernelW), 0);
    uint32_t kAL1DivCheck = 0;
    if (tilingData.kAL1 % tilingData.kL0 == 0 ||
        tilingData.kAL1 == CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
        kAL1DivCheck = 1;
    }
    EXPECT_EQ(kAL1DivCheck, 1);
    bool kBL1DivCheck = false;
    if (tilingData.kBL1 % tilingData.kL0 == 0 ||
        tilingData.kBL1 == CeilDivForConv3dV2(tilingData.singleCoreCi, k0) * k0 * tilingData.kernelH * tilingData.kernelW) {
        kBL1DivCheck = true;
    }
    EXPECT_EQ(kBL1DivCheck, true);
}

void CheckValidTilingData(TilingParam &tilingData,
                          uint64_t k0,
                          DtypeSize dtypeSize,
                          uint64_t tilingKey,
                          bool isConv3dDequant)
{
    bool hasBias = tilingData.hasBias == 1 || tilingData.hasBias_api == 1;
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    if (isConv3dDequant) {
      ASSERT_GT(tilingData.mUB, 0);
      ASSERT_GT(tilingData.nUB, 0);
      EXPECT_EQ(tilingData.nUB % N0_SIZE, 0);
    }
    // check size
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    int8_t pbAL1 = (pBuffer & 0x08) >> 3;
    int8_t pbBL1 = (pBuffer & 0x10) >> 4;
    uint64_t multi_nBL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE, tilingData.nL0);
    uint64_t multi_mAL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreHo, M0_SIZE) * M0_SIZE, tilingData.hoL0);
    uint64_t mL0max = min(L0A_SIZE / (k0 * (pbAL0 + 1) * dtypeSize.fMapDtypeSize), L0C_SIZE / (N0_SIZE * (pbCL0 + 1) * DTYPESIZE_4));
    uint64_t nL0max = min(L0B_SIZE / (k0 * (pbBL0 + 1) * dtypeSize.weightDtypeSize), L0C_SIZE / (M0_SIZE * (pbCL0 + 1) * DTYPESIZE_4));

    if (outputOrder == 1) {
      ASSERT_GT(tilingData.kAL1, 0);
      ASSERT_GT(tilingData.kBL1, 0);
      ASSERT_GT(tilingData.hoL1, 0);
      ASSERT_GT(tilingData.nBL1, 0);
      ASSERT_GT(tilingData.nL0, 0);
      ASSERT_GT(tilingData.hoL0, 0);
      ASSERT_GT(tilingData.kL0, 0);
      EXPECT_LE(tilingData.nBL1, multi_nBL1max * tilingData.nL0);
      EXPECT_LE(tilingData.hoL1, multi_mAL1max * tilingData.hoL0);
      EXPECT_LE(tilingData.kAL1, tilingData.kernelD * tilingData.kernelH * tilingData.kernelW * CeilDivForConv3dV2(tilingData.orgCi, k0) * k0);
      EXPECT_LE(tilingData.kBL1, tilingData.kernelD * tilingData.kernelH * tilingData.kernelW * CeilDivForConv3dV2(tilingData.orgCi, k0) * k0);
      EXPECT_LE(tilingData.hoL0, CeilDivForConv3dV2(mL0max, M0_SIZE) * M0_SIZE);
      EXPECT_LE(tilingData.nL0, CeilDivForConv3dV2(nL0max, N0_SIZE) * N0_SIZE);
      EXPECT_LE(tilingData.kL0, gcd(CeilDivForConv3dV2(tilingData.kAL1, k0), CeilDivForConv3dV2(tilingData.kBL1, k0)) * k0);

      EXPECT_EQ(tilingData.kAL1 % k0, 0);
      EXPECT_EQ(tilingData.kBL1 % k0, 0);
      EXPECT_EQ(tilingData.nBL1 % N0_SIZE, 0);
      EXPECT_EQ(tilingData.woL1 % M0_SIZE, 0);
      EXPECT_EQ(tilingData.nL0 % N0_SIZE, 0);
      EXPECT_EQ(tilingData.kL0 % k0, 0);
      EXPECT_EQ(tilingData.woL0 % M0_SIZE, 0);

      EXPECT_EQ(tilingData.kAL1 % tilingData.kL0, 0);
      EXPECT_EQ(tilingData.kBL1 % tilingData.kL0, 0);
    } else if (outputOrder == 0) {
      CheckHWModeTilingDataValidForConv3dV2(tilingData, k0);
    }

    EXPECT_LE(CalcConv3dUsdL1Size(tilingData, dtypeSize, hasBias, false), L1_SIZE);
    EXPECT_LE(CalcConv3dUsdL0ASize(tilingData, outputOrder, dtypeSize.fMapDtypeSize, pbAL0), L0A_SIZE);
    EXPECT_LE(CalcConv3dUsdL0BSize(tilingData, dtypeSize.weightDtypeSize, pbBL0), L0B_SIZE);
    EXPECT_LE(CalcConv3dUsdL0CSize(tilingData, outputOrder, pbCL0), L0C_SIZE);
}

void GetOriPadFromPadMode(PadModeParams padModeParams, PadRes& padRes)
{
    std::string padMode = padModeParams.padMode;
    uint32_t strideD = padModeParams.strideD;
    uint32_t strideH = padModeParams.strideH;
    uint32_t strideW = padModeParams.strideW;
    uint32_t dilationD = padModeParams.dilationD;
    uint32_t dilationH = padModeParams.dilationH;
    uint32_t dilationW = padModeParams.dilationW;
    int64_t batch = padModeParams.batch;
    int64_t cin = padModeParams.cin;
    int64_t di = padModeParams.di;
    int64_t hi = padModeParams.hi;
    int64_t wi = padModeParams.wi;
    int64_t cout = padModeParams.cout;
    int64_t kD = padModeParams.kD;
    int64_t kH = padModeParams.kH;
    int64_t kW = padModeParams.kW;

    if (padMode == "SPECIFIC") {
        return;
    }

    if (padMode == "VALID") {
        padRes.padh = 0;
        padRes.padt = 0;
        padRes.padu = 0;
        padRes.padd = 0;
        padRes.padl = 0;
        padRes.padr = 0;
        return;
    } else {
        auto padD = (CeilDivForConv3dV2(di, strideD) - 1) * strideD + dilationD * (kD - 1) - di + 1;
        auto padH = (CeilDivForConv3dV2(hi, strideH) - 1) * strideH + dilationH * (kH - 1) - hi + 1;
        auto padW = (CeilDivForConv3dV2(wi, strideW) - 1) * strideW + dilationW * (kW - 1) - wi + 1;
        if (padMode == "SAME" || padMode == "SAME_UPPER") {
            padRes.padt = CeilDivForConv3dV2(padD, NUM_2);
            padRes.padh = padD - padRes.padt;
            padRes.padd = CeilDivForConv3dV2(padH, NUM_2);
            padRes.padu = padH - padRes.padd;
            padRes.padr = CeilDivForConv3dV2(padW, NUM_2);
            padRes.padl = padW - padRes.padr;
        } else {
            // padMode is "SAME_LOWER"
            padRes.padh = CeilDivForConv3dV2(padD, NUM_2);
            padRes.padt = padD - padRes.padh;
            padRes.padu = CeilDivForConv3dV2(padH, NUM_2);
            padRes.padd = padH - padRes.padu;
            padRes.padl = CeilDivForConv3dV2(padW, NUM_2);
            padRes.padr = padW - padRes.padl;
        }
    }
    return;
}

void Conv3DV2TestCase(Conv3dTestParams conv3dTestParams) {
    vector<int64_t> fmShape = conv3dTestParams.fmShape;
    vector<int64_t> weightShape = conv3dTestParams.weightShape;
    vector<uint32_t> pads = conv3dTestParams.pads;
    vector<uint32_t> strides = conv3dTestParams.strides;
    vector<uint32_t> dilations = conv3dTestParams.dilations;
    ge::DataType dtype = conv3dTestParams.dtype;
    uint32_t isHasBias = conv3dTestParams.isHasBias;
    uint32_t groups = conv3dTestParams.groups;
    int64_t fixBatcho = conv3dTestParams.fixBatcho;
    int64_t fixDo = conv3dTestParams.fixDo;
    int64_t fixHo = conv3dTestParams.fixHo;
    int64_t fixWo = conv3dTestParams.fixWo;
    bool isErrorCaseFlag = conv3dTestParams.isErrorCaseFlag;
    string padMode = conv3dTestParams.padMode;
    bool enableHf32 = conv3dTestParams.enableHf32;
    ge::Format format = conv3dTestParams.format;
    ge::Format outformat = conv3dTestParams.outformat;
    bool isConv3dDequant = conv3dTestParams.isConv3dDequant;
    bool hasScale = conv3dTestParams.hasScale;
    ge::DataType biasDtypeIn = conv3dTestParams.biasDtypeIn;
    ge::DataType outputDtypeIn = conv3dTestParams.outputDtypeIn;
    ge::DataType scaleDtypeIn = conv3dTestParams.scaleDtypeIn;

    bool hasBias = (isHasBias == 1);
    uint32_t padh = pads[0];
    uint32_t padt = pads[1];
    uint32_t padu = pads[DIM_2];
    uint32_t padd = pads[DIM_3];
    uint32_t padl = pads[DIM_4];
    uint32_t padr = pads[DIM_5];
    uint32_t strideD = strides[0];
    uint32_t strideH = strides[1];
    uint32_t strideW = strides[DIM_2];
    uint32_t dilationD = dilations[0];
    uint32_t dilationH = dilations[1];
    uint32_t dilationW = dilations[DIM_2];
    int64_t cout = weightShape[0];
    int64_t kD = weightShape[1];
    int64_t kH = weightShape[DIM_2];
    int64_t kW = weightShape[DIM_3];
    int64_t batch = fmShape[0];
    int64_t cin = fmShape[1];
    int64_t di = fmShape[DIM_2];
    int64_t hi = fmShape[DIM_3];
    int64_t wi = fmShape[DIM_4];
    PadModeParams padModeParams = {padMode, strideD, strideH, strideW,
        dilationD, dilationH, dilationW, batch, cin, di, hi, wi, cout, kD, kH, kW};
    PadRes PadRes = {padh, padt, padu, padd, padl, padr};
    GetOriPadFromPadMode(padModeParams, PadRes);
    ConvShape convShapeDo = {di, kD, padh, padt, dilationD, strideD};
    ConvShape convShapeHo = {hi, kH, padu, padd, dilationH, strideH};
    ConvShape convShapeWo = {wi, kW, padl, padr, dilationW, strideW};
    int64_t Do = InferOut(convShapeDo);
    int64_t ho = InferOut(convShapeHo);
    int64_t wo = InferOut(convShapeWo);
    int64_t batcho = batch;
    if (fixBatcho != 0) {
        batcho = fixBatcho;
    }
    if (fixDo != 0) {
        Do = fixDo;
    }
    if (fixHo != 0) {
        ho = fixHo;
    }
    if (fixWo != 0) {
        wo = fixWo;
    }
    ge::Format fmapFormat = ge::FORMAT_NCDHW;
    ge::Format weightFormat = ge::FORMAT_NCDHW;
    ge::Format outputFormat = ge::FORMAT_NCDHW;
    gert::StorageShape featuremap = {{batch, cin, di, hi, wi}, {batch, cin, di, hi, wi}};
    gert::StorageShape weight = {{cout, cin / groups, kD, kH, kW}, {cout, cin / groups, kD, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape scale = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{batcho, cout, Do, ho, wo}, {batcho, cout, Do, ho, wo}};
    if (format == ge::FORMAT_NDHWC) {
        fmapFormat = ge::FORMAT_NDHWC;
        weightFormat = ge::FORMAT_DHWCN;
        outputFormat = ge::FORMAT_NDHWC;
        featuremap = {{batch, di, hi, wi, cin}, {batch, di, hi, wi, cin}};
        weight = {{kD, kH, kW, cin / groups, cout}, {kD, kH, kW, cin / groups, cout}};
        output = {{batcho, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
    }
    if (outformat == ge::FORMAT_NDHWC) {
        outputFormat = ge::FORMAT_NDHWC;
        output = {{batcho, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
    }
    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;
    if(hasBias) {
        input_shape_ref = {&featuremap, &weight, &bias};
    } else {
        input_shape_ref = {&featuremap, &weight, nullptr};
    }

    if(hasScale) {
        input_shape_ref.push_back(&scale);
    } else {
        input_shape_ref.push_back(nullptr);
    }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides_ref = {1, 1, strideD, strideH, strideW};
    std::vector<int64_t> pads_ref = {padh, padt, padu, padd, padl, padr};
    std::vector<int64_t> dilations_ref = {1, 1, dilationD, dilationH, dilationW};
    if (format == ge::FORMAT_NDHWC) {
        strides_ref = {1, strideD, strideH, strideW, 1};
        dilations_ref = {1, dilationD, dilationH, dilationW, 1};
    }

    std::string op_type = "Conv3DV2";
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
        {ge::DT_INT8, 1}, {ge::DT_BF16, DTYPESIZE_2}, {ge::DT_HIFLOAT8, 1}};

    uint32_t featuremapDtypeSize = dtypesizeMap.at(dtype);
    uint32_t weightDtypeSize = dtypesizeMap.at(dtype);
    uint32_t biasDtypeSize = dtypesizeMap.at(dtype);
    uint64_t k0 = C0_SIZE / featuremapDtypeSize;
    ge::DataType fmapDype = dtype;
    ge::DataType weightDtype = dtype;
    ge::DataType biasDtype = dtype == DT_HIFLOAT8 ? ge::DT_FLOAT : dtype;
    ge::DataType outputDtype = dtype;
    ge::DataType scaleDtype = dtype;
    if (isConv3dDequant) {
        biasDtype = biasDtypeIn;
        outputDtype = outputDtypeIn;
        scaleDtype = scaleDtypeIn;
        outputFormat = outformat;
    }
    uint32_t scaleDtypeSize = dtypesizeMap.at(scaleDtype);
    auto holder = gert::TilingContextFaker().SetOpType(op_type)
                                            .NodeIoNum(NUM_4, 1)
                                            .IrInstanceNum({1, 1, 1, 1})
                                            .InputShapes(input_shape_ref)
                                            .OutputShapes(output_shapes_ref)
                                            .CompileInfo(&compile_info)
                                            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                            .NodeInputTd(0, fmapDype, fmapFormat, fmapFormat)
                                            .NodeInputTd(1, weightDtype, weightFormat, weightFormat)
                                            .NodeInputTd(DIM_2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeInputTd(DIM_3, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeOutputTd(0, outputDtype, outputFormat, outputFormat)
                                            .NodeAttrs({
                                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
                                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(padMode)},
                                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32)}
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
    EXPECT_LE(tilingParam.batchDim * tilingParam.doDim * tilingParam.hoDim * tilingParam.nDim, AICORE_NUM);
    EXPECT_GE(tilingParam.batchDim, 1);
    EXPECT_GE(tilingParam.doDim, 1);
    EXPECT_GE(tilingParam.hoDim, 1);
    EXPECT_GE(tilingParam.nDim, 1);
    if((tilingParam.batchDim >= 1) && (tilingParam.doDim >= 1) && (tilingParam.hoDim >= 1) && (tilingParam.nDim >= 1)) {
        DtypeSize dtypeSize = {featuremapDtypeSize, weightDtypeSize, biasDtypeSize, scaleDtypeSize};
        CheckValidTilingData(tilingParam, k0, dtypeSize, tilingKey, isConv3dDequant);
    }
}
} // namespace

class Conv3dv2Tiling : public testing::Test {
    protected:
      static void SetUpTestCase() {}
      static void TearDownTestCase() {}
};

TEST_F(Conv3dv2Tiling, run_conv3dv2_case_cache_1) {
    for (int i = 0; i < NUM_10; i++) {
        Conv3DV2TestCase(Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1});
    }
}

class Conv3dParameterizedTest : public ::testing::TestWithParam<Conv3dTestParams> {};
TEST_P(Conv3dParameterizedTest, RunConv3DCase) {
    Conv3DV2TestCase(GetParam());
}

INSTANTIATE_TEST_CASE_P(AllConv3dCases, Conv3dParameterizedTest, ::testing::Values(
    Conv3dTestParams{{2,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "VALID", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SAME", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SAME_UPPER", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SAME_LOWER", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,256,256,256}, {1,8,3,4}, {9999,9999,9999,9999,9999,9999}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SAME_LOWER", true, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {2,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,15,16,16,16}, {15,2,2,2}, {1,1,0,0,0,0}, {2,1,1}, {1,1,2}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,17,16,16}, {16,3,3,3}, {1,1,1,1,0,0}, {1,2,1}, {2,1,1}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,15,17}, {4,3,4,3}, {2,2,1,1,1,1}, {2,1,1}, {2,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,18,16,16,15}, {9,3,3,4}, {1,1,2,2,1,1}, {1,2,1}, {2,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,16,4096}, {1,4,2,40}, {1,1,0,0,1,1}, {10,3,2}, {2,2,2}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,64,16,2560}, {1,64,3,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {2,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,15,16,16,16}, {15,2,2,2}, {1,1,0,0,0,0}, {2,1,1}, {1,1,2}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,17,16,16}, {16,3,3,3}, {1,1,1,1,0,0}, {1,2,1}, {2,1,1}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,15,17}, {4,3,4,3}, {2,2,1,1,1,1}, {2,1,1}, {2,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,18,16,16,15}, {9,3,3,4}, {1,1,2,2,1,1}, {1,2,1}, {2,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,16,4096}, {1,4,2,40}, {1,1,0,0,1,1}, {10,3,2}, {2,2,2}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,64,16,2560}, {1,64,3,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,4}, {3,1,5,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,1,300,7}, {3,1,5,3}, {1,1,1,1,1,1}, {1,1,63}, {1,3,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,300,6}, {3,1,1,3}, {0,0,0,0,1,1}, {1,2,1}, {1,255,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,2,18,770}, {3,1,1,2}, {0,0,0,0,1,1}, {1,2,3}, {1,1,255}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,256,1}, {1,1,255,1}, {0,0,254,254,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,256}, {1,1,1,255}, {0,0,0,0,254,254}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,6,75,293}, {13,3,15,6}, {0,0,0,0,0,0}, {1,18,27}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,16,23}, {16,3,3,3}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,9,200,123}, {18,1,4,7}, {0,0,1,1,2,2}, {1,32,18}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,1,1,873}, {11,1,1,6}, {0,0,0,0,1,2}, {1,1,18}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,2,11,242,10}, {4,1,1,1}, {0,0,0,0,0,0}, {1,62,7}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,32,32}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{3,4,16,26,36}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,4,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,3,18,130,130}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,10,66,66}, {240,4,4,4}, {0,0,0,0,0,0}, {2,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {240,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,240,6,34,34}, {120,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{8,120,4,32,32}, {240,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,17,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{4,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,257,257}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,1,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,1,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,1,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,3,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,8,1,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,3,26,134,134}, {64,7,7,7}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {64,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,20,128,128}, {64,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,22,66,66}, {128,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,128,20,64,64}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,22,34,34}, {256,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,256,20,32,32}, {1364,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,682,20,32,32}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,22,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,20,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,20,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,12,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,10,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,7,18,18}, {512,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,512,5,16,16}, {2730,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1365,5,16,16}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,64,22,130,130}, {3,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,320,25,40,64}, {320,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,640,25,20,32}, {640,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,10,16}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,1280,25,5,8}, {1280,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,8,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,8,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,8,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,40,64}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,1,80,128}, {512,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,1,160,256}, {256,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,1,320,512}, {128,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,1,320,512}, {3,3,1,1}, {1,1,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,16,224,224}, {768,2,16,16}, {0,0,0,0,0,0}, {2,16,16}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,192,192}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,193,193}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,35,96,96}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,33,96,96}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,33,97,97}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,19,48,48}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,48,48}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,17,49,49}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,24,24}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,24,24}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,9,24,24}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,9,24,24}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,3,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,320,320}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,321,321}, {128,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,128,17,160,160}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,17,161,161}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,11,80,80}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,9,80,80}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,9,81,81}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,5,40,40}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,512,7,40,40}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,8,5,40,40}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,5,40,40}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,17,257,257}, {128,1,3,3}, {0,0,0,0,1,1}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,11,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,9,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,9,129,129}, {256,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,5,64,64}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,65,65}, {512,1,3,3}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,32,32}, {512,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,5,32,32}, {8,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,5,32,32}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,7,32,32}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,5,64,64}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,11,64,64}, {512,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,9,128,128}, {512,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,128,128}, {256,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,17,128,128}, {256,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {256,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,19,256,256}, {128,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,256,17,256,256}, {128,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,128,19,256,256}, {3,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,512,7,32,32}, {8,3,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,8,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,12,16,16,16}, {1152,16,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{16,1,1,1,2}, {17,5,2,4}, {3,3,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,4,8,64,64}, {8,6,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,16,32,64}, {17,3,5,2}, {1,1,2,2,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,31,24,48,56}, {8,6,9,5}, {2,2,4,4,3,3}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,3,8,64,32}, {4,2,3,3}, {0,0,1,1,2,2}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,4,120,16,16}, {1152,1,2,2}, {0,0,0,0,0,0}, {1,2,2}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {4,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,15,16,16}, {2,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,15,16,16,16}, {15,2,2,2}, {1,1,0,0,0,0}, {2,1,1}, {1,1,2}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,17,16,16}, {16,3,3,3}, {1,1,1,1,0,0}, {1,2,1}, {2,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,15,17}, {4,3,4,3}, {2,2,1,1,1,1}, {2,1,1}, {2,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,18,16,16,15}, {9,3,3,4}, {1,1,2,2,1,1}, {1,2,1}, {2,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,16,4096}, {1,4,2,40}, {1,1,0,0,1,1}, {10,3,2}, {2,2,2}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,64,16,2560}, {1,64,3,3}, {1,1,1,1,1,1}, {1,63,1}, {1,3,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1,1,1,1}, {1,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_HIFLOAT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,12,1,1,1}, {2,1,1,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 0, 2, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,15,1,10,10}, {6,1,2,2}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 3, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,1,11,10}, {12,1,3,4}, {0,0,0,0,1,1}, {1,1,2}, {1,1,1}, ge::DT_FLOAT16, 0, 4, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,15,1,10,11}, {20,1,4,3}, {0,0,1,1,0,0}, {1,1,1}, {1,1,2}, ge::DT_FLOAT16, 1, 5, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,12,1,11,11}, {30,1,6,6}, {0,0,1,2,3,4}, {1,2,1}, {1,1,1}, ge::DT_FLOAT16, 0, 6, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,7,1,11,20}, {42,1,6,6}, {0,0,2,1,4,3}, {1,1,1}, {1,2,1}, ge::DT_FLOAT16, 1, 7, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,32,1,64,64}, {32,1,3,3}, {0,0,1,1,1,1}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 2, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_BF16, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_HIFLOAT8, 0, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT16, 0, 2, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,123,3984,16}, {22,3,23,2}, {0,0,14,14,0,0}, {54,47,15}, {20,10,6}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", true, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,20,16,16,1000000}, {1000000,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,1000000,16,1000000,512}, {16,3,3,3}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 0, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,6160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_HIFLOAT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_BF16, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, ge::DT_FLOAT, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {310,310,310,310,310,310}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{2,256,19,160,160}, {256,3,3,3}, {310,310,310,310,310,310}, {1,1,1}, {1,1,1}, ge::DT_FLOAT16, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NDHWC, ge::Format::FORMAT_NCDHW, false, false, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,16,16,17,16}, {20,4,3,3}, {1,1,1,1,2,2}, {1,1,2}, {1,1,2}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, false, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, true, true, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 2, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 0, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, true, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT},
    Conv3dTestParams{{1,6,87,3400,39}, {45,3,18,2}, {0,0,11,11,1,1}, {4,15,52}, {6,9,6}, ge::DT_INT8, 1, 1, 0, 0, 0, 0, true, "SPECIFIC", false, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, true, false, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT})
);
