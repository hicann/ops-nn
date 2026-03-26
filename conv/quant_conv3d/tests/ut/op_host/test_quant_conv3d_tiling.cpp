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
 * \file test_quant_conv3d_tiling.cpp
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

uint64_t CalcConv3dUsdL0ASize(TilingParam &tilingData, uint32_t outputOrder, uint32_t featuremapDtyeSize, int8_t pbAL0)
{
    uint64_t curl0aSize = 0;
    if (outputOrder == 0) {
        curl0aSize = tilingData.hoL0 * tilingData.woL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.hoL0 * tilingData.kL0 * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcConv3dUsdL0BSize(TilingParam &tilingData, uint32_t weightDtyeSize, int8_t pbBL0)
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

void CheckValidTilingData(TilingParam &tilingData, uint64_t k0, DtypeSize dtypeSize, bool hasBias, uint64_t tilingKey)
{
    int32_t outputOrder = tilingData.singleCoreWo == 0 && tilingData.woL1 == 0;
    uint32_t weightDtypeSize = dtypeSize.weightDtypeSize;
    uint32_t featuremapDtypeSize = dtypeSize.fMapDtypeSize;
    uint32_t biasDtypeSize = dtypeSize.biasDtypeSize;
    uint32_t mmadDtypeSize = DTYPESIZE_4;
    // check size
    uint64_t pBuffer = tilingData.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    int8_t pbBL1 = (pBuffer & 0x10) >> NUM_4;
    uint64_t multi_nBL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreCo, N0_SIZE) * N0_SIZE, tilingData.nL0);
    uint64_t multi_mAL1max = CeilDivForConv3dV2(CeilDivForConv3dV2(tilingData.singleCoreHo, M0_SIZE) * M0_SIZE, tilingData.hoL0);
    uint64_t mL0max = min(L0A_SIZE / (k0 * (pbAL0 + 1) * featuremapDtypeSize), L0C_SIZE / (N0_SIZE * (pbCL0 + 1) * mmadDtypeSize));
    uint64_t nL0max = min(L0B_SIZE / (k0 * (pbBL0 + 1) * weightDtypeSize), L0C_SIZE / (M0_SIZE * (pbCL0 + 1) * mmadDtypeSize));

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
    EXPECT_LE(CalcConv3dUsdL0ASize(tilingData, outputOrder, featuremapDtypeSize, pbAL0), L0A_SIZE);
    EXPECT_LE(CalcConv3dUsdL0BSize(tilingData, weightDtypeSize, pbBL0), L0B_SIZE);
    EXPECT_LE(CalcConv3dUsdL0CSize(tilingData, outputOrder, pbCL0), L0C_SIZE);
}

void QuantConv3DTestCase(vector<int64_t> fmShape, vector<int64_t> weightShape, vector<uint32_t> pads,
                        vector<uint32_t> strides, vector<uint32_t> dilations, uint32_t isHasBias = 1,
                        uint32_t groups = 1, string padMode = "SPECIFIC", int64_t fixBatcho = 0,
                        int64_t fixDo = 0, int64_t fixHo = 0, int64_t fixWo = 0, bool isErrorCaseFlag = false,
                        ge::Format format = ge::Format::FORMAT_NCDHW)
{
    ge::DataType dtype = ge::DT_INT8;
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
    gert::StorageShape quantScale = {{cout}, {cout}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{batcho, cout, Do, ho, wo}, {batch, cout, Do, ho, wo}};
    if (format == ge::FORMAT_NDHWC) {
        fmapFormat = ge::FORMAT_NDHWC;
        weightFormat = ge::FORMAT_DHWCN;
        outputFormat = ge::FORMAT_NDHWC;
        featuremap = {{batch, di, hi, wi, cin}, {batch, di, hi, wi, cin}};
        weight = {{kD, kH, kW, cin / groups, cout}, {kD, kH, kW, cin / groups, cout}};
        output = {{batcho, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
    }

    // 对于可选输入，不传时用nullptr占位
    std::vector<void*> input_shape_ref;
    if(hasBias) {
        input_shape_ref = {&featuremap, &weight, &quantScale, &bias, nullptr};
    } else {
        input_shape_ref = {&featuremap, &weight, &quantScale, nullptr, nullptr};
    }
    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides_ref = {1, 1, strideD, strideH, strideW};
    std::vector<int64_t> pads_ref = {padh, padt, padu, padd, padl, padr};
    std::vector<int64_t> dilations_ref = {1, 1, dilationD, dilationH, dilationW};
    if (format == ge::FORMAT_NDHWC) {
        strides_ref = {1, strideD, strideH, strideW, 1};
        dilations_ref = {1, dilationD, dilationH, dilationW, 1};
    }

    std::string op_type = "QuantConv3D";
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
        {ge::DT_FLOAT16, DTYPESIZE_2}, {ge::DT_FLOAT, DTYPESIZE_4}, {ge::DT_INT8, 1}, {ge::DT_BF16, DTYPESIZE_2}};

    uint32_t featuremapDtypeSize = dtypesizeMap.at(dtype);
    uint32_t weightDtypeSize = dtypesizeMap.at(dtype);
    uint32_t biasDtypeSize = DTYPESIZE_4; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8
    uint32_t scaleDtypeSize = DTYPESIZE_8; // biasdetype is int32 for int8; biasdetype is fp32 for hif8/fp8

    uint64_t k0 = C0_SIZE / featuremapDtypeSize;
    auto holder = gert::TilingContextFaker().SetOpType(op_type)
                                            .NodeIoNum(NUM_5, 1)
                                            .IrInstanceNum({1, 1, 1, 1, 1})
                                            .InputShapes(input_shape_ref)
                                            .OutputShapes(output_shapes_ref)
                                            .CompileInfo(&compile_info)
                                            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                            .NodeInputTd(0, ge::DT_INT8, fmapFormat, fmapFormat)
                                            .NodeInputTd(1, ge::DT_INT8, weightFormat, weightFormat)
                                            .NodeInputTd(DIM_2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeInputTd(DIM_3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeInputTd(DIM_4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeOutputTd(0, ge::DT_FLOAT16, outputFormat, outputFormat)
                                            .NodeAttrs({
                                                {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
                                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(padMode)}
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
        CheckValidTilingData(tilingParam, k0, dtypeSize, hasBias, tilingKey);
    }
}
} // namespace

class QuantConv3dTiling : public testing::Test {
    protected:
      static void SetUpTestCase() {}
      static void TearDownTestCase() {}
};

TEST_F(QuantConv3dTiling, run_quantconv3d_case_1) {
  QuantConv3DTestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, 1);
}
TEST_F(QuantConv3dTiling, run_quantconv3d_case_2) {
  QuantConv3DTestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, 1, 1, "VALID");
}
TEST_F(QuantConv3dTiling, run_quantconv3d_case_3) {
  QuantConv3DTestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, 1, 1, "SAME", 0, 0, 0, 0, true);
}
TEST_F(QuantConv3dTiling, run_quantconv3d_case_4) {
  QuantConv3DTestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, 1, 1, "SAME_LOWER", 0, 0, 0, 0, true);
}

TEST_F(QuantConv3dTiling, run_quantconv3d_NDHWC_case_1) {
  QuantConv3DTestCase({1,1,1,256,1}, {1,1,255,1}, {0,0,0,0,0,0}, {1,1,1}, {1,1,1}, 1, 1, "VALID", 0, 0, 0, 0, true, ge::FORMAT_NDHWC);
}

TEST_F(QuantConv3dTiling, run_quantconv3d_pad_ge_kernel_case_1) {
  QuantConv3DTestCase({1,16,6,256,189}, {16,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, 1);
}

TEST_F(QuantConv3dTiling, run_quantconv3d_pad_ge_kernel_case_2) {
  QuantConv3DTestCase({1,16,3,256,5512}, {64,3,3,3}, {10,10,31,31,31,31}, {1,1,1}, {1,1,1}, 1);
}

TEST_F(QuantConv3dTiling, run_quantconv3d_pad_ge_kernel_valid) {
  QuantConv3DTestCase({1,16,6,256,189}, {16,3,3,3}, {310,310,310,310,310,310}, {1,1,1}, {1,1,1}, 1, 1, "SPECIFIC", 0, 0, 0, 0, true);
}
