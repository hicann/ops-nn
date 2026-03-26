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
 * \file test_conv2d_v2_ascendc_cases.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/conv2d_v2_tiling.h"
#include "platform/platform_info.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;
using namespace conv_tiling_utils;

namespace {
struct Conv2DCaseInputParam {
    vector<uint64_t> fmShape;
    vector<uint64_t> weightShape;
    vector<uint32_t> pads;
    vector<uint32_t> strides;
    vector<uint32_t> dilations;
    vector<uint32_t> NumBlocks;
    std::vector<ConvDtype> dtypes;
    std::vector<bool> flags;
    std::vector<uint8_t> modes;
    uint32_t groups;
};

uint64_t CalcUsdL1Size(optiling::TConv2DTiling &tilingData,
                       Conv2dTiling &tiling,
                       int8_t pbAL1,
                       int8_t pbBL1)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint32_t biasDtyeSize = tiling.hasBias ? DTYPE_SIZE_TAB.at(tiling.descInfo.biasType.dtype) : 0;
    uint32_t scaleDtyeSize = tiling.hasScale ? DTYPE_SIZE_TAB.at(tiling.descInfo.scaleType.dtype) : 0;
    uint64_t curl1Size = 0;
    uint64_t al1Size = 0;
    uint64_t bl1Size = 0;
    uint64_t biasL1Size = 0;
    uint64_t scaleL1Size = 0;
    uint64_t fixpSize = 0;

    auto CUBE_N0 = tiling.cubeInfo.n0;

    if (tiling.outputOrder == 1) {
        uint64_t hoAL1Tmp = min(tilingData.get_hoL1() / tilingData.get_orgWo() + NUM_2,  tilingData.get_orgHo());
        uint64_t hiL1Tmp = min((hoAL1Tmp - 1) * tilingData.get_strideH() + (tilingData.get_kernelH() - 1) /
            tilingData.get_dilationH() + 1, tilingData.get_orgHi());
        uint64_t al1Cin = tiling.isC04Flag ? CUBE_C04_SIZE : (tilingData.get_kAL1() /
            (tilingData.get_kernelH() * tilingData.get_kernelW()));
        al1Size = hiL1Tmp * tilingData.get_orgWi() * al1Cin * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.get_nBL1() * tilingData.get_kBL1() * (pbBL1 + 1) * weightDtyeSize;
        fixpSize = tilingData.get_nL0();
    } else {
        uint64_t hiL1 = InferHiL1(tilingData.get_hoL1(), tilingData.get_orgHi(),
            tilingData.get_kernelH(), tilingData.get_dilationH(), tilingData.get_strideH());
        uint64_t wiL1 = InferWiL1(tilingData.get_woL1(), tilingData.get_orgWi(),
            tilingData.get_kernelW(), tilingData.get_dilationW(), tilingData.get_strideW());
        uint64_t al1Cin = tiling.isC04Flag ? CUBE_C04_SIZE : (tilingData.get_kAL1() /
            (tilingData.get_kernelH() * tilingData.get_kernelW()));
        al1Size = hiL1 * wiL1 * al1Cin * (pbAL1 + 1) * featuremapDtyeSize;
        bl1Size = tilingData.get_nBL1() * tilingData.get_kBL1() * weightDtyeSize;
        fixpSize = tilingData.get_nBL1();
    }
    if (tiling.hasBias) {
        if (tilingData.get_biasFullLoadFlag()) {
            biasL1Size = ConvCeilDiv(tilingData.get_singleCoreCo(), CUBE_N0) * CUBE_N0 * biasDtyeSize;
        } else {
            biasL1Size = fixpSize * biasDtyeSize;
        }
    }

    if (tiling.hasScale) {
        if (tilingData.get_fixpParamsFullLoadFlag()) {
            scaleL1Size = ConvCeilDiv(tilingData.get_singleCoreCo(), CUBE_N0) * CUBE_N0 * scaleDtyeSize;
        } else {
            scaleL1Size = fixpSize * scaleDtyeSize;
        }
    }
    curl1Size = al1Size + bl1Size + biasL1Size + scaleL1Size;
    return curl1Size;
}

uint64_t CalcUsdL0ASize(optiling::TConv2DTiling &tilingData,
                        Conv2dTiling &tiling,
                        int8_t pbAL0)
{
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint64_t curl0aSize = 0;
    if (tiling.outputOrder == 0) {
        curl0aSize = tilingData.get_hoL0() * tilingData.get_woL0() *
            tilingData.get_kL0() * (pbAL0 + 1) * featuremapDtyeSize;
    } else {
        curl0aSize = tilingData.get_hoL0() * tilingData.get_kL0() * (pbAL0 + 1) * featuremapDtyeSize;
    }
    return curl0aSize;
}

uint64_t CalcUsdL0BSize(optiling::TConv2DTiling &tilingData,
                        Conv2dTiling &tiling,
                        int8_t pbBL0)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    return tilingData.get_nL0() * tilingData.get_kL0() * (pbBL0 + 1) * weightDtyeSize;
}

uint64_t CalcUsdL0CSize(optiling::TConv2DTiling &tilingData,
                        Conv2dTiling &tiling,
                        int8_t pbCL0)
{
    uint64_t curl0cSize = 0;
    uint32_t mmadDtypeSize = DTYPE_SIZE_TAB.at(tiling.cubeInfo.madType);
    if (tiling.outputOrder == 0) {
        curl0cSize = tilingData.get_hoL0() * tilingData.get_woL0() * tilingData.get_nL0() * (pbCL0 + 1) * mmadDtypeSize;
    } else {
        curl0cSize = tilingData.get_hoL0() * tilingData.get_nL0() * (pbCL0 + 1) * mmadDtypeSize;
    }
    return curl0cSize;
}

void CheckValidCommon(optiling::TConv2DTiling &tilingData,
                      Conv2dTiling &tiling,
                      uint32_t k0)
{
    uint64_t pBuffer = tilingData.get_pBufferFlag();
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    int8_t pbBL1 = (pBuffer & 0x10) >> NUM_4;
    EXPECT_GT(tilingData.get_kAL1(), 0);
    EXPECT_GT(tilingData.get_kBL1(), 0);
    EXPECT_GT(tilingData.get_hoL1(), 0);
    EXPECT_GT(tilingData.get_nBL1(), 0);
    EXPECT_GT(tilingData.get_hoL0(), 0);
    EXPECT_GT(tilingData.get_kL0(), 0);
    EXPECT_GT(tilingData.get_nL0(), 0);

    EXPECT_EQ(tilingData.get_kAL1() % k0, 0);
    EXPECT_EQ(tilingData.get_nBL1() % CUBE_N0, 0);
    EXPECT_EQ(tilingData.get_nL0() % CUBE_N0, 0);
    EXPECT_EQ(tilingData.get_kL0() % k0, 0);
    EXPECT_EQ(tilingData.get_kAL1() % tilingData.get_kL0(), 0);
    EXPECT_EQ(tilingData.get_kBL1() % tilingData.get_kL0(), 0);

    EXPECT_LE(CalcUsdL1Size(tilingData, tiling, pbAL1, pbBL1), MEM_SIZE_512K);
    EXPECT_LE(CalcUsdL0ASize(tilingData, tiling, pbAL0), MEM_SIZE_64K);
    EXPECT_LE(CalcUsdL0BSize(tilingData, tiling, pbBL0), MEM_SIZE_64K);
    EXPECT_LE(CalcUsdL0CSize(tilingData, tiling, pbCL0), MEM_SIZE_256K);
}

void CheckValidMmode(optiling::TConv2DTiling &tilingData,
                     Conv2dTiling &tiling,
                     uint32_t k0)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint32_t mmadDtypeSize = DTYPE_SIZE_TAB.at(tiling.cubeInfo.madType);
    uint64_t pBuffer = tilingData.get_pBufferFlag();
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    EXPECT_LE(tilingData.get_nBL1(), ConvCeilDiv(
        ConvCeilDiv(tilingData.get_singleCoreCo(), CUBE_N0) * CUBE_N0, tilingData.get_nL0()) * tilingData.get_nL0());
    EXPECT_LE(tilingData.get_hoL1(), ConvCeilDiv(
        ConvCeilDiv(tilingData.get_singleCoreHo(), CUBE_M0) * CUBE_M0, tilingData.get_hoL0()) * tilingData.get_hoL0());
    EXPECT_LE(tilingData.get_kAL1(),
        tilingData.get_kernelH() * tilingData.get_kernelW() * ConvCeilDiv(tilingData.get_orgCi(), k0) * k0);
    EXPECT_LE(tilingData.get_kBL1(),
        tilingData.get_kernelH() * tilingData.get_kernelW() * ConvCeilDiv(tilingData.get_orgCi(), k0) * k0);
    EXPECT_LE(tilingData.get_hoL0(),
        ConvCeilDiv(std::min(MEM_SIZE_64K / (k0 * (pbAL0 + 1) * featuremapDtyeSize),
            MEM_SIZE_256K / (CUBE_N0 * (pbCL0 + 1) * mmadDtypeSize)), CUBE_M0) * CUBE_M0);
    EXPECT_LE(tilingData.get_nL0(),
        ConvCeilDiv(std::min(MEM_SIZE_64K / (k0 * (pbBL0 + 1) * weightDtyeSize),
            MEM_SIZE_256K / (CUBE_M0 * (pbCL0 + 1) * mmadDtypeSize)), CUBE_N0) * CUBE_N0);
    EXPECT_LE(tilingData.get_kL0(),
        ConvGcd(ConvCeilDiv(tilingData.get_kAL1(), k0), ConvCeilDiv(tilingData.get_kBL1(), k0)) * k0);

    EXPECT_EQ(tilingData.get_woL1(), 0);
    EXPECT_EQ(tilingData.get_woL0(), 0);
    EXPECT_EQ(tilingData.get_singleCoreWo(), 0);
    EXPECT_EQ(tilingData.get_hoL1() % CUBE_M0, 0);
    if (!tiling.l1TilingInfo.bl1FullLoad) {
        EXPECT_EQ(tilingData.get_nBL1() % tilingData.get_nL0(), 0);
    }
    if (!tiling.l1TilingInfo.al1FullLoad) {
        EXPECT_EQ(tilingData.get_hoL1() % tilingData.get_hoL0(), 0);
    }
}

void CheckValidC04(optiling::TConv2DTiling &tilingData,
                   uint32_t k0)
{
    EXPECT_EQ(tilingData.get_kAL1(),
        ConvCeilDiv(CUBE_C04_SIZE * tilingData.get_kernelH() * tilingData.get_kernelW(), k0) * k0);
    EXPECT_EQ(tilingData.get_kBL1(),
        ConvCeilDiv(CUBE_C04_SIZE * tilingData.get_kernelH() * tilingData.get_kernelW(), k0) * k0);
    if (tilingData.get_orgHi() > 1) {
        EXPECT_EQ(tilingData.get_woL1(), ConvCeilDiv(tilingData.get_orgWo(), CUBE_M0) * CUBE_M0);
    }
}

void CheckValidHWmodePartOne(optiling::TConv2DTiling &tilingData, uint32_t k0)
{
    // K direction check
    EXPECT_GE(tilingData.get_kL0(), k0);
    EXPECT_LE(tilingData.get_kL0(), std::min(tilingData.get_kAL1(), tilingData.get_kBL1()));
    EXPECT_EQ(tilingData.get_kL0() % k0, 0);
    
    // N direction check
    EXPECT_GE(tilingData.get_nL0(), CUBE_N0);
    EXPECT_GE(tilingData.get_nBL1(), tilingData.get_nL0());
    EXPECT_LE(tilingData.get_nBL1(), ConvCeilDiv(
        ConvCeilDiv(tilingData.get_singleCoreCo(), CUBE_N0) * CUBE_N0, tilingData.get_nL0()) * tilingData.get_nL0());
    EXPECT_EQ(tilingData.get_nL0() % CUBE_N0, 0);
    uint32_t nBL1DivCheck = 0;
    if (tilingData.get_nBL1() % tilingData.get_nL0() == 0 ||
        tilingData.get_nBL1() == ConvCeilDiv(tilingData.get_singleCoreCo(), CUBE_N0) * CUBE_N0) {
        nBL1DivCheck = 1;
    }
    EXPECT_EQ(nBL1DivCheck, 1);
    
    // W direction check
    EXPECT_GE(tilingData.get_woL0(), CUBE_M0);
    EXPECT_GE(tilingData.get_woL1(), tilingData.get_woL0());
    EXPECT_LE(tilingData.get_woL1(), 
        ConvCeilDiv(ConvCeilDiv(tilingData.get_singleCoreWo(), CUBE_M0) * CUBE_M0,
          tilingData.get_woL0()) * tilingData.get_woL0());
    EXPECT_EQ(tilingData.get_woL0() % CUBE_M0, 0);
    if (tilingData.get_woL0() < ConvCeilDiv(tilingData.get_orgWo(), CUBE_M0) * CUBE_M0) {
        // woL0 does not reach the upper limit, thus hoL0 must be 1.
        EXPECT_EQ(tilingData.get_hoL0(), 1);
    }
    if (tilingData.get_hoL0() > 1) {
        EXPECT_EQ(tilingData.get_woL0(), ConvCeilDiv(tilingData.get_orgWo(), CUBE_M0) * CUBE_M0);
        EXPECT_EQ(tilingData.get_woL1(), ConvCeilDiv(tilingData.get_orgWo(), CUBE_M0) * CUBE_M0);
    }

    // H direction check
    EXPECT_GE(tilingData.get_hoL0(), 1);
    EXPECT_GE(tilingData.get_hoL1(), tilingData.get_hoL0());
    EXPECT_LE(tilingData.get_hoL1(), tilingData.get_singleCoreHo());
    uint32_t hoL1Check = 0;
    if (tilingData.get_hoL1() % tilingData.get_hoL0() == 0 ||
        tilingData.get_hoL1() == tilingData.get_singleCoreHo()) {
        hoL1Check = 1;
    }
    EXPECT_EQ(hoL1Check, 1);
}

void CheckValidHWmode(optiling::TConv2DTiling &tilingData,
                      Conv2dTiling &tiling,
                      uint32_t k0)
{
    CheckValidHWmodePartOne(tilingData, k0);

    if (tiling.isC04Flag) {
        CheckValidC04(tilingData, k0);
    } else {
        EXPECT_LE(tilingData.get_kAL1(), ConvCeilDiv(
            tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW());
        EXPECT_LE(tilingData.get_kBL1(), ConvCeilDiv(
            tilingData.get_singleCoreCi(), k0) * k0 * tilingData.get_kernelH() * tilingData.get_kernelW());
        EXPECT_EQ(tilingData.get_kAL1() % (k0 * tilingData.get_kernelH() * tilingData.get_kernelW()), 0);
        EXPECT_EQ(tilingData.get_kBL1() % (k0 * tilingData.get_kernelH() * tilingData.get_kernelW()), 0);
        uint32_t kAL1DivCheck = 0;
        if (tilingData.get_kAL1() % tilingData.get_kL0() == 0 ||
            tilingData.get_kAL1() == ConvCeilDiv(tilingData.get_singleCoreCi(), k0) *
                k0 * tilingData.get_kernelH() * tilingData.get_kernelW()) {
            kAL1DivCheck = 1;
        }
        EXPECT_EQ(kAL1DivCheck, 1);
        bool kBL1DivCheck = false;
        if (tilingData.get_kBL1() % tilingData.get_kL0() == 0 ||
            tilingData.get_kBL1() == ConvCeilDiv(tilingData.get_singleCoreCi(), k0) *
                k0 * tilingData.get_kernelH() * tilingData.get_kernelW()) {
            kBL1DivCheck = true;
        }
        EXPECT_EQ(kBL1DivCheck, true);
    }
}

void CheckValidTilingData(optiling::TConv2DTiling &tilingData,
                          Conv2dTiling &tiling)
{
    uint32_t weightDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.weightType.dtype);
    uint32_t featuremapDtyeSize = DTYPE_SIZE_TAB.at(tiling.descInfo.fMapType.dtype);
    uint32_t biasDtyeSize = tiling.hasBias ? DTYPE_SIZE_TAB.at(tiling.descInfo.biasType.dtype) : 0;

    uint32_t mmadDtypeSize = DTYPE_SIZE_TAB.at(tiling.cubeInfo.madType);

    auto CUBE_M0 = tiling.cubeInfo.m0;
    auto k0 = tiling.cubeInfo.k0;
    auto CUBE_N0 = tiling.cubeInfo.n0;

    uint64_t pBuffer = tilingData.get_pBufferFlag();
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> NUM_2;
    int8_t pbAL1 = (pBuffer & 0x08) >> NUM_3;
    int8_t pbBL1 = (pBuffer & 0x10) >> NUM_4;

    CheckValidCommon(tilingData, tiling, k0);

    if (tiling.outputOrder == 1) {
        CheckValidMmode(tilingData, tiling, k0);
    }
    if (tiling.outputOrder == 0) {
        CheckValidHWmode(tilingData, tiling, k0);
    }
}

void Conv2DCase(Conv2DCaseInputParam inputParams)
{
    bool hasBias = inputParams.flags[0];
    bool hasScale = inputParams.flags[1];
    bool isC04Flag = inputParams.flags[DIM_2];

    uint8_t outputOrder = inputParams.modes[0];
    uint8_t roundMode = inputParams.modes[1];
    uint8_t offsetX = inputParams.modes[DIM_2];

    uint64_t orgCi = inputParams.fmShape[0];
    uint64_t orgHi = inputParams.fmShape[1];
    uint64_t orgWi = inputParams.fmShape[DIM_2];
    uint64_t orgCo = inputParams.weightShape[0];
    uint64_t orgKh = inputParams.weightShape[1];
    uint64_t orgKw = inputParams.weightShape[DIM_2];
    uint64_t orgCi_weight = orgCi / inputParams.groups;

    uint32_t hoDim = 0;
    uint32_t moDim = 0;
    if (outputOrder == 1) {
        moDim = inputParams.NumBlocks[0];
    } else {
        hoDim = inputParams.NumBlocks[0];
    }
    uint32_t nDim = inputParams.NumBlocks[1];

    uint32_t padTop = inputParams.pads[0];
    uint32_t padBottom = inputParams.pads[1];
    uint32_t padLeft = inputParams.pads[DIM_2];
    uint32_t padRight = inputParams.pads[DIM_3];
    uint32_t dilationH = inputParams.dilations[0];
    uint32_t dilationW = inputParams.dilations[1];
    uint32_t strideH = inputParams.strides[0];
    uint32_t strideW = inputParams.strides[1];
    ConvShape convShapeH = {orgHi, orgKh, padTop, padBottom, dilationH, strideH};
    ConvShape convShapeW = {orgWi, orgKw, padLeft, padRight, dilationW, strideW};
    uint64_t orgHo = InferOut(convShapeH);
    uint64_t orgWo = InferOut(convShapeW);
    EXPECT_GT(orgHo, 0);
    EXPECT_GT(orgWo, 0);
    int64_t singlekH = orgKh;
    int64_t singlekW = orgKw;
    int64_t singleCi = orgCi;
    int64_t singleCi_weight = orgCi_weight;
    int64_t singleCo = ConvAlignB(ConvCeilDiv(ConvAlignB(orgCo, CUBE_N0), nDim), CUBE_N0);
    int64_t singleWo = -1;
    int64_t singleHo = -1;
    int64_t singleM = -1;
    if (outputOrder == 1) {
        singleM = ConvCeilDiv(orgHo * orgWo, moDim);
    } else {
        singleWo = orgWo;
        singleHo = ConvCeilDiv(orgHo, hoDim);
    }

    ConvDtype featuremapDtype = inputParams.dtypes[0];
    ConvDtype weightDtype = inputParams.dtypes[1];
    ConvDtype biasDtype;
    ConvDtype outputDtype;
    if (hasBias) {
        biasDtype = inputParams.dtypes[DIM_2];
        outputDtype = inputParams.dtypes[DIM_3];
    } else {
        outputDtype = inputParams.dtypes[DIM_2];
    }

    PlatformInfo platformInfo;
    platformInfo.npuArch = NpuArch::DAV_3510;
    platformInfo.l1Size = MEM_SIZE_512K;
    platformInfo.l0ASize = MEM_SIZE_64K;
    platformInfo.l0BSize = MEM_SIZE_64K;
    platformInfo.l0CSize = MEM_SIZE_256K;
    platformInfo.ubSize = MEM_SIZE_256K;
    platformInfo.btSize = MEM_SIZE_4K;
    platformInfo.fbSize = MEM_SIZE_4K;
    platformInfo.aivPerAic = NUM_2;

    Conv2dTiling testTiling(platformInfo);
    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(singleCi_weight, singlekH, singlekW);

    if (outputOrder == 1) {
        testTiling.SetSingleOutputShape(singleCo, singleM, 1);
        testTiling.SetOutputOrder(1);
    } else {
        testTiling.SetSingleOutputShape(singleCo, singleHo, singleWo, 1);
        testTiling.SetOutputOrder(0);
    }
    if (hasScale) {
        testTiling.SetQuantScale(hasScale);
        testTiling.SetRoundMode(roundMode);
        testTiling.SetOffsetx(offsetX);
    }
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, featuremapDtype);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, weightDtype);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, outputDtype);
    if (hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, biasDtype);
    }
    testTiling.SetPadding(padTop, padBottom, padLeft, padRight);
    testTiling.SetDilation(dilationH, dilationW);
    testTiling.SetStride(strideH, strideW);
    testTiling.SetC04Flag(isC04Flag);
    testTiling.SetGroups(inputParams.groups);
    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);

    CheckValidTilingData(tilingData, testTiling);
}
} // namespace

class TestConv2dTilingCases : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};
class TestConv2dHWmode : public TestConv2dTilingCases {};
class TestConv2dMmode : public TestConv2dTilingCases {};

class Conv2dApiParameterizedTest : public ::testing::TestWithParam<Conv2DCaseInputParam> {};
TEST_P(Conv2dApiParameterizedTest, RunConv2DApiCase) {
    Conv2DCase(GetParam());
}

INSTANTIATE_TEST_CASE_P(Conv2dApiTestCases, Conv2dApiParameterizedTest, ::testing::Values(
    Conv2DCaseInputParam{{4,16,16},{16,3,3},{1,1,1,1},{1,1},{1,1},{16,2}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_1
    Conv2DCaseInputParam{{4,1,60000},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_2
    Conv2DCaseInputParam{{4,1,65536},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_3
    Conv2DCaseInputParam{{4,64,2055},{16,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_4
    Conv2DCaseInputParam{{3,214,214},{1408,14,14},{0,0,0,0},{14,14},{1,1},{2,2}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_5
    Conv2DCaseInputParam{{3,1344,1344},{64,7,7},{3,3,3,3},{2,2},{1,1},{16,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,true},{0,0,0}, 1}, // test_c04_Case_6
    Conv2DCaseInputParam{{3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{8,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_7
    Conv2DCaseInputParam{{3,87,97},{13,33,6},{1,2,3,4},{12,11},{1,1},{2,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1}, // test_c04_Case_8
    Conv2DCaseInputParam{{4,1,16},{16,1,3},{1,1,1,1},{1,1},{1,1},{16,2}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_9
    Conv2DCaseInputParam{{4,1,106},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_10
    Conv2DCaseInputParam{{4,1,188},{512,1,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_11
    Conv2DCaseInputParam{{4,1,2560},{16,1,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_12
    Conv2DCaseInputParam{{3,1,267},{1408,1,14},{0,0,0,0},{14,14},{1,1},{2,2}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_13
    Conv2DCaseInputParam{{3,1,56},{64,1,7},{3,3,3,3},{2,2},{1,1},{16,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,true},{0,0,0}, 1}, // test_c04_Case_14
    Conv2DCaseInputParam{{3,1,1216},{128,1,3},{1,1,1,1},{1,1},{1,1},{8,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_15
    Conv2DCaseInputParam{{3,1,97},{13,1,6},{1,2,3,4},{12,11},{1,1},{2,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1}, // test_c04_Case_16
    Conv2DCaseInputParam{{1,29707,2192},{160,21,10},{7,5,4,2},{56,5},{1,8},{8,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,true},{0,0,0}, 1}, // test_c04_Case_17
    Conv2DCaseInputParam{{3,4096,4096},{64,7,7},{3,3,3,3},{2,2},{1,1},{32,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,true},{0,0,0}, 1}, // test_c04_Case_18
    Conv2DCaseInputParam{{1,1,16},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_0
    Conv2DCaseInputParam{{1280,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_1
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_2
    Conv2DCaseInputParam{{640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_3
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_4
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_5
    Conv2DCaseInputParam{{1920,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_6
    Conv2DCaseInputParam{{640,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_7
    Conv2DCaseInputParam{{2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_8
    Conv2DCaseInputParam{{2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_9
    Conv2DCaseInputParam{{1920,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_10
    Conv2DCaseInputParam{{1280,72,56},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_11
    Conv2DCaseInputParam{{960,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_12
    Conv2DCaseInputParam{{640,144,112},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_13
    Conv2DCaseInputParam{{640,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_14
    Conv2DCaseInputParam{{1920,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_15
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_16
    Conv2DCaseInputParam{{960,52,76},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_17
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_18
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_19
    Conv2DCaseInputParam{{640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_20
    Conv2DCaseInputParam{{640,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_21
    Conv2DCaseInputParam{{640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_22
    Conv2DCaseInputParam{{2560,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_23
    Conv2DCaseInputParam{{960,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_24
    Conv2DCaseInputParam{{1280,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_25
    Conv2DCaseInputParam{{1280,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_26
    Conv2DCaseInputParam{{640,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_27
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_28
    Conv2DCaseInputParam{{640,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_29
    Conv2DCaseInputParam{{960,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_30
    Conv2DCaseInputParam{{2560,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_31
    Conv2DCaseInputParam{{320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_32
    Conv2DCaseInputParam{{640,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_33
    Conv2DCaseInputParam{{1280,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_34
    Conv2DCaseInputParam{{1280,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_35
    Conv2DCaseInputParam{{640,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_36
    Conv2DCaseInputParam{{2560,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_37
    Conv2DCaseInputParam{{2560,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_38
    Conv2DCaseInputParam{{1280,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_39
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_40
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_41
    Conv2DCaseInputParam{{1280,52,76},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_42
    Conv2DCaseInputParam{{640,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_43
    Conv2DCaseInputParam{{1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_44
    Conv2DCaseInputParam{{1280,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_45
    Conv2DCaseInputParam{{960,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_46
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_47
    Conv2DCaseInputParam{{1920,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_48
    Conv2DCaseInputParam{{2560,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_49
    Conv2DCaseInputParam{{960,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_50
    Conv2DCaseInputParam{{1920,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_51
    Conv2DCaseInputParam{{320,144,112},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_52
    Conv2DCaseInputParam{{1920,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_53
    Conv2DCaseInputParam{{960,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_54
    Conv2DCaseInputParam{{960,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_55
    Conv2DCaseInputParam{{960,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_56
    Conv2DCaseInputParam{{1920,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_57
    Conv2DCaseInputParam{{960,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_58
    Conv2DCaseInputParam{{1920,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_59
    Conv2DCaseInputParam{{1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_60
    Conv2DCaseInputParam{{960,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_61
    Conv2DCaseInputParam{{960,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_62
    Conv2DCaseInputParam{{640,104,152},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_63
    Conv2DCaseInputParam{{2560,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_64
    Conv2DCaseInputParam{{1280,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_65
    Conv2DCaseInputParam{{1920,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_66
    Conv2DCaseInputParam{{320,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_67
    Conv2DCaseInputParam{{640,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_68
    Conv2DCaseInputParam{{1920,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_69
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_70
    Conv2DCaseInputParam{{1920,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_71
    Conv2DCaseInputParam{{960,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_72
    Conv2DCaseInputParam{{960,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_73
    Conv2DCaseInputParam{{320,112,144},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_74
    Conv2DCaseInputParam{{960,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_75
    Conv2DCaseInputParam{{2560,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_76
    Conv2DCaseInputParam{{960,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_77
    Conv2DCaseInputParam{{3,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_78
    Conv2DCaseInputParam{{960,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_79
    Conv2DCaseInputParam{{1920,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_80
    Conv2DCaseInputParam{{320,152,104},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_81
    Conv2DCaseInputParam{{3,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_82
    Conv2DCaseInputParam{{1920,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_83
    Conv2DCaseInputParam{{2560,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_84
    Conv2DCaseInputParam{{128,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_86
    Conv2DCaseInputParam{{1280,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_87
    Conv2DCaseInputParam{{128,897,1153},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_88
    Conv2DCaseInputParam{{1280,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_89
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_90
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_91
    Conv2DCaseInputParam{{960,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_92
    Conv2DCaseInputParam{{960,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_93
    Conv2DCaseInputParam{{128,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_94
    Conv2DCaseInputParam{{1920,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_95
    Conv2DCaseInputParam{{128,1153,897},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_97
    Conv2DCaseInputParam{{2560,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_98
    Conv2DCaseInputParam{{1920,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_99
    Conv2DCaseInputParam{{640,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_100
    Conv2DCaseInputParam{{128,1025,1025},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_101
    Conv2DCaseInputParam{{1280,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_102
    Conv2DCaseInputParam{{1280,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_103
    Conv2DCaseInputParam{{3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_104
    Conv2DCaseInputParam{{640,152,104},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_105
    Conv2DCaseInputParam{{960,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_106
    Conv2DCaseInputParam{{128,576,448},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_107
    Conv2DCaseInputParam{{1280,56,72},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_108
    Conv2DCaseInputParam{{640,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_109
    Conv2DCaseInputParam{{256,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_110
    Conv2DCaseInputParam{{640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_112
    Conv2DCaseInputParam{{640,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_113
    Conv2DCaseInputParam{{3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_114
    Conv2DCaseInputParam{{128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_115
    Conv2DCaseInputParam{{128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_116
    Conv2DCaseInputParam{{320,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_117
    Conv2DCaseInputParam{{320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_118
    Conv2DCaseInputParam{{3,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_119
    Conv2DCaseInputParam{{128,544,480},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_120
    Conv2DCaseInputParam{{1280,40,96},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_121
    Conv2DCaseInputParam{{1920,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_122
    Conv2DCaseInputParam{{128,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_123
    Conv2DCaseInputParam{{256,208,304},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_124
    Conv2DCaseInputParam{{128,833,1217},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_125
    Conv2DCaseInputParam{{1280,76,52},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_126
    Conv2DCaseInputParam{{256,609,417},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_127
    Conv2DCaseInputParam{{3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_128
    Conv2DCaseInputParam{{256,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_129
    Conv2DCaseInputParam{{1280,76,52},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_130
    Conv2DCaseInputParam{{128,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_131
    Conv2DCaseInputParam{{256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_132
    Conv2DCaseInputParam{{256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_133
    Conv2DCaseInputParam{{128,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_134
    Conv2DCaseInputParam{{1920,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_135
    Conv2DCaseInputParam{{512,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_136
    Conv2DCaseInputParam{{128,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_137
    Conv2DCaseInputParam{{256,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_138
    Conv2DCaseInputParam{{640,112,144},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_139
    Conv2DCaseInputParam{{128,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_140
    Conv2DCaseInputParam{{256,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_141
    Conv2DCaseInputParam{{256,417,609},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_142
    Conv2DCaseInputParam{{1920,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_143
    Conv2DCaseInputParam{{128,448,576},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_144
    Conv2DCaseInputParam{{512,112,144},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_145
    Conv2DCaseInputParam{{256,449,577},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_146
    Conv2DCaseInputParam{{128,1217,833},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_147
    Conv2DCaseInputParam{{1920,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_148
    Conv2DCaseInputParam{{512,144,112},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_149
    Conv2DCaseInputParam{{256,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_150
    Conv2DCaseInputParam{{640,80,192},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_151
    Conv2DCaseInputParam{{256,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_152
    Conv2DCaseInputParam{{512,152,104},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_153
    Conv2DCaseInputParam{{512,144,112},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_154
    Conv2DCaseInputParam{{256,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_155
    Conv2DCaseInputParam{{3,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_156
    Conv2DCaseInputParam{{128,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_157
    Conv2DCaseInputParam{{256,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_158
    Conv2DCaseInputParam{{512,128,128},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_159
    Conv2DCaseInputParam{{128,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_160
    Conv2DCaseInputParam{{4,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_161
    Conv2DCaseInputParam{{256,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_162
    Conv2DCaseInputParam{{512,104,152},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_163
    Conv2DCaseInputParam{{256,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_164
    Conv2DCaseInputParam{{4,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_165
    Conv2DCaseInputParam{{128,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_166
    Conv2DCaseInputParam{{512,289,225},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_167
    Conv2DCaseInputParam{{512,209,305},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_168
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_169
    Conv2DCaseInputParam{{512,112,144},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_170
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_171
    Conv2DCaseInputParam{{128,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_172
    Conv2DCaseInputParam{{512,305,209},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_173
    Conv2DCaseInputParam{{256,545,481},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_174
    Conv2DCaseInputParam{{128,1089,961},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_175
    Conv2DCaseInputParam{{320,128,128},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_176
    Conv2DCaseInputParam{{512,128,128},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_177
    Conv2DCaseInputParam{{512,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_178
    Conv2DCaseInputParam{{256,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_179
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_180
    Conv2DCaseInputParam{{512,136,120},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_181
    Conv2DCaseInputParam{{512,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_182
    Conv2DCaseInputParam{{512,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_183
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_184
    Conv2DCaseInputParam{{512,136,120},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_185
    Conv2DCaseInputParam{{512,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_186
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_187
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_188
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_189
    Conv2DCaseInputParam{{4,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_190
    Conv2DCaseInputParam{{512,152,104},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_191
    Conv2DCaseInputParam{{512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_192
    Conv2DCaseInputParam{{320,64,64},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_193
    Conv2DCaseInputParam{{320,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_194
    Conv2DCaseInputParam{{512,104,152},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_195
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_196
    Conv2DCaseInputParam{{320,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_197
    Conv2DCaseInputParam{{4,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_198
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_199
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_200
    Conv2DCaseInputParam{{4,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_201
    Conv2DCaseInputParam{{640,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_202
    Conv2DCaseInputParam{{320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_203
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_204
    Conv2DCaseInputParam{{256,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_205
    Conv2DCaseInputParam{{1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_206
    Conv2DCaseInputParam{{320,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_207
    Conv2DCaseInputParam{{320,136,120},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_208
    Conv2DCaseInputParam{{8,112,144},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_209
    Conv2DCaseInputParam{{640,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_210
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_211
    Conv2DCaseInputParam{{320,72,56},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_212
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_213
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_214
    Conv2DCaseInputParam{{8,136,120},{8,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_215
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_216
    Conv2DCaseInputParam{{640,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_217
    Conv2DCaseInputParam{{4,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_218
    Conv2DCaseInputParam{{640,34,30},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_219
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_220
    Conv2DCaseInputParam{{1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_221
    Conv2DCaseInputParam{{640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_222
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_223
    Conv2DCaseInputParam{{640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_224
    Conv2DCaseInputParam{{320,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_225
    Conv2DCaseInputParam{{320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_226
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_227
    Conv2DCaseInputParam{{640,68,60},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_228
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_229
    Conv2DCaseInputParam{{640,64,64},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_230
    Conv2DCaseInputParam{{640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_231
    Conv2DCaseInputParam{{640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_232
    Conv2DCaseInputParam{{640,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_233
    Conv2DCaseInputParam{{256,224,288},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_234
    Conv2DCaseInputParam{{128,608,416},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_235
    Conv2DCaseInputParam{{128,512,512},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_236
    Conv2DCaseInputParam{{8,152,104},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_237
    Conv2DCaseInputParam{{512,225,289},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_238
    Conv2DCaseInputParam{{256,513,513},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_239
    Conv2DCaseInputParam{{256,577,449},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_240
    Conv2DCaseInputParam{{128,416,608},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_241
    Conv2DCaseInputParam{{256,256,256},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_242
    Conv2DCaseInputParam{{8,144,112},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_243
    Conv2DCaseInputParam{{512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_244
    Conv2DCaseInputParam{{512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{0,0,0}, 1}, // test_fp16_sdxl_244
    Conv2DCaseInputParam{{3,114,376},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_5
    Conv2DCaseInputParam{{6,234,216},{9,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_6
    Conv2DCaseInputParam{{9,456,234},{10,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_7
    Conv2DCaseInputParam{{12,246,342},{11,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_8
    Conv2DCaseInputParam{{21,53,923},{14,1,3},{0,0,0,0},{6,7},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_9
    Conv2DCaseInputParam{{24,57,821},{15,1,2},{0,0,0,0},{8,9},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_10
    Conv2DCaseInputParam{{27,24,714},{16,1,1},{0,0,0,0},{10,11},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_11
    Conv2DCaseInputParam{{30,42,523},{17,1,2},{0,0,0,0},{12,13},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_12
    Conv2DCaseInputParam{{45,443,1},{22,255,1},{21,22,0,0},{22,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_15
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_59
    Conv2DCaseInputParam{{1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_60
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_62
    Conv2DCaseInputParam{{1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_63
    Conv2DCaseInputParam{{1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_65
    Conv2DCaseInputParam{{1,1,67108832},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_66
    Conv2DCaseInputParam{{1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_68
    Conv2DCaseInputParam{{1,1,67108832},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_69
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_70
    Conv2DCaseInputParam{{22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_77
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_83
    Conv2DCaseInputParam{{8,128,128},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_84
    Conv2DCaseInputParam{{256,304,208},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_85
    Conv2DCaseInputParam{{8,104,152},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_86
    Conv2DCaseInputParam{{256,272,240},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_87
    Conv2DCaseInputParam{{320,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_88
    Conv2DCaseInputParam{{512,257,257},{512,1,1},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_89
    Conv2DCaseInputParam{{512,273,241},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_90
    Conv2DCaseInputParam{{320,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_91
    Conv2DCaseInputParam{{320,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_92
    Conv2DCaseInputParam{{640,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{0,0,0}, 1}, // test_bf16_gen_93
    Conv2DCaseInputParam{{3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_84
    Conv2DCaseInputParam{{8,94,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_94
    Conv2DCaseInputParam{{9,95,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_95
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_106
    Conv2DCaseInputParam{{22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_113
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_119
    Conv2DCaseInputParam{{3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_120
    Conv2DCaseInputParam{{12,86,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_130
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_142
    Conv2DCaseInputParam{{22,59,88},{13,11,9},{4,5,3,2},{7,8},{3,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_149
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{3,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_155
    Conv2DCaseInputParam{{3,58,86},{12,7,5},{3,6,1,2},{3,8},{2,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_156
    Conv2DCaseInputParam{{12,86,96},{12,5,22},{1,2,3,4},{10,9},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_166
    Conv2DCaseInputParam{{1,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_178
    Conv2DCaseInputParam{{24,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_185
    Conv2DCaseInputParam{{7,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_191
    Conv2DCaseInputParam{{6,58,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_192
    Conv2DCaseInputParam{{12,96,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_202
    Conv2DCaseInputParam{{11,97,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_203
    Conv2DCaseInputParam{{64,30,30},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_216
    Conv2DCaseInputParam{{15,15,16},{1,2,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_217
    Conv2DCaseInputParam{{3,4,4},{1000,2,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_219
    Conv2DCaseInputParam{{291,480,480},{291,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_221
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_227
    Conv2DCaseInputParam{{1,1,134217712},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_228
    Conv2DCaseInputParam{{320,68,60},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_223
    Conv2DCaseInputParam{{320,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{0,0,0}, 1}, // test_fp32_gen_224
    Conv2DCaseInputParam{{512,18,18},{2048,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1}, // test_fp32_gen_214
    Conv2DCaseInputParam{{768,16,16},{768,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1}, // test_fp32_gen_220
    Conv2DCaseInputParam{{512,105,105},{1024,1,1},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{0,0,0}, 1}  // test_fp32_gen_222
));

// ===== TestConv2dMmode =====
INSTANTIATE_TEST_CASE_P(TestConv2dMmodeCases,Conv2dApiParameterizedTest, ::testing::Values(
    Conv2DCaseInputParam{{1,1,16},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_0
    Conv2DCaseInputParam{{1280,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_1
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_2
    Conv2DCaseInputParam{{640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_3
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_4
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_5
    Conv2DCaseInputParam{{1920,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_6
    Conv2DCaseInputParam{{640,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_7
    Conv2DCaseInputParam{{2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_8
    Conv2DCaseInputParam{{2560,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_9
    Conv2DCaseInputParam{{1920,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_10
    Conv2DCaseInputParam{{1280,72,56},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_11
    Conv2DCaseInputParam{{960,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_12
    Conv2DCaseInputParam{{640,144,112},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_13
    Conv2DCaseInputParam{{640,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_14
    Conv2DCaseInputParam{{1920,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_15
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_16
    Conv2DCaseInputParam{{960,52,76},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_17
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_18
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_19
    Conv2DCaseInputParam{{640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_20
    Conv2DCaseInputParam{{640,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_21
    Conv2DCaseInputParam{{640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_22
    Conv2DCaseInputParam{{2560,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_23
    Conv2DCaseInputParam{{960,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_24
    Conv2DCaseInputParam{{1280,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_25
    Conv2DCaseInputParam{{1280,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_26
    Conv2DCaseInputParam{{640,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_27
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_28
    Conv2DCaseInputParam{{640,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_29
    Conv2DCaseInputParam{{960,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_30
    Conv2DCaseInputParam{{2560,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_31
    Conv2DCaseInputParam{{320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_32
    Conv2DCaseInputParam{{640,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_33
    Conv2DCaseInputParam{{1280,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_34
    Conv2DCaseInputParam{{1280,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_35
    Conv2DCaseInputParam{{640,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_36
    Conv2DCaseInputParam{{2560,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_37
    Conv2DCaseInputParam{{2560,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_38
    Conv2DCaseInputParam{{1280,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_39
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_40
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_41
    Conv2DCaseInputParam{{1280,52,76},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_42
    Conv2DCaseInputParam{{640,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_43
    Conv2DCaseInputParam{{1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_44
    Conv2DCaseInputParam{{1280,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_45
    Conv2DCaseInputParam{{960,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_46
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_47
    Conv2DCaseInputParam{{1920,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_48
    Conv2DCaseInputParam{{2560,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_49
    Conv2DCaseInputParam{{960,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_50
    Conv2DCaseInputParam{{1920,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_51
    Conv2DCaseInputParam{{320,144,112},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_52
    Conv2DCaseInputParam{{1920,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_53
    Conv2DCaseInputParam{{960,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_54
    Conv2DCaseInputParam{{960,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_55
    Conv2DCaseInputParam{{960,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_56
    Conv2DCaseInputParam{{1920,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_57
    Conv2DCaseInputParam{{960,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_58
    Conv2DCaseInputParam{{1920,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_59
    Conv2DCaseInputParam{{1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_60
    Conv2DCaseInputParam{{960,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_61
    Conv2DCaseInputParam{{960,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_62
    Conv2DCaseInputParam{{640,104,152},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_63
    Conv2DCaseInputParam{{2560,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_64
    Conv2DCaseInputParam{{1280,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_65
    Conv2DCaseInputParam{{1920,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_66
    Conv2DCaseInputParam{{320,80,192},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_67
    Conv2DCaseInputParam{{640,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_68
    Conv2DCaseInputParam{{1920,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_69
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_70
    Conv2DCaseInputParam{{1920,36,28},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_71
    Conv2DCaseInputParam{{960,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_72
    Conv2DCaseInputParam{{960,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_73
    Conv2DCaseInputParam{{320,112,144},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_74
    Conv2DCaseInputParam{{960,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_75
    Conv2DCaseInputParam{{2560,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_76
    Conv2DCaseInputParam{{960,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_77
    Conv2DCaseInputParam{{3,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_78
    Conv2DCaseInputParam{{960,144,112},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_79
    Conv2DCaseInputParam{{1920,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_80
    Conv2DCaseInputParam{{320,152,104},{4,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_81
    Conv2DCaseInputParam{{3,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_82
    Conv2DCaseInputParam{{1920,26,38},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_83
    Conv2DCaseInputParam{{2560,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_84
    Conv2DCaseInputParam{{128,1152,896},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_86
    Conv2DCaseInputParam{{1280,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_87
    Conv2DCaseInputParam{{128,897,1153},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_88
    Conv2DCaseInputParam{{1280,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_89
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_90
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_91
    Conv2DCaseInputParam{{960,104,152},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_92
    Conv2DCaseInputParam{{960,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_93
    Conv2DCaseInputParam{{128,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_94
    Conv2DCaseInputParam{{1920,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_95
    Conv2DCaseInputParam{{128,1153,897},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_97
    Conv2DCaseInputParam{{2560,20,48},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_98
    Conv2DCaseInputParam{{1920,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_99
    Conv2DCaseInputParam{{640,112,144},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_100
    Conv2DCaseInputParam{{128,1025,1025},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_101
    Conv2DCaseInputParam{{1280,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_102
    Conv2DCaseInputParam{{1280,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_103
    Conv2DCaseInputParam{{3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_104
    Conv2DCaseInputParam{{640,152,104},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_105
    Conv2DCaseInputParam{{960,80,192},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_106
    Conv2DCaseInputParam{{128,576,448},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_107
    Conv2DCaseInputParam{{1280,56,72},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_108
    Conv2DCaseInputParam{{640,152,104},{320,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_109
    Conv2DCaseInputParam{{256,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_110
    Conv2DCaseInputParam{{640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_112
    Conv2DCaseInputParam{{640,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_113
    Conv2DCaseInputParam{{3,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_114
    Conv2DCaseInputParam{{128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_115
    Conv2DCaseInputParam{{128,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_116
    Conv2DCaseInputParam{{320,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_117
    Conv2DCaseInputParam{{320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_118
    Conv2DCaseInputParam{{3,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_119
    Conv2DCaseInputParam{{128,544,480},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_120
    Conv2DCaseInputParam{{1280,40,96},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_121
    Conv2DCaseInputParam{{1920,20,48},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_122
    Conv2DCaseInputParam{{128,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_123
    Conv2DCaseInputParam{{256,208,304},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_124
    Conv2DCaseInputParam{{128,833,1217},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_125
    Conv2DCaseInputParam{{1280,76,52},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_126
    Conv2DCaseInputParam{{256,609,417},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_127
    Conv2DCaseInputParam{{3,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_128
    Conv2DCaseInputParam{{256,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_129
    Conv2DCaseInputParam{{1280,76,52},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_130
    Conv2DCaseInputParam{{128,896,1152},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_131
    Conv2DCaseInputParam{{256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_132
    Conv2DCaseInputParam{{256,288,224},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_133
    Conv2DCaseInputParam{{128,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_134
    Conv2DCaseInputParam{{1920,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_135
    Conv2DCaseInputParam{{512,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_136
    Conv2DCaseInputParam{{128,1024,1024},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_137
    Conv2DCaseInputParam{{256,608,416},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_138
    Conv2DCaseInputParam{{640,112,144},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_139
    Conv2DCaseInputParam{{128,416,608},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_140
    Conv2DCaseInputParam{{256,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_141
    Conv2DCaseInputParam{{256,417,609},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_142
    Conv2DCaseInputParam{{1920,40,96},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_143
    Conv2DCaseInputParam{{128,448,576},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_144
    Conv2DCaseInputParam{{512,112,144},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_145
    Conv2DCaseInputParam{{256,449,577},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_146
    Conv2DCaseInputParam{{128,1217,833},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_147
    Conv2DCaseInputParam{{1920,40,96},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_148
    Conv2DCaseInputParam{{512,144,112},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_149
    Conv2DCaseInputParam{{256,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_150
    Conv2DCaseInputParam{{640,80,192},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_151
    Conv2DCaseInputParam{{256,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_152
    Conv2DCaseInputParam{{512,152,104},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_153
    Conv2DCaseInputParam{{512,144,112},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_154
    Conv2DCaseInputParam{{256,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_155
    Conv2DCaseInputParam{{3,1216,832},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_156
    Conv2DCaseInputParam{{128,832,1216},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_157
    Conv2DCaseInputParam{{256,544,480},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_158
    Conv2DCaseInputParam{{512,128,128},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_159
    Conv2DCaseInputParam{{128,1088,960},{128,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_160
    Conv2DCaseInputParam{{4,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_161
    Conv2DCaseInputParam{{256,448,576},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_162
    Conv2DCaseInputParam{{512,104,152},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_163
    Conv2DCaseInputParam{{256,304,208},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_164
    Conv2DCaseInputParam{{4,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_165
    Conv2DCaseInputParam{{128,576,448},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_166
    Conv2DCaseInputParam{{512,289,225},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_167
    Conv2DCaseInputParam{{512,209,305},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_168
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_169
    Conv2DCaseInputParam{{512,112,144},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_170
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_171
    Conv2DCaseInputParam{{128,512,512},{256,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_172
    Conv2DCaseInputParam{{512,305,209},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_173
    Conv2DCaseInputParam{{256,545,481},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_174
    Conv2DCaseInputParam{{128,1089,961},{128,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_175
    Conv2DCaseInputParam{{320,128,128},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_176
    Conv2DCaseInputParam{{512,128,128},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_177
    Conv2DCaseInputParam{{512,256,256},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_178
    Conv2DCaseInputParam{{256,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_179
    Conv2DCaseInputParam{{320,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_180
    Conv2DCaseInputParam{{512,136,120},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_181
    Conv2DCaseInputParam{{512,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_182
    Conv2DCaseInputParam{{512,208,304},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_183
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_184
    Conv2DCaseInputParam{{512,136,120},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_185
    Conv2DCaseInputParam{{512,224,288},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_186
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_187
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_188
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_189
    Conv2DCaseInputParam{{4,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_190
    Conv2DCaseInputParam{{512,152,104},{8,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_191
    Conv2DCaseInputParam{{512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_192
    Conv2DCaseInputParam{{320,64,64},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_193
    Conv2DCaseInputParam{{320,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_194
    Conv2DCaseInputParam{{512,104,152},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_195
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_196
    Conv2DCaseInputParam{{320,128,128},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_197
    Conv2DCaseInputParam{{4,136,120},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_198
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_199
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_200
    Conv2DCaseInputParam{{4,144,112},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_201
    Conv2DCaseInputParam{{640,32,32},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_202
    Conv2DCaseInputParam{{320,56,72},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_203
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_204
    Conv2DCaseInputParam{{256,272,240},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_205
    Conv2DCaseInputParam{{1280,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_206
    Conv2DCaseInputParam{{320,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_207
    Conv2DCaseInputParam{{320,136,120},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_208
    Conv2DCaseInputParam{{8,112,144},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_209
    Conv2DCaseInputParam{{640,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_210
    Conv2DCaseInputParam{{320,104,152},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_211
    Conv2DCaseInputParam{{320,72,56},{640,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_212
    Conv2DCaseInputParam{{640,56,72},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_213
    Conv2DCaseInputParam{{640,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_214
    Conv2DCaseInputParam{{8,136,120},{8,3,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_215
    Conv2DCaseInputParam{{640,72,56},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_216
    Conv2DCaseInputParam{{640,64,64},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_217
    Conv2DCaseInputParam{{4,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_218
    Conv2DCaseInputParam{{640,34,30},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_219
    Conv2DCaseInputParam{{320,112,144},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_220
    Conv2DCaseInputParam{{1280,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_221
    Conv2DCaseInputParam{{640,28,36},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_222
    Conv2DCaseInputParam{{320,152,104},{320,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_223
    Conv2DCaseInputParam{{640,36,28},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_224
    Conv2DCaseInputParam{{320,68,60},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_225
    Conv2DCaseInputParam{{320,76,52},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_226
    Conv2DCaseInputParam{{640,52,76},{640,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_227
    Conv2DCaseInputParam{{640,68,60},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_228
    Conv2DCaseInputParam{{320,144,112},{320,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_229
    Conv2DCaseInputParam{{640,64,64},{640,3,3},{1,1,1,1},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_230
    Conv2DCaseInputParam{{640,38,26},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_231
    Conv2DCaseInputParam{{640,26,38},{1280,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_232
    Conv2DCaseInputParam{{640,28,36},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_233
    Conv2DCaseInputParam{{256,224,288},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_234
    Conv2DCaseInputParam{{128,608,416},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_235
    Conv2DCaseInputParam{{128,512,512},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_236
    Conv2DCaseInputParam{{8,152,104},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_237
    Conv2DCaseInputParam{{512,225,289},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_238
    Conv2DCaseInputParam{{256,513,513},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_239
    Conv2DCaseInputParam{{256,577,449},{256,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_240
    Conv2DCaseInputParam{{128,416,608},{256,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_241
    Conv2DCaseInputParam{{256,256,256},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_242
    Conv2DCaseInputParam{{8,144,112},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_243
    Conv2DCaseInputParam{{512,288,224},{512,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{true,false,false},{1,0,0}, 1}, // test_fp16_sdxl_244
    Conv2DCaseInputParam{{3,114,376},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_5
    Conv2DCaseInputParam{{6,234,216},{9,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_6
    Conv2DCaseInputParam{{9,456,234},{10,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_7
    Conv2DCaseInputParam{{12,246,342},{11,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_8
    Conv2DCaseInputParam{{21,53,923},{14,1,3},{0,0,0,0},{6,7},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_9
    Conv2DCaseInputParam{{24,57,821},{15,1,2},{0,0,0,0},{8,9},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_10
    Conv2DCaseInputParam{{27,24,714},{16,1,1},{0,0,0,0},{10,11},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_11
    Conv2DCaseInputParam{{30,42,523},{17,1,2},{0,0,0,0},{12,13},{1,255},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_12
    Conv2DCaseInputParam{{45,443,1},{22,255,1},{21,22,0,0},{22,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_15
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_59
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_62
    Conv2DCaseInputParam{{1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_65
    Conv2DCaseInputParam{{1,67108832,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_68
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_70
    Conv2DCaseInputParam{{22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_77
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_83
    Conv2DCaseInputParam{{8,128,128},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_84
    Conv2DCaseInputParam{{256,304,208},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_85
    Conv2DCaseInputParam{{8,104,152},{8,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_86
    Conv2DCaseInputParam{{256,272,240},{512,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_87
    Conv2DCaseInputParam{{320,72,56},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_88
    Conv2DCaseInputParam{{512,257,257},{512,1,1},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_89
    Conv2DCaseInputParam{{512,273,241},{512,3,3},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_90
    Conv2DCaseInputParam{{320,52,76},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_91
    Conv2DCaseInputParam{{320,56,72},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_92
    Conv2DCaseInputParam{{640,38,26},{1280,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{true,false,false},{1,0,0}, 1}, // test_bf16_gen_93
    Conv2DCaseInputParam{{3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_84
    Conv2DCaseInputParam{{8,94,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_94
    Conv2DCaseInputParam{{9,95,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_95
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_106
    Conv2DCaseInputParam{{22,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_113
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_119
    Conv2DCaseInputParam{{3,54,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_120
    Conv2DCaseInputParam{{12,86,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_130
    Conv2DCaseInputParam{{14,55,88},{13,11,9},{4,5,3,2},{12,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_142
    Conv2DCaseInputParam{{22,59,88},{13,11,9},{4,5,3,2},{7,8},{3,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_149
    Conv2DCaseInputParam{{6,55,88},{13,11,9},{4,5,3,2},{12,8},{3,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_155
    Conv2DCaseInputParam{{3,58,86},{12,7,5},{3,6,1,2},{3,8},{2,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_156
    Conv2DCaseInputParam{{12,86,96},{12,5,22},{1,2,3,4},{10,9},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_166
    Conv2DCaseInputParam{{1,55,88},{13,11,9},{4,5,3,2},{12,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_178
    Conv2DCaseInputParam{{24,55,88},{13,11,9},{4,5,3,2},{8,8},{4,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_185
    Conv2DCaseInputParam{{7,55,88},{13,11,9},{4,5,3,2},{12,8},{4,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_191
    Conv2DCaseInputParam{{6,58,86},{12,7,5},{3,4,1,2},{3,8},{2,3},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_192
    Conv2DCaseInputParam{{12,96,96},{12,5,22},{1,2,3,4},{10,11},{1,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_202
    Conv2DCaseInputParam{{11,97,97},{13,32,6},{1,2,3,4},{12,13},{3,2},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_203
    Conv2DCaseInputParam{{64,30,30},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_216
    Conv2DCaseInputParam{{15,15,16},{1,2,3},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_217
    Conv2DCaseInputParam{{3,4,4},{1000,2,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_219
    Conv2DCaseInputParam{{291,480,480},{291,3,3},{1,1,1,1},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_221
    Conv2DCaseInputParam{{1,134217712,1},{1,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_227
    Conv2DCaseInputParam{{320,68,60},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_223
    Conv2DCaseInputParam{{320,76,52},{640,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{true,false,false},{1,0,0}, 1}, // test_fp32_gen_224
    Conv2DCaseInputParam{{512,18,18},{2048,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1}, // test_fp32_gen_214
    Conv2DCaseInputParam{{768,16,16},{768,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1}, // test_fp32_gen_220
    Conv2DCaseInputParam{{512,105,105},{1024,1,1},{0,0,0,0},{2,2},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 1}, // test_fp32_gen_222
    Conv2DCaseInputParam{{16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT16,ConvDtype::FLOAT16,ConvDtype::FLOAT16},{false,false,false},{1,0,0}, 2}, // test_opt_group_fp16_base
    Conv2DCaseInputParam{{16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::BFLOAT16,ConvDtype::BFLOAT16,ConvDtype::BFLOAT16},{false,false,false},{1,0,0}, 2}, // test_opt_group_bf16_base
    Conv2DCaseInputParam{{16,1,1},{16,1,1},{0,0,0,0},{1,1},{1,1},{1,1}, {ConvDtype::FLOAT32,ConvDtype::FLOAT32,ConvDtype::FLOAT32},{false,false,false},{1,0,0}, 2}  // test_opt_group_fp32_base
));