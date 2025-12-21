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
 * \file test_conv3d_v2_apt_api_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "graph/tensor.h"
#include "tiling/conv/conv3d/conv3d_tiling.h"
#include "platform/platform_info.h"
#include "../../../op_host/op_tiling/arch35/conv3d_v2_api_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_HWmode.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.h"

using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
// using namespace conv_api_tiling_test;

class TestConv3dV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
    }
    static void TearDownTestCase() {}
    virtual void SetUp() {
        platform.socVersion = platform_ascendc::SocVersion::ASCEND910_95;
        platform.l1Size = 524288;
        platform.l0ASize = 65536;
        platform.l0BSize = 65536;
        platform.l0CSize = 262144;
        platform.ubSize = 262144;
        platform.btSize = 4096;
        platform.fbSize = 4096;
    }
    virtual void TearDown() {}
    conv_tiling::PlatformInfo platform;
};

TEST_F(TestConv3dV2Tiling, L0Range_cout_prime)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->shapeInfo.singleM = 32;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 11;
    testTiling->hasBias = true;

    algo.InitPingPong();
    EXPECT_EQ(algo.dbValue.pbAL1, 2);

    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32};
    std::vector<uint64_t> expRes1 = {16, 32, 64, 128, 176};
    EXPECT_EQ(algo.l0TilingRange.mL0Range, expRes0);
    EXPECT_EQ(algo.l0TilingRange.nL0Range, expRes1);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

TEST_F(TestConv3dV2Tiling, L0Range_m_prime)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 2;
    testTiling->shapeInfo.singleM = 11 * testTiling->cubeInfo.m0;
    testTiling->hasBias = true;

    algo.InitPingPong();
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32, 64, 128, 176};
    std::vector<uint64_t> expRes1 = {16, 32};
    EXPECT_EQ(algo.l0TilingRange.mL0Range, expRes0);
    EXPECT_EQ(algo.l0TilingRange.nL0Range, expRes1);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

TEST_F(TestConv3dV2Tiling, L0Tiling_larger_cout1)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);

    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 64;
    testTiling->shapeInfo.singleM = 4 * testTiling->cubeInfo.m0;
    testTiling->hasBias = true;

    algo.InitPingPong();
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32, 64};
    std::vector<uint64_t> expRes1 = {16, 32, 64, 128, 256, 512, 1024};
    algo.L0TilingDecision();
    bool flag = std::find(expRes0.begin(), expRes0.end(), algo.l0TilingParams.mL0) != expRes0.end();
    EXPECT_EQ(flag, true);
    flag = std::find(expRes1.begin(), expRes1.end(), algo.l0TilingParams.nL0) != expRes1.end();
    EXPECT_EQ(flag, true);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

TEST_F(TestConv3dV2Tiling, L0Tiling_larger_m)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 4;
    testTiling->shapeInfo.singleM = 64 * testTiling->cubeInfo.m0;
    testTiling->hasBias = true;

    algo.InitPingPong();
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32, 64, 128, 256, 512, 1024};
    std::vector<uint64_t> expRes1 = {16, 32, 64};
    algo.L0TilingDecision();
    bool flag = std::find(expRes0.begin(), expRes0.end(), algo.l0TilingParams.mL0) != expRes0.end();
    EXPECT_EQ(flag, true);
    flag = std::find(expRes1.begin(), expRes1.end(), algo.l0TilingParams.nL0) != expRes1.end();
    EXPECT_EQ(flag, true);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

TEST_F(TestConv3dV2Tiling, L0Range_normal)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 16;
    testTiling->shapeInfo.singleM = 16 * testTiling->cubeInfo.m0;
    testTiling->hasBias = true;

    algo.InitPingPong();
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32, 64, 128, 256};
    std::vector<uint64_t> expRes1 = {16, 32, 64, 128, 256};
    algo.L0TilingDecision();
    bool flag = std::find(expRes0.begin(), expRes0.end(), algo.l0TilingParams.mL0) != expRes0.end();
    EXPECT_EQ(flag, true);
    flag = std::find(expRes1.begin(), expRes1.end(), algo.l0TilingParams.nL0) != expRes1.end();
    EXPECT_EQ(flag, true);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

TEST_F(TestConv3dV2Tiling, GetL0Tiling_normal)
{
    conv_tiling::Conv3dTiling* testTiling = new conv_tiling::Conv3dTiling(platform);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    testTiling->InferCubeInfo();
    ConvTilingAlgorithmMmode algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeInfo.orgWo = 8;
    testTiling->shapeInfo.singleCo1 = 16;
    testTiling->shapeInfo.singleM = 16 * testTiling->cubeInfo.m0;
    testTiling->hasBias = true;

    algo.InitPingPong();
    int64_t ret = algo.GetL0Tiling();
    EXPECT_EQ(ret, 0);

    if (testTiling != nullptr) {
        delete testTiling;
        testTiling = nullptr;
    }
}

void SetSingleOutputShapeInTest(conv_tiling::Conv3dTiling &testTiling)
{
    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;

    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;

    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, singleM, 1);
    if (testTiling.platformInfo.socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
        testTiling.SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
        testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    } else {
        testTiling.SetWeightType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
        testTiling.SetFmapType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT16);
        testTiling.SetOutputType(TPosition::CO1, ConvFormat::NDC1HWC0, ConvDtype::FLOAT16);
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    }
}

void SetType(conv_tiling::Conv3dTiling &testTiling, conv_tiling::PlatformInfo &platform)
{
    if (platform.socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
        testTiling.SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
        testTiling.SetOutputType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BFLOAT16);
    } else {
        testTiling.SetWeightType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::BFLOAT16);
        testTiling.SetFmapType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::BFLOAT16);
        testTiling.SetOutputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::BFLOAT16);
    }
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
    }
}

uint64_t Conv3DCeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

uint64_t Conv3DGcd(uint64_t a, uint64_t b)
{
    uint64_t c;
    if (a < b) {
        c = a;
        a = b;
        b = c;
    }

    while (a % b != 0) {
        c = a % b;
        a = b;
        b = c;
    }

    return b;
}

void TestTilingResult(int64_t ret, conv_tiling::Conv3dTiling &testTiling)
{
    EXPECT_EQ(ret, 0);
    if (ret != 0) {
        return;
    }
    uint64_t pBuffer = testTiling.dbValue.pBufferFlag;
    int8_t pbAL0 = pBuffer & 0x01;
    int8_t pbBL0 = (pBuffer & 0x02) >> 1;
    int8_t pbCL0 = (pBuffer & 0x04) >> 2;
    int8_t pbAL1 = (pBuffer & 0x08) >> 3;
    int8_t pbBL1 = (pBuffer & 0x10) >> 4;
    // BFLOAT16
    uint64_t weightDtypeSize = 2;
    uint64_t featuremapDtypeSize = 2;
    uint64_t biasDTypeSize = 2;
    uint64_t mmadDtypesize = 4;


    EXPECT_GT(testTiling.l1TilingInfo.kAL1, 0);
    EXPECT_GT(testTiling.l1TilingInfo.mAL1, 0);
    EXPECT_GT(testTiling.l1TilingInfo.kBL1, 0);
    EXPECT_GT(testTiling.l1TilingInfo.nBL1, 0);
    EXPECT_GT(testTiling.l0TilingInfo.nL0, 0);
    EXPECT_GT(testTiling.l0TilingInfo.mL0, 0);
    EXPECT_GT(testTiling.l0TilingInfo.kL0, 0);


    auto multi_mAL1max = Conv3DCeilDiv(Conv3DCeilDiv(testTiling.shapeInfo.singleM, testTiling.cubeInfo.m0) * testTiling.cubeInfo.m0, testTiling.l0TilingInfo.mL0);
    EXPECT_LE(testTiling.l1TilingInfo.mAL1, multi_mAL1max * testTiling.l0TilingInfo.mL0);
    EXPECT_LE(testTiling.l1TilingInfo.kAL1, testTiling.shapeInfo.singlekD * testTiling.shapeInfo.singlekH * testTiling.shapeInfo.singlekW * Conv3DCeilDiv(testTiling.shapeInfo.orgCi, testTiling.cubeInfo.k0) * testTiling.cubeInfo.k0);
    
    auto multi_nBL1max = Conv3DCeilDiv(Conv3DCeilDiv(testTiling.shapeInfo.singleCo, testTiling.cubeInfo.n0) * testTiling.cubeInfo.n0, testTiling.l0TilingInfo.nL0);
    EXPECT_LE(testTiling.l1TilingInfo.nBL1, multi_nBL1max * testTiling.l0TilingInfo.nL0);
    EXPECT_LE(testTiling.l1TilingInfo.kBL1, testTiling.shapeInfo.singlekD * testTiling.shapeInfo.singlekH * testTiling.shapeInfo.singlekW * Conv3DCeilDiv(testTiling.shapeInfo.orgCi, testTiling.cubeInfo.k0) * testTiling.cubeInfo.k0);
    
    auto mL0max = std::min(testTiling.platformInfo.l0ASize / (testTiling.cubeInfo.k0 * (pbAL0 + 1) * featuremapDtypeSize), testTiling.platformInfo.l0CSize / (testTiling.cubeInfo.n0 * (pbCL0 + 1) * mmadDtypesize));
    auto nL0max = std::min(testTiling.platformInfo.l0BSize / (testTiling.cubeInfo.k0 * (pbBL0 + 1) * weightDtypeSize), testTiling.platformInfo.l0CSize / (testTiling.cubeInfo.m0 * (pbCL0 + 1) * mmadDtypesize));
    EXPECT_LE(testTiling.l0TilingInfo.mL0, Conv3DCeilDiv(mL0max, testTiling.cubeInfo.m0) * testTiling.cubeInfo.m0);
    EXPECT_LE(testTiling.l0TilingInfo.nL0, Conv3DCeilDiv(nL0max, testTiling.cubeInfo.n0) * testTiling.cubeInfo.n0);
    auto tmpkBL1 = testTiling.l1TilingInfo.kAL1;
    tmpkBL1 = testTiling.l1TilingInfo.kBL1;
    EXPECT_LE(testTiling.l0TilingInfo.kL0, Conv3DGcd(Conv3DCeilDiv(testTiling.l1TilingInfo.kAL1, testTiling.cubeInfo.k0), Conv3DCeilDiv(tmpkBL1, testTiling.cubeInfo.k0)) * testTiling.cubeInfo.k0);


    EXPECT_EQ(testTiling.l1TilingInfo.kAL1 % testTiling.cubeInfo.k0, 0);
    EXPECT_EQ(testTiling.l1TilingInfo.nBL1 % testTiling.cubeInfo.n0, 0);
    EXPECT_EQ(testTiling.l1TilingInfo.kBL1 % testTiling.cubeInfo.k0, 0);

    EXPECT_EQ(testTiling.l1TilingInfo.mAL1 % testTiling.cubeInfo.k0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.nL0 % testTiling.cubeInfo.k0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.kL0 % testTiling.cubeInfo.k0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.mL0 % testTiling.cubeInfo.k0, 0);
    if (!testTiling.l1TilingInfo.al1FullLoad) {
        EXPECT_EQ(testTiling.l1TilingInfo.mAL1 % testTiling.l0TilingInfo.mL0, 0);
    }
    EXPECT_EQ(testTiling.l1TilingInfo.kAL1 % testTiling.l0TilingInfo.kL0, 0);
    if (!testTiling.l1TilingInfo.bl1FullLoad) {
        EXPECT_EQ(testTiling.l1TilingInfo.nBL1 % testTiling.l0TilingInfo.nL0, 0);
    }
    EXPECT_EQ(testTiling.l1TilingInfo.kBL1 % testTiling.l0TilingInfo.kL0, 0);


    int64_t hoAL1Tmp = testTiling.l1TilingInfo.mAL1 / testTiling.shapeInfo.orgWo + 2;
    int64_t hiL1Tmp = std::min((hoAL1Tmp - 1) * testTiling.attrInfo.strideH + (testTiling.shapeInfo.singlekH - 1) / testTiling.attrInfo.dilationH + 1, testTiling.shapeInfo.orgHi);
    uint64_t al1Size = static_cast<uint64_t>(hiL1Tmp) * testTiling.shapeInfo.orgWi * (testTiling.l1TilingInfo.kAL1 / (testTiling.shapeInfo.singlekH * testTiling.shapeInfo.singlekW)) * (pbAL1 + 1) * featuremapDtypeSize;
    uint64_t bl1Size = testTiling.l1TilingInfo.nBL1 * testTiling.l1TilingInfo.kBL1 * (pbBL1 + 1) * weightDtypeSize;
    uint64_t biasL1Size = !testTiling.hasBias ? 0 : testTiling.l1TilingInfo.biasFullLoadFlag == 1 ? Conv3DCeilDiv(testTiling.shapeInfo.singleCo, testTiling.cubeInfo.n0) * testTiling.cubeInfo.n0 * biasDTypeSize : testTiling.l0TilingInfo.nL0 * biasDTypeSize;
    uint64_t curl1Size = al1Size + bl1Size + biasL1Size;
    uint64_t curl0aSize = testTiling.l0TilingInfo.mL0 * testTiling.l0TilingInfo.kL0 * (pbAL0 + 1) * featuremapDtypeSize;
    uint64_t curl0bSize = testTiling.l0TilingInfo.nL0 * testTiling.l0TilingInfo.kL0 * (pbBL0 + 1) * weightDtypeSize;
    uint64_t curl0cSize = testTiling.l0TilingInfo.mL0 * testTiling.l0TilingInfo.nL0 * (pbCL0 + 1) * mmadDtypesize;
    EXPECT_LE(curl1Size, testTiling.platformInfo.l1Size);
    EXPECT_LE(curl0aSize, testTiling.platformInfo.l0ASize);
    EXPECT_LE(curl0bSize, testTiling.platformInfo.l0BSize);
    EXPECT_LE(curl0cSize, testTiling.platformInfo.l0CSize);

    if (testTiling.l1TilingInfo.al1FullLoad) {
        testTiling.l1TilingInfo.kAL1 = testTiling.shapeInfo.singlekD * Conv3DCeilDiv(testTiling.shapeInfo.orgCi, testTiling.cubeInfo.k0) * testTiling.cubeInfo.k0 * testTiling.shapeInfo.singlekH * testTiling.shapeInfo.singlekW;
        testTiling.l1TilingInfo.mAL1 = testTiling.cubeInfo.m0 * Conv3DCeilDiv(testTiling.shapeInfo.singleM, testTiling.cubeInfo.m0);
        testTiling.l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    }
    if (testTiling.l1TilingInfo.bl1FullLoad) {
        testTiling.l1TilingInfo.kBL1 = testTiling.shapeInfo.singlekD * Conv3DCeilDiv(testTiling.shapeInfo.orgCi, testTiling.cubeInfo.k0) * testTiling.cubeInfo.k0 * testTiling.shapeInfo.singlekH * testTiling.shapeInfo.singlekW;
        testTiling.l1TilingInfo.nBL1 = testTiling.cubeInfo.n0 * Conv3DCeilDiv(testTiling.shapeInfo.singleCo, testTiling.cubeInfo.n0);
        testTiling.l1TilingInfo.iterateMNOrder = IterateMNOrder::ITER_M_FST;
    }
}

// TestCase for Networks
TEST_F(TestConv3dV2Tiling, NetWorks_001)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 120, 32, 32);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_002)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 16, 26, 36);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_003)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 4, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform); 
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_004)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 18, 130, 130);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_005)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 10, 66, 66);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_006)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(240, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_007)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(120, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_008)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 4, 32, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_009)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_010)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);
    
    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_011)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_012)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_013)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_014)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_015)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_016)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_017)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_018)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_019)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_020)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_021)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_022)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_023)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_024)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 3, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_025)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 3, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_026)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_027)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 3, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_028)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 3, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_029)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_030)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_031)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 3, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_032)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_033)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_034)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_035)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_036)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_037)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 32, 32);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_038)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 1, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_039)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 26, 134, 134);
    testTiling.SetOrgWeightShape(64, 7, 7, 7);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 7, 7);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_040)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 22, 130, 130);
    testTiling.SetOrgWeightShape(64, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_041)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 20, 128, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_042)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 22, 66, 66);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_043)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 20, 64, 64);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_044)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 22, 34, 34);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_045)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 20, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_046)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 20, 32, 32);
    testTiling.SetOrgWeightShape(1364, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_047)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 20, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_048)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 22, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_049)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 20, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_050)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 20, 16, 16);
    testTiling.SetOrgWeightShape(2730, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_051)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 20, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}


TEST_F(TestConv3dV2Tiling, NetWorks_052)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 12, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_053)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 10, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_054)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_055)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_056)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 16, 16);
    testTiling.SetOrgWeightShape(2730, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_057)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 5, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_058)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 22, 130, 130);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_059)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_060)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_061)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_062)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_063)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_064)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_065)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_066)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_067)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_068)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_069)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_070)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_071)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_072)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_073)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_074)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 32, 32);
    testTiling.SetOrgWeightShape(4, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_075)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_076)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_077)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_078)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_079)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_080)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_081)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_082)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_083)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_084)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_085)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 128, 128);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_086)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_087)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_088)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_089)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_090)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_091)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_092)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_093)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_094)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_095)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1; 
    SetType(testTiling, platform);
    SetSingleOutputShapeInTest(testTiling);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_096)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 120, 32, 32);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_097)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(320, 25, 40, 64);
    testTiling.SetOrgWeightShape(320, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_098)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(640, 25, 20, 32);
    testTiling.SetOrgWeightShape(640, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_099)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 10, 16);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_100)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 5, 8);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_101)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(320, 25, 40, 64);
    testTiling.SetOrgWeightShape(320, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_102)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(640, 25, 20, 32);
    testTiling.SetOrgWeightShape(640, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_103)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 10, 16);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_104)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 5, 8);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_105)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 8, 40, 64);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_106)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 8, 80, 128);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_107)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 8, 160, 256);
    testTiling.SetOrgWeightShape(256, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_108)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 8, 320, 512);
    testTiling.SetOrgWeightShape(128, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_109)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 8, 320, 512);
    testTiling.SetOrgWeightShape(3, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_110)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 40, 64);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_111)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 80, 128);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_112)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 160, 256);
    testTiling.SetOrgWeightShape(256, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_113)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 320, 512);
    testTiling.SetOrgWeightShape(128, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_114)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 1, 320, 512);
    testTiling.SetOrgWeightShape(3, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    testTiling.SetPadding(1, 1, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_115)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 4, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_116)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 18, 130, 130);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_117)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 10, 66, 66);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_118)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(240, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_119)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(120, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_120)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 4, 32, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_121)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 16, 224, 224);
    testTiling.SetOrgWeightShape(768, 2, 16, 16);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 2, 16, 16);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(16, 16, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_122)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 35, 192, 192);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_123)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 35, 192, 192);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_124)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 33, 193, 193);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_125)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 35, 96, 96);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_126)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 35, 96, 96);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_127)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 33, 96, 96);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_128)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 33, 97, 97);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_129)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_130)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_131)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 48, 48);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_132)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 49, 49);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_133)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 24, 24);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_134)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_135)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 24, 24);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_136)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 9, 24, 24);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_137)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 9, 24, 24);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_138)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 320, 320);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_139)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 320, 320);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_140)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 321, 321);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_141)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 160, 160);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_142)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 160, 160);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_143)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 160, 160);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_144)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 161, 161);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_145)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 80, 80);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_146)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 80, 80);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_147)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 80, 80);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_148)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 81, 81);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_149)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 40, 40);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_150)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 40, 40);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_151)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 40, 40);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_152)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 40, 40);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_153)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 40, 40);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_154)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_155)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_156)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_157)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_158)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_159)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_160)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_161)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_162)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_163)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_164)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_165)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_166)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_167)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_168)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_169)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 32, 32);
    testTiling.SetOrgWeightShape(4, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_170)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_171)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_172)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_173)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_174)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_175)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_176)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_177)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_178)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_179)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_180)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 128, 128);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_181)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_182)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_183)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_184)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_185)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_186)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_187)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_188)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_189)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_190)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_OpenSora)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 120, 16, 16);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, test_910D_1)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    testTiling.SetPadding(0, 0, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, test_910D_2)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    testTiling.SetPadding(0, 0, 0, 0, 0, 0);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1; 
    SetType(testTiling, platform);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dV2Tiling, NetWorks_ND2NZ_limits)
{
    optiling::TConv3DTiling tilingData;
    conv_tiling::Conv3dTiling testTiling(platform);
 
    testTiling.SetOrgFmapShape(1000000,16,1000000,512);
    testTiling.SetOrgWeightShape(16, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);
 
    testTiling.SetPadding(1, 1, 1, 1, 1, 1);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);
    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;
 
    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;
 
    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, singleM, 1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::DHWCN, ConvDtype::BFLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NDHWC, ConvDtype::BFLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NDHWC, ConvDtype::BFLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BFLOAT16);
 
    
    testTiling.hasBias = 0; 
    
    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}