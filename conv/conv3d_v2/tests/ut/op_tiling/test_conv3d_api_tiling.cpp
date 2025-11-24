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
 * \file test_conv3d_api_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include "graph/tensor.h"
#define private public
#define protected public
#define ENABLE_TILING_DEBUG
#include "tiling/tiling_api.h"
#include "../../../op_kernel/conv3d_v2_tiling_data.h"
#include "../../../op_host/op_tiling/conv3d_api_tiling.h"
#include "../../../op_host/op_tiling/conv3d_api_tiling_utils.h"
#include "../../../op_host/op_tiling/conv3d_api_tiling_algorithm.h"

using namespace std;
using namespace Conv3dApiTiling;

namespace {

constexpr uint32_t L1_SIZE = 524288;
constexpr uint32_t L0A_SIZE = 65536;
constexpr uint32_t L0B_SIZE = 65536;
constexpr uint32_t L0C_SIZE = 131072;
constexpr uint32_t UB_SIZE = 262144;
constexpr uint32_t BT_SIZE = 1024;

class TestConv3dTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        platform.l1Size = L1_SIZE;
        platform.l0CSize = L0C_SIZE;
        platform.ubSize = UB_SIZE;
        platform.l0ASize = L0A_SIZE;
        platform.l0BSize = L0B_SIZE;
        platform.btSize = BT_SIZE;
    }
    virtual void TearDown() {}

protected:
    Conv3dApiTiling::PlatformInfo platform;
};

void SetPlatFormInfo(Conv3dTiling* testTiling)
{
    testTiling->platformInfo.l1Size = L1_SIZE;
    testTiling->platformInfo.l0ASize = L0A_SIZE;
    testTiling->platformInfo.l0BSize = L0B_SIZE;
    testTiling->platformInfo.l0CSize = L0C_SIZE;
    testTiling->platformInfo.ubSize = UB_SIZE;
    testTiling->platformInfo.btSize = BT_SIZE;
}

TEST_F(TestConv3dTiling, L0Range_cout_prime)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->shapeInfo.singleM = 32;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 11;
    testTiling->hasBias = true;

    algo.InitPingPong();
    EXPECT_EQ(algo.doubleBufferValue.pbAL1, 2);

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

TEST_F(TestConv3dTiling, L0Range_m_prime)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 2;
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

TEST_F(TestConv3dTiling, L0Tiling_larger_cout1)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 64;
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

TEST_F(TestConv3dTiling, L0Tiling_larger_m)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 4;
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

TEST_F(TestConv3dTiling, L0Range_normal)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 16;
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

TEST_F(TestConv3dTiling, GetL0Tiling_normal)
{
    Conv3dApiTiling::Conv3dTiling* testTiling = new Conv3dApiTiling::Conv3dTiling();

    SetPlatFormInfo(testTiling);
    testTiling->SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling->SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::BF16);
    testTiling->GetCubeInfo();
    Conv3dApiTiling::Conv3dTilingAlgorithm algo(testTiling);

    testTiling->shapeInfo.orgHi = 32;
    testTiling->shapeInfo.orgWi = 32;
    testTiling->shapeInfo.singlekH = 3;
    testTiling->attrInfo.dilationH = 1;
    testTiling->attrInfo.strideH = 1;
    testTiling->shapeCalc.orgWo = 8;
    testTiling->shapeCalc.singleCo1 = 16;
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

void SetSingleOutputShapeInTest(Conv3dApiTiling::Conv3dTiling &testTiling)
{
    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;

    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;

    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, singleM);
    testTiling.SetGroups(1);
    testTiling.attrInfo.groupOpt = 1;
    testTiling.shapeInfo.cinOpt = testTiling.shapeInfo.orgCi;
    testTiling.shapeInfo.coutOpt = testTiling.shapeInfo.orgCo;
    int64_t singleCoreGroupOpt = CeilDiv(testTiling.attrInfo.groupOpt, 1);
    testTiling.SetOptGroupInfo(static_cast<int64_t>(testTiling.attrInfo.groupOpt),
                                    singleCoreGroupOpt,
                                    static_cast<int64_t>(testTiling.shapeInfo.cinOpt),
                                    static_cast<int64_t>(testTiling.shapeInfo.coutOpt));
}

void SetSingleOutputShapeHwModeInTest(Conv3dApiTiling::Conv3dTiling &testTiling)
{
    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;

    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;

    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, orgHo, orgWo);
    testTiling.SetOutputOrder(Conv3dApiTiling::HW_Mode);
    testTiling.SetGroups(1);
    testTiling.attrInfo.groupOpt = 1;
    testTiling.shapeInfo.cinOpt = testTiling.shapeInfo.orgCi;
    testTiling.shapeInfo.coutOpt = testTiling.shapeInfo.orgCo;
    int64_t singleCoreGroupOpt = CeilDiv(testTiling.attrInfo.groupOpt, 1);
    testTiling.SetOptGroupInfo(static_cast<int64_t>(testTiling.attrInfo.groupOpt),
                                    singleCoreGroupOpt,
                                    static_cast<int64_t>(testTiling.shapeInfo.cinOpt),
                                    static_cast<int64_t>(testTiling.shapeInfo.coutOpt));
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.cinOpt, testTiling.shapeInfo.orgkD, testTiling.shapeInfo.orgkH, testTiling.shapeInfo.orgkW);
}

void SetType(Conv3dApiTiling::Conv3dTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::BF16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::BF16);
    testTiling.SetOutputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::BF16);
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);
    }
}

void SetTypePointWise(Conv3dApiTiling::Conv3dTiling &testTiling)
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling.SetOutputType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);
    }
}

void SetTypePointWise(Conv3dApiTiling::Conv3dTiling &testTiling, Conv3dApiTiling::ConvDtype dtype )
{
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, dtype);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCDHW, dtype);
    testTiling.SetOutputType(TPosition::GM, ConvFormat::NCDHW, dtype);
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);
    }
}

void SetInputPointWise(Conv3dApiTiling::Conv3dTiling &testTiling)
{
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);
}

void TestTilingResult(int64_t ret, Conv3dApiTiling::Conv3dTiling &testTiling)
{
    EXPECT_EQ(ret, 0);
    EXPECT_NE(testTiling.l1TilingInfo.kAL1, 0);
    if (testTiling.l1TilingInfo.isWeightBypass) {
        EXPECT_EQ(testTiling.l1TilingInfo.kBL1, 0);
        EXPECT_EQ(testTiling.l1TilingInfo.nBL1Value, 0);
    } else {
        EXPECT_NE(testTiling.l1TilingInfo.kBL1, 0);

        EXPECT_NE(testTiling.l1TilingInfo.nBL1Value, 0);
        int multiDivFlag = (testTiling.l1TilingInfo.nBL1Value % testTiling.l0TilingInfo.nL0 == 0) |
                            (testTiling.l1TilingInfo.nBL1Value % testTiling.cubeInfo.n0 == 0);
        EXPECT_EQ(multiDivFlag, 1);
    }

    EXPECT_NE(testTiling.l1TilingInfo.mAL1Value, 0);
    int multiDivFlag = (testTiling.l1TilingInfo.mAL1Value % testTiling.l0TilingInfo.mL0 == 0) |
                        (testTiling.l1TilingInfo.mAL1Value % testTiling.cubeInfo.m0 == 0);
    EXPECT_EQ(multiDivFlag, 1);

    EXPECT_NE(testTiling.l0TilingInfo.mL0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.mL0 % testTiling.cubeInfo.m0, 0);
    EXPECT_NE(testTiling.l0TilingInfo.kL0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.kL0 % testTiling.cubeInfo.k0, 0);
    EXPECT_NE(testTiling.l0TilingInfo.nL0, 0);
    EXPECT_EQ(testTiling.l0TilingInfo.nL0 % testTiling.cubeInfo.n0, 0);
}

// TestCase for Networks
TEST_F(TestConv3dTiling, NetWorks_001)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 120, 32, 32);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_002)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 16, 26, 36);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_003)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 4, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_004)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 18, 130, 130);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_005)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 10, 66, 66);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_006)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(240, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_007)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(120, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_008)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 4, 32, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_009)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_010)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_011)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_012)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_013)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_014)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_015)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_016)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_017)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_018)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_019)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_020)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_021)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_022)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_023)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_024)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 3, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_025)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 3, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_026)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_027)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 3, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_028)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 3, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_029)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_030)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_031)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 3, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_032)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_033)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_034)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_035)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_036)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_037)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 3, 32, 32);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_038)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 1, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_039)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 26, 134, 134);
    testTiling.SetOrgWeightShape(64, 7, 7, 7);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 7, 7);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_040)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 22, 130, 130);
    testTiling.SetOrgWeightShape(64, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_041)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 20, 128, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_042)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 22, 66, 66);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_043)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 20, 64, 64);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_044)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 22, 34, 34);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_045)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 20, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_046)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 20, 32, 32);
    testTiling.SetOrgWeightShape(1364, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_047)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 20, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_048)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 22, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_049)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 20, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_050)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 20, 16, 16);
    testTiling.SetOrgWeightShape(2730, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_051)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 20, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}


TEST_F(TestConv3dTiling, NetWorks_052)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 12, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_053)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 10, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_054)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 18, 18);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_055)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_056)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 16, 16);
    testTiling.SetOrgWeightShape(2730, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_057)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 5, 16, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_058)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 22, 130, 130);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_059)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_060)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_061)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_062)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_063)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_064)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_065)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_066)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_067)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_068)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_069)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_070)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_071)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_072)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_073)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_074)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 32, 32);
    testTiling.SetOrgWeightShape(4, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_075)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_076)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_077)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_078)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_079)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_080)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_081)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_082)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_083)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_084)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_085)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 128, 128);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_086)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_087)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_088)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_089)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_090)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_091)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_092)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_093)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_094)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_095)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    testTiling.hasBias = 1;
    SetType(testTiling);
    SetSingleOutputShapeInTest(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_096)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 120, 32, 32);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_097)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(320, 25, 40, 64);
    testTiling.SetOrgWeightShape(320, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_098)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(640, 25, 20, 32);
    testTiling.SetOrgWeightShape(640, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_099)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 10, 16);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_100)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 5, 8);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_101)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(320, 25, 40, 64);
    testTiling.SetOrgWeightShape(320, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_102)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(640, 25, 20, 32);
    testTiling.SetOrgWeightShape(640, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_103)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 10, 16);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_104)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1280, 25, 5, 8);
    testTiling.SetOrgWeightShape(1280, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_105)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 8, 40, 64);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_106)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 8, 80, 128);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_107)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 8, 160, 256);
    testTiling.SetOrgWeightShape(256, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_108)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 8, 320, 512);
    testTiling.SetOrgWeightShape(128, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_109)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 8, 320, 512);
    testTiling.SetOrgWeightShape(3, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_110)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 40, 64);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_111)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 1, 80, 128);
    testTiling.SetOrgWeightShape(512, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_112)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 1, 160, 256);
    testTiling.SetOrgWeightShape(256, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_113)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 1, 320, 512);
    testTiling.SetOrgWeightShape(128, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_114)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 1, 320, 512);
    testTiling.SetOrgWeightShape(3, 3, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 1, 1);

    std::vector<int64_t> padList = {1, 1, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_115)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 4, 32, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_116)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 18, 130, 130);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_117)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 10, 66, 66);
    testTiling.SetOrgWeightShape(240, 4, 4, 4);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 4, 4, 4);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_118)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(240, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_119)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 6, 34, 34);
    testTiling.SetOrgWeightShape(120, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_120)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 4, 32, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_121)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 16, 224, 224);
    testTiling.SetOrgWeightShape(768, 2, 16, 16);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 2, 16, 16);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(16, 16, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_122)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 35, 192, 192);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_123)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 35, 192, 192);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_124)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 33, 193, 193);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_125)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 35, 96, 96);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_126)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 35, 96, 96);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_127)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 33, 96, 96);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_128)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 33, 97, 97);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_129)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_130)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_131)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 48, 48);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_132)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 49, 49);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_133)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 24, 24);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_134)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 48, 48);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_135)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 24, 24);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_136)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 9, 24, 24);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_137)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 9, 24, 24);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_138)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 320, 320);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_139)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 320, 320);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_140)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 321, 321);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_141)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 160, 160);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_142)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 160, 160);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_143)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 160, 160);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_144)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 161, 161);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_145)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 80, 80);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_146)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 80, 80);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_147)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 80, 80);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_148)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 81, 81);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_149)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 40, 40);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_150)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 40, 40);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_151)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 40, 40);
    testTiling.SetOrgWeightShape(8, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_152)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 40, 40);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_153)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 40, 40);
    testTiling.SetOrgWeightShape(1152, 1, 2, 2);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_154)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_155)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_156)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 17, 257, 257);
    testTiling.SetOrgWeightShape(128, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_157)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_158)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 11, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_159)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 9, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_160)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 9, 129, 129);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_161)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_162)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_163)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_164)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_165)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 65, 65);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_166)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_167)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_168)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(8, 5, 32, 32);
    testTiling.SetOrgWeightShape(8, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_169)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 5, 32, 32);
    testTiling.SetOrgWeightShape(4, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_170)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(4, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_171)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_172)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_173)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_174)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_175)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_176)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 32, 32);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_177)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 7, 32, 32);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_178)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 5, 64, 64);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_179)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 11, 64, 64);
    testTiling.SetOrgWeightShape(512, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_180)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 9, 128, 128);
    testTiling.SetOrgWeightShape(512, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_181)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_182)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_183)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 17, 128, 128);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_184)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 128, 128);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_185)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(256, 1, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_186)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_187)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_188)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 17, 256, 256);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_189)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(128, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_190)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 19, 256, 256);
    testTiling.SetOrgWeightShape(3, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_largeK_noFullLoad_001)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(16, 7, 256, 32);
    testTiling.SetOrgWeightShape(1024, 7, 7, 7);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 7, 7);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_largeK_noFullLoad_002)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(17, 7, 256, 32);
    testTiling.SetOrgWeightShape(1024, 7, 5, 5);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 5, 5);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, NetWorks_largeK_noFullLoad_003)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(16, 7, 64, 128);
    testTiling.SetOrgWeightShape(1024, 7, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, InputShape_With_Pad_Negative)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 32, 30, 20);
    testTiling.SetOrgWeightShape(1, 20, 23, 23);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(5, 10, 17);

    SetSingleOutputShapeInTest(testTiling);
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, NetWorks_largeShape_overflow)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(16, 7, 64, 128);
    testTiling.SetOrgWeightShape(UINT64_MAX / 1000, 7, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 7, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, HwMode_net_1)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 1, 448, 448);
    testTiling.SetOrgWeightShape(1664, 1, 14, 14);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_net_2)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 410, 410);
    testTiling.SetOrgWeightShape(1, 1, 51, 51);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_net_3)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(3, 1, 448, 448);
    testTiling.SetOrgWeightShape(1664, 1, 14, 14);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_net_4)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 410, 410);
    testTiling.SetOrgWeightShape(1, 1, 51, 51);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_abL1FullLoad)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 6, 290);
    testTiling.SetOrgWeightShape(1, 1, 6, 51);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_aL1FullLoad)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 51, 290);
    testTiling.SetOrgWeightShape(1, 1, 51, 51);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_bL1FullLoad)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 300, 400);
    testTiling.SetOrgWeightShape(1, 1, 11, 31);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_noneFullLoad_kABL1FullLoad)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 300, 300);
    testTiling.SetOrgWeightShape(10000, 1, 2, 2);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(20, 20, 1);
    testTiling.SetStride(50, 50, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 0;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, HwMode_woLT16)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1, 1, 200, 290);
    testTiling.SetOrgWeightShape(50, 1, 20, 20);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(20, 20, 1);

    SetSingleOutputShapeHwModeInTest(testTiling);
    testTiling.hasBias = 1;
    SetType(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_1)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 32, 4, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_2)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 32, 4, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_3)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 128, 9, 128);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_4)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_5)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 32, 20, 32);
    testTiling.SetOrgWeightShape(682, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_6)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 20, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_7)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 20, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_8)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 10, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_9)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 5, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_10)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 5, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_11)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 128, 20, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_12)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 64, 20, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_13)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 64, 20, 64);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_14)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 32, 20, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_15)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 32, 20, 32);
    testTiling.SetOrgWeightShape(682, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_16)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 20, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_17)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 20, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_18)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 10, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_19)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 5, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_20)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 5, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_21)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 32, 4, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_22)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 128, 9, 128);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_23)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_1)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(240, 32, 4, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_2)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 128, 20, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_3)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 64, 20, 64);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_4)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 32, 20, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_5)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 32, 20, 32);
    testTiling.SetOrgWeightShape(682, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_6)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 20, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_7)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 20, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_8)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 10, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_9)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 5, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_10)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 5, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_11)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(64, 128, 20, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_12)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 64, 20, 128);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_13)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 64, 20, 64);
    testTiling.SetOrgWeightShape(64, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_14)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 32, 20, 32);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_15)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(682, 32, 20, 32);
    testTiling.SetOrgWeightShape(682, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_16)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 20, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_17)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 20, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_18)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 10, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_19)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(512, 16, 5, 16);
    testTiling.SetOrgWeightShape(512, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_20)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(1365, 16, 5, 16);
    testTiling.SetOrgWeightShape(1365, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_21)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(120, 32, 4, 32);
    testTiling.SetOrgWeightShape(240, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_22)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(128, 128, 9, 128);
    testTiling.SetOrgWeightShape(128, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_23)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_fp16)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling, ConvDtype::FLOAT16);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_fp32)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling, ConvDtype::FLOAT32);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_fp16)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling, ConvDtype::FLOAT16);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_nonbias_fp32)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 0;
    SetTypePointWise(testTiling, ConvDtype::FLOAT32);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    TestTilingResult(ret1, testTiling);
}

TEST_F(TestConv3dTiling, PointWise_Padding_negative)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 1, 1, 1, 1};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Dilation_negative)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(2, 2, 2);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Stride_negative)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(2, 2, 2);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Weight_negative)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 3, 3, 3);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 3, 3, 3);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Groups_negative1)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;

    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;

    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, singleM);
    testTiling.SetGroups(2);
    testTiling.attrInfo.groupOpt = 1;
    testTiling.shapeInfo.cinOpt = testTiling.shapeInfo.orgCi;
    testTiling.shapeInfo.coutOpt = testTiling.shapeInfo.orgCo;
    int64_t singleCoreGroupOpt = CeilDiv(testTiling.attrInfo.groupOpt, 1);
    testTiling.SetOptGroupInfo(static_cast<int64_t>(testTiling.attrInfo.groupOpt),
                                    singleCoreGroupOpt,
                                    static_cast<int64_t>(testTiling.shapeInfo.cinOpt),
                                    static_cast<int64_t>(testTiling.shapeInfo.coutOpt));

    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Groups_negative2)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    testTiling.SetSingleWeightShape(testTiling.shapeInfo.orgCi, 1, 1, 1);

    std::vector<int64_t> padList = {0, 0, 0, 0, 0, 0};
    testTiling.SetPadding(padList);
    testTiling.SetDilation(1, 1, 1);
    testTiling.SetStride(1, 1, 1);

    int64_t orgHo = (testTiling.shapeInfo.orgHi + testTiling.attrInfo.padTop + testTiling.attrInfo.padBottom -
             testTiling.attrInfo.dilationH * (testTiling.shapeInfo.orgkH - 1) - 1) / testTiling.attrInfo.strideH + 1;
    int64_t orgWo = (testTiling.shapeInfo.orgWi + testTiling.attrInfo.padLeft + testTiling.attrInfo.padRight -
             testTiling.attrInfo.dilationW * (testTiling.shapeInfo.orgkW - 1) - 1) / testTiling.attrInfo.strideW + 1;

    int64_t singleM = orgHo * orgWo;
    int64_t singleDo = (testTiling.shapeInfo.orgDi + testTiling.attrInfo.padHead + testTiling.attrInfo.padTail -
                        testTiling.attrInfo.dilationD * (testTiling.shapeInfo.orgkD - 1) - 1) / testTiling.attrInfo.strideD + 1;

    testTiling.SetSingleOutputShape(testTiling.shapeInfo.orgCo, singleDo, singleM);
    testTiling.SetGroups(64);
    testTiling.attrInfo.groupOpt = 1;
    testTiling.shapeInfo.cinOpt = testTiling.shapeInfo.orgCi;
    testTiling.shapeInfo.coutOpt = testTiling.shapeInfo.orgCo;
    int64_t singleCoreGroupOpt = CeilDiv(testTiling.attrInfo.groupOpt, 1);
    testTiling.SetOptGroupInfo(static_cast<int64_t>(testTiling.attrInfo.groupOpt),
                                    singleCoreGroupOpt,
                                    static_cast<int64_t>(testTiling.shapeInfo.cinOpt),
                                    static_cast<int64_t>(testTiling.shapeInfo.coutOpt));

    testTiling.hasBias = 1;
    SetTypePointWise(testTiling);

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Format_negative1)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling.SetOutputType(TPosition::GM, ConvFormat::NDC1HWC0, ConvDtype::BF16);
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);
    }

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

TEST_F(TestConv3dTiling, PointWise_Format_negative2)
{
    Ops::NN::Conv3dV2::TConv3DTiling tilingData;
    Conv3dApiTiling::Conv3dTiling testTiling(platform);
    testTiling.SetOrgFmapShape(256, 64, 5, 64);
    testTiling.SetOrgWeightShape(256, 1, 1, 1);
    SetInputPointWise(testTiling);

    SetSingleOutputShapeInTest(testTiling);
    testTiling.hasBias = 1;
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::FRACTAL_Z_3D, ConvDtype::BF16);
    testTiling.SetOutputType(TPosition::GM, ConvFormat::NCDHW, ConvDtype::BF16);
    if (testTiling.hasBias) {
        testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);
    }

    int64_t ret1 = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret1, -1);
}

} // namespace