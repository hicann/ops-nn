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
 * \file test_conv2d_ascendc_api_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/conv2d_tiling.h"
#include "platform/platform_info.h"
#include "test_conv2d_ascendc_utils_tiling.h"

using namespace std;
using namespace conv_tiling;
using namespace conv_tiling_algo_m;
using namespace conv_tiling_algo_hw;
using namespace conv_tiling_algo_bb;
using namespace conv_api_tiling_test;

class TestConv2dTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown()
    {
    }
};

void ClearRange(ConvTilingAlgorithmHWmode* algo)
{
    algo->l0TilingRange.hoL0Range.clear();
    algo->l0TilingRange.woL0Range.clear();
    algo->l0TilingRange.kL0Range.clear();
    algo->l0TilingRange.nL0Range.clear();
}
PlatformInfo SetPlatFormInfo()
{
    PlatformInfo platformInfo;
    platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910_95;
    platformInfo.l1Size = 524288;
    platformInfo.l0ASize = 65536;
    platformInfo.l0BSize = 65536;
    platformInfo.l0CSize = 262144;
    platformInfo.ubSize = 262144;
    platformInfo.btSize = 4096;
    platformInfo.fbSize = 4096;
    return platformInfo;
}
void SetDtype(ConvTilingBase &testTiling,
              ConvDtype fmapDtype,
              ConvDtype weightDtype,
              ConvDtype biasDtype = ConvDtype::FLOAT16,
              ConvDtype quantScaleType = ConvDtype::INT64)
{
    testTiling.descInfo.fMapType.dtype = fmapDtype;
    testTiling.descInfo.weightType.dtype = weightDtype;
    testTiling.descInfo.biasType.dtype = biasDtype;
    testTiling.descInfo.quantScaleType.dtype = quantScaleType;
}

void SetFormat(ConvTilingBase &testTiling,
              ConvFormat fmapFormat,
              ConvFormat weightFormat,
              ConvFormat biasFormat = ConvFormat::ND,
              ConvFormat quantScaleFormat = ConvFormat::ND)
{
    testTiling.descInfo.fMapType.format = fmapFormat;
    testTiling.descInfo.weightType.format = weightFormat;
    testTiling.descInfo.biasType.format = biasFormat;
    testTiling.descInfo.quantScaleType.format = quantScaleFormat;
}

TEST_F(TestConv2dTiling, test_algo_HW_InitPingPong)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.InferCubeInfo();
    testTiling.cubeInfo.k0 = 32;
    testTiling.cubeInfo.n0 = 16;
    testTiling.cubeInfo.m0 = 16;
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = 1;
    testTiling.hasBias = true;
    testTiling.isC04Flag = false;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    algo.InitPingPong();
}

// TEST_F(TestConv2dTiling, test_algo_HW_GetL0Tiling)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     testTiling.InferCubeInfo();
//     testTiling.l1TilingInfo.nBL1 = 16;
//     testTiling.l1TilingInfo.hoAL1 = 16;
//     testTiling.l1TilingInfo.woAL1 = 16;
//     testTiling.l1TilingInfo.kAL1 = 32;
//     testTiling.l1TilingInfo.kBL1 = 32;
//     algo.multiNBL1 = 1;
//     algo.multiHoAL1 = 1;
//     algo.multiWoAL1 = 1;
//     algo.GetL0Tiling();
//     EXPECT_EQ(testTiling.l0TilingInfo.hoL0 > 0, true);
//     EXPECT_EQ(testTiling.l0TilingInfo.woL0 > 0, true);
//     EXPECT_EQ(testTiling.l0TilingInfo.kL0 > 0, true);
//     EXPECT_EQ(testTiling.l0TilingInfo.nL0 > 0, true);
//     EXPECT_EQ(algo.multiNBL1 > 0, true);
// }

/*
TEST_F(TestConv2dTiling, test_algo_HW_GetL0TilingKAL1KBL1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.InferCubeInfo();
    testTiling.l1TilingInfo.nBL1 = 16;
    testTiling.l1TilingInfo.hoAL1 = 16;
    testTiling.l1TilingInfo.woAL1 = 16;
    algo.multiNBL1 = 1;
    algo.multiHoAL1 = 1;
    algo.multiWoAL1 = 1;

    testTiling.l1TilingInfo.kAL1 = 32;
    testTiling.l1TilingInfo.kBL1 = 32;
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32};
    EXPECT_EQ(algo.l0TilingRange.kL0Range, expRes0);
    algo.L0TilingDecision();
    bool flag = std::find(expRes0.begin(), expRes0.end(), algo.l0Params.kL0) != expRes0.end();
    EXPECT_EQ(flag, true);
    ClearRange(&algo);

    testTiling.l1TilingInfo.kAL1 = 32;
    testTiling.l1TilingInfo.kBL1 = 16;
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes1 = {16};
    EXPECT_EQ(algo.l0TilingRange.kL0Range, expRes1);
    algo.L0TilingDecision();
    std::cout << algo.l0Params.kL0 << std::endl;
    flag = std::find(expRes1.begin(), expRes1.end(), algo.l0Params.kL0) != expRes1.end();
    EXPECT_EQ(flag, true);
    ClearRange(&algo);

    testTiling.l1TilingInfo.kAL1 = 64;
    testTiling.l1TilingInfo.kBL1 = 32;
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes2 = {16, 32};
    EXPECT_EQ(algo.l0TilingRange.kL0Range, expRes2);
    algo.L0TilingDecision();
    flag = std::find(expRes2.begin(), expRes2.end(), algo.l0Params.kL0) != expRes2.end();
    EXPECT_EQ(flag, true);
}

TEST_F(TestConv2dTiling, test_algo_HW_GetL0TilingNL0)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling); 
    testTiling.InferCubeInfo();
    testTiling.l1TilingInfo.kAL1 = 16;
    testTiling.l1TilingInfo.kBL1 = 16;
    testTiling.l1TilingInfo.hoAL1 = 16;
    testTiling.l1TilingInfo.woAL1 = 16;
    algo.multiHoAL1 = 1;
    algo.multiWoAL1 = 1;
    algo.multiNBL1 = 0;
    testTiling.l1TilingInfo.nBL1 = 32;
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes0 = {16, 32};
    EXPECT_EQ(algo.l0TilingRange.nL0Range, expRes0);
    algo.L0TilingDecision();
    bool flag = std::find(expRes0.begin(), expRes0.end(), algo.l0Params.nL0) != expRes0.end();
    EXPECT_EQ(flag, true);
    ClearRange(&algo);

    algo.multiNBL1 = 1;
    testTiling.l1TilingInfo.nBL1 = 128;
    algo.GetL0TilingRange();
    std::vector<uint64_t> expRes1 = {128};
    EXPECT_EQ(algo.l0TilingRange.nL0Range, expRes1);
    algo.L0TilingDecision();
    flag = std::find(expRes1.begin(), expRes1.end(), algo.l0Params.nL0) != expRes1.end();
    EXPECT_EQ(flag, true);
}

// TEST_F(TestConv2dTiling, test_algo_HW_GetL0TilingHoL0)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     testTiling.InferCubeInfo();
//     testTiling.l1TilingInfo.kAL1 = 16;
//     testTiling.l1TilingInfo.kBL1 = 16;
//     testTiling.l1TilingInfo.nBL1 = 16;
//     testTiling.l1TilingInfo.woAL1 = 16;
//     algo.multiNBL1 = 1;
//     algo.multiWoAL1 = 1;

//     algo.multiHoAL1 = 0;
//     testTiling.l1TilingInfo.hoAL1 = 30;
//     algo.minBurstNum = 64;
//     testTiling.shapeInfo.singleWo = 16;
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes0 = {1, 2, 3};
//     std::vector<uint64_t> expRes1 = {5, 6, 10, 15, 30};
//     EXPECT_EQ(algo.l0TilingRange.hoL0SpareRange, expRes0);
//     EXPECT_EQ(algo.l0TilingRange.hoL0Range, expRes1);
//     algo.L0TilingDecision();
//     bool flag = std::find(expRes1.begin(), expRes1.end(), algo.l0Params.hoL0) != expRes1.end();
//     EXPECT_EQ(flag, true);
//     ClearRange(&algo);

//     testTiling.l1TilingInfo.hoAL1 = 30;
//     algo.minBurstNum = 64;
//     testTiling.shapeInfo.singleWo = 1;
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes2 = {1, 2, 3, 5, 6, 10, 15, 30};
//     std::vector<uint64_t> expRes3 = {};
//     EXPECT_EQ(algo.l0TilingRange.hoL0SpareRange, expRes2);
//     EXPECT_EQ(algo.l0TilingRange.hoL0Range, expRes3);
//     algo.L0TilingDecision();
//     flag = std::find(expRes2.begin(), expRes2.end(), algo.l0Params.hoL0) != expRes2.end();
//     EXPECT_EQ(flag, true);
//     ClearRange(&algo);

//     algo.multiHoAL1 = 1;
//     testTiling.l1TilingInfo.hoAL1 = 28;
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes4 = {28};
//     EXPECT_EQ(algo.l0TilingRange.hoL0Range, expRes4);
//     algo.L0TilingDecision();
//     flag = std::find(expRes4.begin(), expRes4.end(), algo.l0Params.hoL0) != expRes4.end();
//     EXPECT_EQ(flag, true);
// }

// TEST_F(TestConv2dTiling, test_algo_HW_GetL0TilingWoL0)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     testTiling.InferCubeInfo();
//     testTiling.l1TilingInfo.kAL1 = 16;
//     testTiling.l1TilingInfo.kBL1 = 16;
//     testTiling.l1TilingInfo.nBL1 = 16;
//     testTiling.l1TilingInfo.hoAL1 = 16;
//     testTiling.l1TilingInfo.woAL1 = 150;
//     algo.multiNBL1 = 1;
//     algo.multiHoAL1 = 1;

//     algo.multiWoAL1 = 0;

//     algo.minBurstNum = 64;
//     testTiling.cubeInfo.m0 = 16;
//     algo.InitPingPong();
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes0 = {16, 32};
//     std::vector<uint64_t> expRes1 = {80};
//     EXPECT_EQ(algo.l0TilingRange.woL0SpareRange, expRes0);
//     EXPECT_EQ(algo.l0TilingRange.woL0Range, expRes1);
//     algo.L0TilingDecision();
//     bool flag = std::find(expRes1.begin(), expRes1.end(), algo.l0Params.woL0) != expRes1.end();
//     EXPECT_EQ(flag, false);
//     ClearRange(&algo);

//     algo.multiWoAL1 = 0;
//     testTiling.l1TilingInfo.woAL1 = 63;
//     algo.minBurstNum = 64;
//     testTiling.cubeInfo.m0 = 16;
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes2 = {16, 32};
//     std::vector<uint64_t> expRes3 = {};
//     EXPECT_EQ(algo.l0TilingRange.woL0SpareRange, expRes2);
//     EXPECT_EQ(algo.l0TilingRange.woL0Range, expRes3);
//     algo.L0TilingDecision();
//     flag = std::find(expRes2.begin(), expRes2.end(), algo.l0Params.woL0) != expRes2.end();
//     EXPECT_EQ(flag, true);
//     ClearRange(&algo);

//     algo.multiWoAL1 = 1;
//     testTiling.l1TilingInfo.woAL1 = 120;
//     algo.GetL0TilingRange();
//     std::vector<uint64_t> expRes4 = {120};
//     EXPECT_EQ(algo.l0TilingRange.woL0Range, expRes4);
//     algo.L0TilingDecision();
//     flag = std::find(expRes4.begin(), expRes4.end(), algo.l0Params.woL0) != expRes4.end();
//     EXPECT_EQ(flag, true);
// }

TEST_F(TestConv2dTiling, test_algo_HW_CheckL0DoubleBuffer)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.InitPingPong();
    testTiling.InferCubeInfo();
    testTiling.l0TilingInfo.hoL0 = 15;
    testTiling.l0TilingInfo.woL0 = 12;
    testTiling.l0TilingInfo.kL0 = 10;
    testTiling.l0TilingInfo.nL0 = 13;
    testTiling.l1TilingInfo.hoAL1 = 15;
    testTiling.l1TilingInfo.woAL1 = 12;
    testTiling.l1TilingInfo.kAL1 = 10;
    testTiling.l1TilingInfo.kBL1 = 10;
    testTiling.l1TilingInfo.nBL1 = 13;

    algo.CheckL0DoubleBuffer();

    EXPECT_EQ(algo.dbValue.pbAL0, 2);
    EXPECT_EQ(algo.dbValue.pbBL0, 2);
    EXPECT_EQ(algo.dbValue.pbCL0, 2);
    algo.dbValue.pbAL0 = 2;
    algo.dbValue.pbBL0 = 2;
    algo.dbValue.pbCL0 = 1;

    testTiling.l0TilingInfo.hoL0 = 1;
    testTiling.l0TilingInfo.woL0 = 2;
    testTiling.l0TilingInfo.kL0 = 3;
    testTiling.l0TilingInfo.nL0 = 8193;
    testTiling.l1TilingInfo.hoAL1 = 15;
    testTiling.l1TilingInfo.woAL1 = 12;
    testTiling.l1TilingInfo.kAL1 = 10;
    testTiling.l1TilingInfo.kBL1 = 13;
    testTiling.l1TilingInfo.nBL1 = 1;
    algo.CheckL0DoubleBuffer();
    EXPECT_EQ(algo.dbValue.pbAL0, 2);
    EXPECT_EQ(algo.dbValue.pbBL0, 2);
    EXPECT_EQ(algo.dbValue.pbCL0, 1);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateNL0)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.InitPingPong();
    testTiling.InferCubeInfo();
    algo.l0TilingRange.nL0Range = {1, 2, 4, 5, 8, 9, 10};
    algo.l0Params.nL0Index = 0;
    algo.l0Params.woL0 = 200;
    algo.l0Params.hoL0 = 20;
    algo.l0Params.kL0 = 1;
    auto ret = algo.UpdateNL0();
    EXPECT_EQ(algo.l0Params.nL0, 10);
    EXPECT_EQ(ret, true);

    algo.l0Params.nL0Index = 0;
    algo.l0Params.woL0 = 20;
    algo.l0Params.hoL0 = 2;
    algo.l0Params.kL0 = 1639;
    ret = algo.UpdateNL0();
    EXPECT_EQ(algo.l0Params.nL0, 1);
    EXPECT_EQ(ret, true);

    algo.l0Params.nL0Index = 0;
    algo.l0Params.woL0 = 2;
    algo.l0Params.hoL0 = 2;
    algo.l0Params.kL0 = 1;
    ret = algo.UpdateNL0();
    EXPECT_EQ(algo.l0Params.nL0, 4);
    EXPECT_EQ(ret, false);

    algo.l0Params.nL0Index = 0;
    algo.l0Params.woL0 = 2;
    algo.l0Params.hoL0 = 10;
    algo.l0Params.kL0 = 1;
    ret = algo.UpdateNL0();
    EXPECT_EQ(algo.l0Params.nL0, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateHoL0WoL0)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.InitPingPong();
    algo.l0TilingRange.woL0Range = {1, 2, 4, 5, 8, 9, 10};
    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1639;
    algo.l0Params.nL0 = 11;
    auto ret = algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range);
    EXPECT_EQ(algo.l0Params.woL0, 1);
    EXPECT_EQ(ret, true);

    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 3278;
    ret = algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range);
    EXPECT_EQ(algo.l0Params.woL0, 10);
    EXPECT_EQ(ret, true);

    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 20;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 80;
    ret = algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range);
    EXPECT_EQ(algo.l0Params.woL0, 4);
    EXPECT_EQ(ret, false);

    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 100;
    ret = algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range);
    EXPECT_EQ(algo.l0Params.woL0, 10);
    EXPECT_EQ(ret, true);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateKL0)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.InitPingPong();
    algo.l0TilingRange.kL0Range = {1, 2, 4, 5, 8, 9, 10};
    algo.l0Params.kL0Index = 0;
    algo.l0Params.woL0 = 1639;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 1;
    algo.UpdateKL0();
    EXPECT_EQ(algo.l0Params.kL0, 1);

    algo.l0TilingRange.woL0Range = {1, 2, 4, 5, 8, 9, 10};
    algo.l0Params.kL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 1689;
    algo.UpdateKL0();
    EXPECT_EQ(algo.l0Params.kL0, 1);

    algo.l0TilingRange.woL0Range = {1, 2, 4, 5, 8, 9, 10};
    algo.l0Params.kL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 1;
    algo.UpdateKL0();
    EXPECT_EQ(algo.l0Params.kL0, 10);
}

TEST_F(TestConv2dTiling, test_algo_HW_SetL0TilingRes)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);

    algo.l0Params.hoL0 = 100;
    algo.l0Params.woL0 = 101;
    algo.l0Params.kL0 = 102;
    algo.l0Params.nL0 = 16;
    testTiling.l1TilingInfo.nBL1 = 104;
    algo.SetL0TilingRes();
    EXPECT_EQ(testTiling.l0TilingInfo.hoL0, 100);
    EXPECT_EQ(testTiling.l0TilingInfo.woL0, 101);
    EXPECT_EQ(testTiling.l0TilingInfo.kL0, 102);
    EXPECT_EQ(testTiling.l0TilingInfo.nL0, 16);
    EXPECT_EQ(algo.multiNBL1, 7);
}

TEST_F(TestConv2dTiling, test_algo_HW_InitABL1TilingMode1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.shapeInfo.singleWo = 15;
    testTiling.cubeInfo.m0 = 16;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckBL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::FULL_LOAD_BL1);

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::NONE_FULL_LOAD);
}

TEST_F(TestConv2dTiling, test_algo_HW_InitABL1TilingMode2)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.shapeInfo.singleWo = 16;
    testTiling.cubeInfo.m0 = 16;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckABL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::ALL_FULL_LOAD);
}

TEST_F(TestConv2dTiling, test_algo_HW_InitABL1TilingMode3)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.shapeInfo.singleWo = 16;
    testTiling.cubeInfo.m0 = 16;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsBL1FullLoadFirst).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckABL1FullLoad).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckAL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckBL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::FULL_LOAD_BL1);

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::FULL_LOAD_AL1);

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::NONE_FULL_LOAD);
}

TEST_F(TestConv2dTiling, test_algo_HW_InitABL1TilingMode4)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.shapeInfo.singleWo = 16;
    testTiling.cubeInfo.m0 = 16;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsBL1FullLoadFirst).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckABL1FullLoad).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckAL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckBL1FullLoad).stubs().will(returnValue(true)).then(returnValue(false));

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::FULL_LOAD_AL1);

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::FULL_LOAD_BL1);

    algo.InitABL1TilingMode();
    EXPECT_EQ(algo.l1Flags.abL1Mode, L1TilingMode::NONE_FULL_LOAD);
}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingDecision_ALL_FULL_LOAD)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1Flags.abL1Mode = L1TilingMode::ALL_FULL_LOAD;
    algo.l1TilingInit.kAL1Init = 1;
    algo.l1TilingInit.kBL1Init = 2;
    algo.l1TilingInit.hoAL1Init = 3;
    algo.l1TilingInit.woAL1Init = 4;
    algo.l1TilingInit.nBL1Init = 5;

    algo.CoreL1TilingDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 1);
    EXPECT_EQ(algo.l1Params.kBL1, 2);
    EXPECT_EQ(algo.l1Params.hoAL1, 3);
    EXPECT_EQ(algo.l1Params.woAL1, 4);
    EXPECT_EQ(algo.l1Params.nBL1, 5);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_M_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, true);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, true);


}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingDecision_FULL_LOAD_BL1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_BL1;
    algo.l1TilingInit.kAL1Init = 1;
    algo.l1TilingInit.kBL1Init = 2;
    algo.l1TilingInit.hoAL1Init = 3;
    algo.l1TilingInit.woAL1Init = 4;
    algo.l1TilingInit.nBL1Init = 5;
    algo.l1TilingRange.hoAL1StrictRange = {11,22};
    algo.l1TilingRange.woAL1StrictRange = {33,44};
    algo.l1TilingRange.nBL1StrictRange = {55,66};
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::MultiIterKAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();
    algo.CoreL1TilingDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 0);
    EXPECT_EQ(algo.l1Params.kBL1, 2);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 5);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_M_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, true);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, true);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingDecision_FULL_LOAD_AL1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1Flags.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
    algo.l1TilingInit.kAL1Init = 1;
    algo.l1TilingInit.kBL1Init = 2;
    algo.l1TilingInit.hoAL1Init = 3;
    algo.l1TilingInit.woAL1Init = 4;
    algo.l1TilingInit.nBL1Init = 5;
    algo.l1TilingRange.hoAL1StrictRange = {11,22};
    algo.l1TilingRange.woAL1StrictRange = {33,44};
    algo.l1TilingRange.nBL1StrictRange = {55,66};
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterNBL1).stubs();
    algo.CoreL1TilingDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 1);
    EXPECT_EQ(algo.l1Params.kBL1, 0);
    EXPECT_EQ(algo.l1Params.hoAL1, 3);
    EXPECT_EQ(algo.l1Params.woAL1, 4);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_N_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::MultiIterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 0);
    EXPECT_EQ(algo.l1Params.kBL1, 1);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_M_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision2)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(false)).then(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 1);
    EXPECT_EQ(algo.l1Params.kBL1, 0);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_N_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision3)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::MultiIterKBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 0);
    EXPECT_EQ(algo.l1Params.kBL1, 0);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_N_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision4)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 1);
    EXPECT_EQ(algo.l1Params.kBL1, 0);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_N_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision5)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(false)).then(returnValue(true));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::MultiIterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 0);
    EXPECT_EQ(algo.l1Params.kBL1, 1);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_M_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_CoreL1TilingNoneFullLoadDecision6)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1TilingRange.kAL1StrictRange = {0};
    algo.l1TilingRange.kBL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};

    algo.l1TilingCalc.kBL1MaxSize = 1;
    algo.l1TilingCalc.kAL1MaxSize = 1;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IsKBL1FullLoadFirst).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CheckKL1FullLoad).stubs().will(returnValue(false));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterNBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::MultiIterKAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterKBL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoWoAL1).stubs();

    algo.CoreL1TilingNoneFullLoadDecision();
    EXPECT_EQ(algo.l1Params.kAL1, 0);
    EXPECT_EQ(algo.l1Params.kBL1, 0);
    EXPECT_EQ(algo.l1Params.hoAL1, 0);
    EXPECT_EQ(algo.l1Params.woAL1, 0);
    EXPECT_EQ(algo.l1Params.nBL1, 0);
    EXPECT_EQ(algo.l1Flags.iterateMNOrder, IterateMNOrder::ITER_M_FST);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);

}

TEST_F(TestConv2dTiling, test_algo_HW_IterKAL1_fullload)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.kAL1StrictRange = {1};
    algo.l1TilingCalc.kAL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterKAL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    EXPECT_EQ(kAL1, 1);
    EXPECT_EQ(algo.multiHoAL1, 0);
    EXPECT_EQ(algo.multiWoAL1, 0);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterKAL1_split)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.kAL1StrictRange = {2};
    algo.l1TilingCalc.kAL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterKAL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    algo.UpdateMultiL1Params(kAL1, true);
    EXPECT_EQ(kAL1, 2);
    EXPECT_EQ(algo.multiHoAL1, 1);
    EXPECT_EQ(algo.multiWoAL1, 1);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterKBL1_fullload)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.kBL1StrictRange = {1};
    algo.l1TilingCalc.kBL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterKBL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    EXPECT_EQ(kBL1, 1);
    EXPECT_EQ(algo.multiNBL1, 0);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterKBL1_split)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.kBL1StrictRange = {2};
    algo.l1TilingCalc.kBL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterKBL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    algo.UpdateMultiL1Params(kBL1, false);
    EXPECT_EQ(kBL1, 2);
    EXPECT_EQ(algo.multiNBL1, 1);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterNBL1_fullload)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.nBL1StrictRange = {1};
    algo.l1TilingCalc.nBL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterNBL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    EXPECT_EQ(nBL1, 1);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, true);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, true);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterNBL1_split)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.nBL1StrictRange = {2};
    algo.l1TilingCalc.nBL1MaxSize = 1;
    testTiling.platformInfo.l1Size = 999999;

    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size));
    algo.IterNBL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
    EXPECT_EQ(nBL1, 2);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
}

TEST_F(TestConv2dTiling, test_algo_HW_IterHoWoAL1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t kAL1 = 0;
    uint64_t kBL1 = 0;
    uint64_t hoAL1 = 0;
    uint64_t woAL1 = 0;
    uint64_t nBL1 = 0;
    algo.l1TilingRange.woAL1StrictRange = {0};
    algo.l1TilingRange.hoAL1StrictRange = {0};
    algo.l1TilingRange.nBL1StrictRange = {0};
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoAL1).stubs();
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterWoAL1).stubs();

    algo.IterHoWoAL1(kAL1, kBL1, hoAL1, woAL1, nBL1);
}

TEST_F(TestConv2dTiling, test_algo_HW_ResetHWL1minSize)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.cubeInfo.k0 = 16;
    testTiling.cubeInfo.m0 = 16;
    algo.l1TilingCalc.aL1Minsize = 2048;
    algo.l1TilingCalc.bL1Minsize = 0;
    algo.l1TilingCalc.biasL1MinSize = 0;
    algo.l1TilingCalc.fixpParamsL1MinSize = 0;
    testTiling.platformInfo.l1Size = 1024;
    algo.l1TilingCalc.hoAL1Min = 5;
    algo.l1TilingCalc.hoAL1Min = 5;
    algo.fMapDTypeSize = 2;
    int64_t tmp1 = 2;
    int64_t tmp3 = 512;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::InferHiL1).stubs().will(returnValue(tmp1));
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::InferWiL1).stubs().will(returnValue(tmp3));
    int64_t res = algo.ResetHWL1minSize();
    EXPECT_EQ(res, -1);
}

// TEST_F(TestConv2dTiling, test_algo_HW_MultiIterHoAL1)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterHoAL1).stubs();
//     conv_tiling_algo_hw::L1TilingParams inputTiling;
//     std::vector<uint64_t> inputHoAL1Range;
//     uint64_t hoAL1 = 0;
//     algo.MultiIterHoAL1(inputTiling, inputHoAL1Range, hoAL1);
// }

// TEST_F(TestConv2dTiling, test_algo_HW_MultiIterWoAL1_1)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterWoAL1).stubs();
//     conv_tiling_algo_hw::L1TilingParams inputTiling;
//     std::vector<uint64_t> inputWoAL1Range = {1};
//     std::vector<uint64_t> inputHoAL1Range = {1};
//     uint64_t woAL1 = 0;
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     algo.MultiIterWoAL1(inputTiling, inputWoAL1Range, inputHoAL1Range, woAL1);
// }

// TEST_F(TestConv2dTiling, test_algo_HW_MultiIterWoAL1_2)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterWoAL1).stubs();
//     conv_tiling_algo_hw::L1TilingParams inputTiling;
//     std::vector<uint64_t> inputWoAL1Range = {1};
//     std::vector<uint64_t> inputHoAL1Range = {1};
//     uint64_t woAL1 = 0;
//     algo.l1Flags.useWoAL1SpareRange = false;
//     algo.l1TilingRange.woAL1SpareRange = {1};
//     algo.MultiIterWoAL1(inputTiling, inputWoAL1Range, inputHoAL1Range, woAL1);
// }

// TEST_F(TestConv2dTiling, test_algo_HW_MultiIterWoAL1_3)
// {
//     Conv2dTiling testTiling(SetPlatFormInfo());
//     SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
//     ConvTilingAlgorithmHWmode algo(&testTiling);
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     MOCKER_CPP(&ConvTilingAlgorithmHWmode::IterWoAL1).stubs();
//     conv_tiling_algo_hw::L1TilingParams inputTiling;
//     std::vector<uint64_t> inputWoAL1Range = {1};
//     std::vector<uint64_t> inputHoAL1Range = {1};
//     uint64_t woAL1 = 0;
//     algo.l1Flags.usehoAL1SpareRange = false;
//     algo.l1TilingRange.hoAL1SpareRange = {1};
//     algo.l1Flags.useWoAL1SpareRange = false;
//     algo.l1TilingRange.woAL1SpareRange = {1};
//     algo.MultiIterWoAL1(inputTiling, inputWoAL1Range, inputHoAL1Range, woAL1);
// }

TEST_F(TestConv2dTiling, test_algo_HW_UpdateAL1DoubelBuffer)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.platformInfo.l1Size = 99999;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size + 1));

    algo.UpdateAL1DoubleBuffer();
    EXPECT_EQ(testTiling.dbValue.pbAL1, 1);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateBiasFixpParamsL1Fullload)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.platformInfo.l1Size = 99999;
    algo.l1Flags.isBiasFullLoad = false;
    algo.l1Flags.isFixpParamsFullLoad = false;
    MOCKER_CPP(&ConvTilingAlgorithmHWmode::CalcCurL1Size).stubs().will(returnValue(testTiling.platformInfo.l1Size - 1)).then(returnValue(testTiling.platformInfo.l1Size));

    algo.UpdateBiasFixpParamsL1Fullload();
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, true);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, true);
}

TEST_F(TestConv2dTiling, test_conv2d_check_algorithmLimit)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    testTiling.outputOrder = static_cast<int8_t>(OutputOrder::HW);
    auto ret = testTiling.CheckAlgorithmLimit();
    EXPECT_EQ(ret, true);

    testTiling.outputOrder = static_cast<int8_t>(OutputOrder::M);
    testTiling.hasQuantScale = true;
    ret = testTiling.CheckAlgorithmLimit();
    EXPECT_EQ(ret, true);

    testTiling.hasQuantScale = false;
    testTiling.isC04Flag = true;
    ret = testTiling.CheckAlgorithmLimit();
    EXPECT_EQ(ret, false);

    testTiling.isC04Flag = false;
    ret = testTiling.CheckAlgorithmLimit();
    EXPECT_EQ(ret, true);
}

TEST_F(TestConv2dTiling, test_conv2d_check_attr)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.attrInfo.padLeft = -1;
    auto ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.strideH = -1;
    ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.dilationH = -1;
    ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, true);

}

TEST_F(TestConv2dTiling, test_conv2d_check_quant_attr)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.hasQuantScale = true;
    testTiling.attrInfo.offsetx = 1;
    testTiling.descInfo.fMapType.dtype = ConvDtype::HIFLOAT8;
    auto ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.offsetx = 0;
    testTiling.attrInfo.roundMode = 0;
    ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

    testTiling.descInfo.fMapType.dtype = ConvDtype::FLOAT8_E4M3FN;
    testTiling.attrInfo.roundMode = 0;
    ret = testTiling.CheckAttr();
    EXPECT_EQ(ret, false);

}

TEST_F(TestConv2dTiling, test_conv2d_check_fm_shape)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.shapeInfo.orgHi = -1;
    auto ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgHi = 1;
    testTiling.shapeInfo.orgWi = -1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgWi = 1;
    testTiling.outputOrder = static_cast<int8_t>(OutputOrder::HW);
    testTiling.shapeInfo.singleHo = -1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleHo = 1;
    testTiling.shapeInfo.singleWo = -1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleWo = 1;
    testTiling.outputOrder = static_cast<int8_t>(OutputOrder::M);
    testTiling.shapeInfo.singleM = -1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleM = 1;

    testTiling.shapeInfo.orgCi = -1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgCi = MAX_31_BIT_NUM + 1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgCi = MAX_31_BIT_NUM - 1;
    testTiling.shapeInfo.singleCi = testTiling.shapeInfo.orgCi;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgCi = MAX_31_BIT_NUM;
    testTiling.shapeInfo.singleCi = testTiling.shapeInfo.orgCi;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.singleCi = 2;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleCi = 1;
    testTiling.shapeInfo.orgHi = 1;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.dilationH = 10;
    testTiling.shapeInfo.orgkH = 2;
    testTiling.attrInfo.strideH = 1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);


    testTiling.shapeInfo.orgHi = 1;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    testTiling.attrInfo.dilationH = 1;
    testTiling.shapeInfo.orgkH = 1;
    testTiling.attrInfo.strideH = 1;

    testTiling.shapeInfo.orgWi = 1;
    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.dilationW = 10;
    testTiling.shapeInfo.orgkW = 2;
    testTiling.attrInfo.strideW = 1;
    ret = testTiling.CheckFeaMapShape();
    EXPECT_EQ(ret, false);

}

TEST_F(TestConv2dTiling, test_conv2d_check_weight_shape)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.singleCo = 16;
    testTiling.shapeInfo.orgkW = 1;
    testTiling.shapeInfo.singlekW = 1;

    testTiling.shapeInfo.orgkH = -1;
    auto ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkH = MAX_31_BIT_NUM + 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkH = MAX_31_BIT_NUM;
    testTiling.shapeInfo.singlekH = testTiling.shapeInfo.orgkH;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgkH = MAX_31_BIT_NUM - 1;
    testTiling.shapeInfo.singlekH = testTiling.shapeInfo.orgkH;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgkW = -1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkW = MAX_31_BIT_NUM + 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkW = MAX_31_BIT_NUM;
    testTiling.shapeInfo.singlekW = testTiling.shapeInfo.orgkW;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgkW = MAX_31_BIT_NUM - 1;
    testTiling.shapeInfo.singlekW = testTiling.shapeInfo.orgkW;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgCo = -1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgCo = MAX_31_BIT_NUM + 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgCo = MAX_31_BIT_NUM;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.orgCo = MAX_31_BIT_NUM - 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.singleCo = -1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleCo = MAX_31_BIT_NUM + 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleCo = MAX_31_BIT_NUM;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.singleCo = MAX_31_BIT_NUM - 1;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, true);

    testTiling.shapeInfo.singleCo = 1;
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.orgkH = 2;
    ret = testTiling.CheckWeightShape();
    EXPECT_EQ(ret, false);

}

TEST_F(TestConv2dTiling, test_conv2d_check_format)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.descInfo.weightType.format = ConvFormat::NHWC;
    auto ret = testTiling.CheckFormat();
    EXPECT_EQ(ret, false);

    testTiling.descInfo.weightType.format = ConvFormat::NCHW;
    testTiling.descInfo.fMapType.format = ConvFormat::NHWC;
    ret = testTiling.CheckFormat();
    EXPECT_EQ(ret, false);

    testTiling.descInfo.weightType.format = ConvFormat::NCHW;
    testTiling.descInfo.fMapType.format = ConvFormat::NCHW;
    testTiling.descInfo.biasType.format = ConvFormat::NHWC;
    testTiling.hasBias = true;
    ret = testTiling.CheckFormat();
    EXPECT_EQ(ret, true);

    testTiling.descInfo.weightType.format = ConvFormat::FRACTAL_Z_C04;
    testTiling.descInfo.fMapType.format = ConvFormat::NCHW;
    testTiling.descInfo.biasType.format = ConvFormat::ND;
    ret = testTiling.CheckFormat();
    EXPECT_EQ(ret, true);

}

TEST_F(TestConv2dTiling, test_conv2d_check_dtype)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.hasQuantScale = true;
    testTiling.hasBias = true;
    bool ret = false;
    for (auto group : SUPPORTED_QUANT_TYPES_WITH_BIAS) {
        testTiling.descInfo.fMapType.dtype = group[0];
        testTiling.descInfo.weightType.dtype = group[1];
        testTiling.descInfo.biasType.dtype = group[2];
        testTiling.descInfo.outputType.dtype = group[3];
        ret = testTiling.CheckDtype();
        EXPECT_EQ(ret, true);
    }
    testTiling.hasQuantScale = true;
    testTiling.hasBias = false;
    for (auto group : SUPPORTED_QUANT_TYPES_WITHOUT_BIAS) {
        testTiling.descInfo.fMapType.dtype = group[0];
        testTiling.descInfo.weightType.dtype = group[1];
        testTiling.descInfo.outputType.dtype = group[2];
        EXPECT_EQ(ret, true);
    }

    testTiling.hasQuantScale = false;
    testTiling.hasBias = false;
    for (auto group : SUPPORTED_CONV2D_TYPES_WITH_BIAS) {
        testTiling.descInfo.fMapType.dtype = group[0];
        testTiling.descInfo.weightType.dtype = group[1];
        testTiling.descInfo.biasType.dtype = group[2];
        testTiling.descInfo.outputType.dtype = group[3];
        ret = testTiling.CheckDtype();
        EXPECT_EQ(ret, true);
    }

    testTiling.hasQuantScale = false;
    testTiling.hasBias = false;
    for (auto group : SUPPORTED_CONV2D_TYPES_WITHOUT_BIAS) {
        testTiling.descInfo.fMapType.dtype = group[0];
        testTiling.descInfo.weightType.dtype = group[1];
        testTiling.descInfo.outputType.dtype = group[2];
        ret = testTiling.CheckDtype();
        EXPECT_EQ(ret, true);
    }

    testTiling.hasQuantScale = true;
    testTiling.hasBias = true;
    auto group = SUPPORTED_QUANT_TYPES_WITH_BIAS[0];
    testTiling.descInfo.fMapType.dtype = ConvDtype::FLOAT16;
    testTiling.descInfo.weightType.dtype = group[1];
    testTiling.descInfo.biasType.dtype = group[2];
    testTiling.descInfo.outputType.dtype = group[3];
    ret = testTiling.CheckDtype();
    EXPECT_EQ(ret, false);
    testTiling.hasBias = true;
    ret = testTiling.CheckDtype();
    EXPECT_EQ(ret, false);

    testTiling.hasQuantScale = false;
    testTiling.hasBias = false;
    group = SUPPORTED_CONV2D_TYPES_WITHOUT_BIAS[0];
    testTiling.descInfo.fMapType.dtype = ConvDtype::INT8;
    testTiling.descInfo.weightType.dtype = group[1];
    testTiling.descInfo.biasType.dtype = group[2];
    testTiling.descInfo.outputType.dtype = group[3];
    ret = testTiling.CheckDtype();
    EXPECT_EQ(ret, false);
    testTiling.hasBias = true;
    ret = testTiling.CheckDtype();
    EXPECT_EQ(ret, false);
}

TEST_F(TestConv2dTiling, test_conv2d_check_DataCopyLimits)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    SetFormat(testTiling, ConvFormat::NCHW, ConvFormat::NCHW);
    testTiling.shapeInfo.orgHi = 1;
    testTiling.shapeInfo.orgWi = (MAX_40_BIT_NUM / 2) + 1;
    auto ret = testTiling.CheckDataCopyLimits();
    EXPECT_EQ(ret, false);
}

TEST_F(TestConv2dTiling, test_conv2d_check_FixpipeLimits)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    testTiling.shapeInfo.orgHo = 1;
    testTiling.shapeInfo.orgWo = MAX_32_BIT_NUM + 1;
    auto ret = testTiling.CheckFixpipeLimits();
    EXPECT_EQ(ret, false);
}

TEST_F(TestConv2dTiling, test_conv2d_check_Load3DLimits)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.attrInfo.strideH = LOAD3D_MAX_STRIDE_H_W + 1;
    auto ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.strideH = LOAD3D_MAX_STRIDE_H_W;
    testTiling.attrInfo.dilationH = LOAD3D_MAX_DILATION_H_W + 1;
    ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.dilationH = LOAD3D_MAX_DILATION_H_W;
    testTiling.attrInfo.padLeft = LOAD3D_MAX_PAD + 1;
    ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padLeft = LOAD3D_MAX_PAD;
    testTiling.shapeInfo.orgkH = LOAD3D_MAX_FILTER_H_W_910D + 1;
    ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkH = 1;
    testTiling.shapeInfo.orgkW = LOAD3D_MAX_DDR2L1_SIZE / 16 + 1;
    ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkW = 1;
    ret = testTiling.CheckLoad3DLimits();
    EXPECT_EQ(ret, true);
}

TEST_F(TestConv2dTiling, test_conv2d_algo_choose_Mmode)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;

    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);

    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tilingData.get_woL1(), 0);
    EXPECT_EQ(tilingData.get_woL0(), 0);
    EXPECT_EQ(tilingData.get_singleCoreWo(), 0);
}

TEST_F(TestConv2dTiling, test_conv2d_algo_choose_default_Mmode)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);

    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(tilingData.get_woL1(), 0);
    EXPECT_EQ(tilingData.get_woL0(), 0);
    EXPECT_EQ(tilingData.get_singleCoreWo(), 0);
}

TEST_F(TestConv2dTiling, test_conv2d_algo_choose_default_HWmode)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo, orgWo);
    testTiling.SetOutputOrder(0);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
    EXPECT_NE(tilingData.get_woL1(), 0);
    EXPECT_NE(tilingData.get_woL0(), 0);
    EXPECT_NE(tilingData.get_singleCoreWo(), 0);
}

TEST_F(TestConv2dTiling, test_conv2d_algo_choose_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetOutputOrder(2);
    int64_t ret = testTiling.Compute();
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_dtype_limit_INT8)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::INT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_dtype_limit_HIFLOAT8)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::HIFLOAT8);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::HIFLOAT8);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::HIFLOAT8);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_dtype_limit_FLOAT8_E4M3FN)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT8_E4M3FN);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT8_E4M3FN);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT8_E4M3FN);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_notsupport_quant)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetQuantScale(true);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_inputshape_fail)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo, orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_HWmode_inputshape_fail)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(0);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_notsupport_c04)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetC04Flag(true);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_minloadin_L1_fail)
{
    uint64_t orgCi = 1;
    uint64_t orgHi = 1;
    uint64_t orgWi = 65536;
    uint64_t orgCo = 1;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = 1;
    uint64_t orgWo = 65536;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_conv2d_HWmode_enable_hf32)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo, orgWo);
    testTiling.SetOutputOrder(0);
    testTiling.SetHF32(true, 0);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_conv2d_Mmode_enable_hf32)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo);
    testTiling.SetOutputOrder(1);
    testTiling.SetHF32(true, 0);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT32);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_algo_HW_GetL1TilingRange)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.singleCi1 = 1;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    testTiling.shapeInfo.singleCo1 = 1;
    testTiling.shapeInfo.singleWo = 1;
    testTiling.shapeInfo.singleHo = 1;

    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.GetL1TilingRange();
}

TEST_F(TestConv2dTiling, test_algo_HW_CheckAL1FullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.singleCi1 = 65535;
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = 1;
    ConvTilingAlgorithmHWmode algo(&testTiling);
    bool ret = algo.CheckAL1FullLoad();
    EXPECT_EQ(ret, false);
}

TEST_F(TestConv2dTiling, test_algo_HW_CheckKL1FullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = 2;
    ConvTilingAlgorithmHWmode algo(&testTiling);
    bool ret = algo.CheckKL1FullLoad(65535,1);
    EXPECT_EQ(ret, false);
}
*/

TEST_F(TestConv2dTiling, test_algo_M_CoreL1TilingDecision_FULL_LOAD_AL1)
{
    PlatformInfo platformInfo;
    platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    platformInfo.l1Size = 524288;
    platformInfo.l0ASize = 65536;
    platformInfo.l0BSize = 65536;
    platformInfo.l0CSize = 262144;
    platformInfo.ubSize = 262144;
    platformInfo.btSize = 4096;
    platformInfo.fbSize = 4096;
    Conv2dTiling testTiling(platformInfo);
    SetDtype(testTiling, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32);
    ConvTilingAlgorithmMmode algo(&testTiling);
    algo.l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
    algo.l1TilingCalc.fmapFullLoadL1Size = 524256;
    algo.l1TilingCalc.weightMinLoadL1Size = 32;
    algo.l1TilingCalc.biasMinLoadL1Size = 32;
    algo.l1TilingCalc.fixpMinLoadL1Size = 32;
    algo.CoreL1TilingDecision();
    EXPECT_EQ(algo.l1TilingFlag.iterateMNOrder, IterateMNOrder::ITER_N_FST);

}

TEST_F(TestConv2dTiling, test_algo_M_CoreL1TilingDecision_NoneKABL1FullLoadIter)
{
    PlatformInfo platformInfo;
    platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910B;
    platformInfo.l1Size = 524288;
    platformInfo.l0ASize = 65536;
    platformInfo.l0BSize = 65536;
    platformInfo.l0CSize = 262144;
    platformInfo.ubSize = 262144;
    platformInfo.btSize = 4096;
    platformInfo.fbSize = 4096;
    Conv2dTiling testTiling(platformInfo);
    SetDtype(testTiling, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32);
    ConvTilingAlgorithmMmode algo(&testTiling);
    algo.l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
    algo.l1TilingCalc.fmapMinLoadL1Size = 524256;
    algo.l1TilingCalc.weightMinLoadL1Size = 32;
    algo.l1TilingCalc.biasMinLoadL1Size = 32;
    algo.l1TilingCalc.fixpMinLoadL1Size = 32;
    algo.NoneKABL1FullLoadIter();
    EXPECT_EQ(algo.l1TilingFlag.iterateMNOrder, IterateMNOrder::ITER_M_FST);

}

TEST_F(TestConv2dTiling, test_group_conv2d_CalcOptGroupParams)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    ConvOriGroupInfo oriGroupInfo;
    ConvOptGroupInfo optGroupInfo;

    oriGroupInfo.groups = 0;
    testTiling.CalcOptGroupParams(oriGroupInfo, optGroupInfo);
    EXPECT_EQ(optGroupInfo.enlarge, 0);
    EXPECT_EQ(optGroupInfo.groupOpt, 0);
    EXPECT_EQ(optGroupInfo.cinOpt, 0);
    EXPECT_EQ(optGroupInfo.coutOpt, 0);

    oriGroupInfo.groups = 4;
    oriGroupInfo.ciPerGroup = 0;
    testTiling.CalcOptGroupParams(oriGroupInfo, optGroupInfo);
    EXPECT_EQ(optGroupInfo.enlarge, 0);
    EXPECT_EQ(optGroupInfo.groupOpt, 0);
    EXPECT_EQ(optGroupInfo.cinOpt, 0);
    EXPECT_EQ(optGroupInfo.coutOpt, 0);

    oriGroupInfo.ciPerGroup = 8;
    oriGroupInfo.coPerGroup = 0;
    testTiling.CalcOptGroupParams(oriGroupInfo, optGroupInfo);
    EXPECT_EQ(optGroupInfo.enlarge, 0);
    EXPECT_EQ(optGroupInfo.groupOpt, 0);
    EXPECT_EQ(optGroupInfo.cinOpt, 0);
    EXPECT_EQ(optGroupInfo.coutOpt, 0);

    oriGroupInfo.coPerGroup = 8;
    testTiling.CalcOptGroupParams(oriGroupInfo, optGroupInfo);
    EXPECT_EQ(optGroupInfo.enlarge, 2);
    EXPECT_EQ(optGroupInfo.groupOpt, 2);
    EXPECT_EQ(optGroupInfo.cinOpt, 16);
    EXPECT_EQ(optGroupInfo.coutOpt, 16);

    int64_t ret = Lcm(0, 0);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_group_conv2d_params_check)
{
    Conv2dTiling testTiling(SetPlatFormInfo());

    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    optiling::TConv2DTiling tilingData;

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo, 1);
    testTiling.SetOutputOrder(1);

    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);

    testTiling.SetGroups(0);
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);

    testTiling.SetGroups(4);
    testTiling.SetOptGroupParams(2, 0, 2);
    ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);

    testTiling.SetOptGroupParams(0, 2, 2);
    ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);

    testTiling.SetOptGroupParams(2, 2, 0);
    ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_group_conv2d_notsupport_c04)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 16;
    uint64_t orgWi = 16;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo * orgWo, 1);
    testTiling.SetOutputOrder(0);
    testTiling.SetC04Flag(true);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.SetGroups(2);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_group_conv2d_success)
{
    uint64_t orgCi = 16;
    uint64_t orgHi = 1;
    uint64_t orgWi = 1;
    uint64_t orgCo = 16;
    uint64_t orgKh = 1;
    uint64_t orgKw = 1;
    uint64_t orgHo = orgHi;
    uint64_t orgWo = orgWi;
    Conv2dTiling testTiling(SetPlatFormInfo());

    testTiling.SetOrgWeightShape(orgCo, orgKh, orgKw);
    testTiling.SetOrgFmapShape(orgCi, orgHi, orgWi);
    testTiling.SetSingleWeightShape(orgCi, orgKh, orgKw);
    testTiling.SetSingleOutputShape(orgCo, orgHo, orgWo, 1);
    testTiling.SetOutputOrder(0);
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.SetGroups(2);
    testTiling.SetOptGroupParams(2, 2, 2);

    optiling::TConv2DTiling tilingData;
    int64_t ret = testTiling.GetTiling(tilingData);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_algo_BB_AdjustM)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgHo = 896;
    testTiling.shapeInfo.orgWo = 256;
    testTiling.shapeInfo.orgHi = 1793;
    testTiling.shapeInfo.orgWi = 513;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.strideH = 2;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.fDim = 22;
    conv2DBasicBlockInfo.mTile = 512;
    conv2DBasicBlockInfo.mCut = 448;
    conv2DBasicBlockInfo.batch = 1;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.AdjustM();
    EXPECT_EQ(conv2DBasicBlockInfo.mTile, 512);
    EXPECT_EQ(conv2DBasicBlockInfo.mCut, 448);
    EXPECT_EQ(conv2DBasicBlockInfo.mIn, 4617);

    conv2DBasicBlockInfo.mCut = 22;
    testTiling.shapeInfo.orgHo = 1;
    testTiling.shapeInfo.orgWo = 1;
    algo.AdjustM();
    EXPECT_EQ(conv2DBasicBlockInfo.mTile, 16);
    EXPECT_EQ(conv2DBasicBlockInfo.mCut, 1);
    EXPECT_EQ(conv2DBasicBlockInfo.mIn, 18981);
}

TEST_F(TestConv2dTiling, test_algo_BB_AdjustN)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.nTile = 128;
    conv2DBasicBlockInfo.nCut = 1;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.AdjustN();
    EXPECT_EQ(conv2DBasicBlockInfo.nTile, 128);
    EXPECT_EQ(conv2DBasicBlockInfo.nCut, 1);
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcCoreUtilization)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.groupDim = 1;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.fActive = 16;
    algo.nActive = 1;
    algo.CalcCoreUtilization();
    EXPECT_EQ(conv2DBasicBlockInfo.coreUtilization, 0.5);
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckL1SpaceEnough)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.availableL1Size = 512*1024;
    uint64_t usedAL1Size = 212*1024;
    uint64_t usedBL1Size = 300*1024;
    bool ret = algo.CheckL1SpaceEnough(usedAL1Size, usedBL1Size);
    EXPECT_EQ(ret, true);
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckL1SpaceNotEnough)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.availableL1Size = 512*1024;
    uint64_t usedAL1Size = 212*1024 + 1;
    uint64_t usedBL1Size = 300*1024;
    bool ret = algo.CheckL1SpaceEnough(usedAL1Size, usedBL1Size);
    EXPECT_EQ(ret, false);
}


TEST_F(TestConv2dTiling, test_algo_BB_CalcMNFullLoadFlag)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.mCut = 16;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.CalcMNFullLoadFlag();
    EXPECT_EQ(conv2DBasicBlockInfo.fCut, 32);
    EXPECT_EQ(conv2DBasicBlockInfo.mAl1FullLoad, false);
    EXPECT_EQ(conv2DBasicBlockInfo.nBl1FullLoad, true);
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcAvalibleL1Size)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::INT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::INT32);
    testTiling.InferCubeInfo();

    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.batch = 2;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    testTiling.shapeInfo.singleCo = 8;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    testTiling.hasBias = false;
    testTiling.hasQuantScale = false;
    conv2DBasicBlockInfo.biasFullLoad = false;
    conv2DBasicBlockInfo.fixpFullLoad = false;
    conv2DBasicBlockInfo.nTile = 16;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, 1024);
    EXPECT_EQ(algo.weightL1FullLoadSize, 1152);
    EXPECT_EQ(algo.availableL1Size, 512*1024);

    testTiling.hasBias = true;
    testTiling.hasQuantScale = false;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, 1024);
    EXPECT_EQ(algo.weightL1FullLoadSize, 1152);
    EXPECT_EQ(algo.availableL1Size, 512*1024-64);

    testTiling.hasBias = false;
    testTiling.hasQuantScale = true;
    testTiling.shapeInfo.channelWiseCoeff = 4;
    algo.quantScaleDtypeSize = 8;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, 1024);
    EXPECT_EQ(algo.weightL1FullLoadSize, 1152);
    EXPECT_EQ(algo.availableL1Size, 512*1024-128);

    testTiling.hasBias = true;
    testTiling.hasQuantScale = true;
    testTiling.shapeInfo.channelWiseCoeff = 4;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, 1024);
    EXPECT_EQ(algo.weightL1FullLoadSize, 1152);
    EXPECT_EQ(algo.availableL1Size, 512*1024-64-128);
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcWeightCoeff_groupOpt)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.CalcWeightCoeff();
    EXPECT_EQ(algo.weightCoeff, 1);
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcWeightCoeff)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);

    testTiling.shapeInfo.orgkH = 1;
    testTiling.shapeInfo.orgkW = 1;
    algo.CalcWeightCoeff();
    EXPECT_EQ(algo.weightCoeff, 1);

    testTiling.shapeInfo.orgkH = 2;
    testTiling.shapeInfo.orgkW = 2;
    algo.CalcWeightCoeff();
    EXPECT_EQ(algo.weightCoeff, 4);

    testTiling.shapeInfo.orgkH = 4;
    testTiling.shapeInfo.orgkW = 4;
    algo.CalcWeightCoeff();
    EXPECT_EQ(algo.weightCoeff, 6);
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcL1LoadScore)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    conv2DBasicBlockInfo.batch = 2;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.weightCoeff = 4;
    algo.mRepeats = 2;
    algo.nRepeats = 1;
    algo.l1ScoreBase = 5;
    double l1LoadScoreTemp = algo.CalcL1LoadScore();
    EXPECT_EQ(l1LoadScoreTemp, 5+(1.0/(512/9*2+576*1)));
}

TEST_F(TestConv2dTiling, test_algo_BB_TryKABFullLoadL1Stratgy)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKABFullLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKABFullLoadL1Stratgy(); // update KAndMAl1FullLoad

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKABFullLoadL1Stratgy(); // update KAndNoneFullLoad

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKABFullLoadL1Stratgy(); // update KAndNoneFullLoad

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_TryKABFullLoadL1Stratgy_MFst)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->weightL1FullLoadSize = 100*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKABFullLoadL1Stratgy(); // update KAndMAl1FullLoad MFst

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_TryNFstLoadL1Stratgy)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    algo->weightL1FullLoadSize = 100*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad MFst

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.mIn = 1666;
    conv2DBasicBlockInfo.nTile = 12345;
    algo->weightL1FullLoadSize = 100*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.mIn = 1666;
    conv2DBasicBlockInfo.nTile = 1128;
    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad MFst
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_TryMFstLoadL1Stratgy)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->TryMFstLoadL1Stratgy(); // WeightFullLoad

    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->TryMFstLoadL1Stratgy(); // WeightKFullLoad

    conv2DBasicBlockInfo.nBl1FullLoad = true;
    conv2DBasicBlockInfo.nTile = 999999;
    algo->TryMFstLoadL1Stratgy(); // WeightKFullLoad

    conv2DBasicBlockInfo.nBl1FullLoad = true;
    conv2DBasicBlockInfo.nTile = 128;
    conv2DBasicBlockInfo.mIn = 999999;
    algo->TryMFstLoadL1Stratgy(); // mIn > mInMax

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_TryKAllSplitLoadL1Stratgy_M_FST)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    algo->weightL1FullLoadSize = 100*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKAllSplitLoadL1Stratgy(); // ITER_M_FST

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_TryKAllSplitLoadL1Stratgy_N_FST)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    algo->weightL1FullLoadSize = 300*1024;
    algo->fmapL1FullLoadSize   = 200*1024;
    algo->TryKAllSplitLoadL1Stratgy(); // ITER_N_FST

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcBestL1LoadStratgy)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 3;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false; // no opt
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;

    bool ret = false;
    algo->CalcBestL1LoadStratgy();

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_CalcBestL1LoadStratgy_groupOPt)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;

    bool ret = false;
    algo->CalcBestL1LoadStratgy();

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_KAndMAl1FullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();

    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;

    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 16;
    algo->availableL1Size = 512*1024;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::K_AND_MAL1_FULL_LOAD;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_KAndNBl1FullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::K_AND_NBL1_FULL_LOAD;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_KAndNoneFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::K_AND_NONE_FULL_LOAD;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_FmapFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::FMAP_FULL_LOAD;
    algo->availableL1Size = 512*1024;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_FmapKFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::FMAP_K_FULL_LOAD;
    algo->availableL1Size = 512*1024;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_NFirstKSplit)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::N_FIRST_K_SPLIT;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_WeightFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::WEIGHT_FULL_LOAD;
    algo->availableL1Size = 512*1024;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_WeightKFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::WEIGHT_K_FULL_LOAD;
    algo->availableL1Size = 512*1024;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_MFirstKSplit)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::M_FIRST_K_SPLIT;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_KAllSplit)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::K_ALL_SPLIT;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, true);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_KAndMAl1FullLoad_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();

    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;

    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::K_AND_MAL1_FULL_LOAD;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    conv2DBasicBlockInfo.mIn = 1444;
    conv2DBasicBlockInfo.nTile = 256;

    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_WeightFullLoad_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::WEIGHT_FULL_LOAD;
    algo->availableL1Size = 0;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_WeightKFullLoad_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::WEIGHT_K_FULL_LOAD;
    algo->availableL1Size = 0;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_FmapFullLoad_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::FMAP_FULL_LOAD;
    algo->availableL1Size = 0;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_UpdateL1LoadStrategy_FmapKFullLoad_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->singleCi1xC0 = 128;

    bool ret = false;
    algo->l1LoadStrategyType = L1LoadStrategyType::FMAP_K_FULL_LOAD;
    algo->availableL1Size = 0;
    ret = algo->UpdateL1LoadStrategy(algo);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_InitPingPong)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.InitPingPong();
    EXPECT_EQ(algo.dbValue.pbAL1, 2);
    EXPECT_EQ(algo.dbValue.pbBL1, 2);
    EXPECT_EQ(algo.dbValue.pbAL0, 2);
    EXPECT_EQ(algo.dbValue.pbBL0, 2);
    EXPECT_EQ(algo.dbValue.pbCL0, 1);
}

TEST_F(TestConv2dTiling, test_algo_BB_SetPBufferRes)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode algo(&testTiling, conv2DBasicBlockInfo);
    algo.InitPingPong();
    EXPECT_EQ(algo.dbValue.pbAL1, 2);
    EXPECT_EQ(algo.dbValue.pbBL1, 2);
    EXPECT_EQ(algo.dbValue.pbAL0, 2);
    EXPECT_EQ(algo.dbValue.pbBL0, 2);
    EXPECT_EQ(algo.dbValue.pbCL0, 1);
    algo.SetPBufferRes();
    EXPECT_EQ(testTiling.dbValue.pbAL1, 2);
    EXPECT_EQ(testTiling.dbValue.pbBL1, 2);
    EXPECT_EQ(testTiling.dbValue.pbAL0, 2);
    EXPECT_EQ(testTiling.dbValue.pbBL0, 2);
    EXPECT_EQ(testTiling.dbValue.pbCL0, 1);
    EXPECT_EQ(testTiling.dbValue.pBufferFlag, 27);
}

TEST_F(TestConv2dTiling, test_algo_BB_GetL1Tiling_KABFullLoad)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;
    algo->singleCi1xC0 = 16;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.kAl1FullLoad = true;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    conv2DBasicBlockInfo.iterateMNOrder == conv_tiling::IterateMNOrder::ITER_M_FST;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.nCut = 16;
    conv2DBasicBlockInfo.nDim = 1;
    testTiling.shapeInfo.orgCo = 4096;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.nCut = 4;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.batch= 1;
    testTiling.shapeInfo.orgCo = 1024;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.nCut = 16;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.batch= 1;
    testTiling.shapeInfo.orgCo = 4096;
    algo->GetL1Tiling(algo);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetL1Tiling_GetKABFullLoadL1TilingParams)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;
    algo->singleCi1xC0 = 16;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;
    conv2DBasicBlockInfo.kAl1FullLoad = true;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetKABFullLoadL1TilingParams();

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetKABFullLoadL1TilingParams();

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetKABFullLoadL1TilingParams();

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetKABFullLoadL1TilingParams();
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetL1Tiling_ITER_N_FST)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;
    algo->singleCi1xC0 = 128;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_N_FST;

    conv2DBasicBlockInfo.kAl1FullLoad = true;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = true;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetL1Tiling(algo);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetL1Tiling_ITER_M_FST)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetL1Tiling(algo);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_Process_success)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 256;
    testTiling.shapeInfo.singleCi = 256;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, 0);

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_Process_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 16;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, -1);

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_PostK_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 16;
    testTiling.shapeInfo.singleCi = 16;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = 65536;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, -1);

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmType)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    bool ret = false;
    testTiling.shapeInfo.orgCo = 128;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgHi = 256;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgWi = 256;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkH = 3;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgkW = 3;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.enlarge = 2;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgHo = 16;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.orgWo = 16;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleCi = 32;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.dilationH = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.dilationW = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padLeft = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padRight = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padTop = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.padBottom = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.strideH = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.strideW = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.attrInfo.groups = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.optGroupFlag = false;
    testTiling.shapeInfo.singlekH = 3;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singlekW = 3;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.fDim = 8;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mDim = 4;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.nDim = 2;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.groupDim = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.aicoreNum = 32;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mIn = 3*1024;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mTile = 256;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.nTile = 256;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mCut = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.nCut = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.batch = 4;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);

    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);

    testTiling.optGroupFlag = true;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);

    testTiling.optGroupFlag = false;
    conv2DBasicBlockInfo.batchDim = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);

    testTiling.optGroupFlag = false;
    conv2DBasicBlockInfo.batchDim = 1;
    testTiling.shapeInfo.singleM = 256*256/4;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);
    
    testTiling.shapeInfo.singleCo = 64;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, true);
    
    conv2DBasicBlockInfo.batchDim = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.mDim = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mDim = 1;
    conv2DBasicBlockInfo.nDim = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.mTile = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mIn = 0;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.mIn = 1024;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::INVALID;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 2);
    EXPECT_EQ(ret, true);

    testTiling.optGroupFlag = true;
    testTiling.shapeInfo.singleGroups = 1;
    testTiling.attrInfo.groups = 2;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);

    testTiling.SetQuantScale(true);
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmType_Quant)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::INT8);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::INT32);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;


    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    testTiling.optGroupFlag = false;
    conv2DBasicBlockInfo.batchDim = 1;
    testTiling.SetQuantScale(true);
    bool ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);


    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetTiling)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;

    bool ret = false;
    optiling::TConv2DTiling tilingData;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleM = 256*256/4;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, false);

    testTiling.shapeInfo.singleCo = 64;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, true);

    conv2DBasicBlockInfo.nBl1FullLoad = true;
    testTiling.shapeInfo.singleM = 256;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, true);

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    testTiling.shapeInfo.singleCo = 256;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_N_FST;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, true);

    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    testTiling.shapeInfo.singleCo = 64;
    conv2DBasicBlockInfo.batch = 34;
    conv2DBasicBlockInfo.mIn = 512*1024;
    testTiling.shapeInfo.orgHi = 25;
    testTiling.shapeInfo.orgWi = 25;
    testTiling.shapeInfo.orgHo = 1;
    testTiling.shapeInfo.orgWo = 1;
    testTiling.shapeInfo.orgkH = 25;
    testTiling.shapeInfo.orgkW = 25;
    testTiling.shapeInfo.singlekH = 25;
    testTiling.shapeInfo.singlekW = 25;
    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, false);

    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_N_FST;
    conv2DBasicBlockInfo.batch = 34;
    conv2DBasicBlockInfo.mIn = 512*1024;
    testTiling.shapeInfo.orgHi = 25;
    testTiling.shapeInfo.orgWi = 25;
    testTiling.shapeInfo.orgHo = 1;
    testTiling.shapeInfo.orgWo = 1;
    testTiling.shapeInfo.orgkH = 25;
    testTiling.shapeInfo.orgkW = 25;
    testTiling.shapeInfo.singlekH = 25;
    testTiling.shapeInfo.singlekW = 25;
    testTiling.attrInfo.padLeft = 0;
    testTiling.attrInfo.padRight = 0;
    testTiling.attrInfo.padTop = 0;
    testTiling.attrInfo.padBottom = 0;
    conv2DBasicBlockInfo.nTile = 512;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.kAl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    ret = testTiling.GetTiling(conv2DBasicBlockInfo, tilingData);
    EXPECT_EQ(ret, false);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetCoreBindingDecisionFactor)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.SetWeightType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetFmapType(TPosition::GM, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetOutputType(TPosition::CO1, ConvFormat::NCHW, ConvDtype::FLOAT16);
    testTiling.SetBiasType(TPosition::GM, ConvFormat::ND, ConvDtype::FLOAT16);
    testTiling.InferCubeInfo();
    testTiling.shapeInfo.orgCo = 128;
    testTiling.shapeInfo.orgHi = 256;
    testTiling.shapeInfo.orgWi = 256;
    testTiling.shapeInfo.orgCi = 32;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.orgkH = 3;
    testTiling.shapeInfo.orgkW = 3;
    testTiling.shapeInfo.enlarge = 2;
    testTiling.shapeInfo.orgHo = 16;
    testTiling.shapeInfo.orgWo = 16;
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
    testTiling.attrInfo.padRight = 1;
    testTiling.attrInfo.padTop = 1;
    testTiling.attrInfo.padBottom = 1;
    testTiling.attrInfo.strideH = 1;
    testTiling.attrInfo.strideW = 1;
    testTiling.attrInfo.groups = 1;
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = 512*1024;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = 32;
    conv2DBasicBlockInfo.mIn = 3*1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;

    optiling::TConv2DTiling tilingData;
    bool ret = testTiling.GetCoreBindingDecisionFactor(conv2DBasicBlockInfo);
    EXPECT_EQ(ret, true);
    
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_hw_updatebiasfixparamsl1fullload)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.shapeInfo.singleCo1 = 2100;
    testTiling.cubeInfo.n0 = 16;
    testTiling.platformInfo.l1Size = 524288;
    testTiling.descInfo.fMapType.dtype = ConvDtype::FLOAT16;
    testTiling.descInfo.weightType.dtype = ConvDtype::FLOAT16;
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.l1Params.kAL1 = 1;
    algo.l1Params.kBL1 = 1;
    algo.l1Params.hoAL1 = 1;
    algo.l1Params.woAL1 = 1;
    algo.l1Params.nBL1 = 16;
    algo.biasDTypeSize = 2;
    algo.l1Flags.isBiasFullLoad = true;
    algo.l1Flags.isFixpParamsFullLoad = true;
    algo.UpdateBiasFixpParamsL1Fullload();
    EXPECT_EQ(algo.l1Flags.isBiasFullLoad, false);
    EXPECT_EQ(algo.l1Flags.isFixpParamsFullLoad, false);
}


TEST_F(TestConv2dTiling, test_util_isequal)
{
    std::vector<ConvDtype> arr1 = {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16};
    const std::vector<ConvDtype> arr2 = {ConvDtype::FLOAT16, ConvDtype::FLOAT16};
    uint32_t size = 1;
    auto ret = IsEqual(arr1, arr2, size);
    EXPECT_EQ(ret, false);
}

TEST_F(TestConv2dTiling, test_util_alignB)
{
    uint64_t a = 1;
    uint64_t b = 0;
    auto ret = AlignB(a, b);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_util_CeilDiv)
{
    uint64_t a = 1;
    uint64_t b = 0;
    auto ret = CeilDiv(a, b);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_util_Gcd)
{
    uint64_t a = 1;
    uint64_t b = 0;
    auto ret = Gcd(a, b);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateHoL0WoL0_1)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t valueMax = 1;
    algo.l0TilingRange.woL0Range = {1, 2};
    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range, valueMax,
                        algo.l0Params.hoL0);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateHoL0WoL0_2)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t valueMax = 6;
    algo.l0TilingRange.woL0Range = {1, 2, 4};
    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 1;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 4;
    algo.dbValue.pbAL0 = 1;
    algo.dbValue.pbBL0 = 1;
    algo.dbValue.pbCL0 = 1;
    testTiling.cubeInfo.k0 = 16;
    testTiling.cubeInfo.n0 = 16;
    testTiling.cubeInfo.m0 = 16;
    testTiling.cubeInfo.madType = ConvDtype::FLOAT16;
    algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range, valueMax,
                        algo.l0Params.hoL0);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateHoL0WoL0_3)
{
    PlatformInfo platformInfo;
    platformInfo.socVersion = platform_ascendc::SocVersion::ASCEND910_95;
    platformInfo.l1Size = 524288;
    platformInfo.l0ASize = 1;
    platformInfo.l0BSize = 65536;
    platformInfo.l0CSize = 1;
    platformInfo.ubSize = 262144;
    platformInfo.btSize = 4096;
    platformInfo.fbSize = 4096;
    Conv2dTiling testTiling(platformInfo);
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t valueMax = 6;
    algo.l0TilingRange.woL0Range = {1, 2, 4};
    algo.l0Params.woL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.l0Params.hoL0 = 2;
    algo.l0Params.kL0 = 1;
    algo.l0Params.nL0 = 4;
    algo.dbValue.pbAL0 = 1;
    algo.dbValue.pbBL0 = 1;
    algo.dbValue.pbCL0 = 1;
    testTiling.cubeInfo.k0 = 16;
    testTiling.cubeInfo.n0 = 16;
    testTiling.cubeInfo.m0 = 16;
    testTiling.cubeInfo.madType = ConvDtype::FLOAT16;
    algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range, valueMax,
                        algo.l0Params.hoL0);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateNL0)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    testTiling.InferCubeInfo();
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    uint64_t valueMax = 1;
    algo.l0TilingRange.nL0Range = {1, 2};
    algo.l0Params.nL0Index = 0;
    algo.l0Params.woL0 = 1;
    algo.UpdateNL0();
}

TEST_F(TestConv2dTiling, test_CheckL0DoubleBuffer)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    algo.InitPingPong();
    testTiling.InferCubeInfo();
    algo.l0Params.woL0Index = 0;
    algo.l0Params.hoL0Index = 0;
    testTiling.l0TilingInfo.hoL0 = 15;
    testTiling.l0TilingInfo.woL0 = 12;
    testTiling.l0TilingInfo.kL0 = 10;
    testTiling.l0TilingInfo.nL0 = 13;
    testTiling.l1TilingInfo.hoAL1 = 15;
    testTiling.l1TilingInfo.woAL1 = 12;
    algo.l0TilingRange.woL0Range = {13,13,13};
    algo.l0TilingRange.hoL0Range = {16,16,16};
    testTiling.l1TilingInfo.kAL1 = 10;
    testTiling.l1TilingInfo.kBL1 = 10;
    testTiling.l1TilingInfo.nBL1 = 13;
    algo.dbValue.pbAL0 = 2;
    algo.dbValue.pbBL0 = 2;
    algo.dbValue.pbCL0 = 1;
    testTiling.cubeInfo.k0 = 16;
    testTiling.cubeInfo.n0 = 16;
    testTiling.cubeInfo.m0 = 16;
    testTiling.cubeInfo.madType = ConvDtype::FLOAT16;
    algo.l1TilingCalc.kBL1MaxSize = testTiling.l0TilingInfo.kL0;
    algo.l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    algo.CheckL0DoubleBuffer();
}