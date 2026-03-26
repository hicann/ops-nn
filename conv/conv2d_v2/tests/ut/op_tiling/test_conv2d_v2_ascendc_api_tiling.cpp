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
 * \file test_conv2d_v2_ascendc_api_tiling.cpp
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
using namespace conv_tiling_algo_bb;
using namespace conv_tiling_utils;

namespace {
PlatformInfo SetPlatFormInfo()
{
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
    return platformInfo;
}

void SetDtype(ConvTilingBase &testTiling,
              ConvDtype fmapDtype,
              ConvDtype weightDtype,
              ConvDtype biasDtype = ConvDtype::FLOAT16,
              ConvDtype scaleType = ConvDtype::INT64)
{
    testTiling.descInfo.fMapType.dtype = fmapDtype;
    testTiling.descInfo.weightType.dtype = weightDtype;
    testTiling.descInfo.biasType.dtype = biasDtype;
    testTiling.descInfo.scaleType.dtype = scaleType;
}

void SetTypeNCHWINT8(ConvTilingBase &testTiling)
{
    testTiling.descInfo.weightType = {ConvFormat::NCHW, ConvDtype::INT8, TPosition::GM};
    testTiling.descInfo.fMapType = {ConvFormat::NCHW, ConvDtype::INT8, TPosition::GM};
    testTiling.descInfo.outputType = {ConvFormat::NCHW, ConvDtype::INT32, TPosition::CO1};
    testTiling.descInfo.biasType = {ConvFormat::ND, ConvDtype::INT32, TPosition::GM};
}

void SetTypeNCHWFP16(ConvTilingBase &testTiling)
{
    testTiling.descInfo.weightType = {ConvFormat::NCHW, ConvDtype::FLOAT16, TPosition::GM};
    testTiling.descInfo.fMapType = {ConvFormat::NCHW, ConvDtype::FLOAT16, TPosition::GM};
    testTiling.descInfo.outputType = {ConvFormat::NCHW, ConvDtype::FLOAT16, TPosition::CO1};
    testTiling.descInfo.biasType = {ConvFormat::ND, ConvDtype::FLOAT16, TPosition::GM};
}
struct ShapeParams {
    int64_t orgCo;
    int64_t orgHi;
    int64_t orgWi;
    int64_t orgCi;
    int64_t singleCi;
    int64_t orgkH;
    int64_t orgkW;
    int64_t enlarge;
    int64_t orgHo;
    int64_t orgWo;
};

void SetShape(ConvTilingBase &testTiling, const ShapeParams &shapeParams)
{
    testTiling.shapeInfo.orgCo = shapeParams.orgCo;
    testTiling.shapeInfo.orgHi = shapeParams.orgHi;
    testTiling.shapeInfo.orgWi = shapeParams.orgWi;
    testTiling.shapeInfo.orgCi = shapeParams.orgCi;
    testTiling.shapeInfo.singleCi = shapeParams.singleCi;
    testTiling.shapeInfo.orgkH = shapeParams.orgkH;
    testTiling.shapeInfo.orgkW = shapeParams.orgkW;
    testTiling.shapeInfo.enlarge = shapeParams.enlarge;
    testTiling.shapeInfo.orgHo = shapeParams.orgHo;
    testTiling.shapeInfo.orgWo = shapeParams.orgWo;
}

struct AttrParams {
    int32_t dilationH;
    int32_t dilationW;
    int32_t padLeft;
    int32_t padRight;
    int32_t padTop;
    int32_t padBottom;
    int32_t strideH;
    int32_t strideW;
    int32_t groups;
};

void SetAttrs(ConvTilingBase &testTiling, const AttrParams &attrParams)
{
    testTiling.attrInfo.dilationH = attrParams.dilationH;
    testTiling.attrInfo.dilationW = attrParams.dilationW;
    testTiling.attrInfo.padLeft = attrParams.padLeft;
    testTiling.attrInfo.padRight = attrParams.padRight;
    testTiling.attrInfo.padTop = attrParams.padTop;
    testTiling.attrInfo.padBottom = attrParams.padBottom;
    testTiling.attrInfo.strideH = attrParams.strideH;
    testTiling.attrInfo.strideW = attrParams.strideW;
    testTiling.attrInfo.groups = attrParams.groups;
}
} // namespace

class TestConv2dTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestConv2dTiling, test_algo_HW_InitPingPong)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetDtype(testTiling, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    ConvTilingAlgorithmHWmode algo(&testTiling);
    testTiling.InferCubeInfo();
    testTiling.cubeInfo.k0 = CUBE_K0_32;
    testTiling.cubeInfo.n0 = CUBE_N0;
    testTiling.cubeInfo.m0 = CUBE_M0;
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = 1;
    testTiling.hasBias = true;
    testTiling.isC04Flag = false;
    testTiling.shapeInfo.orgHi = 16;
    testTiling.shapeInfo.orgWi = 16;
    algo.InitPingPong();
}

TEST_F(TestConv2dTiling, test_algo_M_CoreL1TilingDecision_FULL_LOAD_AL1)
{
    PlatformInfo platformInfo;
    platformInfo.npuArch = NpuArch::DAV_3510;
    platformInfo.l1Size = MEM_SIZE_512K;
    platformInfo.l0ASize = MEM_SIZE_64K;
    platformInfo.l0BSize = MEM_SIZE_64K;
    platformInfo.l0CSize = MEM_SIZE_256K;
    platformInfo.ubSize = MEM_SIZE_256K;
    platformInfo.btSize = MEM_SIZE_4K;
    platformInfo.fbSize = MEM_SIZE_4K;
    Conv2dTiling testTiling(platformInfo);
    SetDtype(testTiling, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32);
    ConvTilingAlgorithmMmode algo(&testTiling);
    algo.l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
    algo.l1TilingCalc.fmapFullLoadL1Size = MEM_SIZE_512K - CUBE_C0_SIZE;
    algo.l1TilingCalc.weightMinLoadL1Size = CUBE_C0_SIZE;
    algo.l1TilingCalc.biasMinLoadL1Size = CUBE_C0_SIZE;
    algo.l1TilingCalc.fixpMinLoadL1Size = CUBE_C0_SIZE;
    algo.CoreL1TilingDecision();
    EXPECT_EQ(algo.l1TilingFlag.iterateMNOrder, IterateMNOrder::ITER_N_FST);
}

TEST_F(TestConv2dTiling, test_algo_M_CoreL1TilingDecision_NoneKABL1FullLoadIter)
{
    PlatformInfo platformInfo;
    platformInfo.npuArch = NpuArch::DAV_3510;
    platformInfo.l1Size = MEM_SIZE_512K;
    platformInfo.l0ASize = MEM_SIZE_64K;
    platformInfo.l0BSize = MEM_SIZE_64K;
    platformInfo.l0CSize = MEM_SIZE_256K;
    platformInfo.ubSize = MEM_SIZE_256K;
    platformInfo.btSize = MEM_SIZE_4K;
    platformInfo.fbSize = MEM_SIZE_4K;
    Conv2dTiling testTiling(platformInfo);
    SetDtype(testTiling, ConvDtype::INT8, ConvDtype::INT8, ConvDtype::INT32);
    ConvTilingAlgorithmMmode algo(&testTiling);
    algo.l1TilingFlag.abL1Mode = L1TilingMode::FULL_LOAD_AL1;
    algo.l1TilingCalc.fmapMinLoadL1Size = MEM_SIZE_512K - CUBE_C0_SIZE;
    algo.l1TilingCalc.weightMinLoadL1Size = CUBE_C0_SIZE;
    algo.l1TilingCalc.biasMinLoadL1Size = CUBE_C0_SIZE;
    algo.l1TilingCalc.fixpMinLoadL1Size = CUBE_C0_SIZE;
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
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    algo.availableL1Size = MEM_SIZE_512K;
    uint64_t usedAL1Size = MEM_SIZE_212K;
    uint64_t usedBL1Size = MEM_SIZE_300K;
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
    algo.availableL1Size = MEM_SIZE_512K;
    uint64_t usedAL1Size = MEM_SIZE_212K + 1;
    uint64_t usedBL1Size = MEM_SIZE_300K;
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
    SetTypeNCHWINT8(testTiling);
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
    testTiling.hasScale = false;
    conv2DBasicBlockInfo.biasFullLoad = false;
    conv2DBasicBlockInfo.fixpFullLoad = false;
    conv2DBasicBlockInfo.nTile = 16;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, MEM_SIZE_1K);
    EXPECT_EQ(algo.weightL1FullLoadSize, MEM_SIZE_1K + MEM_SIZE_128B);
    EXPECT_EQ(algo.availableL1Size, MEM_SIZE_512K);

    testTiling.hasBias = true;
    testTiling.hasScale = false;
    algo.biasDTypeSize = 4;  // INT32 dtype size
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, MEM_SIZE_1K);
    EXPECT_EQ(algo.weightL1FullLoadSize, MEM_SIZE_1K + MEM_SIZE_128B);
    EXPECT_EQ(algo.availableL1Size, MEM_SIZE_512K - MEM_SIZE_64B);

    testTiling.hasBias = false;
    testTiling.hasScale = true;
    testTiling.shapeInfo.channelWiseCoeff = 4;
    algo.scaleDtypeSize = 8;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, MEM_SIZE_1K);
    EXPECT_EQ(algo.weightL1FullLoadSize, MEM_SIZE_1K + MEM_SIZE_128B);
    EXPECT_EQ(algo.availableL1Size, MEM_SIZE_512K - MEM_SIZE_128B);

    testTiling.hasBias = true;
    testTiling.hasScale = true;
    testTiling.shapeInfo.channelWiseCoeff = 4;
    algo.biasDTypeSize = 4;
    algo.CalcAvalibleL1Size();
    EXPECT_EQ(algo.fmapL1FullLoadSize, MEM_SIZE_1K);
    EXPECT_EQ(algo.weightL1FullLoadSize, MEM_SIZE_1K + MEM_SIZE_128B);
    EXPECT_EQ(algo.availableL1Size, MEM_SIZE_512K - MEM_SIZE_64B - MEM_SIZE_128B);
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
    algo->availableL1Size = MEM_SIZE_512K;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryKABFullLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryKABFullLoadL1Stratgy(); // update KAndMAl1FullLoad

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryKABFullLoadL1Stratgy(); // update KAndNoneFullLoad

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
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
    algo->availableL1Size = MEM_SIZE_512K;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->weightL1FullLoadSize = MEM_SIZE_100K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
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
    algo->availableL1Size = MEM_SIZE_512K;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    algo->weightL1FullLoadSize = MEM_SIZE_100K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad MFst

    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.mIn = 1666;
    conv2DBasicBlockInfo.nTile = 12345;
    algo->weightL1FullLoadSize = MEM_SIZE_100K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
    algo->TryNFstLoadL1Stratgy(); // update KAndMAl1FullLoad NFst 

    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.mIn = 1666;
    conv2DBasicBlockInfo.nTile = 1128;
    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    algo->availableL1Size = MEM_SIZE_512K;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    algo->weightL1FullLoadSize = MEM_SIZE_100K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
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
    algo->availableL1Size = MEM_SIZE_512K;
    algo->preSetFullLoadFlag.kAl1FullLoad = true;
    algo->preSetFullLoadFlag.kBl1FullLoad = true;

    bool ret = false;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.nTile = 128;

    algo->weightL1FullLoadSize = MEM_SIZE_300K;
    algo->fmapL1FullLoadSize   = MEM_SIZE_200K;
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
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    algo->availableL1Size = MEM_SIZE_512K;
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
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {16, 16, 16, 16, 16, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    algo->singleCi1xC0 = 16;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    testTiling.shapeInfo.orgCo = MEM_SIZE_4K;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.nCut = 4;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.batch= 1;
    testTiling.shapeInfo.orgCo = 1024;
    algo->GetL1Tiling(algo);

    conv2DBasicBlockInfo.nCut = 16;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.batch= 1;
    testTiling.shapeInfo.orgCo = MEM_SIZE_4K;
    algo->GetL1Tiling(algo);
    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetL1Tiling_GetKABFullLoadL1TilingParams)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {16, 16, 16, 16, 16, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    algo->singleCi1xC0 = 16;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {16, 16, 16, 32, 32, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});

    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    algo->singleCi1xC0 = 128;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
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
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
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
    algo->GetL1Tiling(algo.get());

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = true;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetL1Tiling(algo.get());

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = true;
    algo->GetL1Tiling(algo.get());

    conv2DBasicBlockInfo.kAl1FullLoad = false;
    conv2DBasicBlockInfo.kBl1FullLoad = false;
    conv2DBasicBlockInfo.mAl1FullLoad = false;
    conv2DBasicBlockInfo.nBl1FullLoad = false;
    algo->GetL1Tiling(algo.get());
}

TEST_F(TestConv2dTiling, test_algo_BB_Process_success)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 256, 256, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConv2dTiling, test_algo_BB_Process_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {16, 16, 16, 16, 16, 3, 3, 2, 16, 16});
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = true;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 16;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 2;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_algo_BB_PostK_fail)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 16, 16, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singlekH = 1;
    testTiling.shapeInfo.singlekW = MEM_SIZE_64K;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;

    int64_t ret = 0;
    ret = algo->Process();
    EXPECT_EQ(ret, -1);
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmTypePartOne)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmTypePartTwo)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    bool ret = false;
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    testTiling.attrInfo.dilationH = 1;
    testTiling.attrInfo.dilationW = 1;
    testTiling.attrInfo.padLeft = 1;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmTypePartThree)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    bool ret = false;
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, false);
    conv2DBasicBlockInfo.mIn = 3 * 1024;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmTypePartFour)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    bool ret = false;
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    testTiling.optGroupFlag = false;
    conv2DBasicBlockInfo.batchDim = 1;
    ret = testTiling.CheckTilingAlgorithmType(conv2DBasicBlockInfo, 1);
    EXPECT_EQ(ret, true);
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmTypeFive)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    bool ret = false;
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    testTiling.shapeInfo.singleM = 256*256/4;
    testTiling.shapeInfo.singleCo = 64;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.mDim = 1;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmType)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;
    bool ret = false;
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;
    conv2DBasicBlockInfo.batch = 4;
    testTiling.shapeInfo.singleM = 256*256/4;
    testTiling.shapeInfo.singleCo = 64;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.mDim = 1;
    conv2DBasicBlockInfo.nDim = 1;
    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_CheckTilingAlgorithmType_Quant)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWINT8(testTiling);
    testTiling.InferCubeInfo();
    
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

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
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
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
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    ConvTilingAlgorithmBBmode* algo = new ConvTilingAlgorithmBBmode(&testTiling,
        conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
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

    delete algo;
    algo = nullptr;
}

TEST_F(TestConv2dTiling, test_algo_BB_GetTilingPartTwo)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;

    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;
    conv2DBasicBlockInfo.nCut = 1;

    bool ret = false;
    optiling::TConv2DTiling tilingData;

    conv2DBasicBlockInfo.nBl1FullLoad = true;
    testTiling.shapeInfo.singleM = 256;
    conv2DBasicBlockInfo.mAl1FullLoad = true;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_M_FST;
    testTiling.shapeInfo.singleCo = 64;
    conv2DBasicBlockInfo.batch = 34;
    conv2DBasicBlockInfo.mIn = MEM_SIZE_512K;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_GetTilingPartThree)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
    testTiling.shapeInfo.singleCi = 32;
    testTiling.shapeInfo.singlekH = 3;
    testTiling.shapeInfo.singlekW = 3;
    SetAttrs(testTiling, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    testTiling.optGroupFlag = false;
    Conv2DBasicBlockInfo conv2DBasicBlockInfo;
    std::unique_ptr<ConvTilingAlgorithmBBmode> algo = std::make_unique<ConvTilingAlgorithmBBmode>(&testTiling, conv2DBasicBlockInfo);
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;

    conv2DBasicBlockInfo.mTile = 256;
    conv2DBasicBlockInfo.nTile = 256;
    conv2DBasicBlockInfo.mCut = 1;

    bool ret = false;
    optiling::TConv2DTiling tilingData;
    testTiling.shapeInfo.singleM = 256;
    testTiling.shapeInfo.singleCo = 64;
    conv2DBasicBlockInfo.iterateMNOrder = conv_tiling::IterateMNOrder::ITER_N_FST;
    conv2DBasicBlockInfo.batch = 34;
    conv2DBasicBlockInfo.mIn = MEM_SIZE_512K;
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
}

TEST_F(TestConv2dTiling, test_algo_BB_GetCoreBindingDecisionFactor)
{
    Conv2dTiling testTiling(SetPlatFormInfo());
    SetTypeNCHWFP16(testTiling);
    testTiling.InferCubeInfo();
    SetShape(testTiling, {128, 256, 256, 32, 32, 3, 3, 2, 16, 16});
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
    algo->availableL1Size = MEM_SIZE_512K;

    conv2DBasicBlockInfo.fDim = 8;
    conv2DBasicBlockInfo.mDim = 4;
    conv2DBasicBlockInfo.nDim = 2;
    conv2DBasicBlockInfo.batchDim = 1;
    conv2DBasicBlockInfo.groupDim = 1;
    conv2DBasicBlockInfo.aicoreNum = AIC_NUM;
    conv2DBasicBlockInfo.mIn = 3 * 1024;
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
    testTiling.cubeInfo.n0 = CUBE_N0;
    testTiling.platformInfo.l1Size = MEM_SIZE_512K;
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
    testTiling.cubeInfo.k0 = CUBE_K0_16;
    testTiling.cubeInfo.n0 = CUBE_N0;
    testTiling.cubeInfo.m0 = CUBE_M0;
    testTiling.cubeInfo.madType = ConvDtype::FLOAT16;
    algo.UpdateHoL0WoL0(algo.l0Params.woL0, algo.l0Params.woL0Index, algo.l0TilingRange.woL0Range, valueMax,
                        algo.l0Params.hoL0);
}

TEST_F(TestConv2dTiling, test_algo_HW_UpdateHoL0WoL0_3)
{
    PlatformInfo platformInfo;
    platformInfo.npuArch = NpuArch::DAV_3510;
    platformInfo.l1Size = MEM_SIZE_512K;
    platformInfo.l0ASize = 1;
    platformInfo.l0BSize = MEM_SIZE_64K;
    platformInfo.l0CSize = 1;
    platformInfo.ubSize = MEM_SIZE_256K;
    platformInfo.btSize = MEM_SIZE_4K;
    platformInfo.fbSize = MEM_SIZE_4K;
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
    testTiling.cubeInfo.k0 = CUBE_K0_16;
    testTiling.cubeInfo.n0 = CUBE_N0;
    testTiling.cubeInfo.m0 = CUBE_M0;
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
    testTiling.cubeInfo.k0 = CUBE_K0_16;
    testTiling.cubeInfo.n0 = CUBE_N0;
    testTiling.cubeInfo.m0 = CUBE_M0;
    testTiling.cubeInfo.madType = ConvDtype::FLOAT16;
    algo.l1TilingCalc.kBL1MaxSize = testTiling.l0TilingInfo.kL0;
    algo.l1Flags.iterateMNOrder = IterateMNOrder::ITER_N_FST;
    algo.CheckL0DoubleBuffer();
}