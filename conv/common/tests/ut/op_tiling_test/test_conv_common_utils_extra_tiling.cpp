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
 * \file test_conv_common_utils_extra_tiling.cpp
 * \brief UT for conv common tiling utilities: math util, platform util, debug util,
 *        num blocks decision, tiling cache and tbe tiling api.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "kernel_run_context_facker.h"
#include "conv/common/op_host/op_tiling/conv_math_util.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "conv/common/op_host/op_tiling/convbp_tiling_debug_util.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_numblocks_decision.h"
#include "conv/common/op_host/op_tiling/arch35/conv_cache_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_basic_block.h"
#include "conv/common/op_host/op_tiling/arch35/conv2d_basic_block_strategy.h"
#include "conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_api_tiling.h"

using namespace Ops::NN::Conv;
using namespace optiling;
using namespace optiling::conv_ops_tiling;

namespace {
constexpr uint32_t UT_AICORE_NUM = 32;
constexpr uint64_t UT_L1_SIZE = 524288;
constexpr uint64_t UT_L0A_SIZE = 65536;
constexpr uint64_t UT_L0B_SIZE = 65536;
constexpr uint64_t UT_L0C_SIZE = 262144;
constexpr uint64_t UT_UB_SIZE = 253952;
constexpr uint64_t UT_BT_SIZE = 4096;

void BuildPlatformRes(fe::PlatFormInfos& platformInfo)
{
    std::map<std::string, std::string> socInfo = {{"ai_core_cnt", std::to_string(UT_AICORE_NUM)},
                                                  {"l2_size", "134217728"}};
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", std::to_string(UT_UB_SIZE)},
                                                     {"l1_size", std::to_string(UT_L1_SIZE)},
                                                     {"l2_size", "134217728"},
                                                     {"l0_a_size", std::to_string(UT_L0A_SIZE)},
                                                     {"l0_b_size", std::to_string(UT_L0B_SIZE)},
                                                     {"l0_c_size", std::to_string(UT_L0C_SIZE)},
                                                     {"bt_size", std::to_string(UT_BT_SIZE)},
                                                     {"cube_freq", "1000"},
                                                     {"load3d_constraints", "1"}};
    std::map<std::string, std::string> versionInfo = {{"SoC_version", "Ascend960DT"}};
    std::map<std::string, std::string> intrinsics = {
        {"Intrinsic_data_move_l12ub", "float16"},        {"Intrinsic_data_move_l0c2ub", "float16"},
        {"Intrinsic_fix_pipe_l0c2out", "float16"},       {"Intrinsic_fix_pipe_l0c2ub", "float16"},
        {"Intrinsic_data_move_out2l1_nd2nz", "float16"}, {"Intrinsic_matmul_ub_to_ub", "float16"},
        {"Intrinsic_conv_ub_to_ub", "float16"},          {"Intrinsic_data_move_l12bt", "bf16"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfo);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetPlatformRes("version", versionInfo);
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

ConvAscendcTilingInfo BuildTilingInfo(ge::DataType dtype, ge::Format fmapFormat = ge::FORMAT_NCHW)
{
    ConvAscendcTilingInfo tilingInfo;
    tilingInfo.shapeInfo.batch = 1;
    tilingInfo.shapeInfo.ci = 16;
    tilingInfo.shapeInfo.di = 1;
    tilingInfo.shapeInfo.hi = 16;
    tilingInfo.shapeInfo.wi = 16;
    tilingInfo.shapeInfo.co = 16;
    tilingInfo.shapeInfo.kd = 1;
    tilingInfo.shapeInfo.kh = 3;
    tilingInfo.shapeInfo.kw = 3;
    tilingInfo.shapeInfo.dout = 1;
    tilingInfo.shapeInfo.ho = 14;
    tilingInfo.shapeInfo.wo = 14;
    tilingInfo.attrInfo.groups = 1;
    tilingInfo.attrInfo.strideD = 1;
    tilingInfo.attrInfo.strideH = 1;
    tilingInfo.attrInfo.strideW = 1;
    tilingInfo.attrInfo.dilationD = 1;
    tilingInfo.attrInfo.dilationH = 1;
    tilingInfo.attrInfo.dilationW = 1;
    tilingInfo.descInfo.fMapDtype = dtype;
    tilingInfo.descInfo.weightDtype = dtype;
    tilingInfo.descInfo.biasDtype = dtype;
    tilingInfo.descInfo.outDtype = dtype;
    tilingInfo.descInfo.fMapFormat = fmapFormat;
    tilingInfo.descInfo.weightFormat = ge::FORMAT_NCHW;
    tilingInfo.descInfo.outFormat = ge::FORMAT_NCHW;
    tilingInfo.platformInfo.aicoreNum = UT_AICORE_NUM;
    tilingInfo.platformInfo.l1Size = UT_L1_SIZE;
    tilingInfo.platformInfo.l0aSize = UT_L0A_SIZE;
    tilingInfo.platformInfo.l0bSize = UT_L0B_SIZE;
    tilingInfo.platformInfo.l0cSize = UT_L0C_SIZE;
    tilingInfo.platformInfo.ubSize = UT_UB_SIZE;
    tilingInfo.platformInfo.btSize = UT_BT_SIZE;
    tilingInfo.platformInfo.npuArch = NpuArch::DAV_3510;
    tilingInfo.nodeInfo.nodeType = "Conv2DV2";
    return tilingInfo;
}
} // namespace

class ConvCommonUtilsTest : public testing::Test {};

TEST_F(ConvCommonUtilsTest, MathUtilGcdAndFactors)
{
    EXPECT_EQ(MathUtil::GetGcd(48, 18), 6);
    EXPECT_EQ(MathUtil::GetGcd(7, 0), 0);
    EXPECT_EQ(MathUtil::GetGcd(3, 12), 3);
    std::vector<int32_t> factors;
    MathUtil::GetFactors(factors, 24, 8);
    EXPECT_EQ(factors, std::vector<int32_t>({1, 2, 3, 4, 6, 8}));
    std::vector<int32_t> allFactors;
    MathUtil::GetFactors(allFactors, 12, 100);
    EXPECT_EQ(allFactors, std::vector<int32_t>({1, 2, 3, 4, 6, 12}));
}

TEST_F(ConvCommonUtilsTest, MathUtilLcm)
{
    EXPECT_EQ(MathUtil::Lcm(4, 6), 12);
    EXPECT_EQ(MathUtil::Lcm(static_cast<int32_t>(21), static_cast<int32_t>(6)), 42);
}

TEST_F(ConvCommonUtilsTest, PlatformUtilParseRuntimePlatformInfo)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    BuildPlatformRes(platformInfo);
    Conv3DBackpropV2CompileInfo compileInfo;
    PlatformUtil::ParseRuntimePlatformInfo(compileInfo, "Conv3DV2", platformInfo);
    EXPECT_EQ(compileInfo.ub_size, UT_UB_SIZE);
    EXPECT_EQ(compileInfo.l1_size, UT_L1_SIZE);
    EXPECT_EQ(compileInfo.l2_size, 134217728);
    EXPECT_EQ(compileInfo.l0a_size, UT_L0A_SIZE);
    EXPECT_EQ(compileInfo.l0b_size, UT_L0B_SIZE);
    EXPECT_EQ(compileInfo.l0c_size, UT_L0C_SIZE);
    EXPECT_EQ(compileInfo.bt_size, UT_BT_SIZE);
    EXPECT_EQ(compileInfo.cube_freq, 1000);
    EXPECT_EQ(compileInfo.soc_version, "Ascend960DT");
    EXPECT_TRUE(compileInfo.intrinsic_data_move_l12ub);
    EXPECT_TRUE(compileInfo.intrinsic_data_move_l0c2ub);
    EXPECT_TRUE(compileInfo.intrinsic_matmul_ub_to_ub);
    EXPECT_TRUE(compileInfo.intrinsic_conv_ub_to_ub);
    EXPECT_TRUE(compileInfo.load3d_constraints);
    EXPECT_TRUE(compileInfo.intrinsic_data_move_l12bt_bf16);
}

TEST_F(ConvCommonUtilsTest, PlatformUtilParseRuntimePlatformInfoInvalidValue)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> aicoreSpec = {
        {"ub_size", "abc"}, {"l1_size", "abc"}, {"bt_size", "abc"}, {"cube_freq", "abc"}};
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    Conv3DBackpropV2CompileInfo compileInfo;
    PlatformUtil::ParseRuntimePlatformInfo(compileInfo, "Conv3DV2", platformInfo);
    EXPECT_EQ(compileInfo.ub_size, 0);
    EXPECT_EQ(compileInfo.l1_size, 0);
    EXPECT_EQ(compileInfo.bt_size, 0);
    EXPECT_EQ(compileInfo.cube_freq, 0);
    EXPECT_FALSE(compileInfo.intrinsic_data_move_l12ub);
    EXPECT_FALSE(compileInfo.load3d_constraints);
}

TEST_F(ConvCommonUtilsTest, DebugUtilStringAndShape)
{
    EXPECT_EQ(DebugString(std::vector<int64_t>({1, 2, 3})), "[1, 2, 3]");
    EXPECT_EQ(DebugString(std::vector<int64_t>()), "[]");

    gert::StorageShape xShape = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    gert::StorageShape yShape = {{1, 16, 14, 14}, {1, 16, 14, 14}};
    int32_t compileInfo = 0;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingDataPtr = gert::TilingData::CreateCap(1024);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1024);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .TilingData(tilingDataPtr.get())
                      .Workspace(wsSize)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    std::vector<int64_t> inputShape;
    DebugShape(context, 0, inputShape, true);
    EXPECT_EQ(inputShape, std::vector<int64_t>({1, 16, 16, 16}));
    std::vector<int64_t> outputShape;
    DebugShape(context, 0, outputShape, false);
    EXPECT_EQ(outputShape, std::vector<int64_t>({1, 16, 14, 14}));
}

TEST_F(ConvCommonUtilsTest, DebugUtilTensorInfoAndAttrVector)
{
    gert::StorageShape xShape = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    gert::StorageShape yShape = {{1, 16, 14, 14}, {1, 16, 14, 14}};
    int32_t compileInfo = 0;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingDataPtr = gert::TilingData::CreateCap(1024);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1024);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv2DV2")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})}})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .TilingData(tilingDataPtr.get())
                      .Workspace(wsSize)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    TensorInfo info = GetTensorInfo(context, 0, true, 4);
    EXPECT_EQ(info.shape, std::vector<int64_t>({1, 16, 16, 16}));
    EXPECT_EQ(info.format, ge::FORMAT_NCHW);
    EXPECT_EQ(info.dtype, ge::DT_FLOAT16);
    TensorInfo infoDimOne = GetTensorInfo(context, 0, true, 1);
    EXPECT_EQ(infoDimOne.shape, std::vector<int64_t>({1}));

    EXPECT_EQ(GetAttrVector(context, 0, 4, "strides"), std::vector<int64_t>({1, 1, 2, 2}));
    EXPECT_TRUE(GetAttrVector(context, 0, 3, "strides").empty());
    EXPECT_TRUE(GetAttrVector(context, 5, 4, "pads").empty());
}

TEST_F(ConvCommonUtilsTest, NumBlocksDecisionLoad3D)
{
    ConvBaseDeci deci;
    deci.featureFlagInfo_ = ConvAscendcFeatureFlag::IS_LOAD3D_FLAG;
    ConvAscendcTilingInfo tilingInfo = BuildTilingInfo(ge::DT_FLOAT16);
    ASSERT_EQ(deci.GetNumBlocksInfo(tilingInfo), ge::GRAPH_SUCCESS);
    EXPECT_GE(tilingInfo.numBlocksRes.batchDim, 1u);
    EXPECT_GE(tilingInfo.numBlocksRes.nDim, 1u);
    EXPECT_GE(tilingInfo.numBlocksRes.groupDim, 1u);
}

TEST_F(ConvCommonUtilsTest, NumBlocksDecisionDmaMode)
{
    ConvBaseDeci deci;
    deci.featureFlagInfo_ = ConvAscendcFeatureFlag::IS_DMA_FLAG;
    ConvAscendcTilingInfo tilingInfo = BuildTilingInfo(ge::DT_FLOAT16);
    tilingInfo.shapeInfo.ci = 256;
    tilingInfo.shapeInfo.hi = 32;
    tilingInfo.shapeInfo.wi = 32;
    tilingInfo.shapeInfo.co = 256;
    tilingInfo.shapeInfo.ho = 30;
    tilingInfo.shapeInfo.wo = 30;
    tilingInfo.shapeInfo.kh = 5;
    tilingInfo.shapeInfo.kw = 5;
    ASSERT_EQ(deci.GetNumBlocksInfo(tilingInfo), ge::GRAPH_SUCCESS);
    EXPECT_GE(tilingInfo.numBlocksRes.nDim, 1u);
}

TEST_F(ConvCommonUtilsTest, NumBlocksDecisionL1SizeNotEnough)
{
    ConvBaseDeci deci;
    deci.featureFlagInfo_ = ConvAscendcFeatureFlag::IS_LOAD3D_FLAG;
    ConvAscendcTilingInfo tilingInfo = BuildTilingInfo(ge::DT_FLOAT16);
    tilingInfo.platformInfo.l1Size = 0;
    tilingInfo.shapeInfo.ci = 1024;
    tilingInfo.shapeInfo.co = 1024;
    tilingInfo.shapeInfo.hi = 64;
    tilingInfo.shapeInfo.wi = 64;
    tilingInfo.shapeInfo.ho = 62;
    tilingInfo.shapeInfo.wo = 62;
    EXPECT_EQ(deci.GetNumBlocksInfo(tilingInfo), ge::GRAPH_FAILED);
}

TEST_F(ConvCommonUtilsTest, NumBlocksDecisionInstructionLimits)
{
    ConvBaseDeci deci;
    deci.featureFlagInfo_ = ConvAscendcFeatureFlag::IS_LOAD3D_FLAG;
    deci.descInfo_.fMapDtype = ge::DT_FLOAT16;
    deci.nodeInfo_.nodeType = "Conv2DV2";

    deci.shapeInfo_ = ConvAscendcShapesInfo();
    deci.shapeInfo_.wi = 1ULL << 33;
    deci.flagInfo_.enableC04Flag = true;
    EXPECT_FALSE(deci.CheckInstrLimitsMmode());

    deci.flagInfo_.enableC04Flag = false;
    deci.shapeInfo_ = ConvAscendcShapesInfo();
    deci.descInfo_.fMapFormat = ge::FORMAT_NDHWC;
    deci.shapeInfo_.ci = 1ULL << 21;
    deci.shapeInfo_.hi = 1ULL << 10;
    deci.shapeInfo_.wi = 1ULL << 10;
    deci.attrInfo_.dilationD = 2;
    EXPECT_FALSE(deci.CheckInstrLimitsMmode());
    EXPECT_TRUE(deci.CheckInstrLimitsHWmode());

    deci.shapeInfo_ = ConvAscendcShapesInfo();
    deci.shapeInfo_.wo = 100000;
    deci.shapeInfo_.co = 100000;
    EXPECT_FALSE(deci.CheckInstrLimitsHWmode());

    deci.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    deci.shapeInfo_ = ConvAscendcShapesInfo();
    EXPECT_TRUE(deci.CheckInstrLimitsMmode());
    EXPECT_TRUE(deci.CheckInstrLimitsHWmode());
}

TEST_F(ConvCommonUtilsTest, NumBlocksUtilFunctions)
{
    EXPECT_EQ(Gcd(12, 18), 6);
    EXPECT_EQ(Gcd(0, 7), 7);
    EXPECT_EQ(ConvCeilDiv(10, 3), 4);
    EXPECT_EQ(ConvCeilDiv(10, 0), 0);
    EXPECT_EQ(ConvAlignB(10, 16), 16);
    EXPECT_EQ(ConvAlignB(10, 0), 0);
    EXPECT_EQ(ConvInferHiL1(4, 100, 3, 1, 1), 6);
    EXPECT_EQ(ConvInferHiL1(4, 5, 3, 1, 1), 5);
    EXPECT_EQ(ConvInferWiL1(4, 100, 3, 1, 1), 6);
    EXPECT_EQ(ConvInferWiL1(4, 5, 3, 1, 1), 5);
    EXPECT_EQ(ConvComputeHo(16, 3, 1, 1, 1, 1), 16);
    EXPECT_EQ(ConvComputeHo(16, 3, 1, 1, 1, 0), 1);
    EXPECT_EQ(ConvComputeWo(16, 3, 1, 1, 1, 1), 16);
    EXPECT_EQ(ConvComputeWo(16, 3, 1, 1, 1, 0), 1);
    EXPECT_EQ(ConvComputeDo(16, 3, 1, 1, 1, 1), 16);
    EXPECT_EQ(ConvComputeDo(16, 3, 1, 1, 1, 0), 1);
    EXPECT_EQ(FormatString("value=%d", 7), "value=7");
    EXPECT_EQ(FormatString(nullptr), "");
    std::vector<uint32_t> factorList;
    ConvCalcCommFactor(24, 8, factorList);
    EXPECT_EQ(factorList, std::vector<uint32_t>({1, 2, 3, 4, 6, 8}));
    std::vector<uint32_t> rangeList = {1, 2, 3, 4, 8};
    ConvNumBlocksFactorMix(16, rangeList, {1, 2, 4});
    EXPECT_FALSE(rangeList.empty());
    ConvOpsConstParams constParams;
    ConvAscendcDescInfo descInfo;
    descInfo.fMapDtype = ge::DT_FLOAT16;
    descInfo.weightDtype = ge::DT_FLOAT16;
    ConvAscendcShapesInfo shapesInfo;
    shapesInfo.ci = 32;
    shapesInfo.co = 32;
    InitNumBlocksConstParas(constParams, descInfo, shapesInfo);
    EXPECT_EQ(constParams.m0, 16u);
    EXPECT_EQ(constParams.n0, 16u);
    EXPECT_EQ(constParams.k0, 16u);
}

TEST_F(ConvCommonUtilsTest, ConvInputArgsEqualityAndHash)
{
    ConvInputArgs args;
    ConvInputArgs other = args;
    EXPECT_TRUE(args == other);
    other.dual_output = 1;
    EXPECT_FALSE(args == other);
    other = args;
    other.scaleFlag0 = 1;
    EXPECT_FALSE(args == other);
    other = args;
    other.output1Dtype = ge::DT_FLOAT16;
    EXPECT_FALSE(args == other);
    args.inputFormat = ge::FORMAT_NCHW;
    args.weightFormat = ge::FORMAT_HWCN;
    args.outputFormat = ge::FORMAT_NHWC;
    args.inputDtype = ge::DT_FLOAT16;
    args.weightDtype = ge::DT_FLOAT16;
    args.outputDtype = ge::DT_FLOAT16;
    args.biasDtype = ge::DT_FLOAT16;
    args.biasFlag = true;
    args.hf32Flag = true;
    ConvInputArgsHash hasher;
    EXPECT_EQ(hasher(args), hasher(args));
    size_t seed = 0;
    HashCombine(seed, ge::FORMAT_NHWC);
    HashCombine(seed, ge::DT_FLOAT16);
    HashCombine(seed, true);
    HashCombine(seed, static_cast<uint8_t>(3));
    HashCombine(seed, static_cast<int64_t>(7));
    EXPECT_NE(seed, 0u);
}

TEST_F(ConvCommonUtilsTest, ConvTilingCacheBasic)
{
    ConvTilingCache<int> cache;
    int tiling = 0;
    ConvInputArgs args;
    EXPECT_FALSE(cache.GetCachedTiling(args, tiling));
    EXPECT_TRUE(cache.AddCachedTiling(args, 7));
    EXPECT_TRUE(cache.GetCachedTiling(args, tiling));
    EXPECT_EQ(tiling, 7);
    EXPECT_FALSE(cache.AddCachedTiling(args, 8));
    EXPECT_EQ(cache.GetCacheSize(), 1u);
    ConvTilingParseInfo parseInfo;
    parseInfo.aicoreNum = UT_AICORE_NUM;
    cache.SetSocVersion(NpuArch::DAV_3510, parseInfo);
    EXPECT_EQ(cache.GetSocVersion(), NpuArch::DAV_3510);
    EXPECT_EQ(cache.GetPlatFormInfo().aicoreNum, UT_AICORE_NUM);
}

TEST_F(ConvCommonUtilsTest, ConvTilingCacheFull)
{
    ConvTilingCache<int> cache;
    ConvInputArgs args;
    for (size_t idx = 0; idx < MAX_CACHE_SIZE; ++idx) {
        args.inputShapeN = static_cast<int64_t>(idx);
        EXPECT_TRUE(cache.AddCachedTiling(args, static_cast<int>(idx)));
    }
    EXPECT_EQ(cache.GetCacheSize(), MAX_CACHE_SIZE);
    args.inputShapeN = static_cast<int64_t>(MAX_CACHE_SIZE);
    EXPECT_FALSE(cache.AddCachedTiling(args, 0));
}

// ============================================================================
// ConvBasicBlock memory-bound candidate selection (DAV_9201/9202 only in
// production, so drive the private helpers directly).
// ============================================================================
namespace {
using conv_tiling::BasicBlockInfo;
using conv_tiling::BBCommonParams;
using conv_tiling::Conv2DBasicBlockStrategy;
using conv_tiling::ConvBasicBlock;
using conv_tiling::ConvTilingBase;
using FWDimCandidate = ConvBasicBlock::FWDimCandidate;
using MemBoundCmpData = ConvBasicBlock::MemBoundCmpData;

struct BBTestEnv {
    ConvAscendcShapesInfo shapeInfo;
    ConvAscendcDescInfo descInfo;
    ConvAscendcAttrInfo attrInfo;
    ConvOpsConstParams constParams;
    ConvAscendcTilingFlag flagInfo;
    ConvOriGroupInfo oriGroupInfo;
    ConvOptGroupInfo optGroupInfo;
    FixpipeInfo fixpipeInfo;
    conv_tiling::PlatformInfo platformInfo;
    conv_tiling::Conv2dTiling tiling;
    bool enableInnerBatch = false;
    uint32_t innerBatch = 1;
    BBCommonParams params;
    BasicBlockInfo bbInfo;
    std::shared_ptr<Conv2DBasicBlockStrategy> strategy;

    BBTestEnv()
        : tiling(platformInfo),
          params{shapeInfo,    descInfo,     attrInfo,
                 constParams,  flagInfo,     ConvAscendcFeatureFlag::IS_LOAD3D_FLAG,
                 platformInfo, 32,           UT_L1_SIZE,
                 UT_L0C_SIZE,  oriGroupInfo, optGroupInfo,
                 fixpipeInfo,  &tiling,      enableInnerBatch,
                 innerBatch},
          strategy(std::make_shared<Conv2DBasicBlockStrategy>())
    {
        platformInfo.npuArch = NpuArch::DAV_9201;
        shapeInfo.batch = 1;
        shapeInfo.ci = 16;
        shapeInfo.hi = 16;
        shapeInfo.wi = 16;
        shapeInfo.co = 16;
        shapeInfo.kh = 3;
        shapeInfo.kw = 3;
        shapeInfo.ho = 16;
        shapeInfo.wo = 16;
        shapeInfo.dout = 1;
        attrInfo.groups = 1;
        attrInfo.strideH = 1;
        attrInfo.strideW = 1;
        descInfo.fMapDtype = ge::DT_FLOAT16;
        descInfo.weightDtype = ge::DT_FLOAT16;
        constParams.m0 = 16;
        constParams.n0 = 16;
        constParams.k0 = 16;
    }
};

FWDimCandidate MakeFwCandidate(uint32_t fDim, uint32_t nDim, uint32_t mTile, uint32_t nTile, uint32_t mCut,
                               uint32_t nCut, uint64_t mIn, uint64_t total, float util, uint32_t fwDimDiff,
                               double l1Score)
{
    FWDimCandidate cand;
    cand.fwDim = {fDim, nDim};
    cand.strategy = std::make_tuple(conv_tiling::IterateMNOrder::ITER_M_FST, false, false, false, false, false, false,
                                    static_cast<float>(l1Score));
    cand.tileInfo = std::make_tuple(mTile, nTile, mCut, nCut, mIn);
    cand.total = total;
    cand.util = util;
    cand.fwDimDiff = fwDimDiff;
    cand.l1Score = l1Score;
    return cand;
}
} // namespace

TEST_F(ConvCommonUtilsTest, ConvBasicBlockCompareMemoryBoundCandidate)
{
    MemBoundCmpData cur;
    MemBoundCmpData best;
    best.mDataSize = 100;
    best.l1Score = 0.5;
    best.fwDimDiff = 2;
    best.total = 10;

    // mDataSize differs
    cur = best;
    cur.mDataSize = 50;
    EXPECT_TRUE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));
    cur.mDataSize = 200;
    EXPECT_FALSE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));

    // l1Score differs
    cur = best;
    cur.l1Score = 0.9;
    EXPECT_TRUE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));
    cur.l1Score = 0.1;
    EXPECT_FALSE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));

    // fwDimDiff differs
    cur = best;
    cur.fwDimDiff = 1;
    EXPECT_TRUE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));
    cur.fwDimDiff = 3;
    EXPECT_FALSE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));

    // total differs
    cur = best;
    cur.total = 5;
    EXPECT_TRUE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));
    cur.total = 20;
    EXPECT_FALSE(ConvBasicBlock::CompareMemoryBoundCandidate(cur, best));
}

TEST_F(ConvCommonUtilsTest, ConvBasicBlockCollectMemoryBoundCandidates)
{
    BBTestEnv env;
    ConvBasicBlock flow(env.params, env.bbInfo, env.strategy);
    std::vector<FWDimCandidate> evaluated = {
        MakeFwCandidate(2, 2, 16, 16, 16, 1, 256, 8, 0.95f, 0, 0.5),
        MakeFwCandidate(4, 2, 16, 16, 8, 1, 256, 8, 0.5f, 2, 0.1),
    };
    auto candidates = flow.CollectMemoryBoundCandidates(evaluated);
    ASSERT_EQ(candidates.size(), 1U);
    EXPECT_EQ(candidates[0].fwDim.first, 2U);
    EXPECT_GT(candidates[0].util, conv_tiling::MEMORY_BOUND_UTIL_THRESHOLD);
}

TEST_F(ConvCommonUtilsTest, ConvBasicBlockSelectMemoryBoundFallback)
{
    BBTestEnv env;
    ConvBasicBlock flow(env.params, env.bbInfo, env.strategy);

    // util dominates: highest util wins.
    std::vector<FWDimCandidate> byUtil = {
        MakeFwCandidate(2, 2, 16, 16, 16, 1, 256, 8, 0.5f, 0, 0.1),
        MakeFwCandidate(4, 2, 16, 16, 8, 1, 256, 8, 0.95f, 2, 0.1),
    };
    EXPECT_EQ(flow.SelectMemoryBoundFallback(byUtil), 1);

    // util close: l1Score breaks the tie.
    std::vector<FWDimCandidate> byL1 = {
        MakeFwCandidate(2, 2, 16, 16, 16, 1, 256, 8, 0.95f, 0, 0.2),
        MakeFwCandidate(4, 2, 16, 16, 8, 1, 256, 8, 0.95f, 2, 0.8),
    };
    EXPECT_EQ(flow.SelectMemoryBoundFallback(byL1), 1);

    // util and l1Score close: smaller fwDimDiff wins.
    std::vector<FWDimCandidate> byDimDiff = {
        MakeFwCandidate(2, 2, 16, 16, 16, 1, 256, 8, 0.95f, 3, 0.5),
        MakeFwCandidate(4, 2, 16, 16, 8, 1, 256, 8, 0.95f, 0, 0.5),
    };
    EXPECT_EQ(flow.SelectMemoryBoundFallback(byDimDiff), 1);

    // everything close: smaller total wins.
    std::vector<FWDimCandidate> byTotal = {
        MakeFwCandidate(2, 2, 16, 16, 16, 1, 256, 20, 0.95f, 1, 0.5),
        MakeFwCandidate(4, 2, 16, 16, 8, 1, 256, 8, 0.95f, 1, 0.5),
    };
    EXPECT_EQ(flow.SelectMemoryBoundFallback(byTotal), 1);
}

TEST_F(ConvCommonUtilsTest, ConvBasicBlockExecuteMemoryBoundFwdimDecision)
{
    BBTestEnv env;
    env.bbInfo.boundType = conv_tiling::BoundType::MEMORY_BOUND;
    ConvBasicBlock flow(env.params, env.bbInfo, env.strategy);
    std::vector<FWDimCandidate> candidates = {
        MakeFwCandidate(1, 4, 16, 16, 16, 4, 256, 8, 0.95f, 3, 0.2),
        MakeFwCandidate(2, 2, 16, 16, 8, 8, 256, 8, 0.95f, 0, 0.9),
    };
    flow.ExecuteMemoryBoundFWDimDecision(candidates);
    EXPECT_GE(env.bbInfo.mDim, 1U);
    EXPECT_GE(env.bbInfo.nDim, 1U);
    EXPECT_GE(env.bbInfo.batchDim, 1U);
}

TEST_F(ConvCommonUtilsTest, ConvBasicBlockSelectMemoryBoundBest)
{
    BBTestEnv env;
    env.bbInfo.boundType = conv_tiling::BoundType::MEMORY_BOUND;
    ConvBasicBlock flow(env.params, env.bbInfo, env.strategy);
    std::vector<FWDimCandidate> evaluated = {
        MakeFwCandidate(1, 4, 16, 16, 16, 4, 256, 8, 0.95f, 3, 0.2),
        MakeFwCandidate(2, 2, 16, 16, 8, 8, 256, 8, 0.95f, 0, 0.9),
        MakeFwCandidate(4, 1, 16, 16, 4, 16, 256, 8, 0.5f, 3, 0.1),
    };
    int32_t best = flow.BasicBlockSortFWDimScores(evaluated);
    EXPECT_GE(best, 0);
    EXPECT_LT(static_cast<size_t>(best), evaluated.size());
}
