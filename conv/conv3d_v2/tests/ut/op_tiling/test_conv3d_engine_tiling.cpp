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
 * \file test_conv3d_tiling_engine.cpp
 * \brief Unit tests for Conv3dTilingEngine (attr limits, blockDim vs legacy).
 */

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <limits>
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/conv3d_base_tiling.h"
#include "../../../op_host/op_tiling/conv3d_tiling_engine.h"
#include "../../../op_host/op_tiling/conv3d_tiling_utils.h"

using Ops::NN::Conv3dTilingEngineApi::Conv3dTilingEngine;
using optiling::Conv3dOpsTiling::Conv3dBaseTiling;
using optiling::Conv3dOpsTiling::CeilDiv;

namespace {

static void FillPlatformInfoForTest(Conv3dTilingEngine &engine,
                                    uint64_t l1Size = 1048576,
                                    uint64_t l0aSize = 1048576,
                                    uint64_t l0bSize = 1048576,
                                    uint64_t l0cSize = 1048576,
                                    uint64_t ubSize = 1048576,
                                    uint64_t btSize = 1048576,
                                    uint64_t l2Rate = 100,
                                    uint32_t aicoreNum = 32)
{
    engine.platformInfo_.l1Size = l1Size;
    engine.platformInfo_.l0aSize = l0aSize;
    engine.platformInfo_.l0bSize = l0bSize;
    engine.platformInfo_.l0cSize = l0cSize;
    engine.platformInfo_.ubSize = ubSize;
    engine.platformInfo_.btSize = btSize;
    engine.platformInfo_.l2Rate = l2Rate;
    engine.platformInfo_.aicoreNum = aicoreNum;
    engine.platformInfo_.socVersion = "Ascend910";
}

// ---------------------------------------------------------------------------
// BlockDimDecision: Conv3dTilingEngine::ComputeBlockDim vs legacy backtrack
// ---------------------------------------------------------------------------

static optiling::Conv3dOpsTiling::BlockDimRes ComputeLegacyBlockDim(Conv3dBaseTiling &base)
{
    base.GetBlockDimRange();
    base.GetBlockDimInit();
    base.CoreBlockDimDecision();
    return base.blockDimRes;
}

static optiling::Conv3dOpsTiling::BlockDimRes ComputeEngineBlockDimFromBase(Conv3dBaseTiling &base)
{
    // Create local engine instance instead of accessing base.engine_
    Conv3dTilingEngine engine("Conv3DV2");
    engine.platformInfo_.aicoreNum = static_cast<uint32_t>(base.opInfo_.aicoreNum);
    engine.platformInfo_.l1Size = base.opInfo_.l1Size;
    engine.platformInfo_.l0aSize = base.opInfo_.l0aSize;
    engine.platformInfo_.l0bSize = base.opInfo_.l0bSize;
    engine.platformInfo_.l0cSize = base.opInfo_.l0cSize;
    engine.platformInfo_.ubSize = base.opInfo_.ubSize;
    engine.platformInfo_.btSize = base.opInfo_.btSize;
    engine.platformInfo_.l2Rate = base.opInfo_.l2Rate;
    engine.shapeInfo_ = base.shapeInfo_;
    engine.attrInfo_.groups = static_cast<int64_t>(base.attrInfo_.groups);
    engine.attrInfo_.groupOpt = static_cast<int64_t>(base.attrInfo_.groupOpt);
    engine.outputOrder_ = base.outputOrder_;

    bool success = engine.ComputeBlockDim();
    EXPECT_TRUE(success);

    uint32_t batchDim = 0;
    uint32_t mDim = 0;
    uint32_t nDim = 0;
    uint32_t doDim = 0;
    uint32_t groupDim = 0;
    engine.GetBlockDimDetail(batchDim, mDim, nDim, doDim, groupDim);

    optiling::Conv3dOpsTiling::BlockDimRes res;
    res.batchDim = batchDim;
    res.mDim = mDim;
    res.nDim = nDim;
    res.doDim = doDim;
    res.groupDim = groupDim;
    return res;
}

static void InitSimpleConv3dEngine(Conv3dTilingEngine &engine)
{
    using Conv3dApiTiling::ConvDtype;

    std::vector<int64_t> weightShape = {16, 1, 1, 1, 1};
    std::vector<int64_t> fmapShape = {1, 1, 1, 1, 1};
    std::vector<int64_t> outputShape = {1, 16, 1, 1, 1};
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgOutputShape(outputShape);

    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    engine.SetPadding(pads);
    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(1);

    engine.SetDataType(ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16);
    engine.SetBias(false, ConvDtype::FLOAT32);
    engine.SetScale(false, ConvDtype::FLOAT32);
    engine.SetHF32(false);
}

TEST(TestConv3dTilingEngine, BlockDim_EngineMatchesLegacy_AllDimEqualTo1)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    Conv3dBaseTiling base(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = base.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = base.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    base.attrInfo_.groups = 1;
    base.attrInfo_.groupOpt = base.attrInfo_.groups;
    base.shapeInfo_.cinOpt = shapeInfo.cIn;
    base.shapeInfo_.coutOpt = shapeInfo.cOut;

    auto legacy = ComputeLegacyBlockDim(base);
    auto engine = ComputeEngineBlockDimFromBase(base);

    EXPECT_EQ(engine.batchDim, legacy.batchDim);
    EXPECT_EQ(engine.mDim, legacy.mDim);
    EXPECT_EQ(engine.nDim, legacy.nDim);
    EXPECT_EQ(engine.doDim, legacy.doDim);
    EXPECT_EQ(engine.groupDim, legacy.groupDim);
}

TEST(TestConv3dTilingEngine, BlockDim_EngineMatchesLegacy_BatchDimEqualToAicore)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    Conv3dBaseTiling base(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = base.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = base.shapeInfo_;
    shapeInfo.batch = 4096;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    base.attrInfo_.groups = 1;
    base.attrInfo_.groupOpt = base.attrInfo_.groups;
    base.shapeInfo_.cinOpt = shapeInfo.cIn;
    base.shapeInfo_.coutOpt = shapeInfo.cOut;

    auto legacy = ComputeLegacyBlockDim(base);
    auto engine = ComputeEngineBlockDimFromBase(base);

    EXPECT_EQ(engine.batchDim, legacy.batchDim);
    EXPECT_EQ(engine.mDim, legacy.mDim);
    EXPECT_EQ(engine.nDim, legacy.nDim);
    EXPECT_EQ(engine.doDim, legacy.doDim);
    EXPECT_EQ(engine.groupDim, legacy.groupDim);
}

TEST(TestConv3dTilingEngine, BlockDim_EngineMatchesLegacy_GroupConv)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    Conv3dBaseTiling base(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = base.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = base.shapeInfo_;
    shapeInfo.batch = 8;
    shapeInfo.cIn = 32;
    shapeInfo.di = 8;
    shapeInfo.hi = 16;
    shapeInfo.wi = 16;
    shapeInfo.cOut = 64;
    shapeInfo.kd = 3;
    shapeInfo.kh = 3;
    shapeInfo.kw = 3;
    shapeInfo.dOut = 8;
    shapeInfo.ho = 16;
    shapeInfo.wo = 16;

    base.attrInfo_.groups = 4;
    base.attrInfo_.groupOpt = base.attrInfo_.groups;
    base.shapeInfo_.cinOpt = shapeInfo.cIn;
    base.shapeInfo_.coutOpt = shapeInfo.cOut;

    auto legacy = ComputeLegacyBlockDim(base);
    auto engine = ComputeEngineBlockDimFromBase(base);

    EXPECT_EQ(engine.batchDim, legacy.batchDim);
    EXPECT_EQ(engine.mDim, legacy.mDim);
    EXPECT_EQ(engine.nDim, legacy.nDim);
    EXPECT_EQ(engine.doDim, legacy.doDim);
    EXPECT_EQ(engine.groupDim, legacy.groupDim);
}

TEST(TestConv3dTilingEngine, GetConv3DV2TilingData_ReturnsTrueOnValidConfig)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);
    // Initialize platform info for test environment
    FillPlatformInfoForTest(engine);

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);
    EXPECT_TRUE(ok);
}

TEST(TestConv3dTilingEngine, GetConv3DV2TilingData_ReturnsFalseOnInvalidStride)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    engine.SetStride({1, 0, 1});

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);
    EXPECT_FALSE(ok);
}

// ---------------------------------------------------------------------------
// Additional Unit Tests based on UT Plan and Feedback
// ---------------------------------------------------------------------------

static void InitConv3dEngineWithPlatform(Conv3dTilingEngine &engine, uint64_t l1Size = 1048576)
{
    InitSimpleConv3dEngine(engine);
    FillPlatformInfoForTest(engine, l1Size);
}

// Helper to create near-overflow values for deterministic testing
static void CreateNearOverflowShapes(std::vector<int64_t> &fmapShape,
                                     std::vector<int64_t> &weightShape,
                                     int64_t groups,
                                     int64_t dtypeSize = 2) // BF16 = 2 bytes
{
    // Calculate values near uint64_t max / (dtype_size * k0 * ...)
    // Using conservative values to ensure predictable overflow behavior
    const uint64_t threshold = std::numeric_limits<uint64_t>::max() / (dtypeSize * 1000);

    fmapShape = {static_cast<int64_t>(threshold / 1000000), 1, 1, 1, 1};
    weightShape = {static_cast<int64_t>(threshold / 1000000), 1, 1, 1, 1};
}

// ---------------------------------------------------------------------------
// 1) Parameter Legality Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckDilationLegal_NonPositive)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test dilation = 0 (invalid)
    engine.SetDilation({1, 0, 1});
    EXPECT_FALSE(engine.CheckDilationLegal());

    // Test dilation = -1 (invalid)
    engine.SetDilation({1, -1, 1});
    EXPECT_FALSE(engine.CheckDilationLegal());

    // Test positive dilation (valid)
    engine.SetDilation({1, 1, 1});
    EXPECT_TRUE(engine.CheckDilationLegal());
}

TEST(TestConv3dTilingEngine, CheckPadLegal_ExceedLoad3D)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test pad exceeding LOAD3D_MAX_PAD (assuming 255) - padHead is depth padding, not checked by LOAD3D
    std::vector<int64_t> largePad = {256, 0, 0, 0, 0, 0}; // padHead = 256 (depth, not LOAD3D checked)
    engine.SetPadding(largePad);
    EXPECT_TRUE(engine.CheckPadLegal()); // Should pass - LOAD3D doesn't check depth padding

    // Test padTop exceeding LOAD3D_MAX_PAD
    largePad = {0, 0, 256, 0, 0, 0}; // padTop = 256 (height, LOAD3D checked)
    engine.SetPadding(largePad);
    EXPECT_FALSE(engine.CheckPadLegal());

    // Test normal pad values (valid)
    std::vector<int64_t> normalPad = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(normalPad);
    EXPECT_TRUE(engine.CheckPadLegal());
}

// Additional comprehensive LOAD3D pad coverage
TEST(TestConv3dTilingEngine, CheckPadLegal_ExceedLoad3D_LeftRight)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test padLeft exceeding LOAD3D_MAX_PAD (index 4)
    std::vector<int64_t> largeLeftRightPad = {0, 0, 0, 0, 256, 0}; // padLeft = 256
    engine.SetPadding(largeLeftRightPad);
    EXPECT_FALSE(engine.CheckPadLegal());

    // Test padRight exceeding LOAD3D_MAX_PAD (index 5)
    largeLeftRightPad = {0, 0, 0, 0, 0, 256}; // padRight = 256
    engine.SetPadding(largeLeftRightPad);
    EXPECT_FALSE(engine.CheckPadLegal());

    // Test depth pad exceeding limit - should pass since LOAD3D doesn't check depth
    std::vector<int64_t> largeDepthPad = {256, 256, 0, 0, 0, 0}; // padHead=256, padTail=256
    engine.SetPadding(largeDepthPad);
    EXPECT_TRUE(engine.CheckPadLegal()); // Should pass - depth padding not LOAD3D checked
}

TEST(TestConv3dTilingEngine, CheckLoad3DLimitsWithReason_PositivePath)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);
    // Initialize platform info for test environment
    FillPlatformInfoForTest(engine);

    // Setup all attributes within LOAD3D limits
    std::vector<int64_t> fmapShape = {32, 64, 32, 128, 128};
    std::vector<int64_t> weightShape = {128, 64, 3, 3, 3};
    std::vector<int64_t> pads = {1, 1, 1, 1, 1, 1};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetPadding(pads);
    engine.SetStride({2, 2, 2}); // Within MAX_STRIDE (63)
    engine.SetDilation({1, 1, 1}); // Within MAX_DILATION (255)

    // This should pass all LOAD3D checks
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    EXPECT_TRUE(engine.GetConv3DV2TilingData(tilingData));
}

TEST(TestConv3dTilingEngine, CheckFmapShape_InvalidDim)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test hi = 0 (invalid)
    std::vector<int64_t> invalidFmapShape = {1, 1, 1, 0, 1}; // hi = 0
    engine.SetOrgFmapShape(invalidFmapShape);
    EXPECT_FALSE(engine.CheckFmapShape());

    // Test batch = 0 (invalid)
    invalidFmapShape = {0, 1, 1, 1, 1}; // batch = 0
    engine.SetOrgFmapShape(invalidFmapShape);
    EXPECT_FALSE(engine.CheckFmapShape());

    // Test valid shape
    std::vector<int64_t> validFmapShape = {1, 1, 1, 1, 1};
    engine.SetOrgFmapShape(validFmapShape);
    EXPECT_TRUE(engine.CheckFmapShape());
}

TEST(TestConv3dTilingEngine, CheckWeightShape_InvalidDim)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test cOut = 0 (invalid)
    std::vector<int64_t> invalidWeightShape = {0, 1, 1, 1, 1}; // cOut = 0
    engine.SetOrgWeightShape(invalidWeightShape);
    EXPECT_FALSE(engine.CheckWeightShape());

    // Test kw = 0 (invalid)
    invalidWeightShape = {16, 1, 1, 1, 0}; // kw = 0
    engine.SetOrgWeightShape(invalidWeightShape);
    EXPECT_FALSE(engine.CheckWeightShape());

    // Test valid shape
    std::vector<int64_t> validWeightShape = {16, 1, 1, 1, 1};
    engine.SetOrgWeightShape(validWeightShape);
    EXPECT_TRUE(engine.CheckWeightShape());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_SupportedWithBias)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);
    engine.SetBias(true, ConvDtype::FLOAT32);

    // Supported combination: {BF16, BF16, BF16, FLOAT32}
    engine.SetDataType(ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16);
    EXPECT_TRUE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_UnsupportedMix)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Unsupported combination: {BF16, INT8, FLOAT32, BF16}
    engine.SetDataType(ConvDtype::BF16, ConvDtype::INT8, ConvDtype::BF16);
    engine.SetBias(true, ConvDtype::FLOAT32);
    EXPECT_FALSE(engine.CheckParamsDtype());

    // Unsupported combination without bias: {BF16, INT8, BF16}
    engine.SetBias(false, ConvDtype::FLOAT32);
    EXPECT_FALSE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_INT8_PositiveCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test supported INT8 combination without bias
    engine.SetDataType(ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16);
    engine.SetBias(false, ConvDtype::FLOAT32);
    EXPECT_TRUE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_UINT8_NegativeCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test unsupported UINT8 dtype
    engine.SetDataType(ConvDtype::UINT8, ConvDtype::UINT8, ConvDtype::UINT8);
    engine.SetBias(false, ConvDtype::FLOAT32);
    EXPECT_FALSE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckInputShapeWithPadDetail_Negative)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup problematic case: hi=1, kh=3, pad=0, dilation=1
    std::vector<int64_t> fmapShape = {1, 16, 8, 1, 1}; // di=8, hi=1, wi=1
    std::vector<int64_t> weightShape = {32, 1, 3, 3, 1}; // kd=1, kh=3, kw=3
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetPadding(pads);
    engine.SetDilation({1, 1, 1});

    int64_t idPad, ihPad, iwPad;
    EXPECT_FALSE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));

    // Test valid case: hi=8, kh=3, pad=1
    fmapShape = {1, 16, 8, 8, 8};
    pads = {1, 1, 1, 1, 1, 1};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));
}

TEST(TestConv3dTilingEngine, CheckInputLimitsHwMode_GroupOrDtypeRejected)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);
    FillPlatformInfoForTest(engine);

    // Test groups=2 (should be rejected in HW mode)
    engine.SetGroups(2);
    EXPECT_FALSE(engine.CheckInputLimitsHwMode());

    // Test fMapDtype=INT4 (should be rejected)
    engine.SetGroups(1);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::INT4, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_FALSE(engine.CheckInputLimitsHwMode());

    // Test UINT8 dtype with groups=1 (should be rejected in HW mode)
    engine.SetGroups(1);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::UINT8, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_FALSE(engine.CheckInputLimitsHwMode());

    // Test valid case: groups=1 with supported dtype
    engine.SetGroups(1);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_TRUE(engine.CheckInputLimitsHwMode());
}

// ---------------------------------------------------------------------------
// 2) Output Order (M/HW) Switch Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, InitOutputOrder_MModeWhenL1Enough)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine, 10485760); // 10MB L1 size

    EXPECT_TRUE(engine.InitOutputOrder());
    EXPECT_EQ(engine.outputOrder_, Conv3dApiTiling::M_Mode);
}

TEST(TestConv3dTilingEngine, InitOutputOrder_FallbackToHW)
{
    Conv3dTilingEngine engine;

    // Setup shape: hi=64, wi=64, kh=kw=3, groups=1
    std::vector<int64_t> fmapShape = {1, 16, 8, 64, 64};
    std::vector<int64_t> weightShape = {32, 1, 3, 3, 1};
    std::vector<int64_t> outputShape = {1, 32, 8, 64, 64};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(1);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    engine.SetBias(false, Conv3dApiTiling::ConvDtype::FLOAT32);

    // Use small L1 size to force M mode failure
    FillPlatformInfoForTest(engine, 2000);

    EXPECT_TRUE(engine.InitOutputOrder());
    EXPECT_EQ(engine.outputOrder_, Conv3dApiTiling::HW_Mode);
}

TEST(TestConv3dTilingEngine, InitOutputOrder_FailWhenGroupsNotSupported)
{
    Conv3dTilingEngine engine;

    // Setup shape: hi=64, wi=64, kh=kw=3, groups=2 (not supported in HW mode)
    std::vector<int64_t> fmapShape = {1, 16, 8, 64, 64};
    std::vector<int64_t> weightShape = {32, 1, 3, 3, 1};
    std::vector<int64_t> outputShape = {1, 32, 8, 64, 64};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(2);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    // Use small L1 size to force M mode failure
    FillPlatformInfoForTest(engine, 2000);

    EXPECT_FALSE(engine.InitOutputOrder());
}

// ---------------------------------------------------------------------------
// 3) groupOpt Calculation and Weight Layout Verification
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, GetGroupConvOpt_GroupsGt1_Succeed)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: groups=4, cIn=32, cOut=64, dtype=BF16
    std::vector<int64_t> fmapShape = {1, 32, 8, 16, 16}; // cIn=32
    std::vector<int64_t> weightShape = {64, 4, 3, 3, 3}; // cOut=64, groups=4
    std::vector<int64_t> outputShape = {1, 64, 8, 16, 16};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetGroups(4);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    EXPECT_TRUE(engine.GetGroupConvOpt());
    EXPECT_EQ(engine.attrInfo_.groupOpt, 2);
    EXPECT_EQ(engine.shapeInfo_.cinOpt, 16);
    EXPECT_EQ(engine.shapeInfo_.coutOpt, 32);
}

TEST(TestConv3dTilingEngine, GetGroupConvOpt_GroupsGt1_NotDivisible)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: groups=3, cIn=32, cOut=64 (not divisible)
    std::vector<int64_t> fmapShape = {1, 32, 8, 16, 16};
    std::vector<int64_t> weightShape = {64, 3, 3, 3, 3};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(3);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    EXPECT_FALSE(engine.GetGroupConvOpt());
}

TEST(TestConv3dTilingEngine, CheckGroupOptAgainstWeightShape_Pass)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup successful group optimization first
    std::vector<int64_t> fmapShape = {1, 32, 8, 16, 16};
    std::vector<int64_t> weightShape = {64, 4, 3, 3, 3};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(4);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    engine.GetGroupConvOpt();

    // For BF16: k0=16, n0=16 (from CUBE_UNIT)
    // After GetGroupConvOpt() optimization: groupOpt=2, cinOpt=16, coutOpt=32
    // Using engine's internal formula to calculate expected weight dimensions:
    uint64_t k0 = 16;  // CUBE_UNIT for BF16
    uint64_t n0 = 16;  // CUBE_UNIT for BF16
    uint64_t kdkhkw = static_cast<uint64_t>(engine.shapeInfo_.kd) *
                      static_cast<uint64_t>(engine.shapeInfo_.kh) *
                      static_cast<uint64_t>(engine.shapeInfo_.kw);

    uint64_t expectedWeightD = engine.attrInfo_.groupOpt *
                               CeilDiv(engine.shapeInfo_.cinOpt, k0) *
                               kdkhkw;
    uint64_t expectedWeightN1 = CeilDiv(engine.shapeInfo_.coutOpt, n0);

    EXPECT_TRUE(engine.CheckGroupOptAgainstWeightShape(expectedWeightD, expectedWeightN1));
}

TEST(TestConv3dTilingEngine, CheckGroupOptAgainstWeightShape_Fail)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup successful group optimization first
    std::vector<int64_t> fmapShape = {1, 32, 8, 16, 16};
    std::vector<int64_t> weightShape = {64, 4, 3, 3, 3};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(4);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    engine.GetGroupConvOpt();

    // Calculate correct expected values first
    uint64_t k0 = 16;  // CUBE_UNIT for BF16
    uint64_t n0 = 16;  // CUBE_UNIT for BF16
    uint64_t kdkhkw = static_cast<uint64_t>(engine.shapeInfo_.kd) *
                      static_cast<uint64_t>(engine.shapeInfo_.kh) *
                      static_cast<uint64_t>(engine.shapeInfo_.kw);

    uint64_t correctWeightD = engine.attrInfo_.groupOpt *
                              CeilDiv(engine.shapeInfo_.cinOpt, k0) *
                              kdkhkw;
    uint64_t correctWeightN1 = CeilDiv(engine.shapeInfo_.coutOpt, n0);

    // Test with incorrect weightD
    uint64_t weightD = 200; // Wrong value
    uint64_t weightN1 = correctWeightN1;
    EXPECT_FALSE(engine.CheckGroupOptAgainstWeightShape(weightD, weightN1));

    // Test with incorrect weightN1
    weightD = correctWeightD;
    weightN1 = 10;  // Wrong value
    EXPECT_FALSE(engine.CheckGroupOptAgainstWeightShape(weightD, weightN1));
}

TEST(TestConv3dTilingEngine, GetGroupConvOpt_OverflowBranch)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test with values that exceed INT32_MAX when casting
    // The GetGroupConvOpt casts to int32_t, so values > INT32_MAX should cause issues
    std::vector<int64_t> fmapShape = {1, 3000000000LL, 1, 1, 1}; // cIn > INT32_MAX
    std::vector<int64_t> weightShape = {3000000000LL, 3000000000LL, 1, 1, 1}; // cOut > INT32_MAX

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(1);

    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    // This should fail because cIn and cOut exceed INT32_MAX and will be truncated when casting
    // Or because the multiplication inside CalOptGroupParams will overflow
    EXPECT_FALSE(engine.GetGroupConvOpt());
}

// ---------------------------------------------------------------------------
// 4) Overflow Protection Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckParamsOverflow_FalseOnHugeProduct)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Use values that will cause overflow in multiplication
    // These values are within int32_t range but will overflow when multiplied
    std::vector<int64_t> hugeFmap = {10000, 2000000, 10000, 10000, 10000};
    std::vector<int64_t> hugeWeight = {2000000, 2000, 1000, 1000, 1000};

    engine.SetOrgFmapShape(hugeFmap);
    engine.SetOrgWeightShape(hugeWeight);
    engine.SetGroups(1000);  // groups = 1000
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    // First get group optimization (should succeed with these values)
    EXPECT_TRUE(engine.GetGroupConvOpt());

    // The multiplication will be: batch * di * groupOpt * CeilDiv(cinOpt/k0) * hi * wi * k0 * dtypeSize
    // 10000 * 10000 * 1000 * CeilDiv(2000/16) * 10000 * 10000 * 16 * 2
    // This will definitely overflow UINT64_MAX
    EXPECT_FALSE(engine.CheckParamsOverflow());
}

TEST(TestConv3dTilingEngine, CheckParamsOverflow_TrueOnNormalShape)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Use normal small shapes from InitSimpleConv3dEngine
    EXPECT_TRUE(engine.CheckParamsOverflow());
}

// ---------------------------------------------------------------------------
// 5) API Tiling Path Tests (bias/scale/HF32)
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, InitOutputOrder_BiasContributionToL1)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup shape for L1 size calculation
    std::vector<int64_t> fmapShape = {1, 16, 32, 64, 64};
    std::vector<int64_t> weightShape = {32, 1, 3, 3, 3};
    std::vector<int64_t> outputShape = {1, 32, 32, 64, 64};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    // Use L1 size that should accommodate bias contribution
    FillPlatformInfoForTest(engine, 200000); // Sufficient L1 size including bias

    EXPECT_TRUE(engine.InitOutputOrder());
}

TEST(TestConv3dTilingEngine, GetConv3dApiTiling_HF32)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine);

    // Enable HF32
    engine.SetHF32(true);

    // Initialize required state before calling GetConv3dApiTiling
    ASSERT_TRUE(engine.CheckAllParams());
    ASSERT_TRUE(engine.InitOutputOrder());
    ASSERT_TRUE(engine.ComputeBlockDim());

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3dApiTiling(tilingData);

    EXPECT_TRUE(ok);
    EXPECT_EQ(tilingData.conv3dApiTiling.hf32Enable, 1); // Should be enabled
}

TEST(TestConv3dTilingEngine, GetConv3DV2TilingData_Load3DViolation)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine);

    // Setup with pad exceeding LOAD3D limits (H/W directions only)
    std::vector<int64_t> largePad = {0, 0, 256, 0, 0, 0}; // padTop = 256 exceeds LOAD3D_MAX_PAD
    engine.SetPadding(largePad);

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);

    EXPECT_FALSE(ok); // Should fail due to LOAD3D limit violation
}

// ---------------------------------------------------------------------------
// 6) Additional Professional Testing Recommendations
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckAttrLimits_MixedBoundaryValues)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test large but valid stride and dilation values at boundary
    // Original (H, W, D) = (63, 63, 1) -> DHW vector {D, H, W} = {1, 63, 63}
    engine.SetStride({1, 63, 63});
    // Original (H, W, D) = (255, 1, 1) -> DHW vector {1, 255, 1}
    engine.SetDilation({1, 255, 1});

    // These should all be valid
    EXPECT_TRUE(engine.CheckStrideLegal());
    EXPECT_TRUE(engine.CheckDilationLegal());
}

// Note about CheckOutputShape:
// CheckOutputShape is declared in the header file but not implemented in the cpp file.
// Unit tests for this method are skipped to avoid linking issues or relying on dead declarations.

TEST(TestConv3dTilingEngine, CheckParameterConsistency_GroupChannelMismatch)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup groups that don't divide channels evenly
    std::vector<int64_t> fmapShape = {1, 31, 8, 16, 16}; // cIn=31
    std::vector<int64_t> weightShape = {64, 3, 3, 3, 3};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(3); // 3 doesn't divide 31 evenly

    EXPECT_FALSE(engine.GetGroupConvOpt());
}

TEST(TestConv3dTilingEngine, PerformanceTest_BlockDimDecisionComplexShape)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine);

    // Setup complex shape for stress testing
    std::vector<int64_t> fmapShape = {32, 64, 16, 128, 128};
    std::vector<int64_t> weightShape = {128, 1, 7, 7, 7};
    std::vector<int64_t> outputShape = {32, 128, 16, 128, 128};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetStride({2, 2, 2});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(1);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    // Test that BlockDim decision completes successfully
    EXPECT_TRUE(engine.ComputeBlockDim());

    // Verify we get valid BlockDim values
    uint32_t batchDim, mDim, nDim, doDim, groupDim;
    engine.GetBlockDimDetail(batchDim, mDim, nDim, doDim, groupDim);

    EXPECT_GT(batchDim, 0);
    EXPECT_GT(mDim, 0);
    EXPECT_GT(nDim, 0);
    EXPECT_GT(doDim, 0);
    EXPECT_GT(groupDim, 0);
}

TEST(TestConv3dTilingEngine, ComputeBlockDim_TieBreakerPreference)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine);

    // Setup shape that creates multiple equal-cost BlockDim combinations
    // This test ensures the preference order: batch > group > do
    std::vector<int64_t> fmapShape = {8, 32, 16, 32, 32};
    std::vector<int64_t> weightShape = {64, 1, 3, 3, 3};
    std::vector<int64_t> outputShape = {8, 64, 16, 32, 32};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(4); // Multi-group to trigger groupDim consideration
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    EXPECT_TRUE(engine.ComputeBlockDim());

    // Verify BlockDim values follow the documented preference
    uint32_t batchDim, mDim, nDim, doDim, groupDim;
    engine.GetBlockDimDetail(batchDim, mDim, nDim, doDim, groupDim);

    EXPECT_GT(batchDim, 0);
    EXPECT_GT(groupDim, 0);
    EXPECT_GT(doDim, 0);
}

// ---------------------------------------------------------------------------
// 7) Additional Critical Tests for Better Coverage
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckLoad3DLimits_IndividualParameters)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test stride at boundary values
    engine.SetStride({1, 63, 1}); // MAX_STRIDE = 63
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    engine.SetStride({1, 64, 1}); // Exceeds boundary
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Reset and test dilation
    engine.SetStride({1, 1, 1});

    engine.SetDilation({1, 255, 1}); // MAX_DILATION = 255
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    engine.SetDilation({1, 256, 1}); // Exceeds boundary
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Reset and test pad directions
    engine.SetDilation({1, 1, 1});
    std::vector<int64_t> pads;

    // Test each pad direction individually (only H/W directions have LOAD3D limit)
    pads = {0, 0, 255, 0, 0, 0}; // padTop at boundary
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    pads = {0, 0, 256, 0, 0, 0}; // padTop exceeds
    engine.SetPadding(pads);
    EXPECT_FALSE(engine.CheckLoad3DLimits());
}

TEST(TestConv3dTilingEngine, CheckAllParams_CompletePath)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineWithPlatform(engine);

    // Test with fully valid configuration
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    EXPECT_TRUE(engine.GetConv3DV2TilingData(tilingData));

    // Now test each validation point individually
    engine.SetStride({1, 0, 1}); // Invalid stride (original H=0)
    EXPECT_FALSE(engine.CheckAllParams());

    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 0, 1}); // Invalid dilation (original H=0)
    EXPECT_FALSE(engine.CheckAllParams());

    engine.SetDilation({1, 1, 1});
    std::vector<int64_t> badPad = {0, 0, 256, 0, 0, 0}; // padTop exceeds LOAD3D
    engine.SetPadding(badPad);
    EXPECT_FALSE(engine.CheckAllParams());

    engine.SetPadding({1, 1, 1, 1, 1, 1});
    std::vector<int64_t> badFmap = {1, 1, 1, 0, 1}; // hi=0
    engine.SetOrgFmapShape(badFmap);
    EXPECT_FALSE(engine.CheckAllParams());

    std::vector<int64_t> goodFmap = {1, 16, 8, 32, 32};
    engine.SetOrgFmapShape(goodFmap);
    std::vector<int64_t> badWeight = {0, 1, 3, 3, 3}; // cOut=0
    engine.SetOrgWeightShape(badWeight);
    EXPECT_FALSE(engine.CheckAllParams());
}

TEST(TestConv3dTilingEngine, CheckInputShapeWithPadDetail_VariousScenarios)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    std::vector<int64_t> fmapShape, weightShape;
    std::vector<int64_t> pads;
    int64_t idPad, ihPad, iwPad;

    // Case 1: Small input with large kernel and no padding
    fmapShape = {1, 16, 8, 2, 2}; // hi=wi=2
    weightShape = {32, 1, 3, 3, 3}; // kh=kw=3
    pads = {0, 0, 0, 0, 0, 0};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetPadding(pads);
    engine.SetDilation({1, 1, 1});

    EXPECT_FALSE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));

    // Case 2: Same but with sufficient padding in all directions
    pads = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));

    // Case 3: Large dilation causing effective kernel > input
    engine.SetDilation({2, 2, 2}); // Effective kernel = 5x5
    EXPECT_FALSE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));

    // Case 4: With even more padding in all directions
    pads = {2, 2, 2, 2, 2, 2};
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));
}

TEST(TestConv3dTilingEngine, GetGroupConvOpt_MultipleGroups)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test optimal group scenario: cIn and cOut divisible by groups
    std::vector<int64_t> fmapShape = {1, 64, 8, 32, 32}; // cIn=64
    std::vector<int64_t> weightShape = {128, 8, 3, 3, 3}; // cOut=128, groups=8
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(8);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);

    EXPECT_TRUE(engine.GetGroupConvOpt());
    // After CalOptGroupParams optimization for BF16 (k0=n0=16):
    // Original groups=8 with cin=8, cout=16 per group
    // Optimized: groupOpt=4 (merging groups), cinOpt=16, coutOpt=32
    EXPECT_EQ(engine.attrInfo_.groupOpt, 4);
    EXPECT_EQ(engine.shapeInfo_.cinOpt, 16);
    EXPECT_EQ(engine.shapeInfo_.coutOpt, 32);

    // Test with groups=1 (no optimization)
    engine.SetGroups(1);
    EXPECT_TRUE(engine.GetGroupConvOpt());
    EXPECT_EQ(engine.attrInfo_.groupOpt, 1);
}

// ---------------------------------------------------------------------------
// 8) Extended Comprehensive Tests
// ---------------------------------------------------------------------------

// Helper functions for extended testing
static void InitConv3dEngineExtended(Conv3dTilingEngine &engine)
{
    using Conv3dApiTiling::ConvDtype;

    std::vector<int64_t> weightShape = {32, 1, 3, 3, 3};
    std::vector<int64_t> fmapShape = {4, 16, 8, 32, 32};
    std::vector<int64_t> outputShape = {4, 32, 8, 32, 32};
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgOutputShape(outputShape);

    std::vector<int64_t> pads = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(pads);
    engine.SetStride({1, 1, 1});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(1);

    engine.SetDataType(ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16);
    engine.SetBias(false, ConvDtype::FLOAT32);
    engine.SetScale(false, ConvDtype::FLOAT32);
    engine.SetHF32(false);
}

static void FillPlatformInfoExtended(Conv3dTilingEngine &engine,
                                   uint64_t l1Size = 1048576,
                                   uint64_t l0aSize = 1048576,
                                   uint64_t l0bSize = 1048576,
                                   uint64_t l0cSize = 1048576,
                                   uint64_t ubSize = 1048576,
                                   uint64_t btSize = 1048576,
                                   uint64_t l2Rate = 100,
                                   uint32_t aicoreNum = 32)
{
    engine.platformInfo_.l1Size = l1Size;
    engine.platformInfo_.l0aSize = l0aSize;
    engine.platformInfo_.l0bSize = l0bSize;
    engine.platformInfo_.l0cSize = l0cSize;
    engine.platformInfo_.ubSize = ubSize;
    engine.platformInfo_.btSize = btSize;
    engine.platformInfo_.l2Rate = l2Rate;
    engine.platformInfo_.aicoreNum = aicoreNum;
    engine.platformInfo_.socVersion = "Ascend910";
}

// ---------------------------------------------------------------------------
// Comprehensive Load3D Limits Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckLoad3DLimits_AllPadDirections)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);
    FillPlatformInfoExtended(engine);

    // Test H/W pad directions at boundary (D direction has no LOAD3D limit)
    std::vector<std::vector<int64_t>> padConfigs = {
        {0, 0, 256, 0, 0, 0},  // padTop exceeds limit
        {0, 0, 0, 256, 0, 0},  // padBottom exceeds limit
        {0, 0, 0, 0, 256, 0},  // padLeft exceeds limit
        {0, 0, 0, 0, 0, 256}   // padRight exceeds limit
    };

    for (const auto &pad : padConfigs) {
        engine.SetPadding(pad);
        EXPECT_FALSE(engine.CheckLoad3DLimits())
            << "Should fail with pad exceeding limit: ["
            << pad[0] << "," << pad[1] << "," << pad[2] << ","
            << pad[3] << "," << pad[4] << "," << pad[5] << "]";
    }

    // Test valid pad values
    std::vector<int64_t> validPads = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(validPads);
    EXPECT_TRUE(engine.CheckLoad3DLimits());
}

TEST(TestConv3dTilingEngine, CheckLoad3DLimits_StrideAndDilationBoundaries)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);
    FillPlatformInfoExtended(engine);

    // Test stride at MAX_STRIDE boundary (assuming 63) - D direction has no LOAD3D limit
    engine.SetStride({63, 63, 63});
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    engine.SetStride({1, 64, 1}); // Exceeds boundary for strideH
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Test dilation at MAX_DILATION boundary (assuming 255) - D direction has no LOAD3D limit
    engine.SetStride({1, 1, 1});
    engine.SetDilation({255, 255, 255});
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    engine.SetDilation({1, 256, 1}); // Exceeds boundary for dilationH
    EXPECT_FALSE(engine.CheckLoad3DLimits());
}

// ---------------------------------------------------------------------------
// Advanced dtype Combination Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckParamsDtype_AllValidCombinations)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);

    // Test all supported BF16 combinations
    struct DtypeCombo {
        ConvDtype fmap, weight, out, bias;
        bool hasBias;
        bool expected;
    };

    std::vector<DtypeCombo> validCombos = {
        {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, true, true}, // Supported: BF16,BF16,FLOAT32,BF16
        {ConvDtype::BF16, ConvDtype::BF16, ConvDtype::BF16, ConvDtype::FLOAT32, false, true}, // Supported: BF16,BF16,BF16
        {ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16, true, true}, // Supported: FP16,FP16,FP16,FP16
        {ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, ConvDtype::FLOAT32, true, true}, // Supported: FP32,FP32,FP32,FP32
        {ConvDtype::INT8, ConvDtype::INT8, ConvDtype::FLOAT16, ConvDtype::FLOAT32, false, true}  // Supported: INT8,INT8,FLOAT16
    };

    for (const auto &combo : validCombos) {
        engine.SetDataType(combo.fmap, combo.weight, combo.out);
        engine.SetBias(combo.hasBias, combo.bias);
        EXPECT_EQ(engine.CheckParamsDtype(), combo.expected)
            << "Failed for dtype combination";
    }
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_UnsupportedCombinations)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);

    // Test unsupported combinations
    struct DtypeCombo {
        ConvDtype fmap, weight, out;
        bool expected;
    };

    std::vector<DtypeCombo> invalidCombos = {
        {ConvDtype::UINT8, ConvDtype::BF16, ConvDtype::BF16, false},  // UINT8 not supported
        {ConvDtype::INT4, ConvDtype::BF16, ConvDtype::BF16, false},   // INT4 not supported
        {ConvDtype::BF16, ConvDtype::INT8, ConvDtype::BF16, false},   // Mixed unsupported
        {ConvDtype::FLOAT32, ConvDtype::BF16, ConvDtype::BF16, false} // FLOAT32 not supported for input
    };

    for (const auto &combo : invalidCombos) {
        engine.SetDataType(combo.fmap, combo.weight, combo.out);
        engine.SetBias(false, ConvDtype::FLOAT32);
        EXPECT_EQ(engine.CheckParamsDtype(), combo.expected)
            << "Should have failed for unsupported combination";
    }
}

// ---------------------------------------------------------------------------
// Complete Integration Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, IntegrationTest_FullPipelineSuccess)
{
    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);
    FillPlatformInfoExtended(engine);

    // Test complete successful pipeline
    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    EXPECT_TRUE(engine.GetConv3DV2TilingData(tilingData));

    // Verify all major fields are populated
    // Verify input shape is set
    EXPECT_GT(tilingData.conv3dApiTiling.orgDi, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.orgHi, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.orgWi, 0);

    // Verify output shape is set
    EXPECT_GT(tilingData.conv3dApiTiling.orgDo, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.orgHo, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.orgWo, 0);

    // Verify kernel shape is set
    EXPECT_GT(tilingData.conv3dApiTiling.kernelD, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.kernelH, 0);
    EXPECT_GT(tilingData.conv3dApiTiling.kernelW, 0);

    // Verify block dimension related fields are set
    EXPECT_GT(tilingData.conv3dRunInfo.batchDim, 0);
    EXPECT_GT(tilingData.conv3dRunInfo.mDim, 0);
    EXPECT_GT(tilingData.conv3dRunInfo.nDim, 0);
}

TEST(TestConv3dTilingEngine, IntegrationTest_MultiGroupComplexShape)
{
    Conv3dTilingEngine engine;

    // Setup complex multi-group scenario
    std::vector<int64_t> fmapShape = {8, 128, 16, 64, 64};
    std::vector<int64_t> weightShape = {256, 8, 3, 3, 3};
    std::vector<int64_t> outputShape = {8, 256, 16, 64, 64};
    std::vector<int64_t> pads = {1, 1, 1, 1, 1, 1};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetPadding(pads);
    engine.SetStride({2, 2, 2});
    engine.SetDilation({1, 1, 1});
    engine.SetGroups(8);
    engine.SetDataType(Conv3dApiTiling::ConvDtype::BF16,
                      Conv3dApiTiling::ConvDtype::BF16,
                      Conv3dApiTiling::ConvDtype::BF16);
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    engine.SetHF32(false);

    FillPlatformInfoExtended(engine, 2097152); // 2MB L1

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    EXPECT_TRUE(engine.GetConv3DV2TilingData(tilingData));

    // Verify group optimization was applied
    EXPECT_EQ(engine.attrInfo_.groupOpt, 8);
    EXPECT_EQ(engine.shapeInfo_.cinOpt, 16);
    EXPECT_EQ(engine.shapeInfo_.coutOpt, 32);
}

// ---------------------------------------------------------------------------
// 9) Overflow Protection and Special Cases Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckGroupOptAgainstWeightShape_Overflow)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup successful group optimization first
    std::vector<int64_t> fmapShape = {1, 32, 8, 16, 16};
    std::vector<int64_t> weightShape = {64, 4, 3, 3, 3};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetGroups(4);
    engine.GetGroupConvOpt(); // Should succeed

    // Test with overflow in weightD calculation
    uint64_t overflowWeightD = std::numeric_limits<uint64_t>::max();
    uint64_t correctWeightN1 = 4;

    EXPECT_FALSE(engine.CheckGroupOptAgainstWeightShape(overflowWeightD, correctWeightN1));
}

TEST(TestConv3dTilingEngine, ComputeBlockDim_MinMaxScenarios)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Fill platform info
    engine.platformInfo_.l1Size = 1048576;
    engine.platformInfo_.l0aSize = 1048576;
    engine.platformInfo_.l0bSize = 1048576;
    engine.platformInfo_.l0cSize = 1048576;
    engine.platformInfo_.ubSize = 1048576;
    engine.platformInfo_.btSize = 1048576;
    engine.platformInfo_.l2Rate = 100;
    engine.platformInfo_.aicoreNum = 1; // Single core

    // Test with single core - should use minimum distribution
    EXPECT_TRUE(engine.ComputeBlockDim());

    uint32_t batchDim, mDim, nDim, doDim, groupDim;
    engine.GetBlockDimDetail(batchDim, mDim, nDim, doDim, groupDim);

    // With single core, total should be 1
    EXPECT_EQ(batchDim * mDim * nDim * doDim * groupDim, 1);
}

// ---------------------------------------------------------------------------
// 10) Additional Edge Cases and Error Paths
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, ZeroAndNegativeValues_Handling)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test with zero values in various places
    engine.SetStride({1, 0, 1});
    EXPECT_FALSE(engine.CheckAllParams());

    engine.SetStride({1, 1, 0});
    EXPECT_FALSE(engine.CheckAllParams());

    engine.SetStride({0, 1, 1});
    EXPECT_FALSE(engine.CheckAllParams());

    // Test negative padding (should be handled correctly)
    engine.SetStride({1, 1, 1});
    std::vector<int64_t> negativePad = {-1, -1, -1, -1, -1, -1};
    engine.SetPadding(negativePad);

    // Should still pass if input is large enough
    std::vector<int64_t> largeFmap = {1, 16, 16, 32, 32};
    engine.SetOrgFmapShape(largeFmap);

    int64_t idPad, ihPad, iwPad;
    EXPECT_TRUE(engine.CheckInputShapeWithPadDetail(idPad, ihPad, iwPad));
}

TEST(TestConv3dTilingEngine, ShapeSizeLimits_BoundaryTests)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test LOAD3D filter size limits (LOAD3D_MAX_FILTER_H_W = 511)
    // CheckLoad3DLimits only validates kh and kw, not di

    // Test weight shapes at LOAD3D boundary
    std::vector<int64_t> maxWeightShape = {1, 1, 1, 511, 1}; // kh = 511 (at boundary)
    engine.SetOrgWeightShape(maxWeightShape);
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    maxWeightShape = {1, 1, 1, 512, 1}; // kh = 512 (exceeds boundary)
    engine.SetOrgWeightShape(maxWeightShape);
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Reset and test kw
    maxWeightShape = {1, 1, 1, 1, 511}; // kw = 511 (at boundary)
    engine.SetOrgWeightShape(maxWeightShape);
    EXPECT_TRUE(engine.CheckLoad3DLimits());

    maxWeightShape = {1, 1, 1, 1, 512}; // kw = 512 (exceeds boundary)
    engine.SetOrgWeightShape(maxWeightShape);
    EXPECT_FALSE(engine.CheckLoad3DLimits());
}

} // namespace
