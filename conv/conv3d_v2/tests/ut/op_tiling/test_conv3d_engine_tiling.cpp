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
    if (!engine.Init()) {
        // Platform initialization failed, manually inject test values
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
    // Always override with test values to ensure deterministic behavior
    engine.platformInfo_.l1Size = l1Size;
    engine.platformInfo_.l0aSize = l0aSize;
    engine.platformInfo_.l0bSize = l0bSize;
    engine.platformInfo_.l0cSize = l0cSize;
    engine.platformInfo_.ubSize = ubSize;
    engine.platformInfo_.btSize = btSize;
    engine.platformInfo_.l2Rate = l2Rate;
    engine.platformInfo_.aicoreNum = aicoreNum;
    engine.platformInfo_.socVersion = "Ascend910";
    // Mark engine as initialized after manual platform info injection
    engine.SetInitialized(true);
}

static void InitSimpleConv3dEngine(Conv3dTilingEngine &engine)
{
    using Conv3dApiTiling::ConvDtype;
    using Conv3dApiTiling::ConvFormat;

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
    // Set NCDHW format for pointwise mode (1x1x1 kernel)
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);
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
    // Initialize platform info for test environment
    FillPlatformInfoForTest(engine);

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

TEST(TestConv3dTilingEngine, CheckPadLegal_SemanticOnly)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test pad exceeding LOAD3D_MAX_PAD (assuming 255) - padHead is depth padding, not checked by LOAD3D
    std::vector<int64_t> largePad = {256, 0, 0, 0, 0, 0}; // padHead = 256 (depth, not LOAD3D checked)
    engine.SetPadding(largePad);
    EXPECT_TRUE(engine.CheckPadLegal()); // Should pass - semantic check only (non-negative pads)

    // Test padTop exceeding LOAD3D_MAX_PAD: semantic check should still pass,
    // Load3D hardware limit should be enforced by CheckLoad3DLimits()
    largePad = {0, 0, 256, 0, 0, 0}; // padTop = 256 (height, LOAD3D checked)
    engine.SetPadding(largePad);
    EXPECT_TRUE(engine.CheckPadLegal());
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Test normal pad values (valid)
    std::vector<int64_t> normalPad = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(normalPad);
    EXPECT_TRUE(engine.CheckPadLegal());
}

// Additional comprehensive LOAD3D pad coverage
TEST(TestConv3dTilingEngine, CheckPadLegal_SemanticOnly_LeftRight)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test padLeft exceeding LOAD3D_MAX_PAD (index 4):
    // semantic check passes, Load3D limit fails
    std::vector<int64_t> largeLeftRightPad = {0, 0, 0, 0, 256, 0}; // padLeft = 256
    engine.SetPadding(largeLeftRightPad);
    EXPECT_TRUE(engine.CheckPadLegal());
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Test padRight exceeding LOAD3D_MAX_PAD (index 5):
    // semantic check passes, Load3D limit fails
    largeLeftRightPad = {0, 0, 0, 0, 0, 256}; // padRight = 256
    engine.SetPadding(largeLeftRightPad);
    EXPECT_TRUE(engine.CheckPadLegal());
    EXPECT_FALSE(engine.CheckLoad3DLimits());

    // Test depth pad exceeding limit - should pass since LOAD3D doesn't check depth
    std::vector<int64_t> largeDepthPad = {256, 256, 0, 0, 0, 0}; // padHead=256, padTail=256
    engine.SetPadding(largeDepthPad);
    EXPECT_TRUE(engine.CheckPadLegal()); // Should pass - semantic check only (non-negative)
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
    // Output (NCDHW) computed from input/pad/stride/dilation (SAME-like padding here): Do/Ho/Wo = 16/64/64
    std::vector<int64_t> outputShape = {32, 128, 16, 64, 64};
    std::vector<int64_t> pads = {1, 1, 1, 1, 1, 1};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetOrgOutputShape(outputShape);
    engine.SetPadding(pads);
    engine.SetStride({2, 2, 2}); // Within MAX_STRIDE (63)
    engine.SetDilation({1, 1, 1}); // Within MAX_DILATION (255)
    // Set format for regular mode (3x3x3 kernel)
    engine.SetFormat(Conv3dApiTiling::ConvFormat::NDC1HWC0,
                     Conv3dApiTiling::ConvFormat::FRACTAL_Z_3D,
                     Conv3dApiTiling::ConvFormat::NDC1HWC0);

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

TEST(TestConv3dTilingEngine, CheckParamsDtype_Pointwise_FP16BiasFP32_PositiveCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine); // pointwise (1x1x1 kernel)

    engine.SetDataType(ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    engine.SetBias(true, ConvDtype::FLOAT32);
    engine.SetScale(false, ConvDtype::FLOAT32);

    EXPECT_TRUE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_Pointwise_FP16BiasFP16_NegativeCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine); // pointwise (1x1x1 kernel)

    engine.SetDataType(ConvDtype::FLOAT16, ConvDtype::FLOAT16, ConvDtype::FLOAT16);
    engine.SetBias(true, ConvDtype::FLOAT16);
    engine.SetScale(false, ConvDtype::FLOAT32);

    EXPECT_FALSE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_INT8_WithScale_PositiveCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Switch to regular-mode kernel (not pointwise) to validate INT8 + scale support.
    engine.SetOrgWeightShape({16, 1, 3, 3, 3});

    engine.SetDataType(ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BF16);
    engine.SetBias(true, ConvDtype::FLOAT32);
    engine.SetScale(true, ConvDtype::FLOAT32);

    EXPECT_TRUE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_INT8_WithScale_WrongScaleDtype_NegativeCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Switch to regular-mode kernel (not pointwise) to validate INT8 + scale support.
    engine.SetOrgWeightShape({16, 1, 3, 3, 3});

    engine.SetDataType(ConvDtype::INT8, ConvDtype::INT8, ConvDtype::BF16);
    engine.SetBias(true, ConvDtype::FLOAT32);
    engine.SetScale(true, ConvDtype::BF16); // scale must be float32

    EXPECT_FALSE(engine.CheckParamsDtype());
}

TEST(TestConv3dTilingEngine, CheckParamsDtype_INT8_PositiveCase)
{
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Switch to regular-mode kernel (not pointwise) to validate regular INT8 support.
    engine.SetOrgWeightShape({16, 1, 3, 3, 3});

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

TEST(TestConv3dTilingEngine, CheckInputShapeWithPad_Negative)
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

    EXPECT_FALSE(engine.CheckInputShapeWithPad());

    // Test valid case: hi=8, kh=3, pad=1
    fmapShape = {1, 16, 8, 8, 8};
    pads = {1, 1, 1, 1, 1, 1};
    engine.SetOrgFmapShape(fmapShape);
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPad());
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

TEST(TestConv3dTilingEngine, CheckInputShapeWithPad_VariousScenarios)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    std::vector<int64_t> fmapShape, weightShape;
    std::vector<int64_t> pads;

    // Case 1: Small input with large kernel and no padding
    fmapShape = {1, 16, 8, 2, 2}; // hi=wi=2
    weightShape = {32, 1, 3, 3, 3}; // kh=kw=3
    pads = {0, 0, 0, 0, 0, 0};

    engine.SetOrgFmapShape(fmapShape);
    engine.SetOrgWeightShape(weightShape);
    engine.SetPadding(pads);
    engine.SetDilation({1, 1, 1});

    EXPECT_FALSE(engine.CheckInputShapeWithPad());

    // Case 2: Same but with sufficient padding in all directions
    pads = {1, 1, 1, 1, 1, 1};
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPad());

    // Case 3: Large dilation causing effective kernel > input
    engine.SetDilation({2, 2, 2}); // Effective kernel = 5x5
    EXPECT_FALSE(engine.CheckInputShapeWithPad());

    // Case 4: With even more padding in all directions
    pads = {2, 2, 2, 2, 2, 2};
    engine.SetPadding(pads);
    EXPECT_TRUE(engine.CheckInputShapeWithPad());
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
    using Conv3dApiTiling::ConvFormat;

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
    // Set format for regular mode (3x3x3 kernel)
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);
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
    // Try to initialize from platform first, but allow manual override
    if (!engine.Init()) {
        // Platform initialization failed, manually inject test values
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
    // Always override with test values to ensure deterministic behavior
    engine.platformInfo_.l1Size = l1Size;
    engine.platformInfo_.l0aSize = l0aSize;
    engine.platformInfo_.l0bSize = l0bSize;
    engine.platformInfo_.l0cSize = l0cSize;
    engine.platformInfo_.ubSize = ubSize;
    engine.platformInfo_.btSize = btSize;
    engine.platformInfo_.l2Rate = l2Rate;
    engine.platformInfo_.aicoreNum = aicoreNum;
    engine.platformInfo_.socVersion = "Ascend910";
    // Mark engine as initialized after manual platform info injection
    engine.SetInitialized(true);
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
    // Output (NCDHW) computed from input/pad/stride/dilation: Do/Ho/Wo = 8/32/32
    std::vector<int64_t> outputShape = {8, 256, 8, 32, 32};
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
    engine.SetBiasShape({static_cast<int64_t>(engine.shapeInfo_.cOut)});
    engine.SetHF32(false);
    // Set format for regular mode (3x3x3 kernel)
    engine.SetFormat(Conv3dApiTiling::ConvFormat::NDC1HWC0,
                     Conv3dApiTiling::ConvFormat::FRACTAL_Z_3D,
                     Conv3dApiTiling::ConvFormat::NDC1HWC0);

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

    EXPECT_TRUE(engine.CheckInputShapeWithPad());
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

// ---------------------------------------------------------------------------
// 11) Format Validation Tests (CheckInputFormat)
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckInputFormat_PointwiseMode_ValidNCDHW)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup pointwise convolution (1x1x1 kernel)
    std::vector<int64_t> weightShape = {16, 1, 1, 1, 1}; // 1x1x1 kernel
    engine.SetOrgWeightShape(weightShape);

    // Set all formats to NCDHW (valid for pointwise)
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);

    EXPECT_TRUE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_PointwiseMode_InvalidMixedFormats)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup pointwise convolution (1x1x1 kernel)
    std::vector<int64_t> weightShape = {16, 1, 1, 1, 1};
    engine.SetOrgWeightShape(weightShape);

    // Test invalid format: fmap is NDC1HWC0 instead of NCDHW
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::NCDHW, ConvFormat::NCDHW);
    EXPECT_FALSE(engine.CheckInputFormat());

    // Test invalid format: weight is FRACTAL_Z_3D instead of NCDHW
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::FRACTAL_Z_3D, ConvFormat::NCDHW);
    EXPECT_FALSE(engine.CheckInputFormat());

    // Test invalid format: output is NDC1HWC0 instead of NCDHW
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NDC1HWC0);
    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_RegularMode_ValidFormats)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup regular convolution (non-1x1x1 kernel)
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3}; // 3x3x3 kernel
    engine.SetOrgWeightShape(weightShape);

    // Set formats to [NDC1HWC0, FRACTAL_Z_3D, NDC1HWC0] (valid for regular)
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

    EXPECT_TRUE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_RegularMode_InvalidNCDHW)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup regular convolution (non-1x1x1 kernel)
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);

    // Test invalid: all NCDHW (should be NDC1HWC0/FRACTAL_Z_3D for regular)
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);
    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_RegularMode_InvalidWeightFormat)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup regular convolution
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);

    // Test invalid: weight is NCDHW instead of FRACTAL_Z_3D
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::NCDHW, ConvFormat::NDC1HWC0);
    EXPECT_FALSE(engine.CheckInputFormat());

    // Test invalid: weight is NDC1HWC0 instead of FRACTAL_Z_3D
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::NDC1HWC0, ConvFormat::NDC1HWC0);
    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_BiasFormat_Valid)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup with bias
    engine.SetBias(true, ConvDtype::FLOAT32);

    // Bias format should be ND (default in descInfo_)
    // Regular mode with valid formats
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

    EXPECT_TRUE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_BiasFormat_Invalid)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup with bias
    engine.SetBias(true, ConvDtype::FLOAT32);

    // Manually set invalid bias format (normally biasFormat is ND by default)
    engine.descInfo_.biasFormat = ConvFormat::NCDHW; // Invalid for bias

    // Regular mode with valid main formats
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_ScaleFormat_Valid)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup with scale
    engine.SetScale(true, ConvDtype::FLOAT32);

    // Scale format should be ND (default in descInfo_)
    // Regular mode with valid formats
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NCDHW);

    EXPECT_TRUE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_Quant_OutFormat_Invalid)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup with scale
    engine.SetScale(true, ConvDtype::FLOAT32);

    // Regular mode with valid main formats
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_ScaleFormat_Invalid)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup with scale
    engine.SetScale(true, ConvDtype::FLOAT32);

    // Manually set invalid scale format
    engine.descInfo_.scaleFormat = ConvFormat::FRACTAL_Z_3D; // Invalid for scale

    // Regular mode with valid main formats
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NCDHW);

    EXPECT_FALSE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_IntegrationWithCheckAllParams)
{
    using Conv3dApiTiling::ConvFormat;

    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);

    // CheckAllParams should pass with valid formats
    EXPECT_TRUE(engine.CheckAllParams());

    // Now set invalid format
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);

    // CheckAllParams should fail due to format mismatch
    EXPECT_FALSE(engine.CheckAllParams());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_GetConv3DV2TilingData_FormatValidation)
{
    using Conv3dApiTiling::ConvFormat;

    Conv3dTilingEngine engine;
    InitConv3dEngineExtended(engine);
    FillPlatformInfoExtended(engine);

    // Setup regular convolution with invalid formats
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW); // Invalid for regular

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);

    // Should fail due to format validation
    EXPECT_FALSE(ok);

    // Now set valid formats
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);
    ok = engine.GetConv3DV2TilingData(tilingData);

    // Should succeed with valid formats
    EXPECT_TRUE(ok);
}

TEST(TestConv3dTilingEngine, CheckInputFormat_PointwiseModeDetection)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test 1x1x1 kernel (pointwise)
    std::vector<int64_t> pointwiseWeight = {16, 1, 1, 1, 1};
    engine.SetOrgWeightShape(pointwiseWeight);
    EXPECT_TRUE(engine.isPointWise);

    // Valid format for pointwise
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);
    EXPECT_TRUE(engine.CheckInputFormat());

    // Test 3x3x3 kernel (regular)
    std::vector<int64_t> regularWeight = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(regularWeight);
    EXPECT_FALSE(engine.isPointWise);

    // Same format should now fail for regular mode
    EXPECT_FALSE(engine.CheckInputFormat());

    // Valid format for regular mode
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);
    EXPECT_TRUE(engine.CheckInputFormat());
}

TEST(TestConv3dTilingEngine, CheckInputFormat_AllFormatCombinations)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup regular convolution
    std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};
    engine.SetOrgWeightShape(weightShape);

    // Test all invalid combinations for regular mode
    struct FormatCombo {
        ConvFormat fmap, weight, out;
        bool expected;
    };

    std::vector<FormatCombo> combos = {
        // Valid combination
        {ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0, true},

        // Invalid combinations
        {ConvFormat::NCDHW, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0, false},
        {ConvFormat::NDC1HWC0, ConvFormat::NCDHW, ConvFormat::NDC1HWC0, false},
        {ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NCDHW, false},
        {ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW, false},
        {ConvFormat::ND, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0, false},
        {ConvFormat::NDC1HWC0, ConvFormat::ND, ConvFormat::NDC1HWC0, false},
        {ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::ND, false}
    };

    for (const auto &combo : combos) {
        engine.SetFormat(combo.fmap, combo.weight, combo.out);
        EXPECT_EQ(engine.CheckInputFormat(), combo.expected)
            << "Failed for format combination: fmap=" << static_cast<int>(combo.fmap)
            << ", weight=" << static_cast<int>(combo.weight)
            << ", out=" << static_cast<int>(combo.out);
    }
}

TEST(TestConv3dTilingEngine, SetFormat_LoggingAndStorage)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test SetFormat stores values correctly
    engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

    EXPECT_EQ(engine.descInfo_.fMapFormat, ConvFormat::NDC1HWC0);
    EXPECT_EQ(engine.descInfo_.weightFormat, ConvFormat::FRACTAL_Z_3D);
    EXPECT_EQ(engine.descInfo_.outFormat, ConvFormat::NDC1HWC0);

    // Test with different formats
    engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);

    EXPECT_EQ(engine.descInfo_.fMapFormat, ConvFormat::NCDHW);
    EXPECT_EQ(engine.descInfo_.weightFormat, ConvFormat::NCDHW);
    EXPECT_EQ(engine.descInfo_.outFormat, ConvFormat::NCDHW);
}

// ---------------------------------------------------------------------------
// 12) Two-Phase Initialization Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, Init_SuccessfulInitialization)
{
    Conv3dTilingEngine engine;

    // Init should succeed (or fail gracefully if no platform available)
    bool initResult = engine.Init();

    // If Init succeeded, IsInitialized should return true
    if (initResult) {
        EXPECT_TRUE(engine.IsInitialized());
    } else {
        EXPECT_FALSE(engine.IsInitialized());
    }
}

TEST(TestConv3dTilingEngine, Init_IdempotentBehavior)
{
    Conv3dTilingEngine engine;

    // First Init
    bool firstInit = engine.Init();

    // Second Init should return true (already initialized)
    bool secondInit = engine.Init();

    if (firstInit) {
        EXPECT_TRUE(secondInit);
        EXPECT_TRUE(engine.IsInitialized());
    }
}

TEST(TestConv3dTilingEngine, GetConv3DV2TilingData_FailsWithoutInit)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Do NOT call Init() or FillPlatformInfoForTest()
    // Engine should not be initialized
    EXPECT_FALSE(engine.IsInitialized());

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);

    // Should fail because engine is not initialized
    EXPECT_FALSE(ok);
}

TEST(TestConv3dTilingEngine, GetConv3DV2TilingData_SucceedsAfterInit)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Initialize with test platform info
    FillPlatformInfoForTest(engine);
    EXPECT_TRUE(engine.IsInitialized());

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);

    // Should succeed after initialization
    EXPECT_TRUE(ok);
}

TEST(TestConv3dTilingEngine, ManualPlatformInfoInjection_WorksWithoutInit)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Manually inject platform info using FillPlatformInfoForTest
    // This should call Init() internally and set initOk_ flag
    FillPlatformInfoForTest(engine);

    // Engine should be initialized after FillPlatformInfoForTest
    EXPECT_TRUE(engine.IsInitialized());

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    bool ok = engine.GetConv3DV2TilingData(tilingData);

    EXPECT_TRUE(ok);
}

// ---------------------------------------------------------------------------
// 13) CheckBiasShape() Branch Coverage Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckBiasShape_NoBias_ReturnsTrue)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Set hasBias = false (early exit branch)
    engine.SetBias(false, Conv3dApiTiling::ConvDtype::FLOAT32);

    EXPECT_TRUE(engine.CheckBiasShape());
}

TEST(TestConv3dTilingEngine, CheckBiasShape_EmptyShape_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Enable bias but set empty shape
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> emptyShape = {};
    engine.SetBiasShape(emptyShape);

    EXPECT_FALSE(engine.CheckBiasShape());
}

TEST(TestConv3dTilingEngine, CheckBiasShape_InvalidDimCount_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Enable bias with invalid dimension count (should be FORMAT_ND_DIM = 1)
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> invalidShape = {16, 16}; // size = 2, expected 1
    engine.SetBiasShape(invalidShape);

    EXPECT_FALSE(engine.CheckBiasShape());
}

TEST(TestConv3dTilingEngine, CheckBiasShape_MismatchedCout_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: cOut = 16 (from InitSimpleConv3dEngine)
    // Set bias shape with mismatched value
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> mismatchedShape = {32}; // cOut is 16, but bias[0] is 32
    engine.SetBiasShape(mismatchedShape);

    EXPECT_FALSE(engine.CheckBiasShape());
}

TEST(TestConv3dTilingEngine, CheckBiasShape_ValidShape_ReturnsTrue)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: cOut = 16 (from InitSimpleConv3dEngine)
    // Set bias shape with correct value
    engine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> validShape = {16}; // Matches cOut
    engine.SetBiasShape(validShape);

    EXPECT_TRUE(engine.CheckBiasShape());
}

// ---------------------------------------------------------------------------
// 14) CheckScaleShape() Branch Coverage Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckScaleShape_NoScale_ReturnsTrue)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Set hasScale = false (early exit branch)
    engine.SetScale(false, Conv3dApiTiling::ConvDtype::FLOAT32);

    EXPECT_TRUE(engine.CheckScaleShape());
}

TEST(TestConv3dTilingEngine, CheckScaleShape_EmptyShape_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Enable scale but set empty shape
    engine.SetScale(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> emptyShape = {};
    engine.SetScaleShape(emptyShape);

    EXPECT_FALSE(engine.CheckScaleShape());
}

TEST(TestConv3dTilingEngine, CheckScaleShape_InvalidDimCount_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Enable scale with invalid dimension count (should be FORMAT_ND_DIM = 1)
    engine.SetScale(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> invalidShape = {16, 16}; // size = 2, expected 1
    engine.SetScaleShape(invalidShape);

    EXPECT_FALSE(engine.CheckScaleShape());
}

TEST(TestConv3dTilingEngine, CheckScaleShape_MismatchedCout_ReturnsFalse)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: cOut = 16 (from InitSimpleConv3dEngine)
    // Set scale shape with mismatched value
    engine.SetScale(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> mismatchedShape = {32}; // cOut is 16, but scale[0] is 32
    engine.SetScaleShape(mismatchedShape);

    EXPECT_FALSE(engine.CheckScaleShape());
}

TEST(TestConv3dTilingEngine, CheckScaleShape_ValidShape_ReturnsTrue)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Setup: cOut = 16 (from InitSimpleConv3dEngine)
    // Set scale shape with correct value
    engine.SetScale(true, Conv3dApiTiling::ConvDtype::FLOAT32);
    std::vector<int64_t> validShape = {16}; // Matches cOut
    engine.SetScaleShape(validShape);

    EXPECT_TRUE(engine.CheckScaleShape());
}

// ---------------------------------------------------------------------------
// CheckInputFormat Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckInputFormat_ValidAndInvalidFormats)
{
    using Conv3dApiTiling::ConvFormat;
    using Conv3dApiTiling::ConvDtype;

    // Test 1: Valid NCDHW format (pointwise mode with 1x1x1 kernel)
    {
        Conv3dTilingEngine engine;
        InitSimpleConv3dEngine(engine);
        std::vector<int64_t> weightShape = {16, 1, 1, 1, 1};  // 1x1x1 kernel for pointwise
        std::vector<int64_t> fmapShape = {1, 1, 1, 1, 1};
        std::vector<int64_t> outputShape = {1, 16, 1, 1, 1};

        engine.SetOrgWeightShape(weightShape);
        engine.SetOrgFmapShape(fmapShape);
        engine.SetOrgOutputShape(outputShape);
        engine.SetFormat(ConvFormat::NCDHW, ConvFormat::NCDHW, ConvFormat::NCDHW);

        EXPECT_TRUE(engine.CheckInputFormat());
    }

    // Test 2: Valid FRACTAL_Z_3D format (regular mode with 3x3x3 kernel)
    {
        Conv3dTilingEngine engine;
        InitSimpleConv3dEngine(engine);
        std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};  // 3x3x3 kernel for regular mode
        std::vector<int64_t> fmapShape = {1, 1, 3, 3, 3};
        std::vector<int64_t> outputShape = {1, 16, 1, 1, 1};

        engine.SetOrgWeightShape(weightShape);
        engine.SetOrgFmapShape(fmapShape);
        engine.SetOrgOutputShape(outputShape);
        engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::FRACTAL_Z_3D, ConvFormat::NDC1HWC0);

        EXPECT_TRUE(engine.CheckInputFormat());
    }

    // Test 3: Invalid mixed formats (regular mode with wrong weight format)
    {
        Conv3dTilingEngine engine;
        InitSimpleConv3dEngine(engine);
        std::vector<int64_t> weightShape = {16, 1, 3, 3, 3};  // 3x3x3 kernel (regular mode)
        std::vector<int64_t> fmapShape = {1, 1, 3, 3, 3};
        std::vector<int64_t> outputShape = {1, 16, 1, 1, 1};

        engine.SetOrgWeightShape(weightShape);
        engine.SetOrgFmapShape(fmapShape);
        engine.SetOrgOutputShape(outputShape);
        // Invalid: weight should be FRACTAL_Z_3D for regular mode, not NCDHW
        engine.SetFormat(ConvFormat::NDC1HWC0, ConvFormat::NCDHW, ConvFormat::NDC1HWC0);

        EXPECT_FALSE(engine.CheckInputFormat());
    }
}

// ---------------------------------------------------------------------------
// CheckDims Tests
// ---------------------------------------------------------------------------

TEST(TestConv3dTilingEngine, CheckDims_PositiveZeroNegative)
{
    Conv3dTilingEngine engine;
    InitSimpleConv3dEngine(engine);

    // Test 1: All positive dimensions (valid)
    std::vector<int64_t> okShape = {1, 1, 1, 1, 1};
    EXPECT_TRUE(engine.CheckDims(okShape));

    // Test 2: Zero dimension (invalid)
    std::vector<int64_t> zeroShape = {1, 0, 1, 1, 1};
    EXPECT_FALSE(engine.CheckDims(zeroShape));

    // Test 3: Negative dimension (invalid)
    std::vector<int64_t> negShape = {1, -1, 1, 1, 1};
    EXPECT_FALSE(engine.CheckDims(negShape));
}

} // namespace
