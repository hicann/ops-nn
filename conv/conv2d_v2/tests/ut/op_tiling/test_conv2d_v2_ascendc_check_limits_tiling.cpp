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
 * \file test_conv2d_v2_ascendc_check_limits_tiling.cpp
 * \brief UT for conv2d_v2_base_tiling_check_limits.cpp
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace ut_util;
using namespace conv_tiling_utils;

namespace {
struct CheckLimitsParams {
    std::string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    ge::DataType dtype;
    uint32_t isHasBias;
    uint32_t groups;
    std::string padMode;
    bool expectDmaFallback;
    bool isErrorCaseFlag;
    std::string format;
};

void RunConv2DTiling(const CheckLimitsParams& params)
{
    bool hasBias = params.isHasBias == 1;
    uint32_t padu = params.pads[0], padd = params.pads[1], padl = params.pads[2], padr = params.pads[3];
    uint32_t strideH = params.strides[0], strideW = params.strides[1];
    uint32_t dilationH = params.dilations[0], dilationW = params.dilations[1];

    int64_t cout = params.weightShape[0], kH = params.weightShape[2], kW = params.weightShape[3];
    int64_t batch = params.fmShape[0], cin = params.fmShape[1], hi = params.fmShape[2], wi = params.fmShape[3];

    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, cin / params.groups, kH, kW}, {cout, cin / params.groups, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape output = {{batch, cout, hi, wi}, {batch, cout, hi, wi}};

    // Recalculate output with actual params
    ConvShape convShapeH = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kH), padu, padd, dilationH, strideH};
    ConvShape convShapeW = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kW), padl, padr, dilationW, strideW};
    int64_t ho = InferOut(convShapeH);
    int64_t wo = InferOut(convShapeW);
    if (ho > 0 && wo > 0) {
        output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};
    }

    if (params.format == "NHWC") {
        fmapFormat = ge::FORMAT_NHWC;
        weightFormat = ge::FORMAT_HWCN;
        outputFormat = ge::FORMAT_NHWC;
        featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
        weight = {{kH, kW, cin / params.groups, cout}, {kH, kW, cin / params.groups, cout}};
        if (ho > 0 && wo > 0) {
            output = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
        }
    }

    std::vector<void*> input_shape_ref = hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                                   std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::vector<int64_t> stridesVec = params.format == "NHWC" ? std::vector<int64_t>{1, strideH, strideW, 1} :
                                                                std::vector<int64_t>{1, 1, strideH, strideW};
    std::vector<int64_t> padsVec = {static_cast<int64_t>(padu), static_cast<int64_t>(padd), static_cast<int64_t>(padl),
                                    static_cast<int64_t>(padr)};
    std::vector<int64_t> dilationsVec = params.format == "NHWC" ? std::vector<int64_t>{1, dilationH, dilationW, 1} :
                                                                  std::vector<int64_t>{1, 1, dilationH, dilationW};

    std::string op_type = "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    std::string compile_info_string = R"({"hardware_info":
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    ge::DataType bias_dtype = params.dtype;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(NUM_4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, params.dtype, fmapFormat, fmapFormat)
                      .NodeInputTd(1, params.dtype, weightFormat, weightFormat)
                      .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params.dtype, outputFormat, outputFormat)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(params.format)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(params.padMode)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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

    ge::graphStatus ret = tiling_func(tiling_context);

    if (params.isErrorCaseFlag) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case " << params.caseName << " expected FAILED but got SUCCESS";
        return;
    }

    EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case " << params.caseName << " expected SUCCESS but got FAILED";
    if (ret != ge::GRAPH_SUCCESS) {
        return;
    }

    uint64_t tilingKey = tiling_context->GetTilingKey();
    bool dma_flag = (tilingKey & 0x4000) >> NUM_14;

    if (params.expectDmaFallback) {
        EXPECT_TRUE(dma_flag) << "Case " << params.caseName << " expected DMA flag but it was not set";
    } else {
        EXPECT_FALSE(dma_flag) << "Case " << params.caseName << " expected no DMA flag but it was set";
    }
}
} // namespace

class TestConv2dCheckLimits : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// ============================================================================
// CheckLoad3DLimits: stride overflow → DMA fallback
// LOAD3D_MAX_STRIDE_H_W = 63, stride=64 should fail load3d and fall back to DMA
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_load3d_stride_h_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"strideH_overflow",
                                      {1, 16, 128, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {64, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_stride_w_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"strideW_overflow",
                                      {1, 16, 16, 128},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 64},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// CheckLoad3DLimits: pad overflow → DMA fallback
// LOAD3D_MAX_PAD = 255, pad=256 should fail load3d and fall back to DMA
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_load3d_pad_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"padTop_overflow",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {256, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_pad_bottom_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"padBottom_overflow",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {0, 256, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_pad_left_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"padLeft_overflow",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 256, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_pad_right_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"padRight_overflow",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 256},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// CheckLoad3DLimits: dilation overflow → DMA fallback
// LOAD3D_MAX_DILATION_H_W = 255, dilation=256 should fail load3d and fall back to DMA
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_load3d_dilation_h_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"dilationH_overflow",
                                      {1, 16, 512, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {256, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_dilation_w_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"dilationW_overflow",
                                      {1, 16, 16, 512},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 256},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// CheckLoad3DLimits: kh/kw overflow → DMA fallback
// LOAD3D_MAX_FILTER_H_W = 511, kh=512 should fail load3d and fall back to DMA
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_load3d_kh_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"kh_overflow",
                                      {1, 16, 1024, 16},
                                      {16, 16, 512, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

TEST_F(TestConv2dCheckLimits, check_load3d_kw_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"kw_overflow",
                                      {1, 16, 16, 1024},
                                      {16, 16, 1, 512},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// CheckInstructionLimits: fixpipe loop2 dst stride overflow → GRAPH_FAILED
// ho*wo > MAX_32_BIT_NUM(4294967295) when NCHW format
// hi=65536, wi=65536, no_pad, stride=1, dilation=1: ho=65536, wo=65536, ho*wo=4294967296
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_instr_fixpipe_overflow_failed)
{
    RunConv2DTiling(CheckLimitsParams{"fixpipe_overflow",
                                      {1, 16, 65536, 65536},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      0,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      true,
                                      "NCHW"});
}

// ============================================================================
// CheckInstructionLimits: datacopy src stride overflow → GRAPH_FAILED
// hi*wi*dtypeSize > MAX_40_BIT_NUM(1099511627775) when !DMA and NCHW
// With FLOAT(dtypeSize=4): hi=1000000, wi=275000 → 1000000*275000*4 = 1.1e12 > 1.099e12
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_instr_datacopy_overflow_failed)
{
    RunConv2DTiling(CheckLimitsParams{"datacopy_overflow",
                                      {1, 16, 1000000, 275000},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT,
                                      0,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      true,
                                      "NCHW"});
}

// ============================================================================
// CheckL1SizeLimitsKernelFullLoad: L1 overflow → DMA fallback
// With FLOAT16, kh=511, kw=4: weightUsedL1Size = ConvAlignB(32*511*4*16*2, 32) ≈ 2M > 512K
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_l1_size_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"l1size_overflow",
                                      {1, 16, 1024, 16},
                                      {16, 16, 511, 4},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// Borderline valid: params at max allowed values → SUCCESS, no DMA fallback
// pad=255, stride=63 at max, small kernel → should pass CheckLoad3DLimits and CheckL1SizeLimitsKernelFullLoad
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_borderline_valid_no_dma)
{
    RunConv2DTiling(CheckLimitsParams{"borderline_valid",
                                      {1, 16, 256, 16},
                                      {16, 16, 1, 1},
                                      {255, 0, 0, 0},
                                      {63, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// Normal success case: no overflow, no DMA fallback
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_normal_success_no_dma)
{
    RunConv2DTiling(CheckLimitsParams{"normal_success",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// CheckLoad3DLimits: load3dPosk > MAX_16_BIT_NUM(65535) → GRAPH_FAILED
// kh*kw*k0 > 65535, both load3d and DMA fail → return FAILED
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_load3d_posk_overflow_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"posk_overflow",
                                      {1, 16, 1024, 16},
                                      {16, 16, 100, 41},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      true,
                                      "NCHW"});
}

// ============================================================================
// CheckInstructionLimits: NHWC NCHW success path → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_nhwc_format_success_no_dma)
{
    RunConv2DTiling(CheckLimitsParams{"nhwc_success",
                                      {1, 16, 16, 16},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      false,
                                      "NHWC"});
}

// ============================================================================
// CheckInstructionLimits: NHWC with L0C2GM overflow → GRAPH_FAILED
// wo*co = 65536*100000 = 6553600000 > MAX_32_BIT_NUM(4294967295) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_nhwc_l0c2gm_overflow_fail)
{
    RunConv2DTiling(CheckLimitsParams{"nhwc_l0c2gm_fail",
                                      {1, 1, 65536, 65536},
                                      {100000, 1, 1, 1},
                                      {0, 0, 0, 0},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      0,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      true,
                                      "NHWC"});
}

// ============================================================================
// CheckInstructionLimits: strideDilation large → DMA fallback
// stride=64, dilation=64 → CheckLoad3DLimits fails for stride
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_large_stride_dilation_dma_fallback)
{
    RunConv2DTiling(CheckLimitsParams{"large_stride_dil",
                                      {1, 16, 256, 512},
                                      {16, 16, 1, 1},
                                      {0, 0, 0, 0},
                                      {64, 1},
                                      {64, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      true,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// Normal large batch → SUCCESS, no DMA
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_large_batch_success)
{
    RunConv2DTiling(CheckLimitsParams{"large_batch",
                                      {16, 32, 56, 56},
                                      {64, 32, 3, 3},
                                      {1, 1, 1, 1},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      1,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      false,
                                      "NCHW"});
}

// ============================================================================
// Check with no_bias path → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckLimits, check_no_bias_success)
{
    RunConv2DTiling(CheckLimitsParams{"no_bias",
                                      {1, 32, 64, 64},
                                      {64, 32, 3, 3},
                                      {1, 1, 1, 1},
                                      {1, 1},
                                      {1, 1},
                                      ge::DT_FLOAT16,
                                      0,
                                      1,
                                      "SPECIFIC",
                                      false,
                                      false,
                                      "NCHW"});
}
