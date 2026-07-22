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
 * \file test_conv2d_v2_ascendc_check_attrs_tiling.cpp
 * \brief UT for conv2d_v2_base_tiling_check_attrs.cpp
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
struct AttrTestParams {
    std::string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<uint32_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    int64_t groups;
    int64_t offsetX;
    std::string dataFormat;
    std::string padMode;
    bool enableHf32;
    ge::DataType dtype;
    bool hasBias;
    bool expectFail;
};

void RunAttrsCheckTest(const AttrTestParams& params)
{
    int64_t batch = params.fmShape[0], cin = params.fmShape[1];
    int64_t hi = params.fmShape[2], wi = params.fmShape[3];
    int64_t cout = params.weightShape[0], ci_per_group = params.weightShape[1];
    int64_t kH = params.weightShape[2], kW = params.weightShape[3];

    int64_t strideH = params.strides[2], strideW = params.strides[3];
    int64_t dilationH = params.dilations[2], dilationW = params.dilations[3];

    std::vector<void*> input_shape_ref = params.hasBias ? std::vector<void*>{nullptr, nullptr, nullptr, nullptr} :
                                                          std::vector<void*>{nullptr, nullptr, nullptr, nullptr};
    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, ci_per_group, kH, kW}, {cout, ci_per_group, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape output = {{batch, cout, hi, wi}, {batch, cout, hi, wi}};

    ConvShape convShapeH = {static_cast<uint64_t>(hi),
                            static_cast<uint64_t>(kH),
                            static_cast<uint64_t>(params.pads[0]),
                            static_cast<uint64_t>(params.pads[1]),
                            static_cast<uint64_t>(dilationH),
                            static_cast<uint64_t>(strideH > 0 ? strideH : 1)};
    ConvShape convShapeW = {static_cast<uint64_t>(wi),
                            static_cast<uint64_t>(kW),
                            static_cast<uint64_t>(params.pads[2]),
                            static_cast<uint64_t>(params.pads[3]),
                            static_cast<uint64_t>(dilationW),
                            static_cast<uint64_t>(strideW > 0 ? strideW : 1)};
    int64_t ho = InferOut(convShapeH);
    int64_t wo = InferOut(convShapeW);
    if (params.padMode == "SAME" || params.padMode == "SAME_UPPER" || params.padMode == "SAME_LOWER") {
        ho = (hi + strideH - 1) / strideH;
        wo = (wi + strideW - 1) / strideW;
    }
    output = {{batch, cout, ho > 0 ? ho : 1, wo > 0 ? wo : 1}, {batch, cout, ho > 0 ? ho : 1, wo > 0 ? wo : 1}};

    input_shape_ref = params.hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                       std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::vector<int64_t> padsVec = {static_cast<int64_t>(params.pads[0]), static_cast<int64_t>(params.pads[1]),
                                    static_cast<int64_t>(params.pads[2]), static_cast<int64_t>(params.pads[3])};

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
                      .NodeInputTd(0, params.dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, params.dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params.dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(params.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(params.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(params.dataFormat)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(params.offsetX)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(params.padMode)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(params.enableHf32)}})
                      .TilingData(tilingDataPtr.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus ret = tiling_func(tiling_context);

    if (params.expectFail) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << params.caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << params.caseName << "] expected SUCCESS but got FAILED";
    }
}
} // namespace

class TestConv2dCheckAttrs : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// CheckStrideLegal: stride <=0 or > MAX_ATTRS_SHAPE(1000000) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, stride_h_zero_fail)
{
    RunAttrsCheckTest({"stride_h_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 0, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

TEST_F(TestConv2dCheckAttrs, stride_w_zero_fail)
{
    RunAttrsCheckTest({"stride_w_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 0},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckDilationLegal: dilation <=0 or > MAX_ATTRS_SHAPE → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, dilation_h_zero_fail)
{
    RunAttrsCheckTest({"dilation_h_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 0, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

TEST_F(TestConv2dCheckAttrs, dilation_w_zero_fail)
{
    RunAttrsCheckTest({"dilation_w_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 0},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckDataFormatLegal: not "NCHW" or "NHWC" → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, invalid_data_format_fail)
{
    RunAttrsCheckTest({"invalid_format",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "INVALID_XYZ",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckPadLegal: invalid pad_mode → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, invalid_pad_mode_fail)
{
    RunAttrsCheckTest({"invalid_padmode",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "INVALID_MODE",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckGroupsLegal: groups <= 0 → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, groups_zero_fail)
{
    RunAttrsCheckTest({"groups_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       0,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckEnableHf32Legal: enable_hf32=true with non-FP32 dtypes → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, hf32_with_fp16_dtype_fail)
{
    RunAttrsCheckTest({"hf32_fp16",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       true,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// Normal valid attrs → SUCCESS (no hf32, valid stride/dilation/groups/format/pad)
// ============================================================================
TEST_F(TestConv2dCheckAttrs, valid_attrs_success)
{
    RunAttrsCheckTest({"valid_attrs",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// PadMode SAME/UPPER/LOWER: auto-calc pads
// ============================================================================
TEST_F(TestConv2dCheckAttrs, pad_mode_same_success)
{
    RunAttrsCheckTest({"padmode_same",
                       {1, 16, 16, 16},
                       {16, 16, 3, 3},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SAME",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

TEST_F(TestConv2dCheckAttrs, pad_mode_valid_success)
{
    RunAttrsCheckTest({"padmode_valid",
                       {1, 16, 16, 16},
                       {16, 16, 3, 3},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "VALID",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// Groups=1 with implicit groups deducing (ci/weightCi) → SUCCESS
// weightShape={16, 1, 1, 1}, cin=16, groups=1 → implicitly groups=16/1=16
// ============================================================================
TEST_F(TestConv2dCheckAttrs, groups_implicit_deduce_success)
{
    RunAttrsCheckTest({"groups_implicit",
                       {1, 16, 16, 16},
                       {16, 1, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// Valid groups=4 → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckAttrs, groups_4_success)
{
    RunAttrsCheckTest({"groups_4",
                       {1, 16, 16, 16},
                       {16, 4, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       4,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// PadMode SAME_UPPER → auto-calc pads
// ============================================================================
TEST_F(TestConv2dCheckAttrs, pad_mode_same_upper_success)
{
    RunAttrsCheckTest({"padmode_upper",
                       {1, 16, 16, 16},
                       {16, 16, 3, 3},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SAME_UPPER",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// CheckOffsetXLegal: offset_x out of range → FAILED
// OFFSET_X_MAX_VALUE=40000000, OFFSET_X_MIN_VALUE=-40000000
// ============================================================================
TEST_F(TestConv2dCheckAttrs, offset_x_above_max_fail)
{
    RunAttrsCheckTest({"offset_x_max",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       40000001,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

TEST_F(TestConv2dCheckAttrs, offset_x_below_min_fail)
{
    RunAttrsCheckTest({"offset_x_min",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       -40000001,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

TEST_F(TestConv2dCheckAttrs, offset_x_zero_success)
{
    RunAttrsCheckTest({"offset_x_zero",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// Groups: groups > 1 with disContinuous → FAILED (path tested via tiling flow)
// Groups: groups > MAX_GROUP_SHAPE → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, groups_exceed_max_fail)
{
    RunAttrsCheckTest({"groups_max",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1000001,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckStrideLegal: stride_n != 1 → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, stride_n_not_one_fail)
{
    RunAttrsCheckTest({"stride_n",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {2, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

TEST_F(TestConv2dCheckAttrs, stride_c_not_one_fail)
{
    RunAttrsCheckTest({"stride_c",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 2, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckDilationLegal: dilation_n != 1 → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, dilation_n_not_one_fail)
{
    RunAttrsCheckTest({"dilation_n",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {2, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

TEST_F(TestConv2dCheckAttrs, dilation_c_not_one_fail)
{
    RunAttrsCheckTest({"dilation_c",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 2, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckStrideLegal: stride_h > MAX_ATTRS_SHAPE → FAILED (above max)
// ============================================================================
TEST_F(TestConv2dCheckAttrs, stride_h_above_max_fail)
{
    RunAttrsCheckTest({"stride_h_max",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1000001, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckDilationLegal: dilation_h > MAX_ATTRS_SHAPE → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, dilation_h_above_max_fail)
{
    RunAttrsCheckTest({"dilation_h_max",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1000001, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckPadLegal: pad out of range (> MAX_ATTRS_SHAPE) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckAttrs, pad_above_max_fail)
{
    RunAttrsCheckTest({"pad_max",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {1000001, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       true});
}

// ============================================================================
// CheckEnableHf32Legal: hf32=true with FP32 dtypes (no bias) → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckAttrs, hf32_with_fp32_success)
{
    RunAttrsCheckTest({"hf32_fp32",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SPECIFIC",
                       true,
                       ge::DT_FLOAT,
                       false,
                       false});
}

// ============================================================================
// PadMode SAME with kernel 3x3 input 7x7: pad calc ends up adding padding
// ============================================================================
TEST_F(TestConv2dCheckAttrs, pad_mode_same_odd_input_success)
{
    RunAttrsCheckTest({"padmode_same_odd",
                       {1, 16, 7, 7},
                       {16, 16, 3, 3},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NCHW",
                       "SAME",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}

// ============================================================================
// CheckDataFormatLegal: NHWC → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckAttrs, data_format_nhwc_success)
{
    RunAttrsCheckTest({"format_nhwc",
                       {1, 16, 16, 16},
                       {16, 16, 1, 1},
                       {0, 0, 0, 0},
                       {1, 1, 1, 1},
                       {1, 1, 1, 1},
                       1,
                       0,
                       "NHWC",
                       "SPECIFIC",
                       false,
                       ge::DT_FLOAT16,
                       true,
                       false});
}
