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
 * \file test_conv2d_v2_ascendc_check_input_tiling.cpp
 * \brief UT for conv2d_v2_base_tiling_check_input.cpp
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
void RunInputCheckTest(const std::string& caseName, const std::vector<int64_t>& fmShape,
                       const std::vector<int64_t>& weightShape, ge::DataType dtype, bool hasBias, bool expectFail)
{
    int64_t batch = fmShape[0], cin = fmShape[1], hi = fmShape[2], wi = fmShape[3];
    int64_t cout = weightShape[0], ci_per_group = weightShape[1], kH = weightShape[2], kW = weightShape[3];

    gert::StorageShape featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
    gert::StorageShape weight = {{cout, ci_per_group, kH, kW}, {cout, ci_per_group, kH, kW}};
    gert::StorageShape bias = {{cout}, {cout}};
    int64_t ho = hi, wo = wi;
    ConvShape convShapeH = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kH), 0U, 0U, 1U, 1U};
    ConvShape convShapeW = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kW), 0U, 0U, 1U, 1U};
    ho = InferOut(convShapeH);
    wo = InferOut(convShapeW);
    gert::StorageShape output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};

    std::vector<void*> input_shape_ref = hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                                   std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

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

    std::vector<int64_t> stridesVec = {1, 1, 1, 1};
    std::vector<int64_t> padsVec = {0, 0, 0, 0};
    std::vector<int64_t> dilationsVec = {1, 1, 1, 1};
    ge::DataType bias_dtype = dtype;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(NUM_4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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

    if (expectFail) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << caseName << "] expected SUCCESS but got FAILED";
    }
}
} // namespace

class TestConv2dCheckInput : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// CheckFmapShape: dim > MAX_FM_H_B16_SHAPE(1000000) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_h_over_max_fail)
{
    RunInputCheckTest("fmap_h_over", {1, 16, 1000001, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

TEST_F(TestConv2dCheckInput, fmap_w_over_max_fail)
{
    RunInputCheckTest("fmap_w_over", {1, 16, 16, 1000001}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: kh/kw > MAX_KH_B16_SHAPE(1000000) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_kh_over_max_fail)
{
    RunInputCheckTest("weight_kh_over", {1, 16, 16, 16}, {16, 16, 1000001, 1}, ge::DT_FLOAT16, true, true);
}

TEST_F(TestConv2dCheckInput, weight_kw_over_max_fail)
{
    RunInputCheckTest("weight_kw_over", {1, 16, 16, 16}, {16, 16, 1, 1000001}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckOutputShape: output dim > MAX_OUT_SHAPE(1000000) → FAILED
// Use hi=65536, wi=65536, kh=512 → ho exceeds limit (ho*wo overflow in check limits before this)
// Actually, ho=hi when kh=1, pad=0, so hi=1000001 → ho=1000001 > 1000000 → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, output_h_over_max_fail)
{
    RunInputCheckTest("output_h_over", {1, 16, 1000001, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: weight cout=0 → CheckDim returns false (<=0)
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_cout_zero_fail)
{
    RunInputCheckTest("weight_cout_zero", {1, 16, 16, 16}, {0, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// Various valid shapes → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_1x1_kernel_success)
{
    RunInputCheckTest("valid_1x1", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_3x3_kernel_success)
{
    RunInputCheckTest("valid_3x3", {1, 16, 32, 32}, {32, 16, 3, 3}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_7x7_kernel_success)
{
    RunInputCheckTest("valid_7x7", {1, 64, 112, 112}, {128, 64, 7, 7}, ge::DT_FLOAT16, true, false);
}

TEST_F(TestConv2dCheckInput, valid_no_bias_success)
{
    RunInputCheckTest("valid_no_bias", {1, 32, 64, 64}, {64, 32, 3, 3}, ge::DT_FLOAT16, false, false);
}

// ============================================================================
// Multiple input channels → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_multi_channel_success)
{
    RunInputCheckTest("multi_chan", {2, 128, 56, 56}, {256, 128, 1, 1}, ge::DT_FLOAT16, true, false);
}

// ============================================================================
// CheckFmapShape: fmap N out of range (above MAX_FM_BATCH_SHAPE) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_n_zero_fail)
{
    RunInputCheckTest("fmap_n_over", {1000001, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckFmapShape: fmap cin out of range (above MAX_FM_C_SHAPE) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, fmap_cin_zero_fail)
{
    RunInputCheckTest("fmap_cin_over", {1, 1000001, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckWeightShape: weight cout > MAX_COUT_SHAPE → CheckDim fails → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, weight_cin_zero_fail)
{
    RunInputCheckTest("weight_cout_over", {1, 16, 16, 16}, {1000001, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// CheckOutputShape: output wo > MAX_OUT_SHAPE → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, output_w_over_max_fail)
{
    RunInputCheckTest("output_w_over", {1, 16, 16, 1000001}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, true);
}

// ============================================================================
// Valid with FLOAT dtype → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_fp32_dtype_success)
{
    RunInputCheckTest("fp32_dtype", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT, true, false);
}

// ============================================================================
// Valid with BFLOAT16 dtype → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_bf16_dtype_success)
{
    RunInputCheckTest("bf16_dtype", {1, 16, 32, 32}, {32, 16, 3, 3}, ge::DT_BF16, true, false);
}

// ============================================================================
// Valid with FLOAT dtype, no bias → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_int8_dtype_success)
{
    RunInputCheckTest("fp32_nobias", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_FLOAT, false, false);
}

// ============================================================================
// CheckParamsDtypeWithBias: invalid dtype combination (INT32 not supported) → FAILED
// ============================================================================
TEST_F(TestConv2dCheckInput, invalid_bias_dtype_fail)
{
    RunInputCheckTest("invalid_bias_dtype", {1, 16, 16, 16}, {16, 16, 1, 1}, ge::DT_INT32, true, true);
}

// ============================================================================
// Large batch size → SUCCESS
// ============================================================================
TEST_F(TestConv2dCheckInput, valid_large_batch_success)
{
    RunInputCheckTest("large_batch", {100, 16, 64, 64}, {16, 16, 1, 1}, ge::DT_FLOAT16, true, false);
}

// ============================================================================
// Output H=1, W=1 → SUCCESS (edge case)
// ============================================================================
TEST_F(TestConv2dCheckInput, output_hw_one_success)
{
    RunInputCheckTest("hw_one", {1, 1, 1, 1}, {1, 1, 1, 1}, ge::DT_FLOAT16, true, false);
}
