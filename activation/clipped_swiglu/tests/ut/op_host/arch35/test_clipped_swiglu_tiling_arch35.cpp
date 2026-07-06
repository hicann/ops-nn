/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_clipped_swiglu_tiling_arch35.cpp
 * \brief Arch35 (Ascend 950) tiling UT for ClippedSwiglu
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "array_ops.h"
#include "../../../../op_host/clipped_swiglu_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "error_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
struct ClippedSwigluCompileInfo {};
} // namespace optiling

class ClippedSwigluArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ClippedSwigluArch35TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ClippedSwigluArch35TilingTest TearDown" << std::endl; }
};

static const string kCompileInfoStr = R"({
  "hardware_info": {
    "BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 64, "socVersion": "Ascend950"}
})";

static const map<string, string> kSocVersion = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};

struct Arch35TilingTestParam {
    gert::StorageShape xShape;
    gert::StorageShape* groupShape = nullptr;
    gert::StorageShape yShape;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType groupDtype = ge::DT_INT64;
    int64_t dim = -1;
    float alpha = 1.702f;
    float limit = 7.0f;
    float bias = 1.0f;
    bool interleaved = false;
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
};

static void RunArch35TilingTest(const Arch35TilingTestParam& tc)
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfoStr.c_str(), soc_infos, aicore_spec, intrinsics, kSocVersion);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ClippedSwigluCompileInfo compile_info;
    string op_type("ClippedSwiglu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs(
                                 {const_cast<char*>(kCompileInfoStr.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", kSocVersion);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("ClippedSwiglu")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&tc.xShape, tc.groupShape})
                      .OutputShapes({&tc.yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, tc.groupDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.dim)},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(tc.alpha)},
                                  {"limit", Ops::NN::AnyValue::CreateFrom<float>(tc.limit)},
                                  {"bias", Ops::NN::AnyValue::CreateFrom<float>(tc.bias)},
                                  {"interleaved", Ops::NN::AnyValue::CreateFrom<bool>(tc.interleaved)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), tc.expectedStatus);
}

// ========== Normal cases: half (interleaved=false), ungrouped ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_half_ungrouped) { RunArch35TilingTest({{5760}, nullptr, {2880}}); }

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp32_half_ungrouped)
{
    RunArch35TilingTest({{3200, 5760}, nullptr, {1600, 5760}, ge::DT_FLOAT, ge::DT_INT64, 0});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_bf16_half_ungrouped)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_BF16});
}

// ========== Normal cases: interleaved, ungrouped ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_interleaved_ungrouped)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, 7.0f, 1.0f, true});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp32_interleaved_ungrouped)
{
    RunArch35TilingTest({{3200, 5760}, nullptr, {1600, 5760}, ge::DT_FLOAT, ge::DT_INT64, 0, 1.702f, 7.0f, 1.0f, true});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_bf16_interleaved_ungrouped)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_BF16, ge::DT_INT64, -1, 1.702f, 7.0f, 1.0f, true});
}

// ========== Normal cases: half (interleaved=false), grouped ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_half_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_FLOAT16, ge::DT_INT64, 0});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp32_half_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_FLOAT, ge::DT_INT64, 0});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_bf16_half_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_BF16, ge::DT_INT64, 0});
}

// ========== Normal cases: interleaved, grouped ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_interleaved_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_FLOAT16, ge::DT_INT64, 0, 1.702f, 7.0f, 1.0f, true});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp32_interleaved_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_FLOAT, ge::DT_INT64, 0, 1.702f, 7.0f, 1.0f, true});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_bf16_interleaved_grouped)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760}, &gs, {1600, 5760}, ge::DT_BF16, ge::DT_INT64, 0, 1.702f, 7.0f, 1.0f, true});
}

// ========== Normal cases: custom attrs ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_custom_alpha)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.5f});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_custom_bias)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, 7.0f, 0.5f});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_custom_limit)
{
    RunArch35TilingTest({{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, 10.0f});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_positive_dim)
{
    RunArch35TilingTest({{2, 4, 5760}, nullptr, {2, 4, 2880}, ge::DT_FLOAT16, ge::DT_INT64, 2});
}

// ========== Normal: large shape ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_fp16_large_shape)
{
    RunArch35TilingTest({{64, 4096}, nullptr, {32, 4096}, ge::DT_FLOAT16, ge::DT_INT64, 0});
}

// ========== Error cases ==========

TEST_F(ClippedSwigluArch35TilingTest, arch35_x_is_none_wrong)
{
    // To test x shape is nullptr, we set xShape with 0 dims and expect failure
    // when xDims_ <= 0 check triggers
    RunArch35TilingTest(
        {{}, nullptr, {}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, 7.0f, 1.0f, false, ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_x_dim_div2_wrong)
{
    RunArch35TilingTest({{3200, 5761},
                         nullptr,
                         {3200, 2880},
                         ge::DT_FLOAT16,
                         ge::DT_INT64,
                         0,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_x_dtype_wrong)
{
    RunArch35TilingTest(
        {{5760}, nullptr, {2880}, ge::DT_INT8, ge::DT_INT64, -1, 1.702f, 7.0f, 1.0f, false, ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_limit_zero_wrong)
{
    RunArch35TilingTest(
        {{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, 0.0f, 1.0f, false, ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_limit_negative_wrong)
{
    RunArch35TilingTest(
        {{5760}, nullptr, {2880}, ge::DT_FLOAT16, ge::DT_INT64, -1, 1.702f, -1.0f, 1.0f, false, ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_groupindex_dtype_wrong)
{
    gert::StorageShape gs = {{200}};
    RunArch35TilingTest({{3200, 5760},
                         &gs,
                         {1600, 5760},
                         ge::DT_FLOAT16,
                         ge::DT_FLOAT,
                         0,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_groupindex_dims_wrong)
{
    gert::StorageShape gs = {{200, 2}};
    RunArch35TilingTest({{3200, 5760},
                         &gs,
                         {1600, 5760},
                         ge::DT_FLOAT16,
                         ge::DT_INT64,
                         0,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_y_dims_diff_wrong)
{
    RunArch35TilingTest({{3200, 5760},
                         nullptr,
                         {3200, 2880, 2},
                         ge::DT_FLOAT16,
                         ge::DT_INT64,
                         0,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_y_dim_equal_x_dim_div2_wrong)
{
    RunArch35TilingTest({{3200, 5760},
                         nullptr,
                         {3200, 2800},
                         ge::DT_FLOAT16,
                         ge::DT_INT64,
                         0,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}

TEST_F(ClippedSwigluArch35TilingTest, arch35_dim_value_wrong)
{
    RunArch35TilingTest({{3200, 5760},
                         nullptr,
                         {1600, 5760},
                         ge::DT_FLOAT16,
                         ge::DT_INT64,
                         10,
                         1.702f,
                         7.0f,
                         1.0f,
                         false,
                         ge::GRAPH_FAILED});
}
