/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

struct AdaptiveMaxPool2dTilingTestParam {
    string case_name;

    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> y_shape;
    std::initializer_list<int64_t> output_size;

    ge::DataType data_type;

    uint32_t expected_block_dim;
    uint64_t expected_tiling_key;
    string expected_tiling_data;
};

namespace optiling {
struct AdaptiveMaxPool2dCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    size_t sysWorkspaceSize = 0;
};

} // namespace optiling

class AdaptiveMaxPool2dTilingTest : public testing::TestWithParam<AdaptiveMaxPool2dTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveMaxPool2dTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveMaxPool2dTilingTest TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();

    stringstream ss;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        ss << std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)])) << " ";
    }

    return ss.str();
}

// Platform info + compile info + soc infos bundle for tiling setup
struct PlatformTestBundle {
    fe::PlatFormInfos platform_info;
    optiling::AdaptiveMaxPool2dCompileInfo compile_info;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
};

// Shared tiling context holder — keeps dependent objects alive
struct TilingTestContext {
    gert::KernelRunContextHolder holder;
    std::unique_ptr<unsigned char[]> tiling_data;
    std::unique_ptr<unsigned char[]> workspace_holder;
    // Owned data that TilingContext holds pointers to — must outlive holder
    PlatformTestBundle bundle;
    std::shared_ptr<gert::StorageShape> indices_storage;

    gert::TilingContext* GetTilingContext()
    {
        return holder.GetContext<gert::TilingContext>();
    }
};

// Initialize platform info, compile info, soc/ai_core/intrinsics maps
static void InitPlatformAndCompileInfo(PlatformTestBundle& bundle)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    bundle.platform_info.Init();
    GetPlatFormInfos(compile_info_string.c_str(), bundle.soc_infos, bundle.aicore_spec, bundle.intrinsics);

    string op_type("AdaptiveMaxPool2d");
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()),
                     reinterpret_cast<void*>(&bundle.platform_info)})
            .Outputs({&bundle.compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    auto parseCtx = kernel_holder.GetContext<gert::TilingParseContext>();
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", bundle.soc_infos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", bundle.aicore_spec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", bundle.intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    parseCtx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

// Build tiling context faker and set platform res on the resulting context
static void BuildTilingTestContext(
    TilingTestContext& ctx,
    gert::StorageShape* x_shape,
    gert::StorageShape* y_shape,
    gert::StorageShape* indices_shape,
    const std::vector<int64_t>& output_size,
    ge::DataType data_type)
{
    InitPlatformAndCompileInfo(ctx.bundle);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(tiling_data, nullptr);

    ctx.indices_storage = std::make_shared<gert::StorageShape>(*indices_shape);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaptiveMaxPool2d")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({x_shape})
                      .OutputShapes({y_shape, ctx.indices_storage.get()})
                      .CompileInfo(&ctx.bundle.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&ctx.bundle.platform_info))
                      .NodeInputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(output_size)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", ctx.bundle.soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ctx.bundle.aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", ctx.bundle.intrinsics);

    ctx.holder = std::move(holder);
    ctx.tiling_data = std::move(tiling_data);
    ctx.workspace_holder = std::move(workspace_size_holder);
}

TEST_P(AdaptiveMaxPool2dTilingTest, test_case_adaptive_max_pool2d_tiling)
{
    AdaptiveMaxPool2dTilingTestParam param = GetParam();

    gert::StorageShape x_shape = {param.x_shape, param.x_shape};
    gert::StorageShape y_shape = {param.y_shape, param.y_shape};
    gert::StorageShape indices_shape = {param.y_shape, param.y_shape};
    vector<int64_t> output_size = param.output_size;

    TilingTestContext ctx;
    BuildTilingTestContext(ctx, &x_shape, &y_shape, &indices_shape, output_size, param.data_type);

    string op_type("AdaptiveMaxPool2d");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(ctx.GetTilingContext()), ge::GRAPH_SUCCESS);

    auto tiling_key = ctx.GetTilingContext()->GetTilingKey();
    ASSERT_EQ(tiling_key, param.expected_tiling_key);
}

static AdaptiveMaxPool2dTilingTestParam cases[] = {
    {"test_case_simt_float32",
     {1, 64, 1, 1},
     {1, 64, 1, 1},
     {1, 1},
     ge::DT_FLOAT,
     1,
     0UL,
     "1 64 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 1 1 1 "},
    {"test_case_simt_float16",
     {1, 64, 1, 1},
     {1, 64, 1, 1},
     {1, 1},
     ge::DT_FLOAT16,
     1,
     0UL,
     "1 64 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 1 1 1 "},
    {"test_case_simt_bfloat16",
     {1, 64, 1, 1},
     {1, 64, 1, 1},
     {1, 1},
     ge::DT_BF16,
     1,
     0UL,
     "1 64 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 64 1 1 1 "},
};
INSTANTIATE_TEST_CASE_P(AdaptiveMaxPool2d, AdaptiveMaxPool2dTilingTest, testing::ValuesIn(cases));

// ======================== Tiling Failure Cases ========================

static void RunTilingAndExpect(
    gert::StorageShape* x_shape,
    gert::StorageShape* y_shape,
    gert::StorageShape* indices_shape,
    const std::vector<int64_t>& output_size,
    ge::DataType data_type,
    ge::graphStatus expect_result)
{
    string op_type("AdaptiveMaxPool2d");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);

    TilingTestContext ctx;
    BuildTilingTestContext(ctx, x_shape, y_shape, indices_shape, output_size, data_type);
    EXPECT_EQ(tiling_func(ctx.GetTilingContext()), expect_result);
}

// OP_LOGE_FOR_INVALID_DTYPE: x dtype must be float, float16, or bfloat16
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_unsupported_dtype_int32)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    gert::StorageShape indices_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1}, ge::DT_INT32, ge::GRAPH_FAILED);
}

// OP_LOGE_FOR_INVALID_SHAPEDIM: x shape dim must be 4
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_xshape_not_4d)
{
    gert::StorageShape x_shape = {{1, 64, 3}, {1, 64, 3}};
    gert::StorageShape y_shape = {{1, 64, 1}, {1, 64, 1}};
    gert::StorageShape indices_shape = {{1, 64, 1}, {1, 64, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON: The dimensions N, C, Hi, Wi must be at least 1
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_invalid_shape_zero_n)
{
    gert::StorageShape x_shape = {{0, 64, 3, 3}, {0, 64, 3, 3}};
    gert::StorageShape y_shape = {{0, 64, 1, 1}, {0, 64, 1, 1}};
    gert::StorageShape indices_shape = {{0, 64, 1, 1}, {0, 64, 1, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// OP_LOGE_WITH_INVALID_ATTR: the size of output_size only support 2
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_output_size_len_not_2)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    gert::StorageShape indices_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// OP_LOGE_WITH_INVALID_ATTR: the value of output_size should > 0
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_output_size_le_zero)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    gert::StorageShape indices_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {0, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// indices shape dim must be 4
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_indices_shape_not_4d)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    gert::StorageShape indices_shape = {{1, 64, 1}, {1, 64, 1}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// indices shape must match y shape
TEST(AdaptiveMaxPool2dTilingFail, tiling_fail_indices_shape_mismatch)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 1}, {1, 64, 1, 1}};
    gert::StorageShape indices_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    RunTilingAndExpect(&x_shape, &y_shape, &indices_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}