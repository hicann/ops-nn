/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_grouped_quant_max_tiling.cpp
 * @brief Unit test for GroupedQuantMax tiling
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "platform/platform_info.h"

#include "../../../../op_host/arch35/grouped_quant_max_tiling_arch35.h"

using namespace std;
using namespace ge;

class GroupedQuantMaxTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedQuantMaxTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupedQuantMaxTilingTest TearDown" << std::endl; }
};

// Test that GroupedQuantMax OpImpl is registered correctly
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_op_impl_registered)
{
    std::string op_type("GroupedQuantMax");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);

    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto tiling_parse_func = op_impl->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    std::cout << "GroupedQuantMax OpImpl registered successfully" << std::endl;
}

// Test CheckDtype failure: x dtype not supported
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_invalid_x_dtype)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// Test CheckShape failure: x dim > 8
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_x_dim_exceed_max)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// Test CheckAttrs failure: invalid dstType
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_invalid_dst_type)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(99)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// Test CheckAttrs failure: roundMode mismatch with dstType
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_roundmode_mismatch)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("round")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// Test CheckShape failure: scale dim != 1
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_invalid_scale_dim)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3, 2}, {3, 2});
    gert::StorageShape group_list_shape({3}, {3});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// Test CheckShape failure: groupList dim != 1
TEST_F(GroupedQuantMaxTilingTest, grouped_quant_max_tiling_invalid_grouplist_dim)
{
    std::string op_type("GroupedQuantMax");

    optiling::GroupedQuantMaxCompileInfo compile_info;
    compile_info.ubSize = 262144;

    fe::PlatFormInfos platform_info;
    std::map<std::string, std::string> soc_infos = {
        {"SoCInfo_soc_version", "Ascend950"}, {"SoCInfo_core_num", "24"}, {"SoCInfo_ai_core_num", "24"}};
    std::map<std::string, std::string> aicore_spec = {{"ub_size", "262144"}};

    gert::StorageShape x_shape({6, 128}, {6, 128});
    gert::StorageShape scale_shape({3}, {3});
    gert::StorageShape group_list_shape({3, 1}, {3, 1});
    gert::StorageShape y_shape({6, 128}, {6, 128});
    gert::StorageShape amax_shape({3}, {3});

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape, &group_list_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_TRUE(tiling_context != nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
