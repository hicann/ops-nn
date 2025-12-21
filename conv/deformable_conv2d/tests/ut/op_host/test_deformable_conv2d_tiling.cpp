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
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class DeformableConv2dTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeformableConv2dTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeformableConv2dTiling TearDown" << std::endl;
    }
};

TEST_F(DeformableConv2dTiling, DeformableConv2dTiling_ascendc_001)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DeformableConv2dCompileInfo {
        int32_t coreNum = 0;
    } compile_info;

    std::string op_type("DeformableConv2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 11, 3, 7}, {1, 11, 3, 7}};
    gert::StorageShape weight = {{7, 1, 1, 7}, {7, 1, 1, 7}};
    gert::StorageShape offsets = {{1, 11, 3, 3}, {1, 11, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    gert::StorageShape deformable_out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x, &weight, &offsets, &bias})
                      .OutputShapes({&out, &deformable_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs(
                          {{"kernel_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"modulated", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(DeformableConv2dTiling, DeformableConv2dTiling_ascendc_002)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DeformableConv2dCompileInfo {
        int32_t coreNum = 0;
    } compile_info;

    std::string op_type("DeformableConv2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 11, 3, 7}, {1, 11, 3, 7}};
    gert::StorageShape weight = {{7, 1, 1, 7}, {7, 1, 1, 7}};
    gert::StorageShape offsets = {{1, 11, 3, 3}, {1, 11, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    gert::StorageShape deformable_out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x, &weight, &offsets, &bias})
                      .OutputShapes({&out, &deformable_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs(
                          {{"kernel_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"modulated", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(DeformableConv2dTiling, DeformableConv2dTiling_ascendc_003)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DeformableConv2dCompileInfo {
        int32_t coreNum = 0;
    } compile_info;

    std::string op_type("DeformableConv2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 11, 3, 7}, {1, 11, 3, 7}};
    gert::StorageShape weight = {{7, 1, 1, 7}, {7, 1, 1, 7}};
    gert::StorageShape offsets = {{1, 11, 3, 3}, {1, 11, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    gert::StorageShape deformable_out = {{1, 7, 11, 3}, {1, 7, 11, 3}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x, &weight, &offsets, &bias})
                      .OutputShapes({&out, &deformable_out})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs(
                          {{"kernel_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"stride", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"deformable_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"modulated", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1);
}