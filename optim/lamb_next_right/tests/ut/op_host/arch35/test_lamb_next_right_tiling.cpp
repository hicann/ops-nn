/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "optim/lamb_next_right/op_host/arch35/lamb_next_right_tiling_arch35.h"
#include "optim/lamb_next_right/op_kernel/arch35/lamb_next_right_dag.h"
#include "atvoss/broadcast/broadcast_base_struct.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class LambNextRightTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(LambNextRightTilingTest, lamb_next_right_tiling_test_fp32)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{1}, {1}};
    gert::StorageShape in4_shape = {{1}, {1}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextRightCompileInfo compile_info;
    string op_type("LambNextRight");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
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
                      .SetOpType(op_type)
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape})
                      .OutputShapes({&out0_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
TEST_F(LambNextRightTilingTest, lamb_next_right_tiling_test_fp16)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{1}, {1}};
    gert::StorageShape in4_shape = {{1}, {1}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextRightCompileInfo compile_info;
    string op_type("LambNextRight");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
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
                      .SetOpType(op_type)
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape})
                      .OutputShapes({&out0_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
TEST_F(LambNextRightTilingTest, lamb_next_right_tiling_test_dtype_mismatch)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{1}, {1}};
    gert::StorageShape in4_shape = {{1}, {1}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextRightCompileInfo compile_info;
    string op_type("LambNextRight");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
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
                      .SetOpType(op_type)
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape})
                      .OutputShapes({&out0_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(LambNextRightTilingTest, lamb_next_right_tiling_test_invalid_dtype)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{1}, {1}};
    gert::StorageShape in4_shape = {{1}, {1}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextRightCompileInfo compile_info;
    string op_type("LambNextRight");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
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
                      .SetOpType(op_type)
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape})
                      .OutputShapes({&out0_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

namespace {
using EmptyTilingData = Ops::Base::BroadcastBaseTilingData<LambNextRightOp::LambNextRightCompute<float, float>::OpDag>;

void CheckEmptyTilingBuffer(gert::TilingData* data, ge::DataType dtype, ge::graphStatus expected)
{
    gert::StorageShape emptyShape = {{0}, {0}};
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    auto* ws = reinterpret_cast<gert::ContinuousVector*>(workspace.get());
    ASSERT_NE(ws, nullptr);
    optiling::LambNextRightCompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto faker = gert::TilingContextFaker();
    faker.SetOpType("LambNextRight")
        .NodeIoNum(6, 2)
        .InputShapes(std::vector<gert::StorageShape*>(6, &emptyShape))
        .OutputShapes(std::vector<gert::StorageShape*>(2, &emptyShape))
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .TilingData(data)
        .Workspace(ws);
    for (int32_t i = 0; i < 6; ++i) {
        faker.NodeInputTd(i, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 2; ++i) {
        faker.NodeOutputTd(i, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    context->SetBlockDim(7);
    context->SetTilingKey(99);
    context->GetWorkspaceSizes(1)[0] = 123;
    optiling::LambNextRightTiling tiling(context);
    ASSERT_EQ(tiling.DoTiling(), expected);
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetBlockDim(), 1);
        EXPECT_EQ(context->GetTilingKey(), GET_TPL_TILING_KEY(1));
        EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 0);
        ASSERT_NE(data, nullptr);
        EXPECT_EQ(data->GetDataSize(), sizeof(EmptyTilingData));
    } else {
        EXPECT_EQ(context->GetBlockDim(), 7);
        EXPECT_EQ(context->GetTilingKey(), 99);
        EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 123);
        if (data != nullptr) {
            EXPECT_EQ(data->GetDataSize(), 0);
        }
    }
    if (data != nullptr && data->GetData() != nullptr) {
        const auto* bytes = static_cast<const uint8_t*>(data->GetData());
        const size_t initialized = expected == ge::GRAPH_SUCCESS ? sizeof(EmptyTilingData) : 0;
        EXPECT_TRUE(std::all_of(bytes, bytes + initialized, [](uint8_t byte) { return byte == 0; }));
        EXPECT_TRUE(
            std::all_of(bytes + initialized, bytes + data->GetCapacity(), [](uint8_t byte) { return byte == 0x5a; }));
    }
}
} // namespace

TEST_F(LambNextRightTilingTest, empty_tiling_buffer_capacity_boundaries)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (size_t capacity : {size_t(0), size_t(1), sizeof(EmptyTilingData) - 1, sizeof(EmptyTilingData),
                                sizeof(EmptyTilingData) + 16, size_t(2 * 1024 * 1024)}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", capacity=" << capacity);
            auto buffer = gert::TilingData::CreateCap(capacity);
            ASSERT_NE(buffer, nullptr);
            auto* data = reinterpret_cast<gert::TilingData*>(buffer.get());
            std::memset(data->GetData(), 0x5a, capacity);
            CheckEmptyTilingBuffer(data, dtype,
                                   capacity < sizeof(EmptyTilingData) ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS);
        }
    }
}

TEST_F(LambNextRightTilingTest, empty_tiling_rejects_null_buffers)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        CheckEmptyTilingBuffer(nullptr, dtype, ge::GRAPH_FAILED);
        auto buffer = gert::TilingData::CreateCap(sizeof(EmptyTilingData));
        ASSERT_NE(buffer, nullptr);
        auto* data = reinterpret_cast<gert::TilingData*>(buffer.get());
        data->Init(sizeof(EmptyTilingData), nullptr);
        CheckEmptyTilingBuffer(data, dtype, ge::GRAPH_FAILED);
    }
}
