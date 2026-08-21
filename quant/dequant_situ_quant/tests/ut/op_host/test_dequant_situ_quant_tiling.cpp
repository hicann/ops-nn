/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "test_cube_util.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/dequant_situ_quant_tiling.h"
#include "../../../op_graph/dequant_situ_quant_proto.h"

using namespace std;
using namespace ge;

class DequantSituQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantSituQuantTilingTest SetUp" << std::endl;
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantSituQuantTilingTest TearDown" << std::endl;
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    }
};

static const string COMPILE_INFO = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                      "Intrinsic_fix_pipe_l0c2out": false,
                      "Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "Intrinsic_data_move_out2l1_nd2nz": false,
                      "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                      "CORE_NUM": 40}
})";

static void SetupPlatform(fe::PlatFormInfos& platform_info, map<string, string>& soc_infos,
                          map<string, string>& aicore_spec, map<string, string>& intrinsics)
{
    GetPlatFormInfos(COMPILE_INFO.c_str(), soc_infos, aicore_spec, intrinsics);
    platform_info.Init();
}

// Test static quant with scalar quant_scale, no bias → TilingKey=10000 (DSQ_STATIC_QUANT_ONE)
TEST_F(DequantSituQuantTilingTest, tiling_static_quant_scalar)
{
    gert::StorageShape x_shape = {{16, 64}, {16, 64}};
    gert::StorageShape ws_shape = {{64}, {64}};
    gert::StorageShape qs_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{16, 32}, {16, 32}};
    gert::StorageShape y_scale_shape = {{16}, {16}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    fe::PlatFormInfos platform_info;
    SetupPlatform(platform_info, soc_infos, aicore_spec, intrinsics);

    optiling::DequantSituQuantCompileInfo compile_info;

    std::string op_type("DequantSituQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    // 7 inputs: x, weight_scale, activation_scale(absent), bias(absent), quant_scale, quant_offset(absent),
    // group_index(absent)
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSituQuant")
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 0, 0, 1, 0, 0}, {1, 1})
                      .InputShapes({&x_shape, &ws_shape, &qs_shape})
                      .OutputShapes({&y_shape, &y_scale_shape})
                      .NodeAttrs({{"beta", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(25.0f)},
                                  {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"quant_type", Ops::NN::AnyValue::CreateFrom<std::string>("static")}})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 10000);
}

// Test dynamic quant without smooth scale, no bias → TilingKey=20000
TEST_F(DequantSituQuantTilingTest, tiling_dynamic_quant)
{
    gert::StorageShape x_shape = {{32, 128}, {32, 128}};
    gert::StorageShape ws_shape = {{128}, {128}};
    gert::StorageShape y_shape = {{32, 64}, {32, 64}};
    gert::StorageShape y_scale_shape = {{32}, {32}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    fe::PlatFormInfos platform_info;
    SetupPlatform(platform_info, soc_infos, aicore_spec, intrinsics);

    optiling::DequantSituQuantCompileInfo compile_info;

    std::string op_type("DequantSituQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSituQuant")
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 0, 0, 0, 0, 0}, {1, 1})
                      .InputShapes({&x_shape, &ws_shape})
                      .OutputShapes({&y_shape, &y_scale_shape})
                      .NodeAttrs({{"beta", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(25.0f)},
                                  {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"quant_type", Ops::NN::AnyValue::CreateFrom<std::string>("dynamic")}})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 20000);
}

// Test INT32 dynamic → TilingKey=30000
TEST_F(DequantSituQuantTilingTest, tiling_int32_dynamic)
{
    gert::StorageShape x_shape = {{32, 6144}, {32, 6144}};
    gert::StorageShape ws_shape = {{6144}, {6144}};
    gert::StorageShape act_shape = {{32}, {32}};
    gert::StorageShape y_shape = {{32, 3072}, {32, 3072}};
    gert::StorageShape y_scale_shape = {{32}, {32}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    fe::PlatFormInfos platform_info;
    SetupPlatform(platform_info, soc_infos, aicore_spec, intrinsics);

    optiling::DequantSituQuantCompileInfo compile_info;

    std::string op_type("DequantSituQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSituQuant")
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 0, 0, 0, 0}, {1, 1})
                      .InputShapes({&x_shape, &ws_shape, &act_shape})
                      .OutputShapes({&y_shape, &y_scale_shape})
                      .NodeAttrs({{"beta", Ops::NN::AnyValue::CreateFrom<float>(4.0f)},
                                  {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(25.0f)},
                                  {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"quant_type", Ops::NN::AnyValue::CreateFrom<std::string>("dynamic")}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 30000);
}
