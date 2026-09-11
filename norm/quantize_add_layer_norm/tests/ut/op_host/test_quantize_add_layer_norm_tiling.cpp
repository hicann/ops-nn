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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

// must be layout-compatible with optiling::QuantizeAddLayerNormCompileInfo in tiling.h
struct QuantizeAddLayerNormCompileInfo {
    uint32_t aivCoreNum_ = 0;
    uint64_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t vecRegSize_ = 0;
    uint32_t blockSize_ = 0;
    bool isRegbase = false;
};

class QuantizeAddLayerNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantizeAddLayerNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantizeAddLayerNormTiling TearDown" << std::endl; }
};

TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{8192, 5120}, {8192, 5120}};
    gert::StorageShape gamma_shape = {{
                                          5120,
                                      },
                                      {
                                          5120,
                                      }};
    gert::StorageShape out_shape = {{8192, 5120}, {8192, 5120}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                                    &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(65535)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1101);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{40, 11264}, {40, 11264}};
    gert::StorageShape gamma_shape = {{
                                          11264,
                                      },
                                      {
                                          11264,
                                      }};
    gert::StorageShape out_shape = {{40, 11264}, {40, 11264}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                                    &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-65535)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1102);
    // dlog_setlevel(0, 3, 0);
}

// ---------- ascend950 (regbase) tiling cases ----------
// faked Ascend950 platform; quant_shape is the shape used for both scales(idx5) and zero_points(idx6)
// (per-channel -> gamma_shape; per_tensor -> scalar [1]). bias(idx4) defaults to gamma_shape (broadcast);
// pass bias_shape = &input_shape for the elewise-bias keys.
static void RunRegbaseTilingCase(gert::StorageShape& input_shape, gert::StorageShape& gamma_shape,
                                 gert::StorageShape& out_shape, gert::StorageShape& quant_shape, int64_t axis,
                                 ge::DataType xDtype, ge::DataType scaleDtype, uint32_t expectedKey,
                                 uint32_t expectedBlockDim, gert::StorageShape* bias_shape = nullptr)
{
    gert::StorageShape& bias_ref = (bias_shape != nullptr) ? *bias_shape : gamma_shape;
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &bias_ref, &quant_shape,
                                    &quant_shape})
                      .OutputShapes(std::vector<gert::StorageShape*>{&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), expectedKey);
    ASSERT_EQ(tiling_context->GetBlockDim(), expectedBlockDim);
}

// full_load + per_channel(div, axis=-1) + brc bias, fp16 -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8012, 64);
}

// full_load + mul_mode(axis=-65535) + brc bias, fp16 -> key 8002
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_mulmode_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8002,
                         64);
}

// full_load + per_tensor(scalar scale, axis=65535) + brc bias, fp16 -> key 8022
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_pertensor_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape scalar_shape = {{1}, {1}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, scalar_shape, 65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8022,
                         64);
}

// welford + per_channel(div, axis=-1) + brc bias, fp16, large cols -> key 8112
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_welford_perchannel_001)
{
    gert::StorageShape input_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    gert::StorageShape gamma_shape = {{32768}, {32768}};
    gert::StorageShape out_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8112, 64);
}

// full_load + per_channel(div) + elewise bias, fp16 -> key 8011
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8011, 64,
                         &input_shape);
}

// full_load + mul_mode(axis=-65535) + elewise bias, fp16 -> key 8001
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_mulmode_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8001,
                         64, &input_shape);
}

// full_load + per_tensor(axis=65535, scalar scale) + elewise bias, fp16 -> key 8021
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_pertensor_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape scalar_shape = {{1}, {1}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, scalar_shape, 65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8021,
                         64, &input_shape);
}

// welford + per_channel(div) + elewise bias, fp16, large cols -> key 8111
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_welford_perchannel_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    gert::StorageShape gamma_shape = {{32768}, {32768}};
    gert::StorageShape out_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8111, 64,
                         &input_shape);
}

// full_load + per_channel(div) + brc bias, bf16 x with bf16 scales -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_bf16_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_BF16, ge::DT_BF16, 8012, 64);
}

// full_load + per_channel(div) + brc bias, fp32 x with fp32 scales -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_fp32_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT, ge::DT_FLOAT, 8012, 64);
}
