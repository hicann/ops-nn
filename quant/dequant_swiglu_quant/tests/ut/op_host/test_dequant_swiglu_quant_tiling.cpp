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
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_graph/dequant_swiglu_quant_proto.h"
#include "test_dequant_swiglu_quant_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

class DequantSwigluQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantSwigluQuantTiling SetUp" << std::endl;
	setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantSwigluQuantTiling TearDown" << std::endl;
	unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
    }
};

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_30013)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{96, 3072}, {96, 3072}};
    gert::StorageShape weightScaleShape = {{1, 3072}, {1, 3072}};
    gert::StorageShape activationScaleShape = {{96, 1}, {96, 1}};
    gert::StorageShape quantScaleShape = {{1, 1536}, {1, 1536}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{96, 1536}, {96, 1536}};
    gert::StorageShape sCaleShape = {{96}, {96}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, nullptr })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 30013);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_000)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000100);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_000_x_dtype_bf16)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000000);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_000_x_dtype_bf16_groupIndex_fail)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1, 1}, {1, 1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_spe_000)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{16, 2048}, {16, 2048}};
    gert::StorageShape weightScaleShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape activationScaleShape = {{16, 1}, {16, 1}};
    gert::StorageShape quantScaleShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape groupShape = {{128, 2}, {128, 2}};
    gert::StorageShape yShape = {{16, 1024}, {16, 1024}};
    gert::StorageShape sCaleShape = {{16}, {16}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110000100);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_x_dtype)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_group_dtype)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_x_shape)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 30}, {128, 30}};
    gert::StorageShape weightScaleShape = {{1, 30}, {1, 30}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 15}, {1, 15}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 15}, {128, 15}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_bias)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &xShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_weight_scale_dtype)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_activation_scale_dtype)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_not_support_quant_scale_dtype)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 测试合法的输入的数据类型: x_dtype_int32_bias_dtype_bf16_qs_dtype_fp32
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_bf16_qs_dtype_fp32)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000000);

}

// 测试合法的输入的数据类型: x_dtype_int32_bias_dtype_fp16_qs_dtype_fp32
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp16_qs_dtype_fp32)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100001000);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_fp32_qs_dtype_fp32
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp32_qs_dtype_fp32)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100002000);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_int32_qs_dtype_fp32
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_int32_qs_dtype_fp32)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100003000);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_bf16_qs_dtype_fp16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_bf16_qs_dtype_fp16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000100);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_fp16_qs_dtype_fp16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp16_qs_dtype_fp16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100001100);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_fp32_qs_dtype_fp16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp32_qs_dtype_fp16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100002100);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_int32_qs_dtype_fp16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_int32_qs_dtype_fp16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100003100);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_bf16_qs_dtype_bf16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_bf16_qs_dtype_bf16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000200);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_fp16_qs_dtype_bf16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp16_qs_dtype_bf16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100001200);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_fp32_qs_dtype_bf16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_fp32_qs_dtype_bf16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100002200);

}

// 测试合法的输入的数据类型:x_dtype_int32_bias_dtype_int32_qs_dtype_bf16
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_int32_qs_dtype_bf16)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100003200);

}

// 非法类型bias
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_illegal)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 测试错误的bias shape
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_shape_illegal)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2049}, {1, 2049}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 测试swiglumode不为0或1的情况
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_swiglumode_illegal)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(5)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

}

// 测试有bias但speGroupType为true的情况
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_bias_speGroupType_true)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2049}, {1, 2049}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1, 1}, {1, 1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 测试swiglu_mode为1但speGroupType为true的情况
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_swiglumode_1_speGroupType_true)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{1, 2049}, {1, 2049}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1, 1}, {1, 1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_111)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
    EXPECT_NE(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize())
                  .find("60 718 359 29 359 3 3 1 1 1"), // 60 718 359 30 359 2 2 2 1 0 2 0 3
              string::npos);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_011)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 0, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 11);
    EXPECT_NE(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize())
                  .find("60 718 359 29 359 3 3 1 1 1"),   // 60 718 359 30 359 2 2 2 1 0 2 0 3
              string::npos);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_101)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 0, 0, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 101);
    EXPECT_NE(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize())
                  .find("60 718 359 29 359 3 3 1 1 1"),  //60 718 359 30 359 2 2 2 1 0 2 0 3
              string::npos);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_110)
{
    gert::StorageShape xShape = {{11, 5800}, {11, 5800}};
    gert::StorageShape wScaleShape = {{5800}, {5800}};
    gert::StorageShape aScaleShape = {{11}, {11}};
    gert::StorageShape qScaleShape = {{2900}, {2900}};
    gert::StorageShape yShape = {{11, 2900}, {11, 2900}};
    gert::StorageShape scaleShape = {{11}, {11}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 0})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                  {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
    EXPECT_NE(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize())
                  .find("11 5800 2900 3 2900 4 4 0 1 0"),  // 11 5800 2900 3 2900 4 4 0 1 0 2 0 1
              string::npos);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_x_dtype_fail)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_x_shape_fail_0)
{
    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_x_shape_fail_1)
{
    gert::StorageShape xShape = {{4, 15, 1, 717}, {4, 15, 1, 717}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_x_shape_fail_2)
{
    gert::StorageShape xShape = {{4, -15, 1, 718}, {4, -15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_w_scale_no_exist_fail)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 0, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_w_scale_shape_fail_0)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{2, 1, 718}, {2, 1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_w_scale_shape_fail_1)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{2, 718}, {2, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{2, 359}, {2, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_w_scale_shape_fail_2)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{2, 718}, {2, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 0})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_w_scale_fail_3)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 720}, {1, 720}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 0, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_1111_last_dim)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1111);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_1111_nlast_dim)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape yShape = {{2, 15, 1, 718}, {2, 15, 1, 718}};
    gert::StorageShape scaleShape = {{2, 15, 1}, {2, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 0})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_101110_error)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &qScaleShape, &groupIndexShape, &biasShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_activate_scale_type_fail)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_bias_type_fail)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_quant_scale_type_fail)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_dst_type_error)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(20)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_v35_round_mode_error)
{
    gert::StorageShape xShape = {{4, 15, 1, 718}, {4, 15, 1, 718}};
    gert::StorageShape wScaleShape = {{1, 718}, {1, 718}};
    gert::StorageShape aScaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape biasShape = {{1, 718}, {1, 718}};
    gert::StorageShape qScaleShape = {{1, 359}, {1, 359}};
    gert::StorageShape yShape = {{4, 15, 1, 359}, {4, 15, 1, 359}};
    gert::StorageShape scaleShape = {{4, 15, 1}, {4, 15, 1}};
    gert::StorageShape groupIndexShape = {{1}, {1}};
    std::string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910_95" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("DequantSwigluQuant")
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 0, 1})
                      .InputShapes({&xShape, &wScaleShape, &aScaleShape, &biasShape, &qScaleShape, &groupIndexShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rinttt")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement2_activation_scale_is_None)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, nullptr, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    // .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000100);

}


TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_quant_scale_H)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape quantOffsetShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, &quantOffsetShape, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000100);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_quant_scale_1)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1,}, {1,}};
    gert::StorageShape quantOffsetShape = {{1,}, {1,}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, &quantOffsetShape, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100000100);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_quant_scale_nullptr)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape quantOffsetShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, nullptr, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    // .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    // .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_quant_scale_wrong_shape)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 512}, {1, 512}};
    gert::StorageShape quantOffsetShape = {{1, 512}, {1, 512}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, &quantOffsetShape, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_qs_qo_different_shape)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1}, {1, 1}};
    gert::StorageShape quantOffsetShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, &quantOffsetShape, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_qo_nullptr)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape quantOffsetShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    // .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

}

TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_requirement3_qo_qs_different_type)
{


    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 48, "socVersion": "Ascend910B"}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct SwiGluCompileInfo {};
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{1, 2048}, {1, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape quantScaleShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape quantOffsetShape = {{1, 1024}, {1, 1024}};
    gert::StorageShape groupShape = {{1}, {1}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, nullptr, &quantScaleShape, &quantOffsetShape, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();
    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

}


// 测试合法的输入的数据类型: x_dtype_int32_bias_dtype_bf16_qs_dtype_fp32
TEST_F(DequantSwigluQuantTiling, dequant_swiglu_quant_tiling_x_dtype_int32_bias_dtype_bf16_qs_dtype_fp32_more_expert_fewer_tokens)
{
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48, "socVersion": "Ascend910B"}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;

    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version = { { "Short_SoC_version", "Ascend910B" } };
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::DequantSwigluQuantCompileInfo compile_info;

    std::string op_type("DequantSwigluQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({ const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info) })
            .Outputs({ &compile_info })
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
        intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape xShape = {{128, 2048}, {128, 2048}};
    gert::StorageShape weightScaleShape = {{32, 2048}, {32, 2048}};
    gert::StorageShape activationScaleShape = {{128, 1}, {128, 1}};
    gert::StorageShape biasShape = {{32, 2048}, {32, 2048}};
    gert::StorageShape quantScaleShape = {{32, 1024}, {32, 1024}};
    gert::StorageShape groupShape = {{32}, {32}};
    gert::StorageShape yShape = {{128, 1024}, {128, 1024}};
    gert::StorageShape sCaleShape = {{128}, {128}};
    auto holder = gert::TilingContextFaker()
                    .SetOpType("DequantSwigluQuant")
                    .NodeIoNum(7, 2)
                    .IrInstanceNum({ 1, 1, 1, 1, 1, 1, 1 })
                    .InputShapes({ &xShape, &weightScaleShape, &activationScaleShape, &biasShape, &quantScaleShape, nullptr, &groupShape })
                    .OutputShapes({ &yShape, &sCaleShape })
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(6, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                {"activate_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                {"swiglu_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(7.0)},
                                {"glu_alpha", Ops::NN::AnyValue::CreateFrom<float>(1.702)},
                                {"glu_bias", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();


    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110000000);

}