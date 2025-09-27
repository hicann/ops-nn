/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_backprop_input_v2_tiling.cpp
 * \brief
 */
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "graph/graph.h"
#define private public
#define protected public
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../../common/op_host/op_tiling/platform_util.h"
#include "test_cube_util.h"

#define SUCCESS 0

using namespace std;
using namespace ge;

namespace{
extern std::string GetTestSuiteName();
extern std::string GetTestCaseName();

struct Conv3DBpInputV2TilingTestParam {
    string case_name;
    string soc_version;
    string short_soc_version;
    string compile_info;
    string dtype;

    std::initializer_list<int64_t> input_size;
    std::initializer_list<int64_t> filter_ori_shape;
    std::initializer_list<int64_t> filter_shape;
    std::initializer_list<int64_t> out_backprop_ori_shape;
    std::initializer_list<int64_t> out_backprop_shape;
    std::initializer_list<int64_t> y_ori_shape;
    std::initializer_list<int64_t> y_shape;

    ge::Format input_size_format;
    ge::Format filter_ori_format;
    ge::Format filter_format;
    ge::Format out_backprop_ori_format;
    ge::Format out_backprop_format;
    ge::Format y_ori_format;
    ge::Format y_format;

    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string data_format;
    std::string padding;
    int64_t _op_impl_mode_enum;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
};

class Conv3DBackpropInputV2TilingRunTime2 : public testing::TestWithParam<Conv3DBpInputV2TilingTestParam> {};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

static void TestOneParamCase(const Conv3DBpInputV2TilingTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;

    std::string bank_path;
    const char* optype = "Conv3DBackpropInputV2";
    gert::TilingContext* context = nullptr;

    gert::StorageShape input_size = {param.input_size, param.input_size};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_ori_shape, param.y_shape});
    std::vector<void*> output_shapes_ref(1);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(param.compile_info.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    map<string, string> soc_version_infos = {
        {"SoC_version", param.soc_version}, {"Short_SoC_version", param.short_soc_version}};

    std::string op_type("Conv3DBackpropInputV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    if (param.parse_result) {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_FAILED);
        return;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto test_dtype = ge::DT_FLOAT16;
    if (param.dtype == "bfloat16") {
        test_dtype = ge::DT_BF16;
    } else if (param.dtype == "float32") {
        test_dtype = ge::DT_FLOAT;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_size, &filter_shape, &out_backprop_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<void*>(&platform_info))
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)},
                           {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(param._op_impl_mode_enum)}})
                      .NodeInputTd(0, test_dtype, param.input_size_format, param.input_size_format)
                      .NodeInputTd(1, test_dtype, param.filter_ori_format, param.filter_format)
                      .NodeInputTd(2, test_dtype, param.out_backprop_ori_format, param.out_backprop_format)
                      .NodeOutputTd(0, test_dtype, param.y_ori_format, param.y_format)
                      .CompileInfo(&compile_info)
                      .Workspace(workspace)
                      .TilingData(tiling_data.get())
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();

    if (param.tiling_result) {
        ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
        return;
    }
    auto tiling_key = tiling_context->GetOutputPointer<uint64_t>(0);
    auto block_dim = tiling_context->GetOutputPointer<uint32_t>(1);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << ">>>>>tilingData<<<<<" << tiling_data_result << std::endl;
    ASSERT_EQ(*tiling_key, param.tiling_key);
    ASSERT_EQ(*block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

Conv3DBpInputV2TilingTestParam general_20_core_num_cases_params[] = {
    {"Conv3ddx_whitebox_3",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {2, 2, 2, 17, 15},
     {16, 1, 16, 16},
     {1, 10, 30, 30, 15},
     {1, 10, 1, 30, 30, 16},
     {1, 20, 60, 60, 17},
     {1, 20, 2, 60, 60, 16},
     ge::FORMAT_ND,
     ge::FORMAT_DHWCN,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     {1, 2, 2, 2, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     "",
     0,
     true,
     true,
     20,
     513,
     "1 1 1 1 1 1 20 0 1 17 15 1 2 1 2 16 4 10 30 30 20 60 60 2 2 2 1 2 2 2 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 2 2 1 16 "
     "1 2 1 1 256 64 32 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 3600 0 32 0 "},
    {"conv3d_dx_07",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {256, 512, 1, 1, 1},
     {32, 16, 16, 16},
     {1, 256, 17, 128, 128},
     {1, 17, 16, 128, 128, 16},
     {1, 512, 17, 128, 128},
     {1, 17, 32, 128, 128, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     514,
     "1 1 1 1 1 1 20 0 1 512 256 16 32 16 32 16 4 17 128 128 17 128 128 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 "
     "1 2 1 1 256 16 8 1 1 256 64 128 1 1 1 1 1 4 4 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_dx_11",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {256, 128, 1, 1, 1},
     {8, 16, 16, 16},
     {1, 256, 9, 128, 128},
     {1, 9, 16, 128, 128, 16},
     {1, 128, 9, 128, 128},
     {1, 9, 8, 128, 128, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     514,
     "1 1 1 1 1 1 20 0 1 128 256 16 8 16 8 16 4 9 128 128 9 128 128 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 "
     "1 1 256 16 8 1 1 256 64 128 1 1 1 1 1 4 4 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_dx_19",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {512, 512, 3, 3, 3},
     {864, 32, 16, 16},
     {1, 512, 5, 32, 32},
     {1, 5, 32, 32, 32, 16},
     {1, 512, 7, 32, 32},
     {1, 7, 32, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     512,
     "1 1 1 1 1 1 20 0 1 512 512 32 32 32 32 16 4 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 2 1 1 1 1 1 1 1 2 2 1 2 2 "
     "1 512 32 8 1 1 256 48 128 1 1 1 1 1 36 9 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_dx_26",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {8, 512, 3, 3, 3},
     {864, 1, 16, 16},
     {1, 8, 5, 32, 32},
     {1, 5, 1, 32, 32, 16},
     {1, 512, 7, 32, 32},
     {1, 7, 32, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     512,
     "1 1 1 1 1 1 20 0 1 512 8 1 32 1 32 16 4 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 2 1 1 1 1 1 1 1 2 2 1 2 2 1 16 "
     "1 8 1 1 256 48 128 1 1 1 1 1 3 3 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_backprop_input_net_ID4565_bf16_7",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {1280, 1280, 3, 1, 1},
     {240, 80, 16, 16},
     {1, 1280, 25, 5, 8},
     {1, 25, 80, 5, 8, 16},
     {1, 1280, 25, 5, 8},
     {1, 25, 80, 5, 8, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {1, 1, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     514,
     "1 1 1 1 1 1 20 0 1 1280 1280 80 80 80 80 16 4 25 5 8 25 5 8 3 1 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 1 1 1 2 2 1 2 2 "
     "1 1280 80 16 1 1 48 64 256 1 1 1 1 1 20 4 1 1 1 0 0 0 0 1 0 48 0 256 0 "},
    {"conv3d_backprop_input_net_ID4565_bf16_7_DN",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {1280, 1280, 1, 1, 1},
     {80, 80, 16, 16},
     {25, 1280, 1, 5, 8},
     {25, 1, 80, 5, 8, 16},
     {25, 1280, 1, 5, 8},
     {25, 1, 80, 5, 8, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     514,
     "1 1 1 1 1 1 20 0 25 1280 1280 80 80 80 80 16 4 1 5 8 1 5 8 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 2 1 "
     "1280 80 16 1 1 48 64 256 1 1 1 1 1 20 4 1 1 1 0 0 0 0 1 0 48 0 256 0 "},
    {"conv3d_dx_19_fp32",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {512, 512, 3, 3, 3},
     {1728, 32, 16, 8},
     {1, 512, 5, 32, 32},
     {1, 5, 64, 32, 32, 8},
     {1, 512, 7, 32, 32},
     {1, 7, 64, 32, 32, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     14,
     0,
     "1 1 2 1 1 7 14 0 1 512 512 64 64 64 64 8 3 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 2 1 1 1 1 1 1 1 2 2 1 2 2 1 "
     "512 64 64 1 1 128 24 256 1 1 1 1 1 54 6 1 1 1 0 0 0 0 1 0 512 0 512 0 "},
    {"conv3d_dx_050_fp32",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {64, 3, 7, 7, 7},
     {343, 4, 16, 8},
     {4, 64, 6, 112, 112},
     {4, 6, 8, 112, 112, 8},
     {4, 3, 17, 229, 229},
     {4, 17, 1, 229, 229, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     512,
     "1 1 1 1 1 1 20 0 4 3 64 8 1 8 1 8 3 6 112 112 17 229 229 7 7 7 1 2 2 2 0 0 0 0 0 0 6 6 6 6 6 1 1 1 2 2 1 2 2 1 "
     "64 8 1 1 1 1024 8 16 1 1 1 1 1 49 196 1 1 1 0 0 0 0 1 0 10534 0 8 0 "},
    {"conv3d_dx_26_output_format_NCDHW",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {8, 512, 3, 3, 3},
     {864, 1, 16, 16},
     {1, 8, 5, 32, 32},
     {1, 5, 1, 32, 32, 16},
     {1, 512, 7, 32, 32},
     {1, 7, 32, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     512,
     "1 1 1 1 1 1 20 0 1 512 8 1 32 1 32 16 4 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 2 1 1 1 1 1 1 1 2 2 1 2 2 1 16 "
     "1 8 1 1 256 48 128 1 1 1 1 1 3 3 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_dx_bs16_net_ID_11",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {512, 512, 3, 3, 3},
     {864, 32, 16, 16},
     {16, 512, 20, 16, 16},
     {16, 20, 32, 16, 16, 16},
     {16, 512, 22, 18, 18},
     {16, 22, 32, 18, 18, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     512,
     "1 1 1 1 1 1 20 0 16 512 512 32 32 32 32 16 4 20 16 16 22 18 18 3 3 3 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 1 1 1 2 2 1 2 "
     "2 1 512 32 16 1 1 128 48 256 1 1 1 1 1 69 3 1 1 1 0 0 0 0 1 0 126 0 256 0 "}};

static Conv3DBpInputV2TilingTestParam general_24_core_num_cases_params[] = {
    {"Conv3d_bp_input_binary_bf16",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {2, 2, 2, 256, 64},
     {128, 4, 16, 16},
     {64, 2, 15, 15, 64},
     {64, 2, 4, 15, 15, 16},
     {64, 3, 16, 16, 256},
     {64, 3, 16, 16, 16, 16},
     ge::FORMAT_ND,
     ge::FORMAT_DHWCN,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     "",
     0,
     true,
     true,
     24,
     513,
     "1 1 1 1 1 1 24 0 64 256 64 4 16 4 16 16 4 2 15 15 3 16 16 2 2 2 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 2 2 1 "
     "64 4 8 1 1 256 64 128 1 1 1 1 1 4 4 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"conv3d_dx_17",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {512, 512, 1, 1, 1},
     {32, 32, 16, 16},
     {1, 512, 5, 32, 32},
     {1, 5, 32, 32, 32, 16},
     {1, 512, 5, 32, 32},
     {1, 5, 32, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     514,
     "1 1 1 1 1 1 24 0 1 512 512 32 32 32 32 16 4 5 32 32 5 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 1 "
     "1 512 32 8 1 1 256 64 128 1 1 1 1 1 4 8 1 1 1 0 0 0 0 1 0 256 0 128 0 "},
    {"net_ID_5_m_factor",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {128, 128, 3, 3, 3},
     {216, 8, 16, 16},
     {8, 128, 20, 64, 64},
     {8, 20, 8, 64, 64, 16},
     {8, 128, 22, 66, 66},
     {8, 22, 8, 66, 66, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     512,
     "1 1 1 1 1 1 24 0 8 128 128 8 8 8 8 16 4 20 64 64 22 66 66 3 3 3 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 1 1 1 2 2 1 2 2 1 "
     "128 8 8 1 1 256 48 128 1 1 1 1 1 24 6 1 1 1 0 0 0 0 1 0 1452 0 128 0 "},
    {"net_ID_16_m_non_factor",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {512, 1365, 1, 1, 1},
     {86, 32, 16, 16},
     {4, 512, 4, 32, 32},
     {4, 4, 32, 32, 32, 16},
     {4, 1365, 4, 32, 32},
     {4, 4, 86, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     514,
     "1 1 1 1 1 1 24 0 4 1365 512 32 86 32 86 16 4 4 32 32 4 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 1 "
     "1 512 32 16 1 1 128 64 256 1 1 1 1 1 8 8 1 1 1 0 0 0 0 1 0 128 0 256 0 "},
    {"conv3d_dx_n_factor",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {48, 960, 4, 4, 4},
     {3840, 3, 16, 16},
     {1, 48, 4, 6, 6},
     {1, 4, 3, 6, 6, 16},
     {1, 960, 10, 14, 14},
     {1, 10, 60, 14, 14, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     512,
     "1 1 1 1 1 1 24 0 1 960 48 3 60 3 60 16 4 4 6 6 10 14 14 4 4 4 1 2 2 2 0 0 0 0 0 0 3 3 3 3 3 1 1 1 2 2 1 2 2 1 48 "
     "3 16 1 1 128 64 256 1 1 1 1 1 12 4 1 1 1 0 0 0 0 1 0 126 0 256 0 "},
    {"conv3d_dx_17_fp32",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {512, 512, 1, 1, 1},
     {64, 32, 16, 8},
     {1, 512, 5, 32, 32},
     {1, 5, 64, 32, 32, 8},
     {1, 512, 5, 32, 32},
     {1, 5, 64, 32, 32, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     20,
     2,
     "1 1 4 1 1 5 20 0 1 512 512 64 64 64 64 8 3 5 32 32 5 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 2 1 "
     "512 64 64 1 1 128 32 256 1 1 1 1 1 8 1 1 1 1 0 0 0 0 1 0 256 0 512 0 "},
    {"conv3d_dx_050_fp32",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {64, 3, 7, 7, 7},
     {343, 4, 16, 8},
     {4, 64, 6, 112, 112},
     {4, 6, 8, 112, 112, 8},
     {4, 3, 17, 229, 229},
     {4, 17, 1, 229, 229, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     512,
     "1 1 1 1 1 1 24 0 4 3 64 8 1 8 1 8 3 6 112 112 17 229 229 7 7 7 1 2 2 2 0 0 0 0 0 0 6 6 6 6 6 1 1 1 2 2 1 2 2 1 "
     "64 8 1 1 1 1024 8 16 1 1 1 1 1 49 196 1 1 1 0 0 0 0 1 0 10534 0 8 0 "},
    {"conv3d_dx_17_hf32",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {512, 512, 1, 1, 1},
     {64, 32, 16, 8},
     {1, 512, 5, 32, 32},
     {1, 5, 64, 32, 32, 8},
     {1, 512, 5, 32, 32},
     {1, 5, 64, 32, 32, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0x40,
     true,
     true,
     20,
     2,
     "1 1 4 1 1 5 20 0 1 512 512 64 64 64 64 8 3 5 32 32 5 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 2 2 1 "
     "512 64 64 1 1 128 32 256 1 1 1 1 1 8 1 1 1 1 1 0 0 0 1 0 256 0 512 0 "},
    {"group_depth_g320_k2",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {320, 1, 1, 2, 2},
     {80, 1, 16, 16},
     {1, 320, 5, 31, 31},
     {1, 5, 20, 31, 31, 16},
     {1, 320, 5, 32, 32},
     {1, 5, 20, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     320,
     "NCDHW",
     "",
     0,
     true,
     true,
     16,
     1,
     "1 1 16 1 1 1 16 0 1 320 320 20 20 1 1 16 4 5 31 31 5 32 32 1 2 2 20 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 1 1 1 "
     "20 16 1 1 5 1 48 64 16 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 64 0 16 0 "},
    {"group_depth_g320_k2_f32",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "float32",
     {5},
     {320, 1, 1, 2, 2},
     {160, 1, 16, 8},
     {1, 320, 5, 31, 31},
     {1, 5, 40, 31, 31, 8},
     {1, 320, 5, 32, 32},
     {1, 5, 40, 32, 32, 8},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     320,
     "NCDHW",
     "",
     0,
     true,
     true,
     22,
     1,
     "1 1 11 1 2 1 22 0 1 320 320 40 40 2 2 8 3 5 31 31 5 32 32 1 2 2 20 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 1 2 1 "
     "20 16 2 1 5 1 16 32 16 1 1 1 1 1 2 2 1 1 1 0 0 0 0 1 0 96 0 8 0 "},
    {"Conv3d_bp_input_not_fit_conv_formula",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {64, 64, 3, 3, 3},
     {108, 4, 16, 16},
     {1, 64, 8, 8, 8},
     {1, 8, 4, 8, 8, 16},
     {1, 64, 8, 8, 8},
     {1, 8, 4, 8, 8, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 2, 1},
     {1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     false,
     20,
     1,
     " "},
    {"conv3d_dx_13",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {256, 256, 1, 3, 3},
     {144, 16, 16, 16},
     {1, 256, 9, 64, 64},
     {1, 9, 16, 64, 64, 16},
     {1, 256, 9, 129, 129},
     {1, 9, 16, 129, 129, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     512,
     "1 1 1 1 1 1 24 0 1 256 256 16 16 16 16 16 4 9 64 64 9 129 129 1 3 3 1 1 2 2 0 0 0 0 0 0 0 2 2 2 2 1 1 1 2 2 1 2 "
     "2 1 256 16 8 1 1 256 48 128 1 1 1 1 1 18 9 1 1 1 0 0 0 0 1 0 4257 0 128 0 "},
    {"Conv3d_bp_input_dilation",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {64, 64, 3, 3, 3},
     {108, 4, 16, 16},
     {1, 64, 3, 32, 32},
     {1, 3, 4, 32, 32, 16},
     {1, 64, 6, 32, 32},
     {1, 6, 4, 32, 32, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 2, 1, 1},
     {1, 2, 2, 2, 2, 2},
     {1, 1, 2, 2, 2},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     24,
     512,
     "1 1 1 1 1 1 24 0 1 64 64 4 4 4 4 16 4 3 32 32 6 32 32 3 3 3 1 2 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 1 64 4 "
     "4 1 1 336 48 64 1 1 1 1 1 12 12 1 1 1 0 1 0 0 1 0 320 0 64 0 "},
    {"Conv3d_bp_input_exceed_l1_size",
     "Ascend910B2",
     "Ascend910B",
     R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1, "fusion_mode": 0})",
     "bfloat16",
     {5},
     {1, 1, 2, 2, 2},
     {8, 1, 16, 16},
     {1, 1, 1, 255, 255},
     {1, 1, 1, 255, 255, 16},
     {1, 1, 2, 510, 510},
     {1, 2, 1, 510, 510, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NDC1HWC0,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 255, 255},
     1,
     "NCDHW",
     "",
     0,
     true,
     false,
     24,
     512,
     " "}};

static void ThreadFunc(
    const Conv3DBpInputV2TilingTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const Conv3DBpInputV2TilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_P(Conv3DBackpropInputV2TilingRunTime2, general_cases)
{
    TestOneParamCase(GetParam());
}

TEST_F(Conv3DBackpropInputV2TilingRunTime2, general_cases_params_multi_thread)
{
    TestMultiThread(
        general_20_core_num_cases_params,
        sizeof(general_20_core_num_cases_params) / sizeof(Conv3DBpInputV2TilingTestParam), 3);
    TestMultiThread(
        general_24_core_num_cases_params,
        sizeof(general_24_core_num_cases_params) / sizeof(Conv3DBpInputV2TilingTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(
    MilanBinary, Conv3DBackpropInputV2TilingRunTime2, testing::ValuesIn(general_20_core_num_cases_params));
INSTANTIATE_TEST_CASE_P(
    MilanBinary2, Conv3DBackpropInputV2TilingRunTime2, testing::ValuesIn(general_24_core_num_cases_params));

} // namespace
