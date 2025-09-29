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
 * \file test_conv3d_backprop_input_v2_tiling_arch35.cpp
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
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../../common/op_host/op_tiling/platform_util.h"
#include "test_cube_util.h"

#define SUCCESS 0

using namespace std;
using namespace ge;

namespace {
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
    std::string tiling_data_in_repo;
};

class Conv3DBackpropInputV2TilingRunTime3 : public testing::TestWithParam<Conv3DBpInputV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3DBackpropInputV2TilingRunTime3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv3DBackpropInputV2TilingRunTime3 TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;

    // 8个u32的dim分核相关参数
    uint32_t startField = 0;
    uint32_t endField = 8 * sizeof(int32_t);
    for (size_t i = startField; i < endField; i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    // 中间12个u8类型的值
    startField = endField;
    endField += 12 * sizeof(uint8_t);
    for (size_t i = startField; i < endField; i += sizeof(uint8_t)) {
        result += std::to_string((reinterpret_cast<const uint8_t*>(tiling_data->GetData())[i / sizeof(uint8_t)]));
        result += " ";
    }

    startField = endField;
    endField = tiling_data->GetDataSize();
    for (size_t i = startField; i < endField; i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

static void TestOneParamCase(const Conv3DBpInputV2TilingTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape input_size = {param.input_size, param.input_size};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_ori_shape, param.y_shape});
    std::vector<void*> output_shapes_ref(1);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    // platform info init
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
    ASSERT_NE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo(), nullptr);

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
    } else if (param.dtype == "hifloat8") {
        test_dtype = ge::DT_HIFLOAT8;
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

const string COMPILE_INFO_STR_910_95 = R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary",
                          "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
                          "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore"}
                          })";

Conv3DBpInputV2TilingTestParam cases_params_910_95[] = {
    {"conv3d_dx_1",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {512, 512, 1, 1, 1},
     {512, 512, 1, 1, 1},
     {1, 512, 5, 32, 32},
     {1, 512, 5, 32, 32},
     {1, 512, 5, 32, 32},
     {1, 512, 5, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1003,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 1 512 512 512 512 32 32 32 32 5 32 32 5 32 32 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 512 176 1 256 64 176 1 1 4 4 1 0 1 0 256 0 0 0 "},
    {"conv3d_dx_2",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {512, 512, 3, 3, 3},
     {512, 512, 3, 3, 3},
     {1, 512, 5, 32, 32},
     {1, 512, 5, 32, 32},
     {1, 512, 7, 32, 32},
     {1, 512, 7, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     32,
     1001,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 1 512 512 512 512 32 32 32 32 5 32 32 7 32 32 3 3 3 1 1 1 1 1 0 0 1 1 1 1 2 1 1 1 1 1 1 1 1 512 64 1 1024 16 64 1 1 45 45 3 0 1 0 1024 0 0 0 "},
    {"conv3d_dx_3_bf16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1, 41, 1, 26, 26},
     {1, 41, 1, 26, 26},
     {1, 1, 1, 2, 1},
     {1, 1, 1, 2, 1},
     {1, 41, 1, 32, 1},
     {1, 41, 1, 32, 1},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 23, 16},
     {0, 0, 8, 9, 12, 13},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     3,
     1001,
     "1 1 1 1 1 1 3 0 2 2 2 1 1 1 16 4 1 0 0 0 0 1 41 1 41 1 1 3 1 3 1 2 1 1 32 1 1 26 26 1 1 1 23 16 0 0 8 9 12 13 0 17 16 13 12 1 1 1 1 1 16 1 32 208 16 1 1 52 52 1 0 1 0 32 0 0 0 "},
    {"conv3d_dx_4",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {128, 128, 3, 3, 3},
     {128, 128, 3, 3, 3},
     {8, 128, 20, 64, 64},
     {8, 128, 20, 64, 64},
     {8, 128, 22, 66, 66},
     {8, 128, 22, 66, 66},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1001,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 8 128 128 128 128 8 8 8 8 20 64 64 22 66 66 3 3 3 1 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 1 1 1 1 128 128 1 512 32 128 1 1 18 18 3 0 1 0 512 0 0 0 "},
    {"conv3d_dx_5_basic_block_fullload_bl1",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {256, 128, 1, 1, 1},
     {256, 128, 1, 1, 1},
     {1, 256, 9, 128, 128},
     {1, 256, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1003,
     "1 1 1 1 1 1 32 0 2 2 1 2 1 1 16 4 1 0 0 0 0 1 128 256 128 256 16 8 16 8 9 128 128 9 128 128 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 256 128 9 512 32 128 1 1 4 8 1 0 1 0 512 0 0 0 "},
    {"conv3d_dx_6_kernel_split",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {256, 3, 4, 4, 4},
     {256, 3, 4, 4, 4},
     {1, 256, 10, 130, 130},
     {1, 256, 10, 130, 130},
     {1, 3, 16, 256, 256},
     {1, 3, 16, 256, 256},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {3, 3, 3, 3, 3, 3},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 2 2 2 1 16 4 1 0 0 0 1 1 3 256 3 256 16 1 16 1 10 130 130 16 256 256 4 4 4 1 1 2 2 2 3 3 3 3 3 3 0 0 0 0 0 1 1 1 1 256 16 1 512 32 16 1 1 16 16 4 0 1 0 512 0 0 0 "},
    {"conv3d_dx_7_ndhwc",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1280, 1, 3, 3, 640},
     {1280, 1, 3, 3, 640},
     {1, 1, 32, 32, 1280},
     {1, 1, 32, 32, 1280},
     {1, 1, 32, 32, 640},
     {1, 1, 32, 32, 640},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     {1, 1, 1, 1, 1},
     {0, 0, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     "",
     0,
     true,
     true,
     30,
     1002,
     "1 1 1 1 1 1 30 0 2 2 2 2 2 1 16 4 1 0 0 0 0 1 640 1280 640 1280 80 40 80 40 1 32 32 1 32 32 1 3 3 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1280 64 1 352 32 64 1 1 36 36 1 0 1 0 352 0 0 0 "},
    {"conv3d_dx_8_small_m",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {512, 512, 3, 3, 3},
     {512, 512, 3, 3, 3},
     {16, 512, 20, 16, 16},
     {16, 512, 20, 16, 16},
     {16, 512, 20, 16, 16},
     {16, 512, 20, 16, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1002,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 1 16 512 512 512 512 32 32 32 32 20 16 16 20 16 16 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 512 256 1 256 48 256 1 1 48 3 3 0 1 0 256 0 0 0 "},
    {"conv3d_dx_9_small_mn",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {512, 128, 3, 3, 1},
     {512, 128, 3, 3, 1},
     {16, 512, 20, 8, 16},
     {16, 128, 20, 8, 16},
     {16, 128, 20, 8, 16},
     {16, 512, 20, 16, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {1, 1, 1, 1, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1002,
     "1 1 1 1 1 1 32 0 2 2 2 2 2 1 16 4 1 0 0 0 1 16 128 512 128 512 32 8 32 8 20 8 16 20 8 16 3 3 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 0 1 1 1 1 512 128 1 128 96 128 1 1 8 8 3 0 1 0 128 0 0 0 "},
    {"conv3d_dx_10",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {64, 64, 3, 3, 3},
     {64, 64, 3, 3, 3},
     {8, 64, 20, 64, 64},
     {8, 64, 20, 64, 64},
     {8, 64, 22, 66, 66},
     {8, 64, 22, 66, 66},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1001,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 8 64 64 64 64 4 4 4 4 20 64 64 22 66 66 3 3 3 1 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 1 1 1 1 64 64 1 992 16 64 1 1 18 18 3 0 1 0 990 0 0 0 "},
    {"conv3d_dx_11_f16_h256_id3_cin_less_16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {256, 4, 1, 1, 1},
     {256, 4, 1, 1, 1},
     {1, 256, 4, 60, 60},
     {1, 256, 4, 60, 60},
     {1, 4, 4, 60, 60},
     {1, 4, 4, 60, 60},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1003,
     "1 1 1 1 1 1 32 0 2 2 2 2 2 1 16 4 1 0 0 0 0 1 4 256 4 256 16 1 16 1 4 60 60 4 60 60 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 256 16 1 464 32 16 1 1 4 4 1 0 1 0 464 0 0 0 "},
    {"conv3d_dx_12_small_shape",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {1280, 2560, 1, 1, 1},
     {1280, 2560, 1, 1, 1},
     {1, 1280, 1, 32, 32},
     {1, 1280, 1, 32, 32},
     {1, 2560, 1, 32, 32},
     {1, 2560, 1, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1003,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 1 2560 1280 2560 1280 80 160 80 160 1 32 32 1 32 32 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1280 160 1 256 64 160 1 1 4 4 1 0 1 0 256 0 0 0 "},
    {"conv3d_dx_13_small_shape",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float16",
     {5},
     {1280, 2560, 1, 3, 3},
     {1280, 2560, 1, 3, 3},
     {1, 1280, 1, 32, 32},
     {1, 1280, 1, 32, 32},
     {1, 2560, 1, 32, 32},
     {1, 2560, 1, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     32,
     1001,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 1 2560 1280 2560 1280 80 160 80 160 1 32 32 1 32 32 1 3 3 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1280 80 1 512 32 80 1 1 27 27 1 0 1 0 512 0 0 0 "},
    {"conv3d_dx_14_small_shape",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1280, 640, 1, 1, 1},
     {1280, 640, 1, 1, 1},
     {1, 1280, 1, 32, 32},
     {1, 1280, 1, 32, 32},
     {1, 640, 1, 32, 32},
     {1, 640, 1, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1003,
     "1 1 1 1 1 1 32 0 2 2 2 2 2 1 16 4 1 0 0 0 0 1 640 1280 640 1280 80 40 80 40 1 32 32 1 32 32 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1280 160 1 128 96 160 1 1 6 3 1 0 1 0 128 0 0 0 "},
    {"conv3d_dx_15_small_shape",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1280, 640, 1, 3, 3},
     {1280, 640, 1, 3, 3},
     {1, 1280, 1, 32, 32},
     {1, 1280, 1, 32, 32},
     {1, 640, 1, 32, 32},
     {1, 640, 1, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     30,
     1001,
     "1 1 1 1 1 1 30 0 2 2 2 2 2 1 16 4 1 0 0 0 0 1 640 1280 640 1280 80 40 80 40 1 32 32 1 32 32 1 3 3 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1280 64 1 352 32 64 1 1 36 36 1 0 1 0 352 0 0 0 "},
    {"conv3d_dx_16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {3, 64, 3, 3, 3},
     {3, 64, 3, 3, 3},
     {16, 3, 20, 128, 128},
     {16, 3, 20, 128, 128},
     {16, 64, 22, 130, 130},
     {16, 64, 22, 130, 130},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     11000,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 16 64 3 64 3 1 4 1 4 20 128 128 22 130 130 3 3 3 1 1 1 1 1 0 0 0 0 0 0 2 2 2 2 2 1 1 1 1 4 64 1 1024 16 64 1 1 3 3 3 0 1 0 1024 0 0 0 "},
    {"conv3d_dx_17",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {4, 320, 1, 3, 3},
     {4, 320, 1, 3, 3},
     {1, 4, 1, 128, 128},
     {1, 4, 1, 128, 128},
     {1, 320, 1, 128, 128},
     {1, 320, 1, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     32,
     11000,
     "1 1 1 1 1 1 32 0 2 2 2 2 1 1 16 4 1 0 0 0 0 1 320 4 320 4 1 20 1 20 1 128 128 1 128 128 1 3 3 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 4 320 1 96 48 320 1 1 1 1 1 0 1 0 512 0 0 0 "},
    {"conv3d_dx_18_c04_random",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1, 205, 1, 15, 15},
     {1, 205, 1, 15, 15},
     {2, 1, 1, 31, 194},
     {2, 1, 1, 31, 194},
     {2, 205, 1, 31, 194},
     {2, 205, 1, 31, 194},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 7, 7, 7, 7},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     11000,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 2 205 1 205 1 1 13 1 13 1 31 194 1 31 194 1 15 15 1 1 1 1 1 0 0 7 7 7 7 0 7 7 7 7 1 1 1 1 1 64 1 752 16 64 1 1 57 57 1 0 1 0 752 0 0 0 "},
    {"conv3d_dx_19",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1152, 4, 1, 2, 2},
     {1152, 4, 1, 2, 2},
     {1, 1152, 120, 8, 8},
     {1, 1152, 120, 8, 8},
     {1, 4, 120, 16, 16},
     {1, 4, 120, 16, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 2 1 1 1 16 4 1 0 0 0 0 1 4 1152 4 1152 72 1 72 1 120 8 8 120 16 16 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1152 16 1 64 256 16 1 1 5 5 1 0 1 0 64 0 1 0 "},
    {"conv3d_dx_20_f16_h256_weight_ncdhw",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {256, 256, 4, 4, 4},
     {256, 256, 4, 4, 4},
     {1, 256, 4, 60, 60},
     {1, 256, 4, 60, 60},
     {1, 256, 10, 122, 122},
     {1, 256, 10, 122, 122},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 1 1 256 256 256 256 16 16 16 16 4 60 60 10 122 122 4 4 4 1 1 2 2 2 0 0 0 0 0 0 3 3 3 3 3 1 1 1 1 256 128 1 496 32 128 1 1 12 12 4 0 1 0 488 0 0 1 "},
    {"conv3d_dx_21_f16_h256_weight_ndhwc",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {256, 4, 4, 4, 256},
     {256, 4, 4, 4, 256},
     {1, 256, 4, 60, 60},
     {1, 256, 4, 60, 60},
     {1, 256, 10, 122, 122},
     {1, 256, 10, 122, 122},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 1 256 256 256 256 16 16 16 16 4 60 60 10 122 122 4 4 4 1 1 2 2 2 0 0 0 0 0 0 3 3 3 3 3 1 1 1 1 256 128 1 496 32 128 1 1 12 12 4 0 1 0 488 0 0 1 "},
    {"conv3d_dx_22_soc_version_invalid",
     "Ascend910_",
     "Ascend910_",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {1152, 4, 1, 2, 2},
     {1152, 4, 1, 2, 2},
     {1, 1152, 120, 8, 8},
     {1, 1152, 120, 8, 8},
     {1, 4, 120, 16, 16},
     {1, 4, 120, 16, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     false,
     11,
     0,
     ""},
    // 基本块精度看护用例
    {"conv3d_dx_23_hifloat32_bs32",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float32",
     {5},
     {128, 96, 1, 1, 1},
     {128, 96, 1, 1, 1},
     {32, 128, 1, 56, 56},
     {32, 128, 1, 56, 56},
     {32, 96, 1, 56, 56},
     {32, 96, 1, 56, 56},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1002,
     "1 1 1 1 1 1 32 0 2 2 1 2 1 1 8 3 1 0 0 0 0 32 96 128 96 128 16 12 16 6 1 56 56 1 56 56 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 128 96 1 672 8 96 1 1 8 16 1 0 1 0 3136 0 0 0 "},
    // 8bit待优化用例
    {"conv3d_dx_24_hifloat8",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "hifloat8",
     {5},
     {20, 479, 1, 4, 3},
     {20, 479, 1, 4, 3},
     {4, 20, 1, 146, 118},
     {4, 20, 1, 146, 118},
     {4, 479, 1, 147, 118},
     {4, 479, 1, 147, 118},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     32,
     2,
     "4 1 8 1 1 1 32 0 2 2 1 1 1 1 32 5 1 0 0 0 0 4 479 20 479 20 1 15 1 30 1 146 118 1 147 118 1 4 3 1 1 1 1 1 0 0 1 1 1 1 0 2 2 1 1 1 1 1 1 20 479 1 256 96 256 1 1 4 4 1 0 1 0 2242 0 0 0 "},
    // 32bit待优化用例
    {"conv3d_dx_25_hifloat32_bs32",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float32",
     {5},
     {32, 128, 1, 3, 3},
     {32, 128, 1, 3, 3},
     {32, 32, 1, 7, 7},
     {32, 32, 1, 7, 7},
     {32, 128, 1, 7, 7},
     {32, 128, 1, 7, 7},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
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
     32,
     1002,
     "1 1 1 1 1 1 32 0 2 2 2 2 2 1 8 3 1 0 0 0 1 32 128 32 128 32 4 16 4 8 1 7 7 1 7 7 1 3 3 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 32 128 1 64 48 128 1 1 3 3 1 0 1 0 49 0 0 0 "},
    // depthwise group bf16用例
    {"conv3d_dx_26_depthwise_group_bf16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {128, 1, 1, 1, 1},
     {128, 1, 1, 1, 1},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     128,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     102,
     "1 1 32 1 1 1 32 0 2 2 1 1 1 1 16 4 16 0 0 0 0 1 128 128 16 16 8 8 1 1 9 128 128 9 128 128 1 1 1 8 128 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 8 16 16 9 256 16 16 1 1 1 1 1 0 1 0 512 0 0 0 "},
    // general group bf16用例
    {"conv3d_dx_27_general_group_bf16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {128, 2, 1, 1, 1},
     {128, 2, 1, 1, 1},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     64,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     102,
     "1 1 32 1 1 1 32 0 2 2 1 1 1 1 16 4 8 0 0 0 0 1 128 128 16 16 8 8 1 1 9 128 128 9 128 128 1 1 1 8 64 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 8 16 16 9 256 16 16 1 1 1 1 1 0 1 0 512 0 0 0 "},
    // depthwise group fp32用例
    {"conv3d_dx_28_depthwise_group_fp32",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float32",
     {5},
     {128, 1, 1, 1, 1},
     {128, 1, 1, 1, 1},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     128,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     102,
     "1 1 32 1 1 1 32 0 2 2 1 1 1 1 8 3 16 0 0 0 0 1 128 128 16 16 16 16 2 1 9 128 128 9 128 128 1 1 1 8 128 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 8 16 16 9 256 16 16 1 1 1 1 1 0 1 0 512 0 0 0 "},
    // general group fp32用例
    {"conv3d_dx_29_general_group_fp32",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float32",
     {5},
     {128, 2, 1, 1, 1},
     {128, 2, 1, 1, 1},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     {1, 128, 9, 128, 128},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     64,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     102,
     "1 1 32 1 1 1 32 0 2 2 1 1 1 1 8 3 8 0 0 0 0 1 128 128 16 16 16 16 2 1 9 128 128 9 128 128 1 1 1 8 64 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 8 16 16 9 256 16 16 1 1 1 1 1 0 1 0 512 0 0 0 "},
    // A:NCDHW,B:NCDHW;dtype:bf16/fp16
    {"conv3d_dx_30_NCDHW_NCDHW_bf16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {7, 47, 3, 4, 4},
     {7, 47, 3, 4, 4},
     {8, 7, 1, 39, 11},
     {8, 7, 1, 39, 11},
     {8, 47, 3, 80, 24},
     {8, 47, 3, 80, 24},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 2 1 1 1 16 4 1 0 0 0 0 8 47 7 47 7 1 3 1 3 1 39 11 3 80 24 3 4 4 1 1 2 2 2 0 0 0 0 0 0 2 3 3 3 3 1 1 1 1 7 48 1 256 64 48 1 1 1 1 3 0 1 0 252 0 1 1 "},
    // A:NCDHW,B:NCDHW;dtype:hf32/fp32
    {"conv3d_dx_31_NCDHW_NCDHW_fp32",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "float32",
     {5},
     {7, 47, 3, 4, 4},
     {7, 47, 3, 4, 4},
     {8, 7, 1, 39, 11},
     {8, 7, 1, 39, 11},
     {8, 47, 3, 80, 24},
     {8, 47, 3, 80, 24},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1012,
     "1 1 1 1 1 1 32 0 2 2 2 1 1 1 8 3 1 0 0 0 0 8 47 7 47 7 1 6 1 3 1 39 11 3 80 24 3 4 4 1 1 2 2 2 0 0 0 0 0 0 2 3 3 3 3 1 1 1 1 7 48 1 256 32 48 1 1 1 1 3 0 1 0 252 0 1 1 "},
    // A:NCDHW,B:DHWCN;dtype:bf16/fp16
    {"conv3d_dx_32_NCDHW_DHWCN_bf16",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {3, 4, 4, 47, 7},
     {3, 4, 4, 47, 7},
     {8, 7, 1, 39, 11},
     {8, 7, 1, 39, 11},
     {8, 47, 3, 80, 24},
     {8, 47, 3, 80, 24},
     ge::FORMAT_ND,
     ge::FORMAT_DHWCN,
     ge::FORMAT_DHWCN,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1002,
     "1 1 1 1 1 1 32 0 2 2 1 2 2 1 16 4 1 0 0 0 0 8 47 7 47 7 1 3 1 3 1 39 11 3 80 24 3 4 4 1 1 2 2 2 0 0 0 0 0 0 2 3 3 3 3 1 1 1 1 7 48 1 960 16 48 1 1 16 16 3 0 1 0 960 0 0 0 "},
    // kernel拆分stride大于2用例
    {"conv3d_dx_33_kernel_16_stride_2",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {64, 1, 16, 16, 192},
     {64, 1, 16, 16, 192},
     {1, 64, 2, 55, 55},
     {1, 64, 2, 55, 55},
     {1, 192, 2, 124, 124},
     {1, 192, 2, 124, 124},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1022,
     "1 1 1 1 1 1 32 0 2 2 1 2 1 1 16 4 1 0 0 0 0 1 192 64 192 64 4 12 4 12 2 55 55 2 124 124 1 16 16 1 1 1 2 2 0 0 0 0 0 0 0 15 15 15 15 1 1 1 1 64 96 1 496 32 96 1 1 64 64 1 0 1 0 496 0 0 0 "},
    // kernel拆分stride大于2 ncdhw用例
    {"conv3d_dx_34_kernel_16_stride_2_ncdhw",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {64, 192, 1, 16, 16},
     {64, 192, 1, 16, 16},
     {1, 64, 2, 55, 55},
     {1, 64, 2, 55, 55},
     {1, 192, 2, 124, 124},
     {1, 192, 2, 124, 124},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1022,
     "1 1 1 1 1 1 32 0 2 2 1 2 1 1 16 4 1 0 0 0 1 1 192 64 192 64 4 12 4 12 2 55 55 2 124 124 1 16 16 1 1 1 2 2 0 0 0 0 0 0 0 15 15 15 15 1 1 1 1 64 96 1 496 32 96 1 1 64 64 1 0 1 0 496 0 0 0 "},
    {"conv3d_dx_35_kernel_16_stride_2_ncdhw",
     "Ascend910_95",
     "Ascend910_95",
     COMPILE_INFO_STR_910_95,
     "bfloat16",
     {5},
     {64, 192, 1, 16, 24},
     {64, 192, 1, 16, 24},
     {1, 64, 2, 55, 51},
     {1, 64, 2, 55, 51},
     {1, 192, 2, 124, 124},
     {1, 192, 2, 124, 124},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     "",
     0,
     true,
     true,
     32,
     1022,
     "1 1 1 1 1 1 32 0 2 2 2 2 1 1 16 4 1 0 0 0 1 1 192 64 192 64 4 12 4 12 2 55 51 2 124 124 1 16 24 1 1 1 2 2 0 0 0 0 0 0 0 15 15 23 23 1 1 1 1 64 64 1 496 32 64 1 1 96 96 1 0 1 0 496 0 0 0 "},
};

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

TEST_F(Conv3DBackpropInputV2TilingRunTime3, general_cases_params_multi_thread)
{
    TestMultiThread(cases_params_910_95, sizeof(cases_params_910_95) / sizeof(Conv3DBpInputV2TilingTestParam), 3);
}

TEST_P(Conv3DBackpropInputV2TilingRunTime3, general_cases)
{
    TestOneParamCase(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    CONV3DDX_cases_params_910_95, Conv3DBackpropInputV2TilingRunTime3, testing::ValuesIn(cases_params_910_95));

} // namespace
