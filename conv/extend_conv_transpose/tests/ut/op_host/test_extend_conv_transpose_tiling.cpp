/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv_transpose_tilling_runtime.cpp
 * \brief
 */
#include <fstream>
#include <iostream>
#include <vector>
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

struct ExtendConvTransposeTilingTestParam {
    string case_name;
    string soc_version;
    string short_soc_version;
    string compile_info;

    ge::DataType x_dtype;
    ge::DataType filter_dtype;
    ge::DataType y_dtype;

    std::initializer_list<int64_t> input_size;
    std::initializer_list<int64_t> filter_ori_shape;
    std::initializer_list<int64_t> filter_shape;
    std::initializer_list<int64_t> out_backprop_ori_shape;
    std::initializer_list<int64_t> out_backprop_shape;
    std::initializer_list<int64_t> y_ori_shape;
    std::initializer_list<int64_t> y_shape;
    std::initializer_list<int64_t> bias_shape;
    std::initializer_list<int64_t> scale_shape;

    ge::Format input_size_format;
    ge::Format filter_ori_format;
    ge::Format filter_format;
    ge::Format out_backprop_ori_format;
    ge::Format out_backprop_format;
    ge::Format y_ori_format;
    ge::Format y_format;
    ge::Format bias_format;
    ge::Format scale_format;

    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string data_format;
    std::vector<int64_t> output_padding;
    int64_t fusion_mode;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
    std::string tiling_data_in_repo;
};

class ExtendConvTransposeTilingRunTime : public testing::TestWithParam<ExtendConvTransposeTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ExtendConvTransposeTilingRunTime SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ExtendConvTransposeTilingRunTime TearDown" << std::endl;
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

TEST_P(ExtendConvTransposeTilingRunTime, general_cases)
{
    ExtendConvTransposeTilingTestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape input_size = {param.input_size, param.input_size};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    gert::StorageShape bias_shape = {param.bias_shape, param.bias_shape};
    gert::StorageShape scale_shape = {param.scale_shape, param.scale_shape};
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
    soc_infos["cube_vector_combine"] = "fuse"; // set fuse socversion

    map<string, string> soc_version_infos = {
        {"SoC_version", param.soc_version}, {"Short_SoC_version", param.short_soc_version}};

    std::string op_type("ExtendConvTranspose");
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
    auto bias_dtype = ge::DT_FLOAT16;
    if (param.x_dtype == ge::DT_INT8) {
        bias_dtype = ge::DT_INT32;
    }
    auto scale_dtype = ge::DT_UINT64;
    int64_t nodeNum = 5;
    auto inputShape = {&input_size, &out_backprop_shape, &filter_shape, &bias_shape, &scale_shape};
    if (param.scale_shape.size() == 0) {
        nodeNum--;
        inputShape = {&input_size, &out_backprop_shape, &filter_shape, &bias_shape};
    }
    if (param.bias_shape.size() == 0) {
        nodeNum--;
        inputShape = {&input_size, &out_backprop_shape, &filter_shape};
    }
    auto holder = gert::TilingContextFaker()
                        .SetOpType(op_type)
                        .NodeIoNum(nodeNum, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1})
                        .InputShapes(inputShape)
                        .OutputShapes(output_shapes_ref)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
                           {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"fusion_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.fusion_mode)}})
                        .NodeInputTd(0, DT_INT32, param.input_size_format, param.input_size_format)
                        .NodeInputTd(1, param.x_dtype, param.out_backprop_ori_format, param.out_backprop_format)
                        .NodeInputTd(2, param.filter_dtype, param.filter_ori_format, param.filter_format)
                        .NodeInputTd(3, bias_dtype, param.bias_format, param.bias_format)
                        .NodeInputTd(4, scale_dtype, param.scale_format, param.scale_format)
                        .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
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
    std::cout << "transpose>>>>>>>>>>>>>>>>>>" << tiling_data_result << std::endl;
    ASSERT_EQ(*tiling_key, param.tiling_key);
    ASSERT_EQ(*block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

const string COMPILE_INFO_STR_FUSE = R"({"_pattern": "Extend_conv_transpose", "tiling_type": "binary",
                          "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8,
                          "cube_core_cnt": 8, "vector_core_cnt": 8, "core_type_list": "CubeCore,VectorCore"}
                          })";

ExtendConvTransposeTilingTestParam cases_params_fuse[] = {
    {"net_ncdhw_int8_2_int8_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_INT8,
    {5},
    {512, 256, 1, 2, 2},
    {512, 256, 1, 2, 2},
    {4, 512, 1, 20, 16},
    {4, 512, 1, 20, 16},
    {4, 256, 1, 40, 32},
    {4, 256, 1, 40, 32},
    {256},
    {256},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 1 1 1 32 5 5 1 0 0 33554433 4 256 512 256 512 16 8 16 16 1 20 16 1 40 32 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 512 256 1 160 256 256 1 1 2 2 1 0 1 0 160 0 0 0 1 1 "},

    {"net_ncdhw_int8_2_fp16_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {256, 128, 1, 2, 2},
    {256, 128, 1, 2, 2},
    {1, 256, 1, 88, 80},
    {1, 256, 1, 88, 80},
    {1, 128, 1, 176, 160},
    {1, 128, 1, 176, 160},
    {128},
    {128},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 5 5 1 0 0 33554689 1 128 256 128 256 8 8 16 8 1 88 80 1 176 160 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 128 1 480 128 128 1 1 2 2 1 0 1 0 480 0 0 0 0 1 "},

    {"net_ncdhw_fp16_2_fp16_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {64, 64, 1, 2, 2},
    {64, 64, 1, 2, 2},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 352, 320},
    {1, 64, 1, 352, 320},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 16 4 4 1 0 0 33554689 1 64 64 64 64 4 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 64 64 1 960 32 64 1 1 1 1 1 0 1 0 960 0 0 0 0 1 "},

    {"net_ncdhw_fp16_2_fp16_stride_2_pad_1",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {256, 192, 1, 4, 4},
    {256, 192, 1, 4, 4},
    {1, 256, 1, 18, 7},
    {1, 256, 1, 18, 7},
    {1, 192, 1, 36, 14},
    {1, 192, 1, 36, 14},
    {192},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    6,
    16777217,
    "1 1 1 1 1 1 6 0 1 1 2 2 2 1 16 4 4 1 0 0 33554433 1 192 256 192 256 16 12 16 12 1 18 7 1 36 14 1 4 4 1 1 1 2 2 0 0 1 1 1 1 0 2 2 2 2 1 1 1 1 256 64 1 256 128 64 1 1 16 16 1 0 1 0 252 0 0 0 0 0 "},

    {"net_ncdhw_fp16_2_fp16_stride_2_no_scale",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 1, 2, 2},
    {256, 1, 1, 2, 2},
    {1, 256, 1, 36, 64},
    {1, 256, 1, 36, 64},
    {1, 1, 1, 72, 128},
    {1, 1, 1, 72, 128},
    {1},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 2 1 1 1 16 4 4 1 0 0 33554433 1 1 256 1 256 16 1 16 1 1 36 64 1 72 128 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 16 1 256 128 16 1 1 2 2 1 0 1 0 256 0 0 0 1 0 "},

    {"net_ncdhw_int8_2_fp16_stride_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {512, 64, 1, 4, 4},
    {512, 64, 1, 4, 4},
    {12, 512, 1, 4, 8},
    {12, 512, 1, 4, 8},
    {12, 64, 1, 16, 32},
    {12, 64, 1, 16, 32},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 4, 4},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777730,
    "1 1 1 1 1 1 8 0 1 1 2 2 2 1 32 5 5 1 0 0 33554689 12 64 512 64 512 16 4 32 4 1 4 8 1 16 32 1 4 4 1 1 1 4 4 0 0 0 0 0 0 0 3 3 3 3 1 1 1 1 512 64 1 128 512 64 1 1 4 4 1 0 1 0 128 0 0 0 0 0 "},

    {"net_ncdhw_fp16_2_fp16_group_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {64, 32, 1, 2, 2},
    {64, 32, 1, 2, 2},
    {1, 64, 1, 32, 60},
    {1, 64, 1, 32, 60},
    {1, 128, 1, 64, 120},
    {1, 128, 1, 64, 120},
    {128},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    4,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777217,
    "1 1 1 1 1 1 8 0 1 1 2 2 2 1 16 4 4 1 0 0 33554433 1 128 64 32 16 4 8 1 2 1 32 60 1 64 120 1 2 2 4 4 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 16 32 1 960 32 32 1 1 2 2 1 0 1 0 960 0 0 0 0 0 "},

    {"net_ndhwc_int8_2_int8_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_INT8,
    {5},
    {512, 1, 2, 2, 256},
    {512, 1, 2, 2, 256},
    {4, 512, 1, 20, 16},
    {4, 512, 1, 20, 16},
    {4, 256, 1, 40, 32},
    {4, 256, 1, 40, 32},
    {256},
    {256},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 1 1 1 32 5 5 1 0 0 33554433 4 256 512 256 512 16 8 16 16 1 20 16 1 40 32 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 512 256 1 160 256 256 1 1 2 2 1 0 1 0 160 0 0 0 1 1 "},

    {"net_ndhwc_int8_2_fp16_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 2, 2, 128},
    {256, 1, 2, 2, 128},
    {1, 256, 1, 88, 80},
    {1, 256, 1, 88, 80},
    {1, 128, 1, 176, 160},
    {1, 128, 1, 176, 160},
    {128},
    {128},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 5 5 1 0 0 33554433 1 128 256 128 256 8 8 16 8 1 88 80 1 176 160 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 128 1 480 128 128 1 1 2 2 1 0 1 0 480 0 0 0 0 1 "},

    {"net_ndhwc_fp16_2_fp16_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {64, 1, 2, 2, 64},
    {64, 1, 2, 2, 64},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 352, 320},
    {1, 64, 1, 352, 320},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 16 4 4 1 0 0 33554433 1 64 64 64 64 4 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 64 64 1 960 32 64 1 1 1 1 1 0 1 0 960 0 0 0 0 1 "},

    {"net_ndhwc_fp16_2_fp16_stride_2_pad_1",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 4, 4, 192},
    {256, 1, 4, 4, 192},
    {1, 256, 1, 18, 7},
    {1, 256, 1, 18, 7},
    {1, 192, 1, 36, 14},
    {1, 192, 1, 36, 14},
    {192},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    6,
    16777218,
    "1 1 1 1 1 1 6 0 1 1 2 2 2 1 16 4 4 1 0 0 33554433 1 192 256 192 256 16 12 16 12 1 18 7 1 36 14 1 4 4 1 1 1 2 2 0 0 1 1 1 1 0 2 2 2 2 1 1 1 1 256 64 1 256 128 64 1 1 16 16 1 0 1 0 252 0 0 0 0 0 "},

    {"net_ndhwc_fp16_2_fp16_stride_2_no_scale",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 2, 2, 1},
    {256, 1, 2, 2, 1},
    {1, 256, 1, 36, 64},
    {1, 256, 1, 36, 64},
    {1, 1, 1, 72, 128},
    {1, 1, 1, 72, 128},
    {1},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 2 1 1 1 16 4 4 1 0 0 33554433 1 1 256 1 256 16 1 16 1 1 36 64 1 72 128 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 16 1 256 128 16 1 1 2 2 1 0 1 0 256 0 0 0 1 0 "},

    {"net_ndhwc_int8_2_fp16_stride_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {512, 1, 4, 4, 64},
    {512, 1, 4, 4, 64},
    {12, 512, 1, 4, 8},
    {12, 512, 1, 4, 8},
    {12, 64, 1, 16, 32},
    {12, 64, 1, 16, 32},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 4, 4},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777730,
    "1 1 1 1 1 1 8 0 1 1 2 2 2 1 32 5 5 1 0 0 33554433 12 64 512 64 512 16 4 32 4 1 4 8 1 16 32 1 4 4 1 1 1 4 4 0 0 0 0 0 0 0 3 3 3 3 1 1 1 1 512 64 1 128 512 64 1 1 4 4 1 0 1 0 128 0 0 0 0 0 "},

    {"net_ndhwc_fp16_2_fp16_group_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {64, 1, 2, 2, 32},
    {64, 1, 2, 2, 32},
    {1, 64, 1, 32, 60},
    {1, 64, 1, 32, 60},
    {1, 128, 1, 64, 120},
    {1, 128, 1, 64, 120},
    {128},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    4,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777218,
    "1 1 1 1 1 1 8 0 1 1 2 2 2 1 16 4 4 1 0 0 33554433 1 128 64 32 16 4 8 1 2 1 32 60 1 64 120 1 2 2 4 4 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 16 32 1 960 32 32 1 1 2 2 1 0 1 0 960 0 0 0 0 0 "},

    {"net_ndhwc_a16w8_2_fp16_stride_2",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {64, 1, 2, 2, 64},
    {64, 1, 2, 2, 64},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 352, 320},
    {1, 64, 1, 352, 320},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 33554433 1 64 64 64 64 2 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 64 64 1 960 32 64 1 1 2 2 1 0 1 0 960 0 0 0 0 1 "},

    {"net_ndhwc_a16w8_2_fp16_stride_2_no_scale",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 2, 2, 1},
    {256, 1, 2, 2, 1},
    {1, 256, 1, 36, 64},
    {1, 256, 1, 36, 64},
    {1, 1, 1, 72, 128},
    {1, 1, 1, 72, 128},
    {1},
    {},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 2 1 1 1 32 4 5 1 0 0 33554433 1 1 256 1 256 8 1 16 1 1 36 64 1 72 128 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 16 1 256 128 16 1 1 4 4 1 0 1 0 256 0 0 0 1 0 "},

    {"net_ndhwc_a16w8_2_fp16_pertensor_quant_mode",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {64, 1, 2, 2, 64},
    {64, 1, 2, 2, 64},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 352, 320},
    {1, 64, 1, 352, 320},
    {64},
    {1},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 16777217 1 64 64 64 64 2 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 64 64 1 960 32 64 1 1 2 2 1 0 1 0 960 0 0 0 0 1 "},

    {"net_ndhwc_a16w8_2_fp16_enable_relu",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {64, 1, 2, 2, 64},
    {64, 1, 2, 2, 64},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 176, 160},
    {1, 64, 1, 352, 320},
    {1, 64, 1, 352, 320},
    {64},
    {1},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    1,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 16777217 1 64 64 64 64 2 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 64 64 1 960 32 64 1 1 2 2 1 0 1 0 960 0 1 0 0 1 "},

    {"net_ndhwc_a16w16_large_input",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {128, 1, 2, 2, 64},
    {128, 1, 2, 2, 64},
    {1, 128, 1, 288, 112},
    {1, 128, 1, 288, 112},
    {1, 64, 1, 576, 224},
    {1, 64, 1, 576, 224},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 16 4 4 1 0 0 33554433 1 64 128 64 128 8 4 8 4 1 288 112 1 576 224 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 128 64 1 1008 32 64 1 1 2 2 1 0 1 0 1008 0 0 0 0 1 "},

    {"net_ndhwc_a16w8_large_input",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {128, 1, 2, 2, 64},
    {128, 1, 2, 2, 64},
    {1, 128, 1, 288, 112},
    {1, 128, 1, 288, 112},
    {1, 64, 1, 576, 224},
    {1, 64, 1, 576, 224},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 33554433 1 64 128 64 128 4 4 8 4 1 288 112 1 576 224 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 128 64 1 1008 32 64 1 1 4 4 1 0 1 0 1008 0 0 0 0 1 "},

    {"net_ndhwc_a16w16_large_input_stride_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {128, 1, 4, 4, 64},
    {128, 1, 4, 4, 64},
    {1, 128, 1, 112, 56},
    {1, 128, 1, 112, 56},
    {1, 64, 1, 448, 224},
    {1, 64, 1, 448, 224},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 4, 4},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777730,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 16 4 4 1 0 0 33554433 1 64 128 64 128 8 4 8 4 1 112 56 1 448 224 1 4 4 1 1 1 4 4 0 0 0 0 0 0 0 3 3 3 3 1 1 1 1 128 64 1 1024 32 64 1 1 8 8 1 0 1 0 1024 0 0 0 0 0 "},

    {"net_ndhwc_a16w8_large_input_stride_4",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {128, 1, 4, 4, 64},
    {128, 1, 4, 4, 64},
    {1, 128, 1, 112, 56},
    {1, 128, 1, 112, 56},
    {1, 64, 1, 448, 224},
    {1, 64, 1, 448, 224},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 4, 4},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777730,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 33554433 1 64 128 64 128 4 4 8 4 1 112 56 1 448 224 1 4 4 1 1 1 4 4 0 0 0 0 0 0 0 3 3 3 3 1 1 1 1 128 64 1 1024 32 64 1 1 16 16 1 0 1 0 1024 0 0 0 0 0 "},

    {"net_ndhwc_a16w16_multi_batch",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 2, 2, 64},
    {256, 1, 2, 2, 64},
    {4, 256, 1, 40, 32},
    {4, 256, 1, 40, 32},
    {4, 64, 1, 80, 64},
    {4, 64, 1, 80, 64},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 16 4 4 1 0 0 33554433 4 64 256 64 256 16 4 16 4 1 40 32 1 80 64 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 64 1 640 32 64 1 1 4 4 1 0 1 0 640 0 0 0 0 1 "},

    {"net_ndhwc_a16w8_multi_batch",
    "SOC_L1_1024",
    "SOC_L1_1024",
    COMPILE_INFO_STR_FUSE,
    ge::DT_FLOAT16,
    ge::DT_INT8,
    ge::DT_FLOAT16,
    {5},
    {256, 1, 2, 2, 64},
    {256, 1, 2, 2, 64},
    {4, 256, 1, 40, 32},
    {4, 256, 1, 40, 32},
    {4, 64, 1, 80, 64},
    {4, 64, 1, 80, 64},
    {64},
    {64},
    ge::FORMAT_ND,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    {1, 1, 1, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1},
    1,
    "NCDHW",
    {0, 0, 0, 0, 0},
    0,
    true,
    true,
    8,
    16777474,
    "1 1 1 1 1 1 8 0 1 1 1 2 2 1 32 4 5 1 0 0 33554433 4 64 256 64 256 8 4 16 4 1 40 32 1 80 64 1 2 2 1 1 1 2 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 256 64 1 640 32 64 1 1 8 8 1 0 1 0 640 0 0 0 0 1 "},
};

INSTANTIATE_TEST_CASE_P(
    Conv3DDX_cases_params_fuse, ExtendConvTransposeTilingRunTime, testing::ValuesIn(cases_params_fuse));

} // namespace
