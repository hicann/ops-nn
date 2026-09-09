/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv_transpose_v2_tilling_runtime.cpp
 * \brief
 */
#include <fstream>
#include <iostream>
#include <limits>
#include <type_traits>
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
#include "conv/conv3d_backprop_input_v2/op_host/op_tiling/conv3d_backprop_input_v2_tiling_data_arch35.h"
#include "../../../../common/op_host/op_tiling/conv_platform_util.h"
#include "test_cube_util.h"

#define SUCCESS 0

using namespace std;
using namespace ge;

namespace {
struct ExtendConvTransposeV2TilingTestParam {
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
    bool dual_output = false;
    bool legacy_dtype_attrs = false;
    std::initializer_list<int64_t> scale1_shape = {};
    uint8_t expected_quant_mode = std::numeric_limits<uint8_t>::max();
    uint8_t expected_quant_mode1 = std::numeric_limits<uint8_t>::max();
    ge::DataType y1_dtype = ge::DT_UNDEFINED;
};

class ExtendConvTransposeV2TilingRunTime : public testing::TestWithParam<ExtendConvTransposeV2TilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "ExtendConvTransposeV2TilingRunTime SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExtendConvTransposeV2TilingRunTime TearDown" << std::endl; }
};

constexpr size_t TILING_DATA_START_OFFSET = 0U;
constexpr size_t PARALLEL_DIM_FIELD_COUNT = 6U;
constexpr size_t CORE_NUM_FIELD_COUNT = 1U;
constexpr size_t BASE_FLAG_FIELD_COUNT = 16U;
constexpr size_t SHAPE_FIELD_COUNT = 29U;
constexpr size_t BACKPROP_PAD_FIELD_COUNT = 5U;
constexpr size_t BASE_TILING_FIELD_COUNT = 13U;
constexpr size_t UINT32_ALIGNMENT_FIELD_COUNT = 1U;
constexpr size_t SPLIT_K_VALUE_FIELD_COUNT = 6U;
constexpr size_t SPLIT_K_FLAG_FIELD_COUNT = 2U;
constexpr size_t OFFSET_X_FIELD_COUNT = 1U;
constexpr size_t UINT8_ALIGNMENT_FIELD_COUNT = 1U;
constexpr size_t KERNEL_SPLIT_FIELD_COUNT = 6U;
constexpr size_t KERNEL_SHAPE_FIELD_COUNT = 2U;
constexpr size_t FIXED_SHIFT_FIELD_COUNT = 1U;

template <typename T>
static void AppendTilingFields(const void* data, size_t& offset, size_t fieldCount, string& result)
{
    static_assert(std::is_integral_v<T>, "tiling field must be an integral type");
    using UnsignedT = std::make_unsigned_t<T>;
    constexpr size_t BITS_PER_BYTE = std::numeric_limits<uint8_t>::digits;
    const auto bytes = static_cast<const uint8_t*>(data);
    for (size_t fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex) {
        UnsignedT field = 0;
        for (size_t byteIndex = 0; byteIndex < sizeof(T); ++byteIndex) {
            field |= static_cast<UnsignedT>(bytes[offset++]) << (byteIndex * BITS_PER_BYTE);
        }
        result += std::to_string(static_cast<T>(field));
        result += " ";
    }
}

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    const auto data = tiling_data->GetData();
    string result;
    size_t offset = TILING_DATA_START_OFFSET;

    // 核间并行维度
    AppendTilingFields<uint32_t>(data, offset, PARALLEL_DIM_FIELD_COUNT, result);
    // 核数
    AppendTilingFields<uint64_t>(data, offset, CORE_NUM_FIELD_COUNT, result);
    // 基础标志和切分参数
    AppendTilingFields<uint8_t>(data, offset, BASE_FLAG_FIELD_COUNT, result);
    // 输入输出形状参数
    AppendTilingFields<uint32_t>(data, offset, SHAPE_FIELD_COUNT, result);
    // 反向 padding 参数
    AppendTilingFields<int32_t>(data, offset, BACKPROP_PAD_FIELD_COUNT, result);
    // 基础 tiling 参数
    AppendTilingFields<uint32_t>(data, offset, BASE_TILING_FIELD_COUNT, result);
    offset += UINT32_ALIGNMENT_FIELD_COUNT * sizeof(uint32_t);
    // SplitK 计算参数
    AppendTilingFields<uint64_t>(data, offset, SPLIT_K_VALUE_FIELD_COUNT, result);
    // SplitK 标志
    AppendTilingFields<uint8_t>(data, offset, SPLIT_K_FLAG_FIELD_COUNT, result);
    // 输入偏移
    AppendTilingFields<int8_t>(data, offset, OFFSET_X_FIELD_COUNT, result);
    offset += UINT8_ALIGNMENT_FIELD_COUNT * sizeof(uint8_t);
    // Kernel Split 参数
    AppendTilingFields<uint32_t>(data, offset, KERNEL_SPLIT_FIELD_COUNT, result);
    offset += UINT32_ALIGNMENT_FIELD_COUNT * sizeof(uint32_t);
    // 卷积核形状参数
    AppendTilingFields<uint64_t>(data, offset, KERNEL_SHAPE_FIELD_COUNT, result);
    // 定点偏移参数
    AppendTilingFields<uint8_t>(data, offset, FIXED_SHIFT_FIELD_COUNT, result);

    return result;
}

TEST_P(ExtendConvTransposeV2TilingRunTime, general_cases)
{
    ExtendConvTransposeV2TilingTestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape input_size = {param.input_size, param.input_size};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    gert::StorageShape bias_shape = {param.bias_shape, param.bias_shape};
    gert::StorageShape scale_shape = {param.scale_shape, param.scale_shape};
    gert::StorageShape scale1_shape = {param.scale1_shape, param.scale1_shape};
    std::vector<gert::StorageShape> output_shapes(2, {param.y_ori_shape, param.y_shape});
    std::vector<void*> output_shapes_ref(2);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(param.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;

    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    soc_infos["cube_vector_combine"] = "fuse"; // set fuse socversion

    map<string, string> soc_version_infos = {{"SoC_version", param.soc_version},
                                             {"Short_SoC_version", param.short_soc_version}};

    std::string op_type("ExtendConvTransposeV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    ASSERT_NE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo(), nullptr);

    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
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
    std::vector<gert::StorageShape*> inputShape = {&input_size, &out_backprop_shape, &filter_shape};
    std::vector<uint32_t> irInstanceNum = {1U, 1U, 1U, 0U, 0U, 0U};
    if (param.bias_shape.size() != 0) {
        inputShape.emplace_back(&bias_shape);
        irInstanceNum[3] = 1U;
    }
    if (param.scale_shape.size() != 0) {
        inputShape.emplace_back(&scale_shape);
        irInstanceNum[4] = 1U;
    }
    if (param.scale1_shape.size() != 0) {
        inputShape.emplace_back(&scale1_shape);
        irInstanceNum[5] = 1U;
    }
    const int64_t nodeNum = static_cast<int64_t>(inputShape.size());
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> node_attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
        {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
        {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
        {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"fusion_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.fusion_mode)},
        {"y_quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"placehodlder", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"fixed_shift_value", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
        {"dual_output", Ops::NN::AnyValue::CreateFrom<bool>(param.dual_output)}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(nodeNum, 2)
                      .IrInstanceNum(irInstanceNum)
                      .InputShapes(inputShape)
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(node_attrs)
                      .NodeInputTd(0, DT_INT32, param.input_size_format, param.input_size_format)
                      .NodeInputTd(1, param.x_dtype, param.out_backprop_ori_format, param.out_backprop_format)
                      .NodeInputTd(2, param.filter_dtype, param.filter_ori_format, param.filter_format)
                      .NodeInputTd(3, bias_dtype, param.bias_format, param.bias_format)
                      .NodeInputTd(4, scale_dtype, param.scale_format, param.scale_format)
                      .NodeInputTd(5, scale_dtype, param.scale_format, param.scale_format)
                      .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                      .NodeOutputTd(1, param.y1_dtype == ge::DT_UNDEFINED ? param.y_dtype : param.y1_dtype,
                                    param.y_ori_format, param.y_format)
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
    if (!param.tiling_data.empty()) {
        ASSERT_EQ(*tiling_key, param.tiling_key);
        ASSERT_EQ(*block_dim, param.block_dim);
        ASSERT_EQ(tiling_data_result, param.tiling_data);
    }
}

const string COMPILE_INFO_STR_FUSE = R"({"_pattern": "Extend_conv_transpose", "tiling_type": "binary",
                          "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true, "socVersion": "Ascend950",
                          "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8,
                          "cube_core_cnt": 8, "vector_core_cnt": 8, "core_type_list": "CubeCore,VectorCore"}
                          })";

ExtendConvTransposeV2TilingTestParam cases_params_fuse[] = {

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 32 5 5 1 0 0 1 0 0 2 4 256 512 256 512 16 8 16 16 1 20 16 1 40 32 1 2 2 1 1 1 2 2 0 0 "
     "0 0 0 0 0 1 1 1 1 1 1 1 1 512 256 1 256 128 256 1 1 1 1 256 0 0 0 0 0 0 0 0 0 2 2 39 31 4 4 13 ",
     {},
     true},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 32 5 5 1 0 0 1 0 0 2 1 128 256 128 256 8 8 8 8 1 88 80 1 176 160 1 2 2 1 1 1 2 2 0 0 0 "
     "0 0 0 0 1 1 1 1 1 1 1 1 256 128 1 512 64 128 1 1 1 1 512 0 0 0 0 0 0 0 0 0 2 2 175 159 4 4 13 "},

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
     "1 1 1 1 1 1 8 1 1 2 2 2 1 16 4 4 1 0 0 1 0 0 2 1 64 64 64 64 4 4 4 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 64 64 1 480 64 64 1 1 1 1 480 0 0 0 0 0 0 0 0 0 2 2 351 319 4 4 42 "},

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
     "1 1 1 1 1 1 6 1 1 2 2 2 1 16 4 4 1 0 0 1 0 0 0 1 192 256 192 256 16 12 16 12 1 18 7 1 36 14 1 4 4 1 1 1 2 2 0 0 "
     "1 1 1 1 0 2 2 2 2 1 1 1 1 256 64 1 256 128 64 16 16 1 1 252 0 0 0 0 0 0 0 0 0 4 4 35 13 16 16 42 "},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 16 4 4 1 0 0 1 0 0 0 1 1 256 1 256 16 1 16 1 1 36 64 1 72 128 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 256 16 1 576 16 16 1 1 1 1 576 0 0 0 0 0 0 0 0 0 2 2 71 127 4 4 42 "},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 32 5 5 1 0 0 1 0 0 2 12 64 512 64 512 16 4 16 4 1 4 8 1 16 32 1 4 4 1 1 1 4 4 0 0 0 0 "
     "0 0 0 3 3 3 3 1 1 1 1 512 64 1 256 128 64 1 1 1 1 256 0 0 0 0 0 0 0 0 0 4 4 13 29 16 16 13 "},

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
     "1 1 1 1 1 1 8 1 1 2 2 1 1 16 4 4 1 0 0 1 0 1 0 1 128 64 32 16 4 8 1 2 1 32 60 1 64 120 1 2 2 4 4 1 2 2 0 0 0 0 0 "
     "0 0 1 1 1 1 1 1 1 1 16 32 1 960 32 32 2 2 1 1 960 0 0 0 0 0 0 0 0 0 2 2 63 119 4 4 42 "},

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
     16777218,
     "1 1 1 1 1 1 8 1 1 1 2 1 1 32 4 5 1 0 0 1 0 1 2 1 64 64 64 64 2 4 2 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 64 64 1 1024 32 64 8 8 1 1 14080 0 0 0 0 0 0 0 0 0 2 2 351 319 4 4 13 "},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 32 4 5 1 0 0 1 0 0 0 1 1 256 1 256 8 1 8 1 1 36 64 1 72 128 1 2 2 1 1 1 2 2 0 0 0 0 0 "
     "0 0 1 1 1 1 1 1 1 1 256 16 1 512 32 16 1 1 1 1 512 0 0 0 0 0 0 0 0 0 2 2 71 127 4 4 13 "},

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
     16777218,
     "1 1 1 1 1 1 8 1 1 1 2 1 1 32 4 5 1 0 0 1 0 1 1 1 64 64 64 64 2 4 2 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 64 64 1 1024 32 64 8 8 1 1 14080 0 0 0 0 0 0 0 0 0 2 2 351 319 4 4 13 "},

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
     16777218,
     "1 1 1 1 1 1 8 1 1 1 2 1 1 32 4 5 1 0 0 1 0 1 1 1 64 64 64 64 2 4 2 4 1 176 160 1 352 320 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 64 64 1 1024 32 64 8 8 1 1 14080 1 0 0 0 0 0 0 0 0 2 2 351 319 4 4 13 "},

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
     "1 1 1 1 1 1 8 1 1 2 2 2 1 16 4 4 1 0 0 1 0 0 2 1 64 128 64 128 8 4 8 4 1 288 112 1 576 224 1 2 2 1 1 1 2 2 0 0 0 "
     "0 0 0 0 1 1 1 1 1 1 1 1 128 64 1 448 64 64 1 1 1 1 448 0 0 0 0 0 0 0 0 0 2 2 575 223 4 4 42 "},

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
     16777218,
     "1 1 1 1 1 1 8 1 1 1 2 1 1 32 4 5 1 0 0 1 0 1 2 1 64 128 64 128 4 4 4 4 1 288 112 1 576 224 1 2 2 1 1 1 2 2 0 0 0 "
     "0 0 0 0 1 1 1 1 1 1 1 1 128 64 1 1024 32 64 16 16 1 1 16128 0 0 0 0 0 0 0 0 0 2 2 575 223 4 4 13 "},

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
     "1 1 1 1 1 1 8 1 1 1 2 2 1 16 4 4 1 0 0 1 0 0 2 1 64 128 64 128 8 4 8 4 1 112 56 1 448 224 1 4 4 1 1 1 4 4 0 0 0 "
     "0 0 0 0 3 3 3 3 1 1 1 1 128 64 1 512 64 64 4 4 1 1 512 0 0 0 0 0 0 0 0 0 4 4 445 221 16 16 42 "},

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
     16777218,
     "1 1 1 1 1 1 8 1 1 1 2 1 1 32 4 5 1 0 0 1 0 1 2 1 64 128 64 128 4 4 4 4 1 112 56 1 448 224 1 4 4 1 1 1 4 4 0 0 0 "
     "0 0 0 0 3 3 3 3 1 1 1 1 128 64 1 1024 32 64 32 64 1 1 12544 0 0 0 0 0 0 0 0 0 4 4 445 221 16 16 13 "},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 16 4 4 1 0 0 1 0 0 2 4 64 256 64 256 16 4 16 4 1 40 32 1 80 64 1 2 2 1 1 1 2 2 0 0 0 0 "
     "0 0 0 1 1 1 1 1 1 1 1 256 64 1 768 16 64 1 1 1 1 768 0 0 0 0 0 0 0 0 0 2 2 79 63 4 4 42 ",
     {},
     false,
     true},

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
     150994946,
     "1 1 1 1 1 1 8 2 2 1 2 1 1 32 4 5 1 0 0 1 0 0 2 4 64 256 64 256 8 4 8 4 1 40 32 1 80 64 1 2 2 1 1 1 2 2 0 0 0 0 0 "
     "0 0 1 1 1 1 1 1 1 1 256 64 1 512 32 64 1 1 1 1 512 0 0 0 0 0 0 0 0 0 2 2 79 63 4 4 13 "},

    {"dual_output_scale1_only_vector_quant",
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
     0,
     0,
     "",
     "",
     true,
     false,
     {64},
     static_cast<uint8_t>(0),
     static_cast<uint8_t>(2),
     ge::DT_INT8},

    {"dual_output_scale0_only_vector_quant",
     "SOC_L1_1024",
     "SOC_L1_1024",
     COMPILE_INFO_STR_FUSE,
     ge::DT_FLOAT16,
     ge::DT_INT8,
     ge::DT_INT8,
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
     0,
     0,
     "",
     "",
     true,
     false,
     {},
     static_cast<uint8_t>(2),
     static_cast<uint8_t>(0),
     ge::DT_FLOAT16},
};

constexpr size_t DUAL_OUTPUT_CASE_COUNT = 2;
constexpr size_t ALL_CASE_COUNT = sizeof(cases_params_fuse) / sizeof(cases_params_fuse[0]);
INSTANTIATE_TEST_CASE_P(ExtendConvTransposeV2Cases, ExtendConvTransposeV2TilingRunTime,
                        testing::ValuesIn(cases_params_fuse + ALL_CASE_COUNT - DUAL_OUTPUT_CASE_COUNT,
                                          cases_params_fuse + ALL_CASE_COUNT));

} // namespace
