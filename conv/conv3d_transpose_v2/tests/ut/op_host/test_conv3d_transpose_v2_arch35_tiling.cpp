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
 * \file test_conv3d_transpose_v2_tiling_arch35.cpp
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

struct Conv3DTransposeV2TilingTestParam {
    string case_name;
    string soc_version;
    string short_soc_version;

    std::initializer_list<int64_t> input_size;
    std::initializer_list<int64_t> x_ori_shape;
    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> filter_ori_shape;
    std::initializer_list<int64_t> filter_shape;
    std::initializer_list<int64_t> y_ori_shape;
    std::initializer_list<int64_t> y_shape;

    ge::Format input_size_format;
    ge::Format x_ori_format;
    ge::Format x_format;
    ge::Format filter_ori_format;
    ge::Format filter_format;
    ge::Format y_ori_format;
    ge::Format y_format;

    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string data_format;
    std::vector<int64_t> output_padding;
    int64_t offset;
    std::string padding;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
    std::string tiling_data_in_repo;
};

const string COMPILE_INFO_STR_950 = R"({
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

const string COMPILE_INFO_STR_910B = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";

class Conv3DTransposeV2TilingRunTime3 : public testing::TestWithParam<Conv3DTransposeV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3DTransposeV2TilingRunTime3 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv3DTransposeV2TilingRunTime3 TearDown" << std::endl;
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

static void TestOneParamCase(const Conv3DTransposeV2TilingTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape input_size_shape = {param.input_size, param.input_size};
    gert::StorageShape x_shape = {param.x_ori_shape, param.x_shape};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_ori_shape, param.y_shape});
    std::vector<void*> output_shapes_ref(1);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    string compileInfoPtr = COMPILE_INFO_STR_910B;

    if (param.soc_version.compare("3510") == 0) {
        compileInfoPtr = COMPILE_INFO_STR_950;
    }
    GetPlatFormInfos(compileInfoPtr.c_str(), soc_infos, aicore_spec, intrinsics);
    aicore_spec["cube_freq"] = "1800";

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoPtr.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    map<string, string> soc_version_infos = {
        {"SoC_version", param.soc_version}, {"Short_SoC_version", param.soc_version},{"NpuArch", param.short_soc_version}};

    std::string op_type("Conv3DTransposeV2");
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

    size_t total_size = 0;
    std::vector<int64_t> input_size(param.input_size);
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size.size(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(param.input_size_format);
    tensor->SetStorageFormat(param.input_size_format);

    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
        input_size.size() * sizeof(int64_t));

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
                           {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"offset", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .NodeInputTd(0, ge::DT_INT32, param.input_size_format, param.input_size_format)
                      .NodeInputTd(1, ge::DT_BF16, param.x_ori_format, param.x_format)
                      .NodeInputTd(2, ge::DT_BF16, param.filter_ori_format, param.filter_format)
                      .NodeOutputTd(0, ge::DT_BF16, param.y_ori_format, param.y_format)
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

Conv3DTransposeV2TilingTestParam cases_params_950_case[] = {
    {"conv3d_transpose_1",
     "3510",
     "3510",
     {1, 2, 1, 4, 4},
     {1, 2, 1, 4, 4},
     {1, 2, 1, 4, 4},
     {2, 2, 5, 2, 2},
     {2, 2, 5, 2, 2},
     {1, 2, 5, 5, 5},
     {1, 2, 5, 5, 5},
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
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     true,
     true,
     5,
     50331648,
     "1 1 1 1 1 1 5 0 2 2 2 2 2 1 16 4 4 1 0 0 0 1 2 2 2 2 1 1 1 1 1 4 4 5 5 5 5 2 2 1 1 1 1 1 0 0 0 0 0 0 4 1 1 1 1 1 1 1 1 2 16 1 32 16 16 1 1 1 1 5 0 1 0 30 0 0 0 0 0 "},
    {"conv3d_transpose_2_group",
     "Ascend910_95",
     "3510",
     {1, 16, 1, 4, 4},
     {1, 16, 1, 4, 4},
     {1, 16, 1, 4, 4},
     {16, 1, 5, 2, 2},
     {16, 1, 5, 2, 2},
     {1, 16, 5, 5, 5},
     {1, 16, 5, 5, 5},
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
     16,
     "NCDHW",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     true,
     true,
     5,
     16842754,
     "1 1 1 1 1 1 5 0 2 2 2 2 2 1 16 4 4 16 0 0 0 1 16 16 16 16 1 1 1 1 1 4 4 5 5 5 5 2 2 1 16 1 1 1 0 0 0 0 0 0 4 1 1 1 1 1 1 1 1 16 16 1 32 64 16 1 1 1 1 5 0 1 0 30 0 0 0 0 0 "},
    {"conv3d_transpose_3_general_group",
     "Ascend910_95",
     "3510",
     {1, 64, 1, 4, 4},
     {1, 64, 1, 4, 4},
     {1, 64, 1, 4, 4},
     {64, 4, 5, 2, 2},
     {64, 4, 5, 2, 2},
     {1, 64, 5, 5, 5},
     {1, 64, 5, 5, 5},
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
     16,
     "NCDHW",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     true,
     true,
     20,
     16842754,
     "1 1 1 1 1 1 20 0 2 2 2 2 2 1 16 4 4 4 0 0 0 1 64 64 16 16 4 4 1 1 1 4 4 5 5 5 5 2 2 4 16 1 1 1 0 0 0 0 0 0 4 1 1 1 1 1 1 1 1 16 16 1 32 64 16 1 1 1 1 5 0 1 0 30 0 0 0 0 0 "},
    {"conv3d_transpose_4_general_group1_ncdhw_dhwcn",
     "3510",
     "3510",
     {1, 2, 1, 4, 4},
     {1, 2, 1, 4, 4},
     {1, 2, 1, 4, 4},
     {5, 2, 2, 2, 2},
     {5, 2, 2, 2, 2},
     {1, 2, 5, 5, 5},
     {1, 2, 5, 5, 5},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_DHWCN,
     ge::FORMAT_DHWCN,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     true,
     true,
     5,
     16777218,
     "1 1 1 1 1 1 5 0 2 2 2 2 2 1 16 4 4 1 0 0 0 1 2 2 2 2 1 1 1 1 1 4 4 5 5 5 5 2 2 1 1 1 1 1 0 0 0 0 0 0 4 1 1 1 1 1 1 1 1 2 16 1 32 64 16 1 1 1 1 5 0 1 0 30 0 0 0 0 0 "},
    {"conv3d_transpose_5_general_group1_ndhwc_dhwcn",
     "3510",
     "3510",
     {1, 1, 4, 4, 2},
     {1, 1, 4, 4, 2},
     {1, 1, 4, 4, 2},
     {5, 2, 2, 2, 2},
     {5, 2, 2, 2, 2},
     {1, 5, 5, 5, 2},
     {1, 5, 5, 5, 2},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_DHWCN,
     ge::FORMAT_DHWCN,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     true,
     true,
     5,
     16777218,
     "1 1 1 1 1 1 5 0 2 2 2 2 2 1 16 4 4 1 0 0 0 1 2 2 2 2 1 1 1 1 1 4 4 5 5 5 5 2 2 1 1 1 1 1 0 0 0 0 0 0 4 1 1 1 1 1 1 1 1 2 16 1 32 64 16 1 1 1 1 5 0 1 0 30 0 0 0 0 0 "},
};

static void ThreadFunc(
    const Conv3DTransposeV2TilingTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const Conv3DTransposeV2TilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(Conv3DTransposeV2TilingRunTime3, general_cases_params_multi_thread)
{
    TestMultiThread(
        cases_params_950_case, sizeof(cases_params_950_case) / sizeof(Conv3DTransposeV2TilingTestParam), 3);
}

TEST_P(Conv3DTransposeV2TilingRunTime3, general_cases)
{
    TestOneParamCase(GetParam());
}

INSTANTIATE_TEST_CASE_P(
    Conv3DTranpose950_case, Conv3DTransposeV2TilingRunTime3, testing::ValuesIn(cases_params_950_case));

} // namespace
