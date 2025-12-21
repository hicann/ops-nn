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
 * \file test_conv3d_transpose_v2_tiling.cpp
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

#ifdef USE_LEGACY_COMMON

using namespace std;
using namespace ge;

namespace {
extern std::string GetTestSuiteName();
extern std::string GetTestCaseName();

struct Conv3DTransposeV2TilingTestParam {
    string case_name;
    string soc_version;
    string short_soc_version;
    string compile_info;
    string dtype;

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
    int64_t _op_impl_mode_enum;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
    std::string tiling_data_in_repo;
};

class Conv3DTransposeV2TilingRunTime2 : public testing::TestWithParam<Conv3DTransposeV2TilingTestParam> {};

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

static void TestOneParamCase(const Conv3DTransposeV2TilingTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;

    std::string bank_path;
    const char* optype = "Conv3DTransposeV2";
    gert::TilingContext* context = nullptr;

    gert::StorageShape input_size_shape = {param.input_size, param.input_size};
    gert::StorageShape x_shape = {param.x_ori_shape, param.x_shape};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.filter_shape};
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

    auto test_dtype = ge::DT_FLOAT16;
    if (param.dtype == "bfloat16") {
        test_dtype = ge::DT_BF16;
    } else if (param.dtype == "float32") {
        test_dtype = ge::DT_FLOAT;
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<void*>(&platform_info))
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
                           {"offset", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)},
                           {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(param._op_impl_mode_enum)}})
                      .NodeInputTd(0, ge::DT_INT32, param.input_size_format, param.input_size_format)
                      .NodeInputTd(1, test_dtype, param.x_ori_format, param.x_format)
                      .NodeInputTd(2, test_dtype, param.filter_ori_format, param.filter_format)
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
    std::cout << "transpose>>>>>>>>>>>>>>>>>>" << tiling_data_result << std::endl;
    ASSERT_EQ(*tiling_key, param.tiling_key);
    ASSERT_EQ(*block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

static Conv3DTransposeV2TilingTestParam general_20_core_num_cases_params[] = {
    {"Conv3d_transpose_binary_test_kernel_split",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
     "bfloat16",
     {1, 120, 128, 128, 256},
     {1, 62, 66, 66, 256},
     {1, 62, 16, 66, 66, 16},
     {4, 4, 4, 256, 256},
     {1024, 16, 16, 16},
     {1, 120, 128, 128, 256},
     {1, 120, 16, 128, 128, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_DHWCN,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     {1, 2, 2, 2, 1},
     {3, 3, 3, 3, 3, 3},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     0,
     true,
     true,
     20,
     256,
     "1 1 4 1 1 5 20 0 1 256 256 16 16 16 16 16 4 62 66 66 120 128 128 4 4 4 1 2 2 2 3 3 3 3 3 3 0 0 0 0 0 1 1 1 2 2 2 "
     "2 2 1 256 16 16 24 1 128 64 128 1 1 1 1 1 8 8 1 1 1 0 0 0 0 1 0 4096 0 256 0 "},

    {"Conv3d_transpose_binary_test_kernel_split2",
     "Ascend910B3",
     "Ascend910B",
     R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
     "bfloat16",
     {1, 240, 256, 256, 3},
     {1, 122, 130, 130, 256},
     {1, 122, 16, 130, 130, 16},
     {4, 4, 4, 3, 256},
     {64, 16, 16, 16},
     {1, 240, 256, 256, 3},
     {1, 240, 1, 256, 256, 16},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     ge::FORMAT_DHWCN,
     ge::FORMAT_FRACTAL_Z_3D,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDC1HWC0,
     {1, 2, 2, 2, 1},
     {3, 3, 3, 3, 3, 3},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     {0, 0, 0, 0, 0},
     0,
     "VALID",
     0,
     true,
     true,
     20,
     256,
     "1 1 4 1 1 5 20 0 1 3 256 16 1 16 1 16 4 122 130 130 240 256 256 4 4 4 1 2 2 2 3 3 3 3 3 3 0 0 0 0 0 1 1 1 2 2 2 "
     "1 2 1 256 16 1 48 1 256 64 16 1 1 1 1 1 16 32 1 1 1 0 0 0 0 1 0 16384 0 16 0 "}};

Conv3DTransposeV2TilingTestParam general_cases_params[] =
    {
        {"Conv3d_transpose_tiling_dynamic_batch_invalid_C",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "hardware_info": {"CORE_NUM": 24}, "dedy_c1": 233, "tiling_type": "dynamic_tiling","repo_seeds": {"10000": [1,52,112,32]},"repo_range": {"10000": [1,1,24,54,92,122,128,158]},"block_dim": {"CORE_NUM": 24},"_vars": {"10000": ["dedy_d","dedy_h","dedy_w","dedx_d","dedx_h","dedx_w"]}})",
         "bfloat16",
         {1, 24, 32, 92, 128},
         {1, 24, 92, 128, 64},
         {1, 24, 4, 92, 128, 16},
         {3, 3, 3, 32, 64},
         {18, 4, 16, 16},
         {1, 24, 32, 92, 128},
         {1, 24, 2, 92, 128, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {},
         {},
         {},
         1,
         "NDHWC",
         {},
         0,
         "",
         0,
         true,
         false,
         2,
         0,
         "24 24 92 92 128 128 "},

        {"Conv3d_transpose_binary_test_base",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "bfloat16",
         {128, 3, 16, 16, 256},
         {128, 2, 15, 15, 64},
         {128, 2, 4, 15, 15, 16},
         {2, 2, 2, 256, 64},
         {128, 4, 16, 16},
         {128, 3, 16, 16, 256},
         {128, 3, 16, 16, 16, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "VALID",
         0,
         true,
         true,
         24,
         1,
         "8 1 1 1 1 3 24 0 128 256 64 4 16 4 16 16 4 2 15 15 3 16 16 2 2 2 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 1 "
         "2 1 64 4 16 1 1 128 64 256 1 1 1 1 1 4 2 1 1 1 0 0 0 0 16 0 256 0 256 0 "},

        {"Conv3d_transpose_binary_test_padding",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "bfloat16",
         {64, 5, 5, 5, 256},
         {64, 3, 3, 3, 64},
         {64, 3, 4, 3, 3, 16},
         {2, 2, 2, 256, 64},
         {128, 4, 16, 16},
         {64, 5, 5, 5, 256},
         {64, 16, 5, 5, 5, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 2, 2, 2, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "SAME",
         0,
         true,
         true,
         20,
         1,
         "4 1 1 1 1 5 20 0 64 256 64 4 16 4 16 16 4 3 3 3 5 5 5 2 2 2 1 2 2 2 0 1 0 1 0 1 0 1 0 1 0 1 1 1 2 2 1 1 2 1 "
         "64 4 16 1 1 32 64 256 1 1 1 1 1 4 2 1 1 1 0 0 0 0 16 0 25 0 256 0 "},

        {"Conv3d_transpose_binary_test",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "bfloat16",
         {0, 0, 0, 0, 0},
         {63, 2, 15, 15, 64},
         {63, 2, 4, 15, 15, 16},
         {2, 2, 2, 256, 64},
         {128, 4, 16, 16},
         {63, 3, 16, 16, 256},
         {63, 3, 16, 16, 16, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "VALID",
         0,
         true,
         true,
         21,
         1,
         "7 1 1 1 1 3 21 0 63 256 64 4 16 4 16 16 4 2 15 15 3 16 16 2 2 2 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 1 "
         "2 1 64 4 16 1 1 128 64 256 1 1 1 1 1 4 2 1 1 1 0 0 0 0 9 0 256 0 256 0 "},

        {"Conv3d_transpose_binary_test_hit_repo",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "bfloat16",
         {65, 3, 16, 16, 0},
         {65, 2, 15, 15, 64},
         {65, 2, 4, 15, 15, 16},
         {2, 2, 2, 256, 64},
         {128, 4, 16, 16},
         {65, 3, 16, 16, 256},
         {65, 3, 16, 16, 16, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "VALID",
         0,
         true,
         true,
         15,
         1,
         "5 1 1 1 1 3 15 0 65 256 64 4 16 4 16 16 4 2 15 15 3 16 16 2 2 2 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 1 "
         "2 1 64 4 16 1 1 128 64 256 1 1 1 1 1 4 2 1 1 1 0 0 0 0 13 0 256 0 256 0 ",
         R"({"al1_bound":6272,"aub_bound":0,"batch_dim":24,"bl1_bound":32768,"d_dim":1,"d_al1":1,"d_bl1":1,"d_al0":1,"d_bl0":1,"d_cl0":1,"group_dim":1,"k_al1":1,"k_aub":1,"k_bl1":1,"k_l0":1,"m_al1":1,"m_aub":1,"m_dim":1,"m_l0":1,"n_bl1":1,"n_cub":1,"n_dim":1,"n_l0":1,"tiling_id":1,"wo_aub":1})"},

        {"Conv3d_transpose_binary_test_base_fp32",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "float32",
         {128, 3, 16, 16, 256},
         {128, 2, 15, 15, 64},
         {128, 2, 8, 15, 15, 8},
         {2, 2, 2, 256, 64},
         {256, 4, 16, 8},
         {128, 3, 16, 16, 256},
         {128, 3, 32, 16, 16, 8},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "VALID",
         0,
         true,
         true,
         24,
         0,
         "12 1 1 1 2 1 24 0 128 256 64 8 32 8 32 8 3 2 15 15 3 16 16 2 2 2 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 1 2 "
         "1 1 64 8 16 3 1 128 16 256 1 1 1 1 1 2 8 1 1 1 0 0 0 0 11 0 256 0 128 0 "},

        {"Conv3d_transpose_binary_test_padding_fp32",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "float32",
         {64, 5, 5, 5, 256},
         {64, 3, 3, 3, 64},
         {64, 3, 8, 3, 3, 8},
         {2, 2, 2, 256, 64},
         {256, 4, 16, 8},
         {64, 5, 5, 5, 256},
         {64, 32, 5, 5, 5, 8},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 2, 2, 2, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "SAME",
         0,
         true,
         true,
         22,
         0,
         "11 1 1 1 2 1 22 0 64 256 64 8 32 8 32 8 3 3 3 3 5 5 5 2 2 2 1 2 2 2 0 1 0 1 0 1 0 1 0 1 0 1 1 1 2 2 1 2 1 1 "
         "64 8 16 5 1 32 16 256 1 1 1 1 1 2 8 1 1 1 0 0 0 0 6 0 25 0 128 0 "},

        {"Conv3d_transpose_binary_test_padding_hf32",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "float32",
         {64, 5, 5, 5, 256},
         {64, 3, 3, 3, 64},
         {64, 3, 8, 3, 3, 8},
         {2, 2, 2, 256, 64},
         {256, 4, 16, 8},
         {64, 5, 5, 5, 256},
         {64, 32, 5, 5, 5, 8},
         ge::FORMAT_ND,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_DHWCN,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NDHWC,
         ge::FORMAT_NDC1HWC0,
         {1, 2, 2, 2, 1},
         {0, 0, 0, 0, 0, 0},
         {1, 1, 1, 1, 1},
         1,
         "NDHWC",
         {0, 0, 0, 0, 0},
         0,
         "SAME",
         0x40,
         true,
         true,
         22,
         0,
         "11 1 1 1 2 1 22 0 64 256 64 8 32 8 32 8 3 3 3 3 5 5 5 2 2 2 1 2 2 2 0 1 0 1 0 1 0 1 0 1 0 1 1 1 2 2 1 2 1 1 "
         "64 8 16 5 1 32 16 256 1 1 1 1 1 2 8 1 1 1 1 0 0 0 6 0 25 0 128 0 "},

        {"Conv3d_transpose_group",
         "Ascend910B2",
         "Ascend910B",
         R"({"_pattern": "Conv3d_transpose_v2", "tiling_type": "binary", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "aub_num": 0, "bub_num": 0, "cub_num": 0, "ub_size": 196608, "binary_mode": 1})",
         "bfloat16",
         {1, 64, 64, 32, 32},
         {1, 64, 63, 31, 32},
         {1, 63, 4, 31, 32, 16},
         {64, 1, 2, 4, 1},
         {32, 1, 16, 16},
         {1, 64, 64, 32, 32},
         {1, 64, 4, 32, 32, 16},
         ge::FORMAT_ND,
         ge::FORMAT_NCDHW,
         ge::FORMAT_NDC1HWC0,
         ge::FORMAT_NCDHW,
         ge::FORMAT_FRACTAL_Z_3D,
         ge::FORMAT_NCDHW,
         ge::FORMAT_NDC1HWC0,
         {1, 1, 1, 1, 1},
         {0, 0, 1, 1, 0, 0},
         {1, 1, 1, 1, 1},
         64,
         "NCDHW",
         {0, 0, 0, 0, 0},
         0,
         "VALID",
         0,
         true,
         true,
         16,
         1,
         "1 1 16 1 1 1 16 0 1 64 64 4 4 1 1 16 4 63 31 32 64 32 32 2 4 1 4 1 1 1 0 0 1 1 0 0 1 2 2 0 0 1 1 1 2 2 1 1 1 "
         "4 16 1 1 64 1 64 64 16 1 1 1 1 1 1 1 1 1 1 0 0 0 0 1 0 64 0 16 0 "},
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

TEST_P(Conv3DTransposeV2TilingRunTime2, general_cases)
{
    TestOneParamCase(GetParam());
}

TEST_F(Conv3DTransposeV2TilingRunTime2, general_cases_params_multi_thread)
{
    TestMultiThread(general_cases_params, sizeof(general_cases_params) / sizeof(Conv3DTransposeV2TilingTestParam), 3);
    TestMultiThread(
        general_20_core_num_cases_params,
        sizeof(general_20_core_num_cases_params) / sizeof(Conv3DTransposeV2TilingTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(
    MilanBinary, Conv3DTransposeV2TilingRunTime2, testing::ValuesIn(general_cases_params));
INSTANTIATE_TEST_CASE_P(
    MilanBinary2, Conv3DTransposeV2TilingRunTime2, testing::ValuesIn(general_20_core_num_cases_params));
}
#endif