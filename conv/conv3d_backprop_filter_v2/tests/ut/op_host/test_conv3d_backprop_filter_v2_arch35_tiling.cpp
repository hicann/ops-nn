/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "graph/graph.h"
#define private public
#define protected public
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/platform_util.h"

using namespace std;
using namespace ge;
namespace {
struct Conv3DBpFilterV2TilingTestParam {
    string case_name;
    string soc_version;
    string short_soc_version;
    string compile_info;

    std::initializer_list<int64_t> fmap_ori_shape;
    std::initializer_list<int64_t> fmap_shape;
    std::initializer_list<int64_t> filter_ori_shape;
    std::initializer_list<int64_t> filter_shape;
    std::initializer_list<int64_t> out_backprop_ori_shape;
    std::initializer_list<int64_t> out_backprop_shape;

    ge::Format fmap_ori_format;
    ge::Format fmap_format;
    ge::Format filter_ori_format;
    ge::Format filter_format;
    ge::Format out_backprop_ori_format;
    ge::Format out_backprop_format;
    ge::DataType dtype;

    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string data_format;
    std::string padding;

    int32_t deterministic;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
};

class Conv3DBackpropFilterV2TilingRunTime3 : public testing::TestWithParam<Conv3DBpFilterV2TilingTestParam> {};

static string TilingData2Str(const gert::TilingData *tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

static void TestOneParamCase(const Conv3DBpFilterV2TilingTestParam &param)
{
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape filter_sizes = {param.filter_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    gert::StorageShape fmap_shape = {param.fmap_ori_shape, param.fmap_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.filter_ori_shape, param.filter_shape});
    std::vector<void *> output_shapes_ref(1);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(param.compile_info.c_str()), reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    map<string, string> soc_version_infos = {{"SoC_version", param.soc_version},
                                           {"Short_SoC_version", param.short_soc_version}};
                                           
    map<string, string> npuarchs = {{"SoC_version", param.soc_version},
                                           {"Short_SoC_version", param.short_soc_version}, {"NpuArch", param.short_soc_version}};

    std::string op_type("Conv3DBackpropFilterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                          soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                          npuarchs);                                                                                      
    if (param.parse_result) {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_FAILED);
        return;
    }

    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto tiling_data = gert::TilingData::CreateCap(2048);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&fmap_shape, &filter_sizes, &out_backprop_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<void *>(&platform_info))
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .NodeInputTd(0, param.dtype, param.fmap_ori_format, param.fmap_format)
                      .NodeInputTd(1, ge::DT_INT32, param.filter_ori_format, param.filter_format)
                      .NodeInputTd(2, param.dtype, param.out_backprop_ori_format, param.out_backprop_format)
                      .NodeOutputTd(0, ge::DT_FLOAT, param.filter_ori_format, param.filter_format)
                      .DeterministicInfo(reinterpret_cast<int32_t*>(param.deterministic))
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
    ASSERT_EQ(*tiling_key, param.tiling_key);
    ASSERT_EQ(*block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

TEST_P(Conv3DBackpropFilterV2TilingRunTime3, general_cases) { TestOneParamCase(GetParam()); }

const string COMPILE_INFO_STR_950 = R"({"_pattern": "Conv3d_backprop_filter_v2", "tiling_type": "binary",
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

const string COMPILE_INFO_STR_950_36_CORE = R"({"_pattern": "Conv3d_backprop_filter_v2", "tiling_type": "binary",
                          "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 36,
                          "cube_core_cnt": 36, "vector_core_cnt": 72, "core_type_list": "CubeCore,VectorCore"}
                          })";

Conv3DBpFilterV2TilingTestParam cases_params_950[] = {
    {"conv_stdit_01_fp16", "3510", "3510", COMPILE_INFO_STR_950,
     {33, 4, 1, 32, 32}, {33, 4, 1, 32, 32}, {1152, 4, 1, 2, 2}, {1152, 4, 1, 2, 2}, {33, 1152, 1, 16, 16}, {33, 1152, 1, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_FLOAT16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 32, 1, "1 0 1 1 1 1 1 524288 33 4 1152 4 1152 1 16 16 1 32 32 1 2 2 1 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 144 112 64 16 16 16 1 1 3 3 1 21504 48384 0 1 1 144 16 16 0 1 0 16 0 9 0 1 32 144 64 256 0 "},

    {"conv_stdit_01_hifp8", "3510", "3510", COMPILE_INFO_STR_950,
     {16, 4, 1, 32, 32}, {16, 4, 1, 32, 32}, {1152, 4, 1, 2, 2}, {1152, 4, 1, 2, 2}, {16, 1152, 1, 16, 16}, {16, 1152, 1, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_HIFLOAT8,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, false, 32, 0, ""},

    {"conv_stdit_01_depthwise", "3510", "3510", COMPILE_INFO_STR_950,
     {1, 512, 1, 32, 32}, {1, 512, 1, 32, 32}, {512, 1, 1, 2, 2}, {512, 1, 1, 2, 2}, {1, 512, 1, 16, 16}, {1, 512, 1, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 512, "NCDHW", "VALID", 0,
     true, true, 32, 2, "1 0 32 1 1 1 1 524288 1 512 512 16 16 1 16 16 1 32 32 1 2 2 512 32 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 256 64 16 16 16 1 1 1 1 0 16384 4096 0 1 1 16 16 16 0 1 0 16 0 1 0 0 32 16 64 256 0 "},

    {"conv_stdit_01_group", "3510", "3510", COMPILE_INFO_STR_950,
     {1, 4, 1, 32, 32}, {1, 4, 1, 32, 32}, {1152, 1, 1, 2, 2}, {1152, 1, 1, 2, 2}, {1, 1152, 1, 16, 16}, {1, 1152, 1, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 4, "NCDHW", "VALID", 0,
     true, true, 32, 1, "1 0 4 1 1 1 1 524288 1 4 1152 1 288 1 16 16 1 32 32 1 2 2 4 4 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 144 64 64 16 16 16 1 1 1 1 1 4096 9216 0 1 1 144 4 16 0 1 0 16 0 1 0 2 32 144 64 64 0 "},

    {"magvit_07_b16_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
     {16, 256, 22, 34, 34}, {16, 256, 22, 34, 34}, {256, 256, 3, 3, 3}, {256, 256, 3, 3, 3}, {16, 256, 20, 32, 32}, {16, 256, 20, 32, 32},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
     true, true, 32, 2, "1 0 1 1 1 1 1 524288 16 256 256 256 256 20 32 32 22 34 34 3 3 3 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 1 2 2 256 64 144 16 16 16 1 1 7 7 0 8704 114688 0 1 1 256 32 32 0 1 0 16 0 160 0 1 32 256 144 1024 0 "},

    {"conv_NCDHW_vqvae_net_ID_1_modified_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 128, 17, 129, 129}, {1, 128, 17, 129, 129}, {128, 128, 1, 3, 3}, {128, 128, 1, 3, 3}, {1, 128, 17, 64, 64}, {1, 128, 17, 64, 64},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 128 128 128 128 17 64 64 17 129 129 1 3 3 1 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 64 144 16 16 16 1 1 8 8 0 35088 65536 0 1 1 128 64 64 0 1 0 16 0 5 0 1 32 128 144 4096 0 "},

    {"aclnnConvolutionBackward_bf16_NCDHW_convPlan1_000001_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 256, 8, 128, 128}, {1, 256, 8, 128, 128}, {256, 256, 4, 4, 4}, {256, 256, 4, 4, 4}, {1, 256, 6, 66, 66}, {1, 256, 6, 66, 66},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 256 256 256 256 6 66 66 8 128 128 4 4 4 1 1 2 2 2 3 3 3 3 3 3 1 1 1 16 2 2 1 2 2 256 64 256 16 16 16 1 1 6 6 0 32768 98304 0 1 1 256 66 66 0 1 0 16 0 6 0 0 32 256 256 4356 0 "},

    {"aclnnConvolutionBackward_bf16_NCDHW_convPlan1_000002_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 128, 4, 64, 64}, {1, 128, 4, 64, 64}, {256, 128, 1, 1, 1}, {256, 128, 1, 1, 1}, {1, 256, 4, 64, 64}, {1, 256, 4, 64, 64},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 128 256 128 256 4 64 64 4 64 64 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 256 64 128 16 16 16 1 1 2 2 1 16384 32768 0 1 1 256 2 64 0 1 0 128 0 4 0 2 32 256 128 128 0 "},

    {"aclnnConvolutionBackward_3DDW_conv_bfloat16_NCDHW_net_ID4403_0002_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {2, 4, 16, 34, 46}, {2, 4, 16, 34, 46}, {1152, 4, 1, 2, 2}, {1152, 4, 1, 2, 2}, {2, 1152, 16, 17, 23}, {2, 1152, 16, 17, 23},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 2 4 1152 4 1152 16 17 23 16 34 46 1 2 2 1 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 144 112 64 16 16 16 1 1 4 4 1 30912 64512 0 1 1 144 17 23 0 1 0 16 0 8 0 1 32 144 64 391 0 "},

    {"aclnnConvolutionBackward_3DDW_SplitW_001", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 4, 15, 4, 539}, {1, 4, 15, 4, 539}, {2, 4, 12, 3, 3}, {2, 4, 12, 3, 3}, {1, 2, 5, 4, 539}, {1, 2, 5, 4, 539},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 3, 1, 1}, {4, 5, 6, 6, 6, 6}, {1, 1, 1, 6, 6}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 4 2 4 2 5 4 539 15 4 539 12 3 3 1 1 3 1 1 4 5 6 6 6 6 1 6 6 16 2 2 2 2 2 16 112 144 16 16 16 1 1 8 8 0 42560 15360 0 1 1 16 4 128 0 1 0 16 0 3 0 1 32 16 144 2156 0 "},

    // NDHWC key=1
    {"3DDW_bfloat16_NDHWC_1_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {2, 1, 28, 28, 224}, {2, 1, 28, 28, 224}, {128, 1, 1, 1, 224}, {128, 1, 1, 1, 224}, {2, 1, 28, 28, 128}, {2, 1, 28, 28, 128},
    ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", "VALID", 1,
    true, true, 32, 2, "1 0 1 1 1 1 1 524288 2 224 128 224 128 1 28 28 1 28 28 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 64 224 16 16 16 1 1 5 5 0 81536 40960 0 1 1 128 28 28 0 1 0 224 0 2 0 0 32 128 224 784 0 "},

    // NDHWC KEY=3
    {"3DDW_bfloat16_NDHWC_2_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 16, 16, 256}, {1, 1, 16, 16, 256}, {512, 1, 3, 3, 256}, {512, 1, 3, 3, 256}, {1, 1, 16, 16, 512}, {1, 1, 16, 16, 512},
    ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NDHWC", "VALID", 1,
    true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 256 512 256 512 1 16 16 1 16 16 1 3 3 1 1 1 1 1 0 0 1 1 1 1 1 1 1 16 2 2 1 2 2 256 64 144 16 16 16 1 1 4 4 0 4608 65536 0 1 1 256 16 16 0 1 0 16 0 1 0 0 32 256 144 256 0 "},

    // NDHWC FP32
    {"3DDW_float32_NDHWC_1_deterministic", "3510", "3510", COMPILE_INFO_STR_950,
    {2, 1, 28, 28, 224}, {2, 1, 28, 28, 224}, {128, 1, 1, 1, 224}, {128, 1, 1, 1, 224}, {2, 1, 28, 28, 128}, {2, 1, 28, 28, 128},
    ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", "VALID", 1,
    true, true, 32, 2, "1 0 1 1 1 1 1 524288 2 224 128 224 128 1 28 28 1 28 28 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 8 2 2 2 2 2 128 32 224 16 8 16 1 1 5 5 0 43904 20480 0 1 1 128 28 28 0 1 0 224 0 2 0 0 32 128 224 784 0 "},

    // NDHWC group
    {"3DDW_bfloat16_NDHWC_1_deterministic_group", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 16, 16, 720}, {1, 1, 16, 16, 720}, {720, 1, 1, 1, 120}, {720, 1, 1, 1, 120}, {1, 1, 16, 16, 720}, {1, 1, 16, 16, 720},
    ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 6, "NDHWC", "VALID", 1,
    true, true, 32, 2, "1 0 6 1 1 1 1 524288 1 720 720 120 120 1 16 16 1 16 16 1 1 1 6 6 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 2 2 0 32768 32768 0 1 1 128 16 16 0 1 0 128 0 1 0 0 32 128 128 256 0 "},

     // groupEnlarge dose not allow to adjust baseBlock
    {"3DDW_group_enlarge_baseM80", "3510", "3510", COMPILE_INFO_STR_950,
     {17, 14, 1, 1, 240}, {17, 14, 1, 1, 240}, {78, 7, 1, 1, 1}, {78, 7, 1, 1, 1}, {17, 78, 1, 1, 240}, {17, 78, 1, 1, 240},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_FLOAT16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 2, "NCDHW", "VALID", 1,
     true, true, 32, 1, "1 0 1 1 1 1 1 524288 17 14 78 14 78 1 1 240 1 1 240 1 1 1 2 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 80 192 16 16 16 16 1 1 2 2 1 11520 30720 0 1 1 80 1 240 0 1 0 16 0 1 0 1 32 80 16 240 0 "},

     // when batchdout=1 or howo=1,streamkType will set 0, and tilingKey must be 2
    {"3DDW_all_1", "3510", "3510", COMPILE_INFO_STR_950,
     {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_FLOAT16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
     true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 16 16 16 16 1 1 1 1 0 256 256 0 1 1 16 1 1 0 1 0 16 0 1 0 0 32 16 16 1 0 "},

    {"aclnnConvolutionBackward_3DDW_DilationMin", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 1, 12, 12}, {1, 1, 1, 12, 12}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 11, 11}, {1, 1, 1, 11, 11},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 11 11 1 12 12 1 2 2 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 64 16 16 16 1 1 1 1 0 768 256 0 1 1 16 1 11 0 1 0 16 0 1 0 2 32 16 64 11 0 "},

    {"aclnnConvolutionBackward_3DDW_DilationMax", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 1, 12, 12}, {1, 1, 1, 12, 12}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 11, 11}, {1, 1, 1, 11, 11},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 2147483646, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 11 11 1 12 12 1 2 2 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 64 16 16 16 1 1 1 1 0 768 256 0 1 1 16 1 11 0 1 0 16 0 1 0 2 32 16 64 11 0 "},

    {"aclnnConvolutionBackward_3DDW_DilationOver", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 1, 12, 12}, {1, 1, 1, 12, 12}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 11, 11}, {1, 1, 1, 11, 11},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 2147483647, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, false, 32, 1, ""},

    {"aclnnConvolutionBackward_3DDW_StrideMin", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 1, 3, 301}, {1, 1, 1, 3, 301}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 300}, {1, 1, 1, 2, 300},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 2 300 1 3 301 1 2 2 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 256 64 16 16 16 1 1 2 2 0 19264 8192 0 1 1 16 1 300 0 1 0 16 0 1 0 2 32 16 64 300 0 "},

 	 {"aclnnConvolutionBackward_3DDW_StrideHMax", "3510", "3510", COMPILE_INFO_STR_950,
 	 {1, 1, 1, 2147483646, 1}, {1, 1, 1, 2147483646, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
 	 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
 	 {1, 1, 1, 2147483646, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
 	 true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 1 1 1 2147483646 1 1 1 1 1 1 1 2147483646 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 16 16 16 16 1 1 1 1 0 256 256 0 1 1 16 1 1 1 1 0 16 0 1 0 0 32 16 16 1 0 "},

 	 {"aclnnConvolutionBackward_3DDW_StrideWMax", "3510", "3510", COMPILE_INFO_STR_950,
 	 {1, 1, 1, 1, 2147483646}, {1, 1, 1, 1, 2147483646}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
 	 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
 	 {1, 1, 1, 1, 2147483646}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
 	 true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 1 1 1 1 2147483646 1 1 1 1 1 1 1 2147483646 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 16 16 16 16 1 1 1 1 0 256 256 0 1 1 16 1 1 1 1 0 16 0 1 0 0 32 16 16 1 0 "},

 	 {"aclnnConvolutionBackward_3DDW_StrideWMax_NDHWC", "3510", "3510", COMPILE_INFO_STR_950,
 	 {1, 1, 1, 2147483646, 1}, {1, 1, 1, 2147483646, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1},
 	 ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::DT_BF16,
 	 {1, 1, 1, 2147483646, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", "VALID", 1,
 	 true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 1 1 1 1 2147483646 1 1 1 1 1 1 1 2147483646 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 16 16 16 16 16 1 1 1 1 0 256 256 0 1 1 16 1 1 1 1 0 16 0 1 0 0 32 16 16 1 0 "},

 	 {"aclnnConvolutionBackward_3DDW_SplitW0", "3510", "3510", COMPILE_INFO_STR_950,
 	 {16, 1, 1, 962, 32770}, {16, 1, 1, 962, 32770}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {16, 1, 1, 16, 513}, {16, 1, 1, 16, 513},
 	 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
 	 {1, 1, 1, 64, 64}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
 	 true, true, 32, 1, "1 0 1 1 1 1 1 524288 16 1 1 1 1 1 16 513 1 962 32770 1 2 2 1 1 1 64 64 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 256 64 16 16 16 1 1 8 8 0 32768 32768 0 1 1 16 16 128 1 1 0 16 0 1 0 1 32 16 64 8208 0 "},

 	 {"aclnnConvolutionBackward_3DDW_ShrinkBaseBlock", "3510", "3510", COMPILE_INFO_STR_950,
 	 {1, 3, 1, 54, 142}, {1, 3, 1, 54, 142}, {1, 3, 1, 1, 40}, {1, 3, 1, 1, 40}, {1, 3, 1, 1, 3}, {1, 3, 1, 1, 3},
 	 ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
 	 {1, 1, 1, 61, 57}, {0, 0, 0, 0, 25, 26}, {1, 1, 1, 8, 2}, 1, "NCDHW", "VALID", 1,
 	 true, true, 32, 2, "1 0 1 1 1 1 1 524288 1 3 3 3 3 1 1 3 1 54 142 1 1 40 1 1 1 61 57 0 0 0 0 25 26 1 1 2 16 2 2 2 2 2 16 16 224 16 16 16 1 1 1 1 0 336 512 0 1 1 16 1 16 1 1 0 16 0 1 0 0 32 16 640 3 0 "},

    {"aclnnConvolutionBackward_3DDW_CheckHoShapeErr", "3510", "3510", COMPILE_INFO_STR_950_36_CORE,
    {32, 16, 1, 1, 1}, {32, 16, 1, 1, 1}, {16, 16, 1, 1, 3}, {16, 16, 1, 1, 3}, {32, 16, 1, 1, 751},{32, 16, 1, 1, 751},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 1, 2}, {0, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, false, 36, 1, ""},

    {"aclnnConvolutionBackward_3DDW_CheckWoShapeErr", "3510", "3510", COMPILE_INFO_STR_950_36_CORE,
    {32, 16, 1, 3, 1}, {32, 16, 1, 3, 1}, {16, 16, 1, 1, 3}, {16, 16, 1, 1, 3}, {32, 16, 1, 1, 1},{32, 16, 1, 1, 1},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 1, 2}, {0, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, false, 36, 1, ""},

    {"aclnnConvolutionBackward_3DDW_CheckDoShapeErr", "3510", "3510", COMPILE_INFO_STR_950_36_CORE,
    {32, 16, 3, 1, 1}, {32, 16, 3, 1, 1}, {16, 16, 1, 1, 3}, {16, 16, 1, 1, 3}, {32, 16, 1, 1, 1},{32, 16, 1, 1, 1},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 1, 2}, {0, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, false, 36, 1, ""},
    
    {"aclnnConvolutionBackward_3DDW_padW256", "3510", "3510", COMPILE_INFO_STR_950,
    {64, 16, 5, 9, 38}, {64, 16, 5, 9, 38}, {64, 16, 4, 5, 10}, {164, 16, 4, 5, 10}, {16, 16, 2, 5, 10}, {16, 16, 2, 5, 10},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 1, 60}, {0, 0, 0, 0, 256, 256}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 16 16 16 16 16 2 5 10 5 9 38 4 5 10 1 1 1 1 60 0 0 0 0 256 256 1 1 1 16 2 2 2 2 2 16 64 208 16 16 16 1 1 1 1 0 1280 1024 0 1 1 16 5 10 1 1 0 16 0 4 0 1 32 16 800 50 0 "},

    {"aclnnConvolutionBackward_3DDW_padH256", "3510", "3510", COMPILE_INFO_STR_950,
    {64, 16, 5, 443, 518}, {64, 16, 5, 443, 518}, {64, 16, 4, 5, 10}, {164, 16, 4, 5, 10}, {16, 16, 2, 20, 10}, {16, 16, 2, 20, 10},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::DT_BF16,
    {1, 1, 1, 50, 60}, {0, 0, 256, 256, 16, 16}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 16 16 16 16 16 2 20 10 5 443 518 4 5 10 1 1 1 50 60 0 0 256 256 16 16 1 1 1 16 2 2 2 2 2 16 64 208 16 16 16 1 1 4 4 0 4320 4096 0 1 1 16 20 10 1 1 0 16 0 4 0 1 32 16 800 200 0 "},

    {"aclnnConvolutionBackward_3DDW_AdjustSmallBlock", "3510", "3510", COMPILE_INFO_STR_950_36_CORE,
    {32, 1024, 1, 14, 14}, {32, 1024, 1, 14, 14}, {1024, 1024, 1, 1, 1}, {1024, 1024, 1, 1, 1}, {32, 1024, 1, 14, 14},{32, 1024, 1, 14, 14},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 36, 1, "1 0 1 1 1 1 1 524288 32 1024 1024 1024 1024 1 14 14 1 14 14 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 8 2 2 1 2 2 256 32 256 16 8 16 1 1 3 3 0 28672 24576 0 1 1 256 14 14 0 1 0 256 0 16 0 1 36 256 256 196 0 "},

    {"aclnnConvolutionBackward_2DDW_SplitKernelOnKStartPtExceedLimit", "3510", "3510", COMPILE_INFO_STR_950,
    {1, 1, 1, 625, 1}, {1, 1, 1, 625, 1}, {1, 1, 1, 172, 142}, {1, 1, 1, 172, 142}, {1, 1, 1, 192, 43},{1, 1, 1, 192, 43},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 3, 3}, {0, 0, 60, 60, 134, 134}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 1 1 1 1 1 1 192 43 1 625 1 1 172 142 1 1 1 3 3 0 0 60 60 134 134 1 1 1 8 2 2 2 2 2 16 32 256 16 8 16 1 1 8 8 0 4816 4096 0 1 1 16 6 43 1 1 0 16 0 1 0 2 32 16 390784 258 0 "},

    {"aclnnConvolutionBackward_2DDW_AdjustSmallCaseBaseBlockForSingleCoreN", "3510", "3510", COMPILE_INFO_STR_950,
    {21, 18, 1, 570, 4155}, {21, 18, 1, 570, 4155}, {18, 18, 1, 15, 15}, {18, 18, 1, 15, 15}, {21, 18, 1, 38, 277},{21, 18, 1, 38, 277},
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,ge::DT_FLOAT,
    {1, 1, 1, 15, 15}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
    true, true, 32, 1, "1 0 1 1 1 1 1 524288 21 18 18 18 18 1 38 277 1 570 4155 1 15 15 1 1 1 15 15 0 0 0 0 0 0 1 1 1 8 2 2 2 1 1 32 16 256 16 8 16 1 1 1 1 0 90000 1024 0 1 1 32 3 16 0 1 0 16 0 21 0 2 32 32 3600 831 0 "}
};

static void ThreadFunc(const Conv3DBpFilterV2TilingTestParam *params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const Conv3DBpFilterV2TilingTestParam *params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(Conv3DBackpropFilterV2TilingRunTime3, cases_params_950_thread)
{
    TestMultiThread(cases_params_950, sizeof(cases_params_950) / sizeof(Conv3DBpFilterV2TilingTestParam), 1);
}

INSTANTIATE_TEST_CASE_P(CONV3DDW950, Conv3DBackpropFilterV2TilingRunTime3, testing::ValuesIn(cases_params_950));
}
