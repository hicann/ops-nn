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

class Conv3DBackpropFilterV2TilingRunTime2 : public testing::TestWithParam<Conv3DBpFilterV2TilingTestParam> {};

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

TEST_P(Conv3DBackpropFilterV2TilingRunTime2, general_cases) { TestOneParamCase(GetParam()); }

Conv3DBpFilterV2TilingTestParam milan_binary_params[] = {
    {"stdit_01_fp16", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 4, 1, 32, 32}, {1, 1, 1, 32, 32, 16}, {1152, 4, 1, 2, 2}, {4, 72, 16, 16}, {1, 1152, 1, 16, 16}, {1, 1, 72, 16, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 0, "1 0 1 12 2 1 1 524288 1 4 1152 1 72 1 16 16 1 32 32 1 2 2 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 1 1 96 128 64 16 16 16 1 1 1 1 1 8192 0 1 1 96 8 1 0 16 0 0 0 0 0 0 0 "},

    {"stdit_01_bf16", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 4, 1, 32, 32}, {1, 1, 1, 32, 32, 16}, {1152, 4, 1, 2, 2}, {4, 72, 16, 16}, {1, 1152, 1, 16, 16}, {1, 1, 72, 16, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 0, "1 0 1 12 2 1 1 524288 1 4 1152 1 72 1 16 16 1 32 32 1 2 2 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 1 1 96 128 64 16 16 16 1 1 1 1 1 8192 0 1 1 96 8 1 0 16 0 0 0 0 0 0 0 "},

    {"03_b16", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 256, 17, 256, 256}, {1, 17, 16, 256, 256, 16}, {128, 256, 1, 1, 1}, {16, 8, 16, 16}, {1, 128, 17, 256, 256}, {1, 17, 8, 256, 256, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 1, "17 0 1 1 1 1 1 524288 1 256 128 16 8 17 256 256 17 256 256 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 4 4 0 65536 0 1 1 128 2 1 0 256 0 1 24 128 256 512 0 "},

    {"01_b16", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 128, 17, 257, 257}, {1, 17, 8, 257, 257, 16}, {128, 128, 1, 3, 3}, {72, 8, 16, 16}, {1, 128, 17, 128, 128}, {1, 17, 8, 128, 128, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 1, "1 0 1 1 5 4 1 524288 1 128 128 8 8 17 128 128 17 257 257 1 3 3 1 1 2 2 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 4 2 0 41120 0 1 1 128 4 17 0 128 0 1 20 128 1152 512 0 "},

     {"25_b16", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 512, 11, 64, 64}, {1, 11, 32, 64, 64, 16}, {512, 512, 3, 3, 3}, {864, 32, 16, 16}, {1, 512, 9, 64, 64}, {1, 9, 32, 64, 64, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 1, "1 0 1 4 5 1 1 524288 1 512 512 32 32 9 64 64 11 64 64 3 3 3 1 1 1 1 0 0 1 1 1 1 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 4 4 0 20480 0 1 1 128 8 9 0 1536 0 1 20 128 13824 512 0 "},

    {"07_b16", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {16, 256, 22, 34, 34}, {1, 22, 16, 34, 34, 16}, {256, 256, 3, 3, 3}, {432, 32, 16, 16}, {16, 256, 20, 32, 32}, {16, 20, 16, 32, 32, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 1, "20 0 1 1 1 1 1 524288 16 256 256 16 16 20 32 32 22 34 34 3 3 3 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 4 4 0 19584 0 1 1 128 16 16 0 768 0 1 20 128 6912 512 0 "},

    {"26_b16", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 512, 7, 32, 32}, {1, 7, 32, 32, 32, 16}, {8, 512, 3, 3, 3}, {864, 1, 16, 16}, {1, 8, 5, 32, 32}, {1, 5, 1, 32, 32, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 0, "5 0 1 1 1 4 1 524288 1 512 8 32 1 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 1 1 1 16 2 2 1 2 2 16 32 512 16 16 16 1 1 32 32 1 98304 0 1 1 16 32 1 0 384 0 0 0 0 0 0 0 "},

     {"conv3d_backprop_filter_v2_bf16_ncdhw_0136", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 156, 2, 184, 193}, {1, 2, 10, 184, 193, 16}, {92, 156, 1, 1, 45}, {450, 6, 16, 16}, {1, 92, 2, 184, 75}, {1, 2, 6, 184, 75, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 0, "2 0 1 1 10 1 1 524288 1 156 92 10 6 2 184 75 2 184 193 1 1 45 1 1 1 2 0 0 0 0 0 0 1 1 1 16 2 2 1 1 1 96 48 240 16 16 16 1 1 29 29 1 61760 0 1 1 96 19 1 0 160 0 0 0 0 0 0 0 "},

    {"stdit_01_fp32", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 4, 1, 32, 32}, {1, 1, 1, 32, 32, 16}, {1152, 4, 1, 2, 2}, {4, 72, 16, 16}, {1, 1152, 1, 16, 16}, {1, 1, 72, 16, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 0, "1 0 1 12 2 1 1 524288 1 4 1152 1 144 1 16 16 1 32 32 1 2 2 1 1 2 2 0 0 0 0 0 0 1 1 1 8 2 2 2 1 1 96 64 64 16 8 16 1 1 2 2 1 8192 0 1 1 96 8 1 0 8 0 0 0 0 0 0 0 "},

    {"float32_NCDHW_net_ID_1", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
    {1, 128, 17, 257, 257}, {1, 17, 16, 257, 257, 8}, {128, 128, 1, 3, 3}, {144, 16, 16, 8}, {1, 128, 17, 128, 128}, {1, 17, 16, 128, 128, 8},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
    true, true, 24, 0, "1 0 1 1 6 4 1 524288 1 128 128 16 16 17 128 128 17 257 257 1 3 3 1 1 2 2 0 0 0 0 0 0 1 1 1 8 2 2 1 2 2 128 48 144 16 8 16 1 1 3 3 1 28784 0 1 1 128 22 17 0 32 0 0 0 0 0 0 0 "},

    {"float32_NCDHW_net_ID_5", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
    {1, 3, 19, 256, 256}, {1, 19, 1, 256, 256, 8}, {128, 3, 3, 3, 3}, {27, 16, 16, 8}, {1, 128, 17, 256, 256}, {1, 17, 16, 256, 256, 8},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME", 0,
    true, true, 24, 0, "1 0 1 1 24 1 1 524288 1 3 128 1 16 17 256 256 19 256 256 3 3 3 1 1 1 1 0 0 1 1 1 1 1 1 1 8 2 2 1 2 1 128 48 144 16 8 16 1 1 3 57 1 57344 0 1 1 128 11 17 0 24 0 0 0 0 0 0 0 "},

    {"float32_NCDHW_net_ID_15", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
    {1, 4, 5, 32, 32}, {1, 5, 1, 32, 32, 8}, {4, 4, 1, 1, 1}, {1, 1, 16, 8}, {1, 4, 5, 32, 32}, {1, 5, 1, 32, 32, 8},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
    true, true, 20, 0, "5 0 1 1 4 1 1 524288 1 4 4 1 2 5 32 32 5 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 8 2 2 2 1 1 16 256 16 16 8 16 1 1 1 1 1 4096 0 1 1 16 8 1 0 8 0 0 0 0 0 0 0 "},

    {"float32_NCDHW_net_ID_26", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
    {1, 512, 7, 32, 32}, {1, 7, 64, 32, 32, 8}, {8, 512, 3, 3, 3}, {1728, 1, 16, 8}, {1, 8, 5, 32, 32}, {1, 5, 1, 32, 32, 8},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME", 0,
    true, true, 24, 0, "1 0 1 1 8 3 1 524288 1 512 8 64 2 5 32 32 7 32 32 3 3 3 1 1 1 1 0 0 1 1 1 1 1 1 1 8 2 2 2 1 1 16 16 512 16 8 16 1 1 8 8 1 15360 0 1 1 16 4 5 0 512 0 0 0 0 0 0 0 "},

    {"c256_case", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
    {1, 256, 11, 64, 64}, {1, 11, 16, 64, 64, 16}, {256, 1, 3, 3, 3}, {432, 1, 16, 16}, {1, 256, 9, 64, 64}, {1, 9, 16, 64, 64, 16},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
    {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 256, "NCDHW", "VALID", 0,
    true, true, 20, 0, "1 0 4 1 5 1 1 524288 1 256 256 1 1 9 64 64 11 64 64 3 3 3 16 1 1 1 0 0 1 1 1 1 1 1 1 16 2 2 2 1 1 16 64 144 16 16 16 1 1 13 13 1 15360 0 3 4 16 13 9 0 16 0 0 0 0 0 0 0 "},

    {"test_group_upper_bound", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
    {1, 65536, 11, 64, 64}, {1, 11, 4096, 64, 64, 16}, {65536, 1, 3, 3, 3}, {110592, 1, 16, 16}, {1, 65536, 9, 64, 64}, {1, 9, 4096, 64, 64, 16},
    ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 1, 1, 1}, {0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 256, "NCDHW", "VALID", 0,
    true, false, 20, 0, "1 0 2 1 10 1 1 524288 1 256 256 2 2 9 64 64 11 64 64 3 3 3 16 1 1 1 0 0 1 1 1 1 1 1 1 8 2 2 2 2 2 16 32 144 16 8 16 1 1 13 13 1 10240 0 3 8 16 7 9 0 16 0 0 0 0 0 0 0 "},

    {"net_ID_03_b16", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {4, 256, 4, 64, 64}, {4, 4, 16, 64, 64, 16}, {1364, 256, 1, 1, 1}, {16, 86, 16, 16}, {4, 1364, 4, 64, 64}, {4, 4, 86, 64, 64, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 2, "4 0 1 1 5 1 1 524288 4 256 1364 16 86 4 64 64 4 64 64 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 128 128 128 16 16 16 1 1 4 4 1 65536 0 1 1 1376 8 4 0 128 0 1 20 1376 128 512 0 "},

    {"stdit_01_fp32_deterministic", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 4, 1, 32, 32}, {1, 1, 1, 32, 32, 16}, {1152, 4, 1, 2, 2}, {4, 72, 16, 16}, {1, 1152, 1, 16, 16}, {1, 1, 72, 16, 16, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
     {1, 1, 1, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 1,
     true, true, 24, 0, "1 0 1 12 2 1 1 524288 1 4 1152 1 144 1 16 16 1 32 32 1 2 2 1 1 2 2 0 0 0 0 0 0 1 1 1 8 2 2 2 1 1 96 64 64 16 8 16 1 1 2 2 1 8192 0 1 1 96 8 1 0 8 0 0 0 0 0 0 0 "},

    {"Conv3d_bp_filter_no_support_dma_flag", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
    {1, 1, 14, 20, 128}, {1, 1, 8, 14, 20, 16}, {1, 3, 3, 128, 128}, {72, 8, 16, 16}, {1, 1, 1, 18, 128}, {1, 1, 8, 1, 18, 16},
    ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, ge::FORMAT_DHWCN, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
    {1, 1, 64, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", "SAME", 0,
    true, false, 24, 58632, " "},

    {"f240_h256_net_id_2", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 256, 120, 128, 128}, {1, 120, 16, 128, 128, 16}, {256, 256, 4, 4, 4}, {1024, 16, 16, 16}, {1, 256, 62, 66, 66}, {1, 62, 16, 66, 66, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 3, "1 0 1 2 10 1 1 524288 1 256 256 16 16 62 66 66 120 128 128 4 4 4 1 2 2 2 3 3 3 3 3 3 1 1 1 16 2 2 1 2 2 256 64 128 16 16 16 1 1 6 6 0 32768 0 1 1 256 66 62 0 16 0 1 20 256 256 4368 0 "},

    {"f240_h256_net_id_2", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 256, 120, 128, 128}, {1, 120, 16, 128, 128, 16}, {256, 256, 4, 4, 4}, {1024, 16, 16, 16}, {1, 256, 62, 66, 66}, {1, 62, 16, 66, 66, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 3, "1 0 1 2 10 1 1 524288 1 256 256 16 16 62 66 66 120 128 128 4 4 4 1 2 2 2 3 3 3 3 3 3 1 1 1 16 2 2 1 2 2 256 64 128 16 16 16 1 1 6 6 0 32768 0 1 1 256 66 62 0 16 0 1 20 256 256 4368 0 "},

    {"conv3d_dw_v2_stride2_channel128", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 128, 120, 128, 128}, {1, 120, 8, 128, 128, 16}, {128, 128, 4, 4, 4}, {512, 16, 16, 16}, {1, 128, 62, 66, 66}, {1, 62, 8, 66, 66, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 3, "1 0 1 1 10 2 1 524288 1 128 128 8 8 62 66 66 120 128 128 4 4 4 1 2 2 2 3 3 3 3 3 3 1 1 1 16 2 2 1 2 2 128 64 256 16 16 16 1 1 8 8 0 40960 0 1 1 128 66 62 0 16 0 1 20 128 256 4368 0 "},

    {"conv3d_dw_v2_large_w", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 256, 120, 128, 1280}, {1, 120, 16, 128, 1280, 16}, {256, 256, 4, 4, 4}, {1024, 16, 16, 16}, {1, 256, 62, 66, 642}, {1, 62, 16, 66, 642, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {3, 3, 3, 3, 3, 3}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 20, 3, "1 0 1 1 17 1 1 524288 1 256 256 16 16 62 66 642 120 128 1280 4 4 4 1 2 2 2 3 3 3 3 3 3 1 1 1 16 2 2 1 1 1 256 64 128 16 16 16 1 1 1 1 0 122880 0 1 1 256 66 62 0 16 0 1 20 256 256 42384 0 "},

    {"conv3d_dw_v2_w0", "Ascend910B3", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20}, "block_dim": {"CORE_NUM": 20}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 1, 1, 1, 2}, {1, 1, 1, 1, 2, 8}, {1, 1, 1, 1, 1}, {1, 1, 16, 8}, {1, 1, 1, 1, 0}, {1, 1, 1, 1, 0, 8},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_FLOAT,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, -1, -1}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, false, 20, 3, ""},

    {"conv3d_dw_v2_dilation", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 20, "binary_mode": 1})",
     {1, 32, 6, 32, 32}, {1, 6, 2, 32, 32, 16}, {32, 32, 3, 3, 3}, {54, 2, 16, 16}, {1, 32, 3, 32, 32}, {1, 3, 2, 32, 32, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 1, 1}, {1, 2, 2, 2, 2, 2}, {1, 1, 2, 2, 2}, 1, "NCDHW", "SAME", 0,
     true, true, 24, 0, "1 0 1 1 4 2 3 524288 1 32 32 2 2 3 32 32 6 32 32 3 3 3 1 2 1 1 1 2 2 2 2 2 2 2 2 16 2 2 2 1 1 32 64 144 16 16 16 1 1 4 4 1 6144 0 1 1 32 8 3 0 16 0 0 0 0 0 0 0 "},

    {"conv3d_dw_v2_large_w_1", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 14, 6, 404, 1027}, {1, 6, 1, 404, 1027, 16}, {50, 14, 20, 1, 1}, {20, 4, 16, 16}, {1, 50, 3, 202, 514}, {1, 3, 4, 202, 514, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {9, 9, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME", 0,
     true, true, 24, 1, "3 0 1 1 8 1 1 524288 1 14 50 1 4 3 202 514 6 404 1027 20 1 1 1 2 2 2 9 9 0 0 0 0 1 1 1 16 2 2 2 1 1 64 192 80 16 16 16 1 1 1 1 0 246480 0 1 1 64 1 1 0 320 0 1 24 64 320 514 0 "},

    {"conv3d_dw_v2_large_w_3", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {2, 186, 3, 727, 1071}, {2, 3, 12, 727, 1071, 16}, {64, 186, 3, 1, 1}, {36, 4, 16, 16}, {2, 64, 1, 727, 1071}, {2, 1, 4, 727, 1071, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 1, "2 0 1 1 1 12 1 524288 2 186 64 12 4 1 727 1071 3 727 1071 3 1 1 1 2 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 1 1 64 144 112 16 16 16 1 1 1 1 0 239904 0 1 1 64 1 1 0 576 0 1 24 64 576 1071 0 "},

    {"conv3d_dw_v2_large_w_4", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 58, 55, 67, 3180}, {1, 55, 4, 64, 3180, 16}, {36, 58, 18, 1, 1}, {72, 3, 16, 16}, {1, 36, 19, 34, 1590}, {1, 19, 3, 34, 1590, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "VALID", 0,
     true, true, 24, 3, "1 0 1 1 2 12 1 524288 1 58 36 4 3 19 34 1590 55 67 3180 18 1 1 1 2 2 2 0 0 0 0 0 0 1 1 1 16 2 2 1 1 1 48 336 16 16 16 16 1 1 1 1 0 152640 0 1 1 48 34 19 0 16 0 1 24 48 16 54064 0 "},

    {"conv3d_dw_v2_large_w_5", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {21, 56, 150, 2, 753}, {21, 150, 4, 2, 753, 16}, {64, 56, 2, 1, 1}, {8, 4, 16, 16}, {21, 64, 75, 1, 753}, {21, 75, 4, 1, 753, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 2, 2, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", "SAME", 0,
     true, true, 24, 3, "21 0 1 1 1 1 1 524288 21 56 64 4 4 75 1 753 150 2 753 2 1 1 1 2 2 1 0 0 0 0 0 0 1 1 1 16 2 2 1 1 1 64 128 112 16 16 16 1 1 1 1 0 253008 0 1 1 64 1 75 0 112 0 1 24 64 112 768 0 "},

    {"conv3d_dw_v2_exceed_l1_size", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 1, 2, 257, 257}, {1, 2, 1, 257, 257, 16}, {1, 1, 2, 2, 2}, {8, 1, 16, 16}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 1, 255, 255}, 1, "NCDHW", "VALID", 0,
     true, false, 24, 0, ""},

    {"conv3d_dilation_kernel1_test", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 32, 32, 32, 32}, {1, 32, 2, 32, 32, 16}, {16, 32, 1, 1, 1}, {8, 2, 16, 16}, {1, 16, 32, 32, 32}, {1, 32, 2, 32, 32, 16},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0}, {1, 1, 2, 2, 2}, 1, "NCDHW", "SAME", 0,
     true, true, 24, 1, "8 0 1 1 3 1 1 524288 1 32 16 2 1 32 32 32 32 32 32 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 16 2 2 2 2 2 16 512 32 16 16 16 1 1 2 2 0 32768 0 1 1 16 32 4 0 32 0 1 24 16 32 1024 0 "},

    {"conv3d_dilation_kernel2_test", "Ascend910B2", "Ascend910B", R"({"_pattern": "Conv3d_backprop_filter", "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24}, "block_dim": {"CORE_NUM": 24}, "tiling_type": "binary", "max_core_num": 24, "binary_mode": 1})",
     {1, 1, 34, 1, 1}, {1, 1, 34, 1, 1, 16}, {1, 1, 336, 135, 26}, {1179360, 1, 16, 16}, {1, 1, 1, 27, 4}, {1, 1, 1, 27, 4, 1},
     ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::DT_BF16,
     {1, 1, 53, 4, 52}, {151, 151, 253, 253, 140, 140}, {1, 1, 1, 3, 3}, 1, "NCDHW", "SAME", 0,
     true, true, 24, 0, "1 0 1 1 1 24 1 524288 1 1 1 1 1 1 27 4 34 1 1 336 135 26 1 53 4 52 151 151 253 253 140 140 1 3 3 16 2 2 2 1 1 16 112 144 16 16 16 1 390 1 1 1 64 0 1 1 16 27 1 0 224 0 0 0 0 0 0 0 "},
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

TEST_F(Conv3DBackpropFilterV2TilingRunTime2, milan_binary_multi_thread)
{
    TestMultiThread(milan_binary_params, sizeof(milan_binary_params) / sizeof(Conv3DBpFilterV2TilingTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(MilanBinary, Conv3DBackpropFilterV2TilingRunTime2, testing::ValuesIn(milan_binary_params));
}