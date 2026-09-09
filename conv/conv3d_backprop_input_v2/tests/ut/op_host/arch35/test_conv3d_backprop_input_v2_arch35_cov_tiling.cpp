/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_backprop_input_v2_arch35_cov_tiling.cpp
 * \brief coverage supplement cases for Conv3DBackpropInputV2 arch35 tiling (inner_product /
 *        kernel_split / fullLoad / small_shape / small_kernel templates)
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "graph/graph.h"
#define private public
#define protected public
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../../../common/op_host/op_tiling/conv_platform_util.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;

namespace {
struct Conv3DBpInputV2CovParam {
    string case_name;
    string dtype_y;      // dedx(y) dtype
    string dtype_filter; // filter dtype
    string dtype_obp;    // out_backprop dtype
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
    vector<int64_t> strides;
    vector<int64_t> pads;
    vector<int64_t> dilations;
    int64_t groups;
    string data_format;
    bool enable_hf32;
    string padding;
    int64_t opImplModeEnum;
    bool tiling_result;
    bool one_core; // true: use 1-core compile info (Has1CoreKernelSplitAlternative branches)
};

const string COV_CI_950 = R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary",
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

const string COV_CI_950_1CORE = R"({"_pattern": "Conv3d_backprop_input_v2", "tiling_type": "binary",
                          "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 1,
                          "cube_core_cnt": 1, "vector_core_cnt": 2, "core_type_list": "CubeCore,VectorCore"}
                          })";

static ge::DataType CovDtype(const string& dtype)
{
    if (dtype == "bfloat16") {
        return ge::DT_BF16;
    }
    if (dtype == "float32") {
        return ge::DT_FLOAT;
    }
    if (dtype == "hifloat8") {
        return ge::DT_HIFLOAT8;
    }
    if (dtype == "int32") {
        return ge::DT_INT32;
    }
    if (dtype == "int8") {
        return ge::DT_INT8;
    }
    return ge::DT_FLOAT16;
}

static void TestOneCovCase(const Conv3DBpInputV2CovParam& param, const string& compileInfoStr)
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

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs(
                                 {const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compileInfoStr.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    map<string, string> soc_version_infos = {{"SoC_version", "3510"}, {"Short_SoC_version", "3510"}};
    map<string, string> npuarchs = {{"SoC_version", "3510"}, {"Short_SoC_version", "3510"}, {"NpuArch", "3510"}};

    std::string op_type("Conv3DBackpropInputV2");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_SUCCESS);

    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto tiling_data = gert::TilingData::CreateCap(2048);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_size, &filter_shape, &out_backprop_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<void*>(&platform_info))
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(param.enable_hf32)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)},
                                  {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(param.opImplModeEnum)}})
                      .NodeInputTd(0, ge::DT_INT64, param.input_size_format, param.input_size_format)
                      .NodeInputTd(1, CovDtype(param.dtype_filter), param.filter_ori_format, param.filter_format)
                      .NodeInputTd(2, CovDtype(param.dtype_obp), param.out_backprop_ori_format,
                                   param.out_backprop_format)
                      .NodeOutputTd(0, CovDtype(param.dtype_y), param.y_ori_format, param.y_format)
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
    std::cout << ">>>>>tilingKey=" << *tiling_key << " blockDim=" << *block_dim << std::endl;
}

class Conv3DBpInputV2CovSuite : public testing::TestWithParam<Conv3DBpInputV2CovParam> {};

TEST_P(Conv3DBpInputV2CovSuite, cov_cases)
{
    auto p = GetParam();
    TestOneCovCase(p, p.one_core ? COV_CI_950_1CORE : COV_CI_950);
}

static Conv3DBpInputV2CovParam dx_cov_cases[] = {
    // inner_product int32: dtypeByteL0a=4 branches (L1536-L1619 region), int32 tiling rejected
    {"cov_ip_int32",
     "int32",
     "int32",
     "int32",
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
     false,
     "",
     0,
     false,
     false},
    // inner_product float32 success path
    {"cov_ip_fp32",
     "float32",
     "float32",
     "float32",
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
     false,
     "",
     0,
     true,
     false},
    // hifloat8 with NCDHW filter: rejected by fullLoad 8bit gate (L54-L57), tiling fails
    {"cov_ip_hif8_ncdhw",
     "hifloat8",
     "hifloat8",
     "hifloat8",
     {5},
     {64, 64, 1, 3, 3},
     {64, 64, 1, 3, 3},
     {1, 64, 5, 16, 16},
     {1, 64, 5, 16, 16},
     {1, 64, 5, 32, 32},
     {1, 64, 5, 32, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     false},
    // hifloat8 + NDHWC filter format (8bit only allowed with NDHWC, L54-L57 gate branch)
    {"cov_fl_hif8_ndhwc",
     "hifloat8",
     "hifloat8",
     "hifloat8",
     {5},
     {1, 1, 3, 64, 32},
     {1, 1, 3, 64, 32},
     {1, 5, 8, 8, 64},
     {1, 5, 8, 8, 64},
     {1, 5, 8, 16, 32},
     {1, 5, 8, 16, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     ge::FORMAT_NDHWC,
     {1, 1, 1, 1, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     false,
     "",
     0,
     false,
     false},
    // fullLoad int32: int32 tiling rejected (dtypeByteL0b=4 path evaluated)
    {"cov_fl_int32",
     "int32",
     "int32",
     "int32",
     {5},
     {128, 64, 1, 1, 1},
     {128, 64, 1, 1, 1},
     {1, 128, 9, 32, 32},
     {1, 128, 9, 32, 32},
     {1, 64, 9, 64, 64},
     {1, 64, 9, 64, 64},
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
     false,
     "",
     0,
     false,
     false},
    // inner_product w>512: L1354-L1359 alignedWiAl1>=512 / baseM=256 special adjust
    {"cov_ip_w_gt_512",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 128, 1, 1, 1},
     {256, 128, 1, 1, 1},
     {1, 256, 9, 128, 600},
     {1, 256, 9, 128, 600},
     {1, 128, 9, 128, 600},
     {1, 128, 9, 128, 600},
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
     false,
     "",
     0,
     true,
     false},
    // fullLoad unbalanced mCnt across cores: L196-L202 balance search branch
    {"cov_fl_unbalanced",
     "bfloat16",
     "bfloat16",
     "bfloat16",
     {5},
     {256, 128, 1, 1, 1},
     {256, 128, 1, 1, 1},
     {1, 256, 9, 127, 127},
     {1, 256, 9, 127, 127},
     {1, 128, 9, 127, 127},
     {1, 128, 9, 127, 127},
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
     false,
     "",
     0,
     true,
     false},
    // kernel_split variants: L137/L471/L149-L187/L658-L663 routing and check branches
    {"cov_ks_unaligned",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 3, 4, 4, 4},
     {256, 3, 4, 4, 4},
     {1, 3, 16, 256, 255},
     {1, 3, 16, 256, 255},
     {1, 256, 10, 131, 129},
     {1, 256, 10, 131, 129},
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
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_dilation2",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 3, 4, 4, 4},
     {256, 3, 4, 4, 4},
     {1, 3, 16, 256, 256},
     {1, 3, 16, 256, 256},
     {1, 256, 10, 130, 130},
     {1, 256, 10, 130, 130},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 2, 2, 2},
     {3, 3, 3, 3, 3, 3},
     {1, 1, 2, 2, 2},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_groups2",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 6, 4, 4, 4},
     {256, 6, 4, 4, 4},
     {1, 3, 16, 256, 256},
     {1, 3, 16, 256, 256},
     {1, 256, 10, 130, 130},
     {1, 256, 10, 130, 130},
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
     2,
     "NCDHW",
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_hw11_h_gt_w",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 3, 4, 4, 4},
     {256, 3, 4, 4, 4},
     {1, 3, 16, 64, 256},
     {1, 3, 16, 64, 256},
     {1, 256, 10, 130, 66},
     {1, 256, 10, 130, 66},
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
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_kd1",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 3, 1, 4, 4},
     {256, 3, 1, 4, 4},
     {1, 3, 16, 256, 256},
     {1, 3, 16, 256, 256},
     {1, 256, 10, 130, 130},
     {1, 256, 10, 130, 130},
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
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_batch8",
     "float16",
     "float16",
     "float16",
     {5},
     {256, 3, 4, 4, 4},
     {256, 3, 4, 4, 4},
     {8, 3, 16, 256, 256},
     {8, 3, 16, 256, 256},
     {8, 256, 10, 130, 130},
     {8, 256, 10, 130, 130},
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
     false,
     "",
     0,
     false,
     false},
    {"cov_ks_cout260",
     "float16",
     "float16",
     "float16",
     {5},
     {260, 5, 4, 4, 4},
     {260, 5, 4, 4, 4},
     {1, 5, 16, 256, 256},
     {1, 5, 16, 256, 256},
     {1, 260, 10, 130, 130},
     {1, 260, 10, 130, 130},
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
     false,
     "",
     0,
     false,
     false},
    // small_shape L1 stress: L107-L110 fallback path
    {"cov_ss_l1_stress",
     "float16",
     "float16",
     "float16",
     {5},
     {1024, 1024, 1, 3, 3},
     {1024, 1024, 1, 3, 3},
     {1, 1024, 1, 8, 8},
     {1, 1024, 1, 8, 8},
     {1, 1024, 1, 8, 8},
     {1, 1024, 1, 8, 8},
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
     false,
     "",
     0,
     false,
     false},
    // small_shape batch128: nValue>=128 branch
    {"cov_ss_batch128",
     "bfloat16",
     "bfloat16",
     "bfloat16",
     {5},
     {64, 32, 1, 1, 1},
     {64, 32, 1, 1, 1},
     {128, 64, 1, 8, 8},
     {128, 64, 1, 8, 8},
     {128, 32, 1, 8, 8},
     {128, 32, 1, 8, 8},
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
     false,
     "",
     0,
     true,
     false},
    // ===== 1-core compile info: Has1CoreKernelSplitAlternative branches (small_kernel L412-L464) =====
    // stride 2x2 + kernel 2x2 + even w: HW11 hard admission branch L443-L453
    {"cov_sk_1c_hw_k2",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 2, 2},
     {32, 32, 1, 2, 2},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 64, 64},
     {1, 32, 4, 64, 64},
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
     false,
     "",
     0,
     true,
     true},
    // stride 2x2 + kernel 3x3
    {"cov_sk_1c_hw_k3",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 3, 3},
     {32, 32, 1, 3, 3},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 64, 64},
     {1, 32, 4, 64, 64},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {1, 1, 1, 1, 1, 1},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     true},
    // stride 2x2 + kernel 4x4
    {"cov_sk_1c_hw_k4",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 4, 4},
     {32, 32, 1, 4, 4},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 68, 68},
     {1, 32, 4, 68, 68},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {2, 2, 2, 2, 2, 2},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     true},
    // stride 2x2 + kernel 2x2 + cin_g=1/cout_g=1 reject branch L446-L448, mValue>=256
    {"cov_sk_1c_c1k2",
     "float16",
     "float16",
     "float16",
     {5},
     {1, 1, 1, 2, 2},
     {1, 1, 1, 2, 2},
     {1, 1, 4, 64, 64},
     {1, 1, 4, 64, 64},
     {1, 1, 4, 128, 128},
     {1, 1, 4, 128, 128},
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
     false,
     "",
     0,
     true,
     true},
    // stride_h only (TryKernelSplitH branch L457-L460): stride_h=2, stride_w=1, kernel_h=3
    {"cov_sk_1c_h_only",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 3, 3},
     {32, 32, 1, 3, 3},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 64, 32},
     {1, 32, 4, 64, 32},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 1},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     true},
    // dilation 2 reject branch L425-L427
    {"cov_sk_1c_dil2",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 2, 2},
     {32, 32, 1, 2, 2},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 64, 64},
     {1, 32, 4, 64, 64},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 0, 0, 0},
     {1, 1, 2, 2, 2},
     1,
     "NCDHW",
     false,
     "",
     0,
     false,
     true},
    // asymmetric pad reject branch L428-L430
    {"cov_sk_1c_asym_pad",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 2, 2},
     {32, 32, 1, 2, 2},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 64, 64},
     {1, 32, 4, 64, 64},
     ge::FORMAT_ND,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     ge::FORMAT_NCDHW,
     {1, 1, 1, 2, 2},
     {0, 0, 0, 1, 0, 0},
     {1, 1, 1, 1, 1},
     1,
     "NCDHW",
     false,
     "",
     0,
     true,
     true},
    // small mValue + groups=1 reject branch L433-L435
    {"cov_sk_1c_small_m",
     "float16",
     "float16",
     "float16",
     {5},
     {32, 32, 1, 2, 2},
     {32, 32, 1, 2, 2},
     {1, 32, 4, 16, 16},
     {1, 32, 4, 16, 16},
     {1, 32, 4, 32, 32},
     {1, 32, 4, 32, 32},
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
     false,
     "",
     0,
     true,
     true},
};

INSTANTIATE_TEST_SUITE_P(Conv3DBpInputV2Cov, Conv3DBpInputV2CovSuite, testing::ValuesIn(dx_cov_cases));
} // namespace
