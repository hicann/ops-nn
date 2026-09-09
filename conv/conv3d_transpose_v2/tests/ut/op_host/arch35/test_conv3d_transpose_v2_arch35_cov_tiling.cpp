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
 * \file test_conv3d_transpose_v2_arch35_tiling_cov.cpp
 * \brief coverage supplement: fall through all sub templates (96-101) to the base wrapper
 *        Conv3DTransposeV2TilingArch35 (102), covering its constructor branches.
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
struct Conv3DTransposeV2CovParam {
    string case_name;
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
    vector<int64_t> strides;
    vector<int64_t> pads;
    vector<int64_t> dilations;
    int64_t groups;
    string data_format;
    vector<int64_t> output_padding;
    int64_t offset;
    string padding;
    bool tiling_result;
};

const string TP_COV_CI_950 = R"({
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

static void TestOneTransposeCovCase(const Conv3DTransposeV2CovParam& param)
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
    GetPlatFormInfos(TP_COV_CI_950.c_str(), soc_infos, aicore_spec, intrinsics);
    aicore_spec["cube_freq"] = "1800";

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    Ops::NN::Conv::Conv3DBackpropV2CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs(
                                 {const_cast<char*>(TP_COV_CI_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_version_infos = {
        {"SoC_version", "3510"}, {"Short_SoC_version", "3510"}, {"NpuArch", "3510"}};

    std::string op_type("Conv3DTransposeV2");
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
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_SUCCESS);

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
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"output_padding",
                                   Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .NodeInputTd(0, ge::DT_INT64, param.input_size_format, param.input_size_format)
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
    std::cout << ">>>>>tilingKey=" << *tiling_key << " blockDim=" << *block_dim << std::endl;
}

class Conv3DTransposeV2CovSuite : public testing::TestWithParam<Conv3DTransposeV2CovParam> {};

TEST_P(Conv3DTransposeV2CovSuite, cov_cases) { TestOneTransposeCovCase(GetParam()); }

static Conv3DTransposeV2CovParam tp_cov_cases[] = {
    // huge kernel (kd=7): falls through sub templates to base wrapper 102
    Conv3DTransposeV2CovParam{"cov_tp_fallthrough_big_kernel",
                              {1, 16, 2, 10, 10},
                              {1, 16, 2, 10, 10},
                              {1, 16, 2, 10, 10},
                              {16, 16, 7, 7, 7},
                              {16, 16, 7, 7, 7},
                              {1, 16, 2, 30, 30},
                              {1, 16, 2, 30, 30},
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
                              {0, 0, 0},
                              0,
                              "",
                              false},
    // tiny shape with d=1 kernel d=1: small mValue may fall through to base wrapper
    Conv3DTransposeV2CovParam{"cov_tp_fallthrough_tiny",
                              {1, 16, 1, 2, 2},
                              {1, 16, 1, 2, 2},
                              {1, 16, 1, 2, 2},
                              {16, 16, 1, 1, 1},
                              {16, 16, 1, 1, 1},
                              {1, 16, 1, 2, 2},
                              {1, 16, 1, 2, 2},
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
                              {0, 0, 0},
                              0,
                              "",
                              false},
    // groups=4 exotic: group split makes all sub templates incapable
    Conv3DTransposeV2CovParam{"cov_tp_fallthrough_groups4",
                              {1, 64, 2, 10, 10},
                              {1, 16, 2, 10, 10},
                              {1, 16, 2, 10, 10},
                              {64, 16, 3, 3, 3},
                              {64, 16, 3, 3, 3},
                              {1, 64, 2, 12, 12},
                              {1, 64, 2, 12, 12},
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
                              4,
                              "NCDHW",
                              {0, 0, 0},
                              0,
                              "",
                              false},
};

INSTANTIATE_TEST_SUITE_P(Conv3DTransposeV2Cov, Conv3DTransposeV2CovSuite, testing::ValuesIn(tp_cov_cases));
} // namespace
