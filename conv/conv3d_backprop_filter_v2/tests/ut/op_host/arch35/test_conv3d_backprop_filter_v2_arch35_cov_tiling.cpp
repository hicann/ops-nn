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
 * \file test_conv3d_backprop_filter_v2_arch35_tiling_cov.cpp
 * \brief coverage supplement cases for Conv3DBackpropFilterV2 arch35 tiling (stream_k / winograd /
 *        basic block shared branches)
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
struct Conv3DBpFilterV2CovParam {
    string case_name;
    string dtype; // "float16"/"bfloat16"/"float32"/"hifloat8"
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
    vector<int64_t> strides;
    vector<int64_t> pads;
    vector<int64_t> dilations;
    int64_t groups;
    string data_format;
    string padding;
    bool enable_hf32;
    bool tiling_result;
};

const string DW_COV_CI_950 = R"({"_pattern": "Conv3d_backprop_filter_v2", "tiling_type": "binary",
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

static ge::DataType DwCovDtype(const string& dtype)
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
    return ge::DT_FLOAT16;
}

static void TestOneFilterCovCase(const Conv3DBpFilterV2CovParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape filter_sizes = {param.filter_shape, param.filter_shape};
    gert::StorageShape out_backprop_shape = {param.out_backprop_ori_shape, param.out_backprop_shape};
    gert::StorageShape fmap_shape = {param.fmap_ori_shape, param.fmap_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.filter_ori_shape, param.filter_shape});
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
                                 {const_cast<char*>(DW_COV_CI_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(DW_COV_CI_950.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    map<string, string> soc_version_infos = {{"SoC_version", "3510"}, {"Short_SoC_version", "3510"}};
    map<string, string> npuarchs = {{"SoC_version", "3510"}, {"Short_SoC_version", "3510"}, {"NpuArch", "3510"}};

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::TilingParseContext>()), ge::GRAPH_SUCCESS);

    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto dtype = DwCovDtype(param.dtype);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&fmap_shape, &filter_sizes, &out_backprop_shape})
                      .OutputShapes(output_shapes_ref)
                      .PlatformInfo(reinterpret_cast<void*>(&platform_info))
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(param.enable_hf32)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .NodeInputTd(0, dtype, param.fmap_ori_format, param.fmap_format)
                      .NodeInputTd(1, ge::DT_INT32, param.filter_ori_format, param.filter_format)
                      .NodeInputTd(2, dtype, param.out_backprop_ori_format, param.out_backprop_format)
                      .NodeOutputTd(0, ge::DT_FLOAT, param.filter_ori_format, param.filter_format)
                      .DeterministicInfo(reinterpret_cast<int32_t*>(0))
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

class Conv3DBpFilterV2CovSuite : public testing::TestWithParam<Conv3DBpFilterV2CovParam> {};

TEST_P(Conv3DBpFilterV2CovSuite, cov_cases) { TestOneFilterCovCase(GetParam()); }

static Conv3DBpFilterV2CovParam dw_cov_cases[] = {
    // exact replica of existing passing case conv_stdit_01_fp16 for infra sanity check
    Conv3DBpFilterV2CovParam{"cov_dw_replica_stdit01",
                             "float16",
                             {33, 4, 1, 32, 32},
                             {33, 4, 1, 32, 32},
                             {1152, 4, 1, 2, 2},
                             {1152, 4, 1, 2, 2},
                             {33, 1152, 1, 16, 16},
                             {33, 1152, 1, 16, 16},
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
                             "VALID",
                             false,
                             true},
    // stream_k L91-L108: exceed L1 buffer size error path
    Conv3DBpFilterV2CovParam{"cov_dw_l1_exceed",
                             "float16",
                             {4, 512, 4, 128, 128},
                             {4, 512, 4, 128, 128},
                             {1024, 512, 4, 5, 5},
                             {1024, 512, 4, 5, 5},
                             {4, 1024, 4, 62, 62},
                             {4, 1024, 4, 62, 62},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 1, 1},
                             {0, 0, 2, 2, 2, 2},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             false},
    // kd=2: stream_k L72-L74 splitWo/splitWi + basic_block L267 nValue*=kd
    Conv3DBpFilterV2CovParam{"cov_dw_kd2",
                             "float16",
                             {4, 64, 6, 32, 32},
                             {4, 64, 6, 32, 32},
                             {128, 64, 2, 3, 3},
                             {128, 64, 2, 3, 3},
                             {4, 128, 3, 16, 16},
                             {4, 128, 3, 16, 16},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             false},
    // small case baseN adjust: L163-L198 + L229-L231
    Conv3DBpFilterV2CovParam{"cov_dw_small_case_adjust",
                             "float16",
                             {1, 64, 1, 16, 16},
                             {1, 64, 1, 16, 16},
                             {2048, 64, 1, 3, 3},
                             {2048, 64, 1, 3, 3},
                             {1, 2048, 1, 8, 8},
                             {1, 2048, 1, 8, 8},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             true},
    // batch dout select: L274-L276 BatchDout branch
    Conv3DBpFilterV2CovParam{"cov_dw_batch_dout",
                             "float16",
                             {32, 4, 8, 32, 32},
                             {32, 4, 8, 32, 32},
                             {1152, 4, 1, 2, 2},
                             {1152, 4, 1, 2, 2},
                             {32, 1152, 8, 16, 16},
                             {32, 1152, 8, 16, 16},
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
                             "VALID",
                             false,
                             true},
    // stepK L1 adjust loop: L352-L378 (big cin forces stepK ladder)
    Conv3DBpFilterV2CovParam{"cov_dw_stepk_l1",
                             "float16",
                             {4, 512, 1, 64, 64},
                             {4, 512, 1, 64, 64},
                             {128, 512, 1, 3, 3},
                             {128, 512, 1, 3, 3},
                             {4, 128, 1, 64, 64},
                             {4, 128, 1, 64, 64},
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
                             false,
                             true},
    // groups=2: basic_block L185-L186 disableGroupEnlarge branch
    Conv3DBpFilterV2CovParam{"cov_dw_groups2",
                             "float16",
                             {8, 64, 1, 32, 32},
                             {8, 64, 1, 32, 32},
                             {256, 64, 1, 3, 3},
                             {256, 64, 1, 3, 3},
                             {8, 128, 1, 16, 16},
                             {8, 128, 1, 16, 16},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             2,
                             "NCDHW",
                             "",
                             false,
                             false},
    // fp32 dtype: dtypeByte != FLOAT branches
    Conv3DBpFilterV2CovParam{"cov_dw_fp32",
                             "float32",
                             {33, 4, 1, 32, 32},
                             {33, 4, 1, 32, 32},
                             {1152, 4, 1, 2, 2},
                             {1152, 4, 1, 2, 2},
                             {33, 1152, 1, 16, 16},
                             {33, 1152, 1, 16, 16},
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
                             "VALID",
                             false,
                             true},
    // bf16 dtype variant
    Conv3DBpFilterV2CovParam{"cov_dw_bf16",
                             "bfloat16",
                             {16, 32, 1, 64, 64},
                             {16, 32, 1, 64, 64},
                             {256, 32, 1, 3, 3},
                             {256, 32, 1, 3, 3},
                             {16, 256, 1, 32, 32},
                             {16, 256, 1, 32, 32},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             true},
    // kernel 3x3 with odd alignment: basic_block L572-L573 bL1N % kernelHW branch
    Conv3DBpFilterV2CovParam{"cov_dw_k3_align",
                             "float16",
                             {16, 40, 1, 56, 56},
                             {16, 40, 1, 56, 56},
                             {200, 40, 1, 3, 3},
                             {200, 40, 1, 3, 3},
                             {16, 200, 1, 28, 28},
                             {16, 200, 1, 28, 28},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             true},
    // large wo split: basic_block L601-L603 / L694-L695 / L712-L713 blockBaseK adjust branches
    Conv3DBpFilterV2CovParam{"cov_dw_big_wo",
                             "float16",
                             {2, 32, 1, 128, 512},
                             {2, 32, 1, 128, 512},
                             {64, 32, 1, 3, 3},
                             {64, 32, 1, 3, 3},
                             {2, 64, 1, 64, 256},
                             {2, 64, 1, 64, 256},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             true},
    // L1 invalid shrink loops: basic_block L622-L668
    Conv3DBpFilterV2CovParam{"cov_dw_l1_shrink",
                             "float16",
                             {2, 256, 1, 128, 128},
                             {2, 256, 1, 128, 128},
                             {512, 256, 1, 5, 5},
                             {512, 256, 1, 5, 5},
                             {2, 512, 1, 62, 62},
                             {2, 512, 1, 62, 62},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             false},
    // fmap NDHWC format: stream_k L404-L405 format reject -> NO_STREAMK
    Conv3DBpFilterV2CovParam{"cov_dw_fmap_ndhwc",
                             "float16",
                             {4, 64, 1, 32, 32},
                             {4, 64, 1, 32, 32},
                             {128, 64, 1, 3, 3},
                             {128, 64, 1, 3, 3},
                             {4, 128, 1, 16, 16},
                             {4, 128, 1, 16, 16},
                             ge::FORMAT_NDHWC,
                             ge::FORMAT_NDHWC,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NDHWC,
                             ge::FORMAT_NDHWC,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NDHWC",
                             "",
                             false,
                             false},
    // winograd: hf32 reject branch L47-L48 (2D, stride 1, kernel 3x3)
    Conv3DBpFilterV2CovParam{"cov_wg_hf32_reject",
                             "float16",
                             {4, 64, 1, 56, 56},
                             {4, 64, 1, 56, 56},
                             {64, 64, 1, 3, 3},
                             {64, 64, 1, 3, 3},
                             {4, 64, 1, 56, 56},
                             {4, 64, 1, 56, 56},
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
                             true,
                             true},
    // winograd: kd=2 "only supported for 2d" reject branch L57-L58
    Conv3DBpFilterV2CovParam{"cov_wg_kd2_reject",
                             "float16",
                             {4, 64, 6, 56, 56},
                             {4, 64, 6, 56, 56},
                             {64, 64, 2, 3, 3},
                             {64, 64, 2, 3, 3},
                             {4, 64, 3, 28, 28},
                             {4, 64, 3, 28, 28},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 1, 1, 1, 1},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             false},
    // winograd: pad too large reject branch L74-L75
    Conv3DBpFilterV2CovParam{"cov_wg_padlarge_reject",
                             "float16",
                             {4, 64, 1, 56, 56},
                             {4, 64, 1, 56, 56},
                             {64, 64, 1, 3, 3},
                             {64, 64, 1, 3, 3},
                             {4, 64, 1, 24, 24},
                             {4, 64, 1, 24, 24},
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             ge::FORMAT_NCDHW,
                             {1, 1, 1, 2, 2},
                             {0, 0, 16, 16, 16, 16},
                             {1, 1, 1, 1, 1},
                             1,
                             "NCDHW",
                             "",
                             false,
                             false},
    // winograd: cout/cin too small reject branch L90-L91
    Conv3DBpFilterV2CovParam{"cov_wg_smallc_reject",
                             "float16",
                             {4, 2, 1, 56, 56},
                             {4, 2, 1, 56, 56},
                             {2, 2, 1, 3, 3},
                             {2, 2, 1, 3, 3},
                             {4, 2, 1, 56, 56},
                             {4, 2, 1, 56, 56},
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
                             false,
                             true},
    // winograd: fp32 2D valid case, L133 isFp32 branch + tile selection L210-L211
    Conv3DBpFilterV2CovParam{"cov_wg_fp32_2d",
                             "float32",
                             {4, 64, 1, 56, 56},
                             {4, 64, 1, 56, 56},
                             {64, 64, 1, 3, 3},
                             {64, 64, 1, 3, 3},
                             {4, 64, 1, 56, 56},
                             {4, 64, 1, 56, 56},
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
                             false,
                             true},
    // winograd: fp16 2D valid case (B16H4W16 tile branch L210-L211)
    Conv3DBpFilterV2CovParam{"cov_wg_fp16_2d",
                             "float16",
                             {4, 128, 1, 56, 56},
                             {4, 128, 1, 56, 56},
                             {128, 128, 1, 3, 3},
                             {128, 128, 1, 3, 3},
                             {4, 128, 1, 56, 56},
                             {4, 128, 1, 56, 56},
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
                             false,
                             true},
};

INSTANTIATE_TEST_SUITE_P(Conv3DBpFilterV2Cov, Conv3DBpFilterV2CovSuite, testing::ValuesIn(dw_cov_cases));
} // namespace
