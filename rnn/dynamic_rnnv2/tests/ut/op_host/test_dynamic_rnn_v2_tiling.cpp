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
 * \file test_dynamic_rnn_v2_tiling.cpp
 * \brief
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "array_ops.h"
#include "graph/graph.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "op_common/op_host/util/platform_util.h"

using namespace std;
using namespace ge;
using namespace op;

namespace optiling {

class CompileInfoBase;
using CompileInfoPtr = std::shared_ptr<CompileInfoBase>;

class CompileInfoBase {
public:
  CompileInfoBase() {}
  virtual ~CompileInfoBase() {}
};

struct DynamicRNNV2CompileInfo : public optiling::CompileInfoBase {
  std::vector<std::vector<int64_t>> tuneShapeList;
};
} // namespace optiling

struct DynamicRNNV2TilingTestParam {
  string case_name;
  string compile_info;

  std::initializer_list<int64_t> x_shape;
  std::initializer_list<int64_t> w_shape;
  std::initializer_list<int64_t> b_shape;
  std::initializer_list<int64_t> init_shape;
  std::initializer_list<int64_t> y_shape;

  ge::Format x_format;
  ge::Format w_format;
  ge::Format b_format;
  ge::Format y_format;

  ge::DataType x_dtype;
  ge::DataType w_dtype;
  ge::DataType b_dtype;
  ge::DataType y_dtype;

  // attr
  string cell_type;
  string direction;
  int cell_depth;
  bool use_peephole;
  float keep_prob;
  float cell_clip;
  int num_proj;
  bool time_major;
  string activation;
  string recurrent_activation;

  float forget_bias;
  string gate_order;
  bool stateful;
  string merge_mode;
  bool is_training;

  bool isInithc;
  bool isSeqLength;

  bool parse_result;
  bool tiling_result;

  // output
  // uint32_t block_dim;
  // uint64_t tiling_key;
  // std::string tiling_data;
};

class DynamicRNNV2TilingRunTime2 : public testing::TestWithParam<DynamicRNNV2TilingTestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicRNNV2TilingTestParam SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicRNNV2TilingTestParam TearDown" << std::endl;
  }
};

class DynamicRNNV2TilingRunTime22 : public testing::TestWithParam<DynamicRNNV2TilingTestParam> {
 protected:
  static void SetUpTestCase() {
    std::cout << "DynamicRNNV2TilingTestParam SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DynamicRNNV2TilingTestParam TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

static std::unique_ptr<uint8_t[]> InitListIntAttr(const std::vector<int32_t> &list_int_attr) {
  auto attr_ptr = gert::ContinuousVector::Create<int32_t>(list_int_attr.size());
  auto attr = reinterpret_cast<gert::ContinuousVector *>(attr_ptr.get());
  size_t copy_size = list_int_attr.size() * sizeof(int32_t);
  (void)memcpy_s(attr->MutableData(), copy_size, list_int_attr.data(), copy_size);
  return attr_ptr;
}

void RNNV2Test(DynamicRNNV2TilingTestParam param) {
  std::cout << "run case " << param.case_name << std::endl;

  gert::StorageShape x_shape = {param.x_shape, param.x_shape};
  gert::StorageShape w1_shape = {param.w_shape, param.w_shape};
  gert::StorageShape w2_shape = {param.w_shape, param.w_shape};
  gert::StorageShape b_shape = {param.b_shape, param.b_shape};
  gert::StorageShape seq_shape = {};
  gert::StorageShape h0_shape = {param.init_shape, param.init_shape};
  gert::StorageShape c0_shape = {param.init_shape, param.init_shape};
  std::vector<gert::StorageShape> output_shapes(8, {param.y_shape, param.y_shape});
  std::vector<void *> output_shapes_ref(8);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;
  optiling::DynamicRNNV2CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(param.compile_info.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics);

  std::string op_type("DynamicRNNV2");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());

  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);
  if (param.parse_result) {
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  } else {
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
    return;
  }

  auto tiling_data = gert::TilingData::CreateCap(2048);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(7, 8)
                    .IrInstanceNum({1,1,1,1,1,1,1})
                    .InputShapes({&x_shape, &w1_shape, &w2_shape, &b_shape, nullptr, &h0_shape, &c0_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"cell_type", Ops::NN::AnyValue::CreateFrom<std::string>(param.cell_type)},
                                {"direction", Ops::NN::AnyValue::CreateFrom<std::string>(param.direction)},
                                {"cell_depth", Ops::NN::AnyValue::CreateFrom<int64_t>(param.cell_depth)},
                                {"use_peephole", Ops::NN::AnyValue::CreateFrom<bool>(param.use_peephole)},
                                {"keep_prob", Ops::NN::AnyValue::CreateFrom<float>(param.keep_prob)},
                                {"cell_clip", Ops::NN::AnyValue::CreateFrom<float>(param.cell_clip)},
                                {"num_proj", Ops::NN::AnyValue::CreateFrom<int64_t>(param.num_proj)},
                                {"time_major", Ops::NN::AnyValue::CreateFrom<bool>(param.time_major)},
                                {"activation", Ops::NN::AnyValue::CreateFrom<std::string>(param.activation)},
                                {"recurrent_activation", Ops::NN::AnyValue::CreateFrom<std::string>(param.recurrent_activation)},
                                {"forget_bias", Ops::NN::AnyValue::CreateFrom<float>(param.forget_bias)},
                                {"gate_order", Ops::NN::AnyValue::CreateFrom<string>(param.gate_order)},
                                {"stateful", Ops::NN::AnyValue::CreateFrom<bool>(param.stateful)},
                                {"merge_mode", Ops::NN::AnyValue::CreateFrom<std::string>(param.merge_mode)},
                                {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(param.is_training)},})
                    .NodeInputTd(0, ge::DT_FLOAT, param.x_format, param.x_format)
                    .NodeInputTd(1, ge::DT_FLOAT, param.w_format, param.w_format)
                    .NodeInputTd(2, ge::DT_FLOAT, param.w_format, param.w_format)
                    .NodeInputTd(3, ge::DT_FLOAT, param.b_format, param.b_format)
                    .NodeInputTd(4, ge::DT_FLOAT, param.b_format, param.b_format)
                    .NodeInputTd(5, ge::DT_FLOAT, param.b_format, param.b_format)
                    .NodeInputTd(6, ge::DT_FLOAT, param.b_format, param.b_format)
                    .NodeOutputTd(0, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(1, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(2, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(3, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(4, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(5, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(6, ge::DT_FLOAT, param.y_format, param.y_format)
                    .NodeOutputTd(7, ge::DT_FLOAT, param.y_format, param.y_format)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .Workspace(ws_size)
                    .TilingData(tiling_data.get())
                    .Build();


  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_TRUE(tiling_context->GetPlatformInfo()->Init());

  tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  if (param.tiling_result) {
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  } else {
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    return;
  }
}

TEST_P(DynamicRNNV2TilingRunTime2, general_cases) {
  DynamicRNNV2TilingTestParam param = GetParam();
  RNNV2Test(param);
}

TEST_P(DynamicRNNV2TilingRunTime22, general_cases2) {
  DynamicRNNV2TilingTestParam param = GetParam();
  RNNV2Test(param);
}

static DynamicRNNV2TilingTestParam general_cases_params[] = {
  {"DynamicRNNV2_fuzz_test1", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32},
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false},
        "binary_attrs":{"bias_flag":false,"nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz":false, "l2_size":33554432},"binary_mode_flag":true,
        "block_dim": {"133124": 32}, "correct_range_flag":null,
        "_vars":{"133124":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"133124":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"133124":[]},
        "_attr_vars":{"133124":[]}})",
    {1, 16, 16}, {16, 64}, {64}, {1, 16, 16}, {1, 16, 16},
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    "LSTM", "UNIDIRECTIONAL", 1, false, 1.0, -1.0, 0, true, "tanh", "sigmoid",0.0,  "ifco", false,"concat",true,
    true, false,
    true, true,

  }
};

static DynamicRNNV2TilingTestParam general_cases_params2[] = {
    {"DynamicRNNV2_fuzz_test2", R"({"_pattern": "MatMul", "format_a": "FRACTAL_NZ", "format_b": "FRACTAL_NZ", "dynamic_mode":"dynamic_mknb",
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32},
        "repo_seeds": {}, "repo_range": {}, "attrs":{"transpose_a": false, "transpose_b": false},
        "binary_attrs":{"bias_flag":false,"nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz":false, "l2_size":33554432},"binary_mode_flag":true,
        "block_dim": {"133124": 32}, "correct_range_flag":null,
        "_vars":{"133124":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]},
        "_custom_vars":{"133124":["m", "k", "n", "batch_single_core", "m_single_core", "n_single_core", "batch_dim",
        "n_dim", "m_dim", "m_al1", "n_bl1", "cub_n1", "m_l0", "k_l0", "n_ub_l0_time", "kal0_factor", "kbl0_factor",
        "kal1_factor", "kbl1_factor", "kal1_16", "kbl1_16", "kl1_times", "batch"]}, "_normal_vars":{"133124":[]},
        "_attr_vars":{"133124":[]}, "vars":{"tune_shape_list":[[1,256,3]]}})",
        {1, 16, 16}, {16, 64}, {64}, {1, 16, 16}, {1, 16, 16},
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,

        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
        "LSTM", "UNIDIRECTIONAL", 1, false, 1.0, -1.0, 0, true, "tanh", "sigmoid",0.0,  "ifco", false,"concat",true,
        true, false,
        true, true,
    }
};

INSTANTIATE_TEST_CASE_P(DynamicRNNV2, DynamicRNNV2TilingRunTime2, testing::ValuesIn(general_cases_params));
INSTANTIATE_TEST_CASE_P(DynamicRNNV2, DynamicRNNV2TilingRunTime22, testing::ValuesIn(general_cases_params2));