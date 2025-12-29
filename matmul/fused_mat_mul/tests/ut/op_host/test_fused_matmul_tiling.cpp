/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;

namespace {
static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
    result += " ";
  }

  return result;
}

struct TilingTestParam {
  string case_name;
  string op_type;
  string fused_op_type;
  string compile_info;

  // input
  ge::Format x1_format;
  ge::Format x1_ori_format;
  ge::Format x2_format;
  ge::Format x2_ori_format;
  ge::Format y_format;
  ge::Format y_ori_format;
  bool trans_a;
  bool trans_b;
  int32_t offset_x;
  bool enable_hf32;
  std::initializer_list<int64_t> x1_shape;
  std::initializer_list<int64_t> x2_shape;
  std::initializer_list<int64_t> y_shape;
  std::initializer_list<int64_t> x1_orishape;
  std::initializer_list<int64_t> x2_orishape;
  std::initializer_list<int64_t> y_orishape;
  bool private_attr;
  int32_t input_size;
  int32_t hidden_size;

  // output
  uint32_t block_dim;
  uint64_t tiling_key;
  string tiling_data;

  ge::DataType input_dtype = DT_FLOAT16;
  ge::DataType y_dtype = DT_FLOAT16;
  ge::DataType bias_dtype = DT_FLOAT16;
};

class FusedMatMulTilingRuntime : public testing::TestWithParam<TilingTestParam> {
  virtual void SetUp() {
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

static void TestOneParamCase(const TilingTestParam &param) {
  gert::StorageShape x1_shape = {param.x1_orishape, param.x1_shape};
  gert::StorageShape x2_shape = {param.x2_orishape, param.x2_shape};

  std::vector<int64_t> vec(param.y_orishape);
  gert::StorageShape bias_shape = {{vec[1]},{vec[1]}};
  std::vector<gert::StorageShape> output_shapes(1, {param.y_orishape, param.y_shape});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();

  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(param.compile_info.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->gen_simplifiedkey;
  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);
  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  auto tiling_data = gert::TilingData::CreateCap(2048);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());

  gert::KernelRunContextHolder holder;
  holder = gert::TilingContextFaker()
                    .SetOpType(param.op_type.c_str())
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape, &output_shapes[0]})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<int64_t>(param.enable_hf32)},
                                {"fused_op_type", Ops::NN::AnyValue::CreateFrom<string>(param.fused_op_type)}})
                    .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                    .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                    .NodeInputTd(2, param.input_dtype, param.x2_ori_format, param.x2_format)
                    .NodeInputTd(3, param.input_dtype, param.x2_ori_format, param.x2_format)
                    .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  ge::char_t simplifiedKey[100] = {0};
  ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);
  uint64_t tiling_key = tiling_context->GetTilingKey();
  uint32_t block_dim = tiling_context->GetBlockDim();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  cout<<"===== "<<tiling_key<<" === "<<tiling_data_result<<std::endl;
  EXPECT_EQ(tiling_key, param.tiling_key);
  EXPECT_EQ(block_dim, param.block_dim);
  EXPECT_EQ(tiling_data_result, param.tiling_data);
}

TEST_P(FusedMatMulTilingRuntime, general_cases) {
  TestOneParamCase(GetParam());
}

static TilingTestParam ascend910D_cases_params[] = {
  {
    "FusedMatMul_basic_test00", "FusedMatMul", "mul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {1280, 8192}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 33554465UL,
    "32 4096 1280 8192 256 256 128 256 256 64 8192 2 1 1 1 0 0 16843264 0 256 1 1 0 "
  },
  {
    "FusedMatMul_910D1_basic_01", "FusedMatMul", "mul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {128, 8192}, {4096, 128}, {4096, 8192}, {128, 8192}, {4096, 128}, false, 0, 0, 32, 33554465UL,
    "32 4096 128 8192 128 128 256 128 128 128 8192 1 1 1 1 0 0 16908800 0 128 1 1 0 "
  },
  {
    "FusedMatMul_910D1_basic_02", "FusedMatMul", "add", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {1280, 8192}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 16777249UL,
    "32 4096 1280 8192 256 256 128 256 256 64 8192 2 1 1 1 0 0 16843264 0 256 1 1 0 "
  },
  {
    "FusedMatMul_910D1_basic_03", "FusedMatMul", "add", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {128, 8192}, {4096, 128}, {4096, 8192}, {128, 8192}, {4096, 128}, false, 0, 0, 32, 16777249UL,
    "32 4096 128 8192 128 128 256 128 128 128 8192 1 1 1 1 0 0 16908800 0 128 1 1 0 "
  },
  // FudesMatmul + "" -> streamK
  {
    "FusedMatMul_910D1_basic_04", "FusedMatMul", "", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {1280, 8192}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 4129UL,
    "32 4096 1280 8192 256 256 192 256 256 64 4096 1 1 1 1 0 0 16843264 0 256 1 1 0 "
  },
  {
    "FusedMatMul_910D1_basic_05", "FusedMatMul", "", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {128, 8192}, {4096, 128}, {4096, 8192}, {128, 8192}, {4096, 128}, false, 0, 0, 32, 4129UL,
    "32 4096 128 8192 256 128 256 256 128 64 4096 1 1 1 1 0 0 16843264 0 256 1 1 0 "
  },
  {
    "FusedMatMul_910D1_basic_06", "FusedMatMul", "gelu_erf", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {1280, 8192}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 50331680UL,
    "4096 1280 8192 1 "
  },
  {
    "FusedMatMul_910D1_basic_07", "FusedMatMul", "gelu_tanh", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {128, 8192}, {4096, 128}, {4096, 8192}, {128, 8192}, {4096, 128}, false, 0, 0, 32, 67108896UL,
    "4096 128 8192 1 "
  },
  {
    "FusedMatMul_910D1_mmoe_fused_op_type_none_basic_asw", "FusedMatMul", "", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":28},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 28, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {128, 18856}, {4096, 18856}, {128, 4096}, {128, 18856}, {4096, 18856}, {128, 4096}, false, 0, 0, 26, 33UL,
    "26 128 4096 18856 128 160 192 128 160 48 18856 1 1 1 1 0 0 16908800 0 128 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
  "FusedMatMul_910D1_mmoe_fused_op_type_relu_basic_asw", "FusedMatMul", "relu", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32": 1},
    "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
    "block_dim":{"CORE_NUM":28},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 28, "socVersion": "Ascend910_95" },
    "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {128, 4096}, {4096, 4096}, {128, 4096}, {128, 4096}, {4096, 4096}, {128, 4096}, false, 0, 0, 26, 83886113UL,
  "26 128 4096 4096 128 160 192 128 160 48 4096 1 1 1 1 0 0 16908800 0 128 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
  },
  // tilingKey 33 289   block 26  28  data？
  {
    "FusedMatMul_910D1_mmoe_fused_op_type_none_streamk", "FusedMatMul", "", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":28},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 28, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {128, 4096}, {2048, 4096}, {128, 2048}, {128, 4096}, {2048, 4096}, {128, 2048}, false, 0, 0, 28, 4129UL,
    "28 128 2048 4096 128 256 256 128 256 64 1366 1 1 1 1 0 0 16843264 0 128 1 1 0 ", ge::DT_BF16, ge::DT_BF16, ge::DT_BF16
  },
  // 5243169   5242913   16  28
  {
  "FusedMatMul_910D1_mmoe_fused_op_type_relu_streamk", "FusedMatMul", "relu", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32": 0},
    "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
    "block_dim":{"CORE_NUM":28},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 28, "socVersion": "Ascend910_95" },
    "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {128, 18856}, {64, 18856}, {128, 64}, {128, 18856}, {64, 18856}, {128, 64}, false, 0, 0, 28, 83890209UL,
  "28 128 64 18856 128 64 512 128 64 128 674 1 1 1 1 0 0 16843264 0 128 1 1 0 ", ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT
  }
};

INSTANTIATE_TEST_CASE_P(FusedMatMulAscend910D, FusedMatMulTilingRuntime, testing::ValuesIn(ascend910D_cases_params));
} // namespace