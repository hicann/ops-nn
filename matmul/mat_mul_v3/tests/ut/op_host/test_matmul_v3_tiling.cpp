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
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ge;

namespace {
bool IsDisplayTilingdata(const string& case_name, size_t index, const uint64_t& tiling_key)
{
    // 18 22-27 30-32 48-57 表示mm实际用到的tilingdata
    if (index < 18UL || (index >= 22UL && index <= 27UL) || (index >= 30UL && index <= 32UL) || (index >= 48UL && index <= 57UL)) {
        return true;
    }
    // 非910Dtiling要检查57位以后的tilingdata
    if (case_name.find("MatMulV3_910D1") == string::npos && index > 57UL) {
        return true;
    }
    // 基础API校验全部的tilingdata
    string tilingKey = std::to_string(tiling_key);
    if (!tilingKey.empty() && tilingKey.find("100019") == 0UL) {
        return true;
    }

    return false;
}

static string TilingData2Str(const gert::TilingData* tiling_data, const string& case_name, const uint64_t& tiling_key)
{
    if (tiling_data == nullptr) {
        return "";
    }
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        if (IsDisplayTilingdata(case_name, i / sizeof(int32_t), tiling_key)) {
            result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
            result += " ";
        }
    }
    return result;
}

static string GenGoldenTilingData(const string& tiling_data, const string& case_name, const uint64_t& tiling_key)
{
    istringstream iss(tiling_data);
    vector<string> data_list;
    string tmp;
    while (iss >> tmp) {
        data_list.push_back(tmp);
    }
    string golden_tiling_data;
    for (size_t i = 0; i < data_list.size(); i++) {
        if (IsDisplayTilingdata(case_name, i, tiling_key)) {
            golden_tiling_data += data_list[i];
            golden_tiling_data += " ";
        }
    }
    return golden_tiling_data;
}

struct TilingTestParam {
  string case_name;
  string op_type;
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

  std::initializer_list<int64_t> perm_x1;
  std::initializer_list<int64_t> perm_x2;
  std::initializer_list<int64_t> perm_y;
};

class MatMulV3TilingRuntime : public testing::TestWithParam<TilingTestParam> {
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
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset_x)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(param.enable_hf32)}})
                    .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                    .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
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
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), param.case_name, tiling_key);
  auto golden_tiling_data = GenGoldenTilingData(param.tiling_data, param.case_name, param.tiling_key);
  cout<<"===== "<<tiling_key<<" === "<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, param.tiling_key);
  ASSERT_EQ(block_dim, param.block_dim);
  ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_P(MatMulV3TilingRuntime, general_cases) {
  TestOneParamCase(GetParam());
}

static TilingTestParam ascend910B_cases_params[] = {
  {
    "MatMulV3_basic_test00", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {249, 512}, {512, 311}, {249, 311}, {249, 512}, {512, 311}, {249, 311}, false, 0, 0, 24, 0,
    "24 249 311 512 512 32 128 512 32 128 64 64 16 1 1 0 0 0 0 278528 16384 0 1 1 1 1 32 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 8 3 0 0 0 0 0 1 0 0 0 0 0 0 0 0 16 320 "
  },
  {
    "MatMulV3_basic_test01", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {8192, 8192}, {8192, 8192}, {8192, 8192}, {8192, 8192}, {8192, 8192}, {8192, 8192}, false, 0, 0, 24, 65536,
    "24 8192 8192 8192 8192 128 256 8192 128 256 64 16 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 2 8 16 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test02", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {320, 513}, {513, 311}, {320, 311}, {320, 513}, {513, 311}, {320, 311}, false, 0, 0, 24, 0,
    "24 320 311 513 513 128 48 513 128 48 64 16 32 1 1 0 0 0 0 354816 20480 0 1 1 1 1 8 16 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 3 7 0 0 0 0 1 1 0 0 0 0 0 0 16 528 16 320 "
  },
  {
    "MatMulV3_basic_test03", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {256, 1024}, {256, 768}, {1024, 768}, {256, 1024}, {256, 768}, {1024, 768}, false, 0, 0, 24, 65536,
    "24 1024 768 256 256 256 128 256 256 128 64 8 16 1 1 0 0 0 0 262144 131072 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 4 6 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test04", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {640, 513}, {513, 320}, {640, 320}, {640, 513}, {513, 320}, {640, 320}, false, 0, 0, 24, 0,
    "24 640 320 513 513 128 80 513 128 80 64 16 16 1 1 0 0 0 0 371712 40960 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 5 4 0 0 0 0 1 1 0 0 0 0 0 0 16 528 16 320 "
  },
  {
    "MatMulV3_basic_test05", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {4224, 10832}, {1408, 4224}, {10832, 1408}, {4224, 10832}, {1408, 4224}, {10832, 1408}, false, 0, 0, 24, 0,
    "24 10832 1408 4224 4224 256 128 4224 256 128 64 8 16 1 1 0 0 0 0 360448 98304 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 15 11 0 0 1 1 1 0 0 0 0 0 0 0 23 2048 0 0 "
  },
  {
    "MatMulV3_basic_test06", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {150, 1024}, {1024, 131328}, {150, 131328}, {150, 1024}, {1024, 131328}, {150, 131328}, false, 0, 0, 24, 0,
    "24 150 131328 1024 1024 128 128 1024 128 128 64 16 16 1 1 0 0 0 0 475136 92160 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 171 2 6 0 0 0 0 0 1 0 0 0 0 0 0 0 0 33 1024 "
  },
  {
    "MatMulV3_basic_test07", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {30720, 101376}, {101376, 122880}, {30720, 122880}, {30720, 101376}, {101376, 122880}, {30720, 122880}, false, 0, 0, 24, 32,
    "24 30720 122880 101376 101376 30720 2560 384 128 128 128 9 6 3 1 0 0 1 0 458752 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 60 80 4 6 0 0 0 0 1 1 0 0 0 0 0 0 20 2048 23 2048 "
  },
  {
    "MatMulV3_basic_test11", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {640, 127}, {127, 320}, {640, 320}, {640, 127}, {127, 320}, {640, 320}, false, 0, 0, 24, 0,
    "24 640 320 127 127 128 80 127 128 80 64 16 16 1 1 0 0 0 0 139264 32768 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 5 4 0 0 0 0 1 1 0 0 0 0 0 0 16 128 16 320 "
  },
  {
    "MatMulV3_basic_test12", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {640, 127}, {640, 320}, {127, 320}, {640, 127}, {640, 320}, {127, 320}, false, 0, 0, 24, 0,
    "24 127 320 640 640 128 128 640 128 128 128 8 8 1 1 0 0 0 0 122880 9216 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 3 0 0 1 0 1 1 0 0 0 0 0 0 16 128 16 320 "
  },
  {
    "MatMulV3_basic_test13", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1024, 30720}, {30720, 1024}, {1024, 1024}, {1024, 30720}, {30720, 1024}, {1024, 1024}, false, 0, 0, 24, 65584,
    "24 1024 1024 30720 30720 384 1024 384 128 128 128 9 6 3 1 0 0 1 0 458752 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 1 6 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test14", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {3753, 261}, {2146, 3753}, {261, 3753}, {3753, 261}, {2146, 3753}, {261, 3753}, false, 0, 0, 24, 0,
    "24 261 2146 3753 3753 128 256 3753 128 256 64 16 8 1 1 0 0 0 0 61440 110592 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 3 9 0 0 1 1 1 1 0 0 0 0 0 0 79 272 18 2048 "
  },
  {
    "MatMulV3_basic_test15_unalignN", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {150000, 128}, {84, 128}, {150000, 84}, {150000, 128}, {84, 128}, {150000, 84}, false, 0, 0, 24, 73730,
    "24 150000 84 128 128 128 96 128 128 96 64 2 2 1 1 0 0 0 0 180224 49152 0 1 1 1 1 2 2 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_basic_test16_unalignN", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
    "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
    "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
    "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {14768, 192}, {192, 240}, {14768, 240}, {14768, 192}, {192, 240}, {14768, 240}, false, 0, 0, 24, 69634,
   "24 14768 240 192 192 128 240 192 128 240 32 6 6 1 1 0 0 0 0 393728 65536 0 1 1 1 1 6 6 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_basic_testNKM", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":20},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4000, 30000}, {16, 30000}, {4000, 16}, {4000, 30000}, {16, 30000}, {4000, 16}, false, 0, 0, 20, 65616,
    "20 4000 16 30000 30000 200 64 256 128 128 64 8 8 1 1 0 0 1 0 401408 13312 0 1 1 1 1 4 4 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 1 4 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_incre_test01", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {32, 1408}, {11264, 1408}, {32, 11264}, {32, 1408}, {11264, 1408}, {32, 11264}, false, 0, 0, 22, 65536,
    "22 32 11264 1408 1408 32 128 1408 32 128 128 4 4 1 1 0 0 0 0 221184 32768 0 1 1 1 1 2 2 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 88 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_incre_testxx", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {16384, 5121}, {5121, 1709}, {16384, 1709}, {16384, 5121}, {5121, 1709}, {16384, 1709}, false, 0, 0, 24, 0,
    "24 16384 1709 5121 5121 128 256 5121 128 256 64 16 8 1 1 0 0 0 0 360448 131072 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 1 26 7 0 0 0 0 1 1 0 0 1 0 0 0 38 1040 16 2048 "
  },
  {
    "MatMulV3_incre_test_splitk", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {26812, 12288}, {26812, 6144}, {12288, 6144}, {26812, 12288}, {26812, 6144}, {12288, 6144}, false, 0, 0, 24, 65632,
    "24 12288 6144 26812 26812 1024 3072 384 128 128 128 9 6 3 1 0 0 1 0 98304 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 4 4 6 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_with_bias", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {32, 1280}, {1280, 10240}, {32, 10240}, {32, 1280}, {1280, 10240}, {32, 10240}, false, 0, 0, 24, 65536,
    "24 32 10240 1280 1280 32 512 1280 32 512 32 128 8 1 1 0 0 0 0 358400 55296 0 1 1 1 1 64 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 20 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_split_k_small_mn", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {256, 15000}, {15000, 256}, {256, 256}, {256, 15000}, {15000, 256}, {256, 256}, false, 0, 0, 24, 65584,
    "24 256 256 15000 15000 384 256 384 128 128 128 9 6 3 1 0 0 1 0 514048 16384 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 16 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_vnchw_test_01", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {8200, 88}, {88, 32}, {8200, 32}, {8200, 88}, {88, 32}, {8200, 32}, false, 0, 0, 24, 131074,
    "24 8200 32 88 88 256 32 88 128 32 64 4 2 1 1 0 0 0 0 200704 16384 0 1 1 1 1 2 2 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 33 1 1 0 0 0 1 0 0 0 1 0 0 0 171 96 0 0 "
  },
  {
    "MatMulV3_basic_testNK", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {35264, 1184}, {35264, 352}, {1184, 352}, {35264, 1184}, {35264, 352}, {1184, 352}, false, 0, 0, 24, 65584,
    "24 1184 352 35264 35264 1184 384 384 128 128 128 6 9 1 3 0 0 0 0 98304 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 4 2 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test_TLB_singleK", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {16384, 7144}, {16384, 8192}, {7144, 8192}, {16384, 7144}, {16384, 8192}, {7144, 8192}, false, 0, 0, 24, 96,
    "24 7144 8192 16384 16384 896 2816 384 128 128 128 9 6 3 1 0 0 1 0 196608 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 8 7 8 0 0 1 0 1 0 0 0 0 0 0 0 38 1024 0 0 "
  },
  {
    "MatMulV3_basic_testNK2", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {30720, 3456}, {30720, 1152}, {3456, 1152}, {30720, 3456}, {30720, 1152}, {3456, 1152}, false, 0, 0, 24, 65584,
    "24 3456 1152 30720 30720 1792 384 384 128 128 128 6 9 1 3 0 0 0 0 196608 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 2 4 6 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_testNK3", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {65536, 720}, {65536, 128}, {720, 128}, {65536, 720}, {65536, 128}, {720, 128}, false, 0, 0, 24, 65584,
    "24 720 128 65536 65536 384 128 384 128 128 128 9 6 3 1 0 0 1 0 262144 15360 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 1 4 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_testMK_L2_split", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {66048, 128}, {66048, 8192}, {128, 8192}, {66048, 128}, {66048, 8192}, {128, 8192}, false, 0, 0, 24, 65584,
    "24 128 8192 66048 66048 384 4096 384 128 128 128 9 6 3 1 0 0 1 0 196608 131072 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 6 1 6 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_testMK_N_16", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {32768, 15104}, {16, 32768}, {15104, 16}, {32768, 15104}, {16, 32768}, {15104, 16}, false, 0, 0, 24, 65584,
    "24 15104 16 32768 32768 384 16 384 128 128 128 9 6 3 1 0 0 1 0 425984 40960 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 1 4 1 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_testMK", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {66048, 4000}, {66048, 16}, {4000, 16}, {66048, 4000}, {66048, 16}, {4000, 16}, false, 0, 0, 24, 65584,
    "24 4000 16 66048 66048 384 16 384 128 128 128 9 6 3 1 0 0 1 0 393216 11264 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 1 4 1 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_testMK", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1, 7168}, {7168, 16}, {1, 16}, {1, 7168}, {7168, 16}, {1, 16}, false, 0, 0, 11, 65584,
    "11 1 16 7168 7168 384 16 192 128 128 64 9 6 3 1 0 0 1 0 131072 1024 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_basic_testMK", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":6},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1048, 40}, {1048, 16}, {40, 16}, {1048, 40}, {1048, 16}, {40, 16}, false, 0, 0, 6, 65584,
    "6 40 16 1048 1048 384 16 192 128 128 64 9 6 3 1 0 0 1 0 196608 1024 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_basic_test_nd2nz_fix_2", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {82533, 4}, {4, 5}, {82533, 5}, {82533, 4}, {4, 5}, {82533, 5}, false, 0, 0, 24, 0,
    "24 82533 5 4 4 128 16 4 128 16 64 16 128 1 1 0 0 0 0 131584 65536 0 1 1 1 1 8 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 645 1 0 0 0 0 1 1 0 0 0 0 0 0 1720 16 16 16 "
  },
  {
    "MatMulV3_basic_test_nd2nz_fix_3", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {8253, 4}, {4, 5}, {8253, 5}, {8253, 4}, {4, 5}, {8253, 5}, false, 0, 0, 24, 0,
    "24 8253 5 4 4 128 16 4 128 16 64 16 128 1 1 0 0 0 0 11776 22528 0 1 1 1 1 8 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 65 1 0 0 0 0 1 1 0 0 0 0 0 0 172 16 16 16 "
  },
  {
    "MatMulV3_basic_test_nd2nz_fix_4", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {12366, 223}, {223, 9}, {12366, 9}, {12366, 223}, {223, 9}, {12366, 9}, false, 0, 0, 24, 0,
    "24 12366 9 223 223 128 16 223 128 16 64 16 128 1 1 0 0 0 0 401408 16384 0 1 1 1 1 8 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 97 1 0 0 0 0 1 1 0 0 0 0 0 0 129 224 16 16 "
  },
  {
    "MatMulV3_basic_test_mata_opt", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 16384}, {5120, 16384}, {4096, 5120}, {4096, 16384}, {5120, 16384}, {4096, 5120}, false, 0, 0, 24, 65632,
    "24 4096 5120 16384 16384 352 2560 384 128 128 128 9 6 3 1 0 0 1 0 491520 114688 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 2 4 10 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_test_al1_full_load", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {8, 7168}, {256, 7168}, {8, 256}, {8, 7168}, {256, 7168}, {8, 256}, false, 0, 0, 24, 65537,
    "24 8 256 7168 7168 8 16 7168 16 16 256 28 2 1 1 0 0 0 0 262144 1024 0 1 1 1 1 28 1 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 16 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ", DT_FLOAT, DT_FLOAT
  },
  {
    "MatMulV3_mata_case0", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 16384}, {7168, 16384}, {4096, 7168}, {4096, 16384}, {7168, 16384}, {4096, 7168}, false, 0, 0, 24, 0,
    "24 4096 7168 16384 16384 128 256 16384 128 256 64 16 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 3 4 10 0 0 0 1 0 1 0 0 0 0 0 0 0 0 23 2048 "
  },
  {
    "MatMulV3_mata_case1", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {4096, 7168}, {7168, 16384}, {4096, 16384}, {4096, 7168}, {7168, 16384}, {4096, 16384}, false, 0, 0, 24, 0,
    "24 4096 16384 7168 7168 128 256 7168 128 256 64 16 8 1 1 0 0 0 0 393216 131072 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 5 16 13 0 0 0 0 0 1 0 0 0 0 0 0 0 0 23 2048 "
  },
  {
    "MatMulV3_basic_test_tilingOptimization", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {8192, 3072}, {3072, 1536}, {8192, 1536}, {8192, 3072}, {3072, 1536}, {8192, 1536}, false, 0, 0, 24, 65536,
    "24 8192 1536 3072 3072 128 256 3072 128 256 64 8 8 1 1 0 0 0 0 458752 131072 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 64 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test_weightNz_N_not_align_16", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": true, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4, 7168}, {448, 127, 16, 16}, {4, 2020}, {4, 7168}, {2020, 7168}, {4, 2020}, false, 0, 0, 24, 65536,
    "24 4 2020 7168 7168 16 96 7168 16 96 128 64 8 1 1 0 0 0 0 327680 6144 0 1 1 1 1 32 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 22 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test_weightNz_tiling_select_BASE", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": true, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {24576, 16}, {1, 1536, 16, 16}, {16, 16}, {24576, 16}, {24576, 16}, {16, 16}, false, 0, 0, 24, 65536,
    "24 16 16 24576 24576 16 1024 24576 16 1024 16 512 8 1 1 0 0 0 0 393216 1024 0 1 1 1 1 256 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  }

};

static TilingTestParam ascend310P_cases_params[] = {
{
    "MatMulV3_basic_test08", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":8},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8, "socVersion": "Ascend310P" },
      "format_a":"FRACTAL_NZ","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, false, true, 0, false, {32, 28, 16, 16}, {32, 16, 16, 16}, {16, 28, 16, 16}, {448, 512}, {256, 512}, {448, 256}, false, 0, 0, 8, 65536,
    "8 448 256 512 512 112 128 512 112 128 128 16 16 1 1 0 131072 0 0 245760 57344 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 4 2 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test09", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":8},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8, "socVersion": "Ascend310P" },
      "format_a":"FRACTAL_NZ","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, false, false, 0, false, {32, 4, 16, 16}, {16, 32, 16, 16}, {16, 4, 16, 16}, {64, 512}, {512, 256}, {64, 256}, false, 0, 0, 8, 65536,
    "8 64 256 512 512 32 64 512 32 64 256 16 16 1 1 0 131072 0 0 278528 16384 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_test10", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":8},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8, "socVersion": "Ascend310P" },
      "format_a":"FRACTAL_NZ","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, true, false, 0, false, {4, 32, 16, 16}, {16, 32, 16, 16}, {16, 4, 16, 16}, {512, 64}, {512, 256}, {64, 256}, false, 0, 0, 8, 65536,
    "8 64 256 512 512 16 256 512 16 256 64 16 8 1 1 0 131072 0 0 98304 8192 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 4 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_basic_310p_no_full_load", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":8},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8, "socVersion": "Ascend310P" },
      "format_a":"FRACTAL_NZ","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {6, 154, 16, 16}, {4, 6, 16, 16}, {2464, 64}, {2464, 96}, {96,64}, {2464,64}, false, 0, 0, 8, 65536,
    "8 2464 64 96 96 256 64 96 256 64 64 16 2 1 1 0 131072 0 0 396288 65536 0 1 1 1 1 8 2 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 4 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
  {
    "MatMulV3_tilingFuzz_test1", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {229801796, 1}, {5, 229801796}, {1, 5}, {229801796, 1}, {5, 229801796}, {1, 5}, false, 0, 0, 24, 48,
    "24 1 5 229801796 229801796 384 5 384 128 128 128 9 6 3 1 0 0 1 0 393216 1024 0 1 1 1 1 3 3 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 0 0 0 0 0 0 3067 16 23 2048 "
  },
  {
    "MatMulV3_longseq_test1", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {121926, 64}, {64, 64}, {121926, 64}, {121926, 64}, {64, 64}, {121926, 64}, false, 0, 0, 24, 65538,
    "24 121926 64 64 64 256 64 64 128 64 64 2 1 1 1 0 0 0 0 335872 65536 0 1 1 1 1 1 1 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 477 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 "
  },
};

static TilingTestParam ascend910D_cases_params[] = {
  {
    "MatMulV3_910D1_basic_testNZ", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {512, 80, 16, 16}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 10001900009000290002UL,
    "32 4096 1280 8192 256 256 256 256 256 64 4096 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_basic_test15", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {1280, 8192}, {4096, 1280}, {4096, 8192}, {1280, 8192}, {4096, 1280}, false, 0, 0, 32, 10001900009000290002UL,
    "32 4096 1280 8192 256 256 256 256 256 64 4096 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_basic_test16", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4096, 8192}, {128, 8192}, {4096, 128}, {4096, 8192}, {128, 8192}, {4096, 128}, false, 0, 0, 32, 10001900009000290002UL,
    "32 4096 128 8192 256 128 256 256 128 64 4096 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_streamK_fp16_test17", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4, 8192}, {1280, 8192}, {4, 1280}, {4, 8192}, {1280, 8192}, {4, 1280}, false, 0, 0, 32, 10001900009000290002UL,
    "32 4 1280 8192 16 256 256 16 256 64 1366 1 1 1 1 0 0 16843264 "
  },
  {
      "MatMulV3_910D1_basic_test20", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false,{3952, 224}, {192, 3952}, {224, 192}, {3952, 224}, {192, 3952}, {224, 192}, false, 0, 0, 32, 10001900009000290003UL,
    "32 224 192 3952 224 192 256 224 192 64 124 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_streamK_fp16_test21", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {523, 10866}, {10866, 246}, {523, 246}, {523, 10866}, {10866, 246}, {523, 246}, false, 0, 0, 32, 10001902009000290000UL,
    "32 523 246 10866 176 256 256 176 256 64 1087 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_al1_full_load_22", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {48, 953}, {3306, 953}, {48, 3306}, {48, 953}, {3306, 953}, {48, 3306}, false, 0, 0, 30, 10001900009000090002UL,
    "30 48 3306 953 48 112 576 48 112 144 953 1 1 1 1 0 0 16843264"
  },
  // {
  //   "MatMulV3_910D1_al1_full_load_23", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
  //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1166, 48}, {1166, 673}, {48, 673}, {1166, 48}, {1166, 673}, {48, 673}, false, 0, 0, 32, 10000900009001090001UL,
  //   "32 48 673 1166 1166 48 32 1166 48 32 160 8 2 1 1 0 0 0 0 409600 6144 0 1 1 1 1 8 1 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  // },
  {
    "MatMulV3_910D1_abl1_full_load_03", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {190, 16}, {2130, 16}, {190, 2130}, {190, 16}, {2130, 16}, {190, 2130}, false, 0, 0, 30, 10001900009000090002UL,
    "30 190 2130 16 96 144 64 96 144 16 16 1 1 1 1 0 0 16843264"
  },
  {
    "MatMulV3_910D1_abl1_full_load_04", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {304, 112}, {3152, 112}, {304, 3152}, {304, 112}, {3152, 112}, {304, 3152}, false, 0, 0, 32, 10001900009000090002UL,
    "32 304 3152 112 160 208 256 160 208 64 112 1 1 1 1 0 0 16843264"
  },
  {
    "MatMulV3_910D1_al1_full_load_05", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {4, 8192}, {32000, 8192}, {4, 32000}, {4, 8192}, {32000, 8192}, {4, 32000}, false, 0, 0, 32, 10000900009001090002UL,
    "32 4 32000 8192 8192 4 256 8192 16 256 64 128 8 1 1 0 0 0 0 434176 21504 0 1 1 1 1 128 4 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 "
  },
  {
    "MatMulV3_910D1_stream_k_black_24", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {1024, 8192}, {2048, 8192}, {1024, 2048}, {1024, 8192}, {2048, 8192}, {1024, 2048}, false, 0, 0, 32, 10001900009000090002UL,
    "32 1024 2048 8192 256 256 128 256 256 32 8192 1 1 1 1 0 0 16843264", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_stream_k_fp16_black_25", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {5120, 821}, {5120, 32}, {821, 32}, {5120, 821}, {5120, 32}, {821, 32}, false, 0, 0, 32, 10001900009000290001,
    "32 821 32 5120 208 32 256 208 32 64 640 1 1 1 1 0 0 16843264 ",ge::DT_FLOAT16, ge::DT_FLOAT16
  },
  {
    "MatMulV3_910D1_stream_k_fp32_white_26", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {32, 8192}, {8192, 64}, {32, 64}, {32, 8192}, {8192, 64}, {32, 64}, true, 0, 0, 32, 10001900009000290000UL,
    "32 32 64 8192 32 64 512 32 64 128 256 1 1 1 1 0 0 16843264 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_stream_k_dpsk_tf32_white_27", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1024, 8192}, {8192, 2304}, {1024, 2304}, {1024, 8192}, {8192, 2304}, {1024, 2304}, true, 0, 0, 32, 10001900009000290000UL,
    "32 1024 2304 8192 256 256 256 256 256 64 1024 1 1 1 1 0 0 16843264 ",ge::DT_FLOAT16, ge::DT_FLOAT16
  },
  {
    "MatMulV3_910D1_stream_k_dpsk_fp16_black_28", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1024, 8192}, {8192, 2048}, {1024, 2048}, {1024, 8192}, {8192, 2048}, {1024, 2048}, true, 0, 0, 32, 10001900009000090000UL,
    "32 1024 2048 8192 256 256 128 256 256 32 8192 1 1 1 1 0 0 16843264", ge::DT_FLOAT, ge::DT_FLOAT
  },
    {
    "MatMulV3_910D1_stream_k_dpsk_fp16_black_29", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1024, 8192}, {8192, 2303}, {1024, 2303}, {1024, 8192}, {8192, 2303}, {1024, 2303}, true, 0, 0, 32, 10000900009000090000UL,
    "32 1024 2303 8192 8192 256 256 8192 256 256 32 8 8 1 1 0 0 0 0 196608 131072 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 2 1 1 1 0 0 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_bl1_full_load_26", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {560, 953}, {80, 953}, {560, 80}, {560, 953}, {80, 953}, {560, 80}, false, 0, 0, 24, 10001900009000090002UL,
    "24 560 80 953 48 48 1344 48 48 336 953 1 1 1 1 0 0 16843264"
  },
  {
    "MatMulV3_910D1_abl1_full_load_27", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {32, 640}, {32, 480}, {640, 480}, {32, 640}, {32, 480}, {640, 480}, false, 0, 0, 32, 10001900009000090001UL,
    "32 640 480 32 80 128 128 80 128 32 32 1 1 1 1 0 0 16843264"
  },
{
    "MatMulV3_910D1_abl1_full_load_28", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {512, 1024}, {16, 1024}, {512, 1024}, {512, 1024}, {16, 1024}, {512, 1024}, false, 0, 0, 32, 10001900009000090002UL,
    "32 512 16 1024 16 16 2048 16 16 512 1024 1 1 1 1 0 0 16843264", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_abl1_full_load_29", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {300, 560}, {300, 20}, {560, 20}, {300, 560}, {300, 20}, {560, 20}, false, 0, 0, 18, 10001900009000090001UL,
    "18 560 20 300 32 32 1024 32 32 256 300 1 1 1 1 0 0 16843264", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_abl1_full_load_31", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {944, 48}, {80, 48}, {944, 80}, {944, 48}, {80, 48}, {944, 80}, false, 0, 0, 30, 10001900009000090002UL,
    "30 944 80 48 64 48 192 64 48 48 48 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_bl1_full_load_32", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1024, 1022}, {1024, 25}, {1022, 25}, {1024, 1022}, {1024, 25}, {1022, 25}, false, 0, 0, 32, 10001900009000090001UL,
    "32 1022 25 1024 32 32 1024 32 32 256 1024 1 1 1 1 0 0 16843264 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_fixpipe_opti_01", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {1024, 128}, {4090, 128}, {1024, 4090}, {1024, 128}, {4090, 128}, {1024, 4090}, false, 0, 0, 32, 10001901009000090002UL,
    "32 1024 4090 128 256 256 256 256 256 64 128 1 1 1 1 0 0 16843264 "
  },
  {
    "MatMulV3_910D1_fixpipe_opti_02", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {10256, 32}, {32, 720}, {10256, 720}, {10256, 32}, {32, 720}, {10256, 720}, false, 0, 0, 32, 10001902009000090000UL,
    "32 10256 720 32 256 256 64 256 256 32 32 1 1 15 1 240 0 16843264 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_asw_big_k_01", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {160, 2080000}, {2080000, 128}, {160, 128}, {160, 2080000}, {2080000, 128}, {160, 128}, false, 0, 0, 20, 10000900009000090000UL,
    "20 160 128 2080000 2080000 32 32 2080000 32 32 256 8 8 1 1 0 0 0 0 81920 3072 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  {
    "MatMulV3_910D1_asw_load_balance_m", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1188, 64}, {64, 2524}, {1188, 2524}, {1188, 64}, {64, 2524}, {1188, 2524}, false, 0, 0, 32, 10000900009000090000UL,
    "32 1188 2524 64 64 256 256 64 256 256 32 8 8 1 1 0 0 0 0 163840 102400 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 5 1 240 0 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  },
  // {
  //   "MatMulV3_910D1_asw_load_balance_n", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
  //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {263, 6499}, {2602, 263}, {6499, 2602}, {263, 6499}, {2602, 263}, {6499, 2602}, false, 0, 0, 32, 10000900009000090003UL,
  //   "32 6499 2602 263 263 256 256 263 256 256 64 8 8 1 1 0 0 0 0 155648 98604 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 4 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  // },
  // {
  //   "MatMulV3_910D1_al1_full_load_06", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
  //     "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {7632, 1}, {7632, 10512}, {1, 10512}, {7632, 1}, {7632, 10512}, {1, 10512}, false, 0, 0, 32, 10000900009000090001UL,
  //   "32 1 10512 7632 7632 256 256 7632 256 256 32 8 8 1 1 0 0 0 0 509952 21504 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  // },
  // {
  //   "MatMulV3_910D1_al1_full_load_06", "MatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
  //     "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {0, 1}, {0, 10512}, {1, 10512}, {0, 1}, {0, 10512}, {1, 10512}, false, 0, 0, 32, 0UL,
  //   "32 1 10512 7632 7632 256 256 7632 256 256 32 8 8 1 1 0 0 0 0 509952 21504 0 1 1 1 1 4 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT
  // },
};
INSTANTIATE_TEST_CASE_P(MatMulV3Ascend910B, MatMulV3TilingRuntime, testing::ValuesIn(ascend910B_cases_params));
INSTANTIATE_TEST_CASE_P(MatMulV3Ascend310P, MatMulV3TilingRuntime, testing::ValuesIn(ascend310P_cases_params));
INSTANTIATE_TEST_CASE_P(MatMulV3Ascend910D, MatMulV3TilingRuntime, testing::ValuesIn(ascend910D_cases_params));

static void ThreadFunc(const TilingTestParam *params, size_t testcase_num, size_t thread_idx, size_t thread_num) {
  for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
    TestOneParamCase(params[idx]);
  }
}

static void TestMultiThread(const TilingTestParam *params, size_t testcase_num, size_t thread_num) {
  std::thread threads[thread_num];
  for (size_t idx = 0; idx < thread_num; ++idx) {
    threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
  }

  for (size_t idx = 0; idx < thread_num; ++idx) {
    threads[idx].join();
  }
}

TEST_F(MatMulV3TilingRuntime, ascend910B_thread_cases) {
  TestMultiThread(ascend910B_cases_params, sizeof(ascend910B_cases_params) / sizeof(TilingTestParam), 3);
}
TEST_F(MatMulV3TilingRuntime, ascend910D_thread_cases) {
  TestMultiThread(ascend910D_cases_params, sizeof(ascend910D_cases_params) / sizeof(TilingTestParam), 3);
}
TEST_F(MatMulV3TilingRuntime, ascend310P_thread_cases) {
  TestMultiThread(ascend310P_cases_params, sizeof(ascend310P_cases_params) / sizeof(TilingTestParam), 3);
}

class MatMulV3BiasTilingRuntime : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MatMulV3BiasTilingRuntime SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MatMulV3BiasTilingRuntime TearDown" << std::endl;
  }
};

TEST_F(MatMulV3BiasTilingRuntime, bias_cases) {
  gert::StorageShape x1_shape = {{640, 127}, {640, 127}};
  gert::StorageShape x2_shape = {{640, 320}, {640, 320}};
  gert::StorageShape bias_shape = {{320,}, {320,}};
  std::vector<gert::StorageShape> output_shapes(1, {{127, 320}, {127, 320}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_bias_test";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  auto golden_tiling_data = GenGoldenTilingData(
    "24 127 320 640 640 128 128 640 128 128 128 4 8 1 1 1 0 0 0 122976 9216 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 3 0 0 1 0 1 1 0 0 0 0 0 0 16 128 16 320 ",
    case_name, tiling_key);
  cout<<"===== "<<tiling_key<<" === "<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, 0);
  ASSERT_EQ(block_dim, 24);
  ASSERT_EQ(tiling_data_result,golden_tiling_data);
}

TEST_F(MatMulV3BiasTilingRuntime, 910d_bias_cases) {
  gert::StorageShape x1_shape = {{1024, 8192}, {1024, 8192}};
  gert::StorageShape x2_shape = {{1024, 8192}, {1024, 8192}};
  gert::StorageShape bias_shape = {{1024,}, {1024,}};
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 1024}, {1024, 1024}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;
  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_910D1_bias";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  auto golden_tiling_data = GenGoldenTilingData(
      "32 1024 1024 8192 256 256 128 256 256 64 4096 1 1 1 1 0 0 16843264 ",
      case_name, tiling_key);
  cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
  ASSERT_EQ(tiling_key, 10001900009000290002UL);
  ASSERT_EQ(block_dim, 32);
  ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_F(MatMulV3BiasTilingRuntime, big_axis_cases) {
  gert::StorageShape x1_shape = {{2, INT32_MAX + 1L}, {2, INT32_MAX + 1L}};
  gert::StorageShape x2_shape = {{INT32_MAX + 1L, 2}, {INT32_MAX + 1L, 2}};
  gert::StorageShape bias_shape = {{320,}, {320,}};
  std::vector<gert::StorageShape> output_shapes(1, {{2, 2}, {2, 2}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(MatMulV3BiasTilingRuntime, abnormal_bias_dim_value_case) {
  gert::StorageShape x1_shape = {{127, 640}, {127, 640}};
  gert::StorageShape x2_shape = {{640, 320}, {640, 320}};
  gert::StorageShape bias_shape = {{321}, {321}};
  std::vector<gert::StorageShape> output_shapes(1, {{127, 320}, {127, 320}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(MatMulV3BiasTilingRuntime, abnormal_wrong_k_case) {
  gert::StorageShape x1_shape = {{640, 127}, {640, 127}};
  gert::StorageShape x2_shape = {{640, 320}, {640, 320}};
  gert::StorageShape bias_shape = {{320}, {320}};
  std::vector<gert::StorageShape> output_shapes(1, {{127, 320}, {127, 320}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_fp32) {
  gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_fp32";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_fp32:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_indivisibleM) {
  gert::StorageShape x1_shape = {{62081, 1536}, {62081, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{62081, 384}, {62081, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_indivisibleM";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_indivisibleM:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_indivisibleN) {
  gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
  gert::StorageShape x2_shape = {{1536, 62081}, {1536, 62081}};
  std::vector<gert::StorageShape> output_shapes(1, {{384, 62081}, {384, 62081}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_indivisibleN";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_indivisibleN:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddK) {
  gert::StorageShape x1_shape = {{62080, 768}, {62080, 768}};
  gert::StorageShape x2_shape = {{768, 384}, {768, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_oddK";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_oddK:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddM) {
  gert::StorageShape x1_shape = {{512, 1536}, {512, 1536}};
  gert::StorageShape x2_shape = {{1536, 65536}, {1536, 65536}};
  std::vector<gert::StorageShape> output_shapes(1, {{512, 65536}, {512, 65536}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_oddM";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_oddM:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddN) {
  gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
  gert::StorageShape x2_shape = {{1536, 256}, {1536, 256}};
  std::vector<gert::StorageShape> output_shapes(1, {{62080, 256}, {62080, 256}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_oddN";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_oddN:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_invalidRatio) {
  gert::StorageShape x1_shape = {{2560, 1536}, {2560, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{2560, 384}, {2560, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_invalidRatio";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_invalidRatio:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_bigMN) {
  gert::StorageShape x1_shape = {{98304, 1536}, {98304, 1536}};
  gert::StorageShape x2_shape = {{1536, 768}, {1536, 768}};
  std::vector<gert::StorageShape> output_shapes(1, {{98304, 768}, {98304, 768}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_bigMN";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_bigMN: "<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_NE(tiling_key, 65568);
  ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_targetShape) {
  gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_targetShape";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_targetShape:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_randomShape1) {
  gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
  gert::StorageShape x2_shape = {{1536, 49152}, {1536, 49152}};
  std::vector<gert::StorageShape> output_shapes(1, {{384, 49152}, {384, 49152}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_randomShape1";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_randomShape1:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_randomShape2) {
  gert::StorageShape x1_shape = {{50176, 1536}, {50176, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{50176, 384}, {50176, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_splitK_nkm_randomShape2";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_nkm_randomShape2:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_mkn_bigK) {
  gert::StorageShape x1_shape = {{3840, 30720}, {3840, 30720}};
  gert::StorageShape x2_shape = {{30720, 3840}, {30720, 3840}};
  std::vector<gert::StorageShape> output_shapes(1, {{3840, 3840}, {3840, 3840}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_shift_mkn_bigK";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_shift_mkn_bigK:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65632);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_mkn_bigN) {
  gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
  gert::StorageShape x2_shape = {{1536, 50816}, {1536, 50816}};
  std::vector<gert::StorageShape> output_shapes(1, {{384, 50816}, {384, 50816}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_shift_mkn_bigN";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_shift_mkn_bigN:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65632);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_nkm_bigM) {
  gert::StorageShape x1_shape = {{49152, 1536}, {49152, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{49152, 384}, {49152, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_shift_nkm_bigM";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_shift_nkm_bigM:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_nkm_targetShape) {
  gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
  gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
  std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
  optiling::MatmulV3CompileInfo compile_info;
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
  auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                    .SetOpType("MatMulV3")
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
  string case_name = "MatMulV3_shift_nkm_targetShape";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  cout<<"===== splitK_shift_nkm_targetShape:"<<tiling_key<<" === \n"<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, 65616);
}
} // namespace