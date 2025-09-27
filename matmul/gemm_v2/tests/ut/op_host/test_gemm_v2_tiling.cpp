/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ge;

namespace {
bool IsDisplayTilingdata(const string& case_name, size_t index)
{
  if (index <= 18 || (index >= 22 && index <= 27) || (index >= 30 && index <= 32) || index >= 48) {
        return true;
    }

    return false;
}

static string TilingData2Str(const gert::TilingData* tiling_data, const string& case_name)
{
    if (tiling_data == nullptr) {
        return "";
    }
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        if (IsDisplayTilingdata(case_name, i / sizeof(int32_t))) {
            result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
            result += " ";
        }
    }
    return result;
}

static string GenGoldenTilingData(const string& tiling_data, const string& case_name)
{
    istringstream iss(tiling_data);
    vector<string> data_list;
    string tmp;
    while (iss >> tmp) {
        data_list.push_back(tmp);
    }
    string golden_tiling_data;
    for (size_t i = 0; i < data_list.size(); i++) {
        if (IsDisplayTilingdata(case_name, i)) {
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

  bool private_attr;
  int32_t input_size;
  int32_t hidden_size;

  // output
  uint32_t block_dim;
  uint64_t tiling_key;
  string tiling_data;

  std::initializer_list<int64_t> perm_x1;
  std::initializer_list<int64_t> perm_x2;
  std::initializer_list<int64_t> perm_y;

  ge::DataType input_dtype = DT_FLOAT16;
  ge::DataType y_dtype = DT_FLOAT16;
};

class GemmV2TilingRuntime : public testing::TestWithParam<TilingTestParam> {
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

TEST_P(GemmV2TilingRuntime, general_cases) {
  TilingTestParam param = GetParam();
  gert::StorageShape x1_shape = {param.x1_shape, param.x1_shape};
  gert::StorageShape x2_shape = {param.x2_shape, param.x2_shape};
  std::vector<gert::StorageShape> output_shapes(1, {param.y_shape, param.y_shape});
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
  GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
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
  uint64_t tiling_key = tiling_context->GetTilingKey();
  uint32_t block_dim = tiling_context->GetBlockDim();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), param.case_name);
  auto golden_tiling_data = GenGoldenTilingData(param.tiling_data, param.case_name);
  cout<<"===== "<<tiling_key<<" === "<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, param.tiling_key);
  ASSERT_EQ(block_dim, param.block_dim);
  ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

static TilingTestParam general_cases_params[] = {
  {
    "MatMulV3_basic_test01", "GemmV2", R"({"_pattern": "GemmV2", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {249, 512}, {512, 311}, {249, 311}, false, 0, 0, 24, 0,
    "24 249 311 512 512 32 128 512 32 128 64 64 16 1 1 0 0 0 0 278528 16384 0 1 1 1 1 32 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 8 3 0 0 0 0 0 1 0 0 0 0 0 0 0 0 16 320 "
  },
};

INSTANTIATE_TEST_CASE_P(GemmV2, GemmV2TilingRuntime, testing::ValuesIn(general_cases_params));
} // namespace