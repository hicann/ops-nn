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

#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ge;
using namespace gert;

namespace {
bool IsDisplayTilingdata(const string& case_name, size_t index, uint64_t tilingKey)
{
    // 0-18 22-27 30-32 48之后 表示bmm实际用到的tilingdata
    if (index < 18 || (index >= 22 && index <= 27) || (index >= 30 && index <= 32) || index >= 48) {
        return true;
    }
    // 基础API校验全部的tilingdata
    stringstream ss;
    ss << hex << uppercase << tilingKey;
    string tilingKeyToVerified = ss.str();
    if (!tilingKeyToVerified.empty() && tilingKeyToVerified.back() == '1') {
        return true;
    }
    return false;
}

static string TilingData2Str(const gert::TilingData* tiling_data, const string& case_name, uint64_t tilingKey)
{
    if (tiling_data == nullptr) {
        return "";
    }
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        if (IsDisplayTilingdata(case_name, i / sizeof(int32_t), tilingKey)) {
            result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
            result += " ";
        }
    }
    return result;
}

static string GenGoldenTilingData(const string& tiling_data, const string& case_name, uint64_t tilingKey)
{
    istringstream iss(tiling_data);
    vector<string> data_list;
    string tmp;
    while (iss >> tmp) {
        data_list.push_back(tmp);
    }
    string golden_tiling_data;
    for (size_t i = 0; i < data_list.size(); i++) {
        if (IsDisplayTilingdata(case_name, i, tilingKey)) {
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

    ge::DataType input_dtype = DT_FLOAT;
    ge::DataType y_dtype = DT_FLOAT;
};

class BatchMatMulV3TilingRuntime : public testing::TestWithParam<TilingTestParam> {
  virtual void SetUp() {
  }

protected:
  static void SetUpTestCase() {
    std::cout << "BatchMatMulV3TilingRuntime SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchMatMulV3TilingRuntime TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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


static TilingTestParam ascend910B_cases_params[] = {
    {"BatchMatMulV3_basic_test01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {2, 240, 288}, {2, 288, 96}, {2, 240, 96}, false, 0, 0, 24, 65536,
    "24 240 96 288 288 240 96 288 240 128 32 8 16 1 1 0 0 0 0 163840 2048 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 240 2 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_02", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {10, 10, 32, 32}, {10, 10, 32, 32}, {10, 10, 32, 32}, false, 0, 0, 24, 65552,
    "24 32 32 32 32 32 32 32 32 32 32 64 64 1 1 0 0 0 0 8192 4096 0 1 1 1 1 32 32 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 1 1 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 100 100 100 1 1 1 1 1 1 10 10 10 10 10 10 4 0 32 100 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_03", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":20},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {57, 3, 1434, 380}, {57, 3, 813, 1434}, {57, 3, 380, 813}, false, 0, 0, 20, 65536,
    "20 380 813 1434 1434 256 128 1434 256 128 32 8 16 1 1 0 0 0 0 325632 73728 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 7 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 20 171 171 171 1 1 1 1 1 1 57 57 57 3 3 3 0 0 380 171 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_03", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {15, 993, 9472}, {15, 993, 224}, {15, 9472, 224}, false, 0, 0, 24, 65536,
    "24 9472 224 993 993 512 64 993 512 64 16 8 64 1 1 0 0 0 0 131072 16384 0 1 1 1 1 4 32 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 15 1 19 4 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 15 15 15 1 1 1 1 1 1 1 1 1 15 15 15 0 0 9472 15 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_04", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1024, 13, 12}, {12,12}, {1024, 13, 12}, false, 0, 0, 24, 65536,
    "24 13312 12 12 12 512 12 12 512 16 16 8 256 1 1 0 0 0 0 66560 32768 0 1 1 1 1 4 128 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 104 1 0 0 0 0 1 0 0 0 0 0 0 0 278 16 0 0 24 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 13 1 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_07", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1500, 21, 16}, {1500, 21, 4}, {1500, 16, 4}, false, 0, 0, 24, 4112,
    "24 16 4 21 21 16 4 21 16 16 32 128 128 1 1 0 0 0 0 3072 1024 0 1 1 1 1 64 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 62 0 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 62 0 16 1500 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_08", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1500, 21, 124}, {1500, 21, 124}, {1500, 124, 124}, false, 0, 0, 24, 4112,
    "24 124 124 21 21 124 124 21 128 128 32 16 16 1 1 0 0 0 0 6144 3072 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 14 0 1 1 1 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 14 0 124 1500 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_test_09", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, false, {1, 67664, 224}, {1, 400, 67664}, {1, 224, 400}, false, 0, 0, 24, 0,
    "24 224 400 67664 67664 32 16 67664 32 16 32 64 128 1 1 0 0 0 0 180224 28672 0 1 1 1 1 32 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 3 7 9 0 0 1 1 0 1 0 0 0 0 0 0 0 0 17 1032 24 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 224 1 0 0 "
  },
  {
    "BatchMatMulV3_AL1FullLoad_boundary_test_fp32", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {256, 256}, {48, 256, 256}, {48, 256, 256}, false, 0, 0, 24, 65536,
    "24 256 256 256 256 256 128 256 256 128 32 4 12 1 1 0 0 0 0 278528 16384 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 8 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 48 48 1 1 1 1 1 1 1 1 1 1 48 48 0 0 256 48 0 0 "
  },
  {
    "BatchMatMulV3_AL1FullLoad_boundary_test_fp16", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {256, 512}, {48, 512, 256}, {48, 256, 256}, false, 0, 0, 24, 65536,
    "24 256 256 512 512 256 128 512 256 128 64 4 12 1 1 0 0 0 0 278528 16384 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 8 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 48 48 1 1 1 1 1 1 1 1 1 1 48 48 0 0 256 48 0 0 ",
    DT_FLOAT16, DT_FLOAT16
  },
  {
    "BatchMatMulV3_AL1FullLoad_general_test_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {50, 26}, {1500, 50, 124}, {1500, 26, 124}, false, 0, 0, 24, 65792,
    "24 26 124 50 50 26 124 50 32 128 64 1 8 1 1 0 0 0 0 35840 16384 0 1 1 1 1 1 4 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 1500 1500 1 1 1 1 1 1 1 1 1 1 1500 1500 0 0 26 1500 0 0 "
  },
  {
    "BatchMatMulV3_multi_batch_unaligned_test_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {49638, 1166, 2}, {49638, 2, 1}, {49638, 1166, 1}, false, 0, 0, 24, 16,
    "24 1166 1 2 2 1166 1 2 400 16 16 8 256 1 1 0 0 0 0 2560 4096 0 1 1 1 1 4 128 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6 0 1 1 10 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 24 49638 49638 49638 1 1 1 1 1 1 1 1 1 49638 49638 49638 6 0 1166 1346 0 0 "
  },
  {
    "BatchMatMulV3_BL1FullLoad_boundary_test_fp32", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {48, 256, 256}, {256, 256}, {48, 256, 256}, false, 0, 0, 24, 65536,
    "24 256 256 256 256 256 128 256 256 128 32 4 12 1 1 0 0 0 0 114688 12288 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 48 1 48 1 1 1 1 1 1 1 1 1 48 1 48 0 0 256 48 0 0 "
  },
  {
    "BatchMatMulV3_BL1FullLoad_boundary_test_fp16", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {48, 512, 256}, {512, 256}, {48, 256, 256}, false, 0, 0, 24, 65536,
    "24 256 256 512 512 256 128 512 256 128 64 4 12 1 1 0 0 0 0 114688 12288 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 48 1 48 1 1 1 1 1 1 1 1 1 48 1 48 0 0 256 48 0 0 ",
    DT_FLOAT16, DT_FLOAT16
  },
  {
    "BatchMatMulV3_BL1FullLoad_general_test_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, false, {1500, 50, 124}, {50, 26}, {1500, 124, 26}, false, 0, 0, 24, 66048,
    "24 124 26 50 50 124 26 50 128 32 64 8 1 1 1 0 0 0 0 35840 16384 0 1 1 1 1 4 1 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1 1500 1 1 1 1 1 1 1 1 1 1500 1 1500 0 0 124 1500 0 0 "
  },
  {"BatchMatMulV3_MultiBatch_AL1FullLoad_general_test_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {1500, 1, 512}, {1500, 512, 128}, {1500, 1, 128}, false, 0, 0, 24, 65537,
    "24 1 128 512 512 1 128 512 16 128 64 64 8 1 1 0 0 0 0 65536 1024 0 1 1 1 1 32 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 0 0 1 1500 7 1 "
  },
  {"BatchMatMulV3_MultiBatch_AL1FullLoad_general_test_02",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, false, {1500, 1, 128}, {1500, 512, 128}, {1500, 1, 512}, false, 0, 0, 24, 65537,
    "24 1 512 128 128 1 512 128 16 64 128 16 8 1 1 0 0 0 0 24576 2048 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 16 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 0 0 1 1500 31 1 "
  }
};

static TilingTestParam ascend910_95_cases_params[] = {
  {
    "BatchMatMulV3_910D1_test_iterbatch_1", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {10, 10, 320, 32}, {10, 10, 32, 32}, {10, 10, 320, 32}, false, 0, 0, 32, 1UL,
    "32 320 32 32 256 32 64 256 256 32 32 1 1 1 1 0 0 16843776 0 256 1 100 0 "
  },
  // singleCoreM 256->16
  {
    "BatchMatMulV3_910D1_test_asw_1", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {1, 1}, {1, 1}, {1, 1}, false, 0, 0, 32, 1UL,
    "32 1 1 1 16 16 128 256 256 32 1 1 1 1 1 0 0 16843265 0 256 1 1 0"
  },
  // singeCoreK / baseK < 8  -> 之前stepK= 1 全载场景去掉上述判断代码以后, stepK = 2
  // bmm aFullLoad basic
  {
    "BatchMatMulV3_910D1_test_al1fullload_2", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {1, 29, 11}, {230, 29, 2687}, {230, 11, 2687}, false, 0, 0, 32, 65553UL,
    "32 11 2687 29 16 256 32 16 256 32 29 1 1 1 1 0 0 16909313 0 16 1 230 0 "
  },
  // singleCoreM 256->64
  // 拆分tiling后修复bmm b全载tilingKey和adjustTiling不匹配问题, apiLevel_未赋值导致ubDb未正确计算
  {
    "BatchMatMulV3_910D1_test_bl1fullload_1", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {47, 3680, 64}, {1, 3680, 16}, {47, 64, 16}, false, 0, 0, 32, 131089UL,
    "32 64 16 3680 64 16 128 256 16 32 3680 1 1 1 1 0 0 33686017 0 256 1 47 0 "
  },
  {
    "BatchMatMulV3_910D1_test_iterbatch_basicapi_fp32_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":false,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {1200, 32, 32}, {1200, 32, 32}, {1200, 32, 32}, false, 0, 0, 32, 257UL,
    "32 32 32 1200 32 8 1 32 32 32 0 0 "
  },
  {
    "BatchMatMulV3_910D1_test_iterbatch_basicapi_fixpip_opt_fp32_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":false,"adj_x2":true, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, true, 0, true, {10, 9, 47, 8}, {10, 9, 77, 8}, {10, 9, 47, 77}, false, 0, 0, 32, 2097729UL,
    "47 77 8 90 3 3 1 48 80 16 0 0 "
  },
  {
    "BatchMatMulV3_910D1_test_batchmatmultomul_fp32_01", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {2400, 1, 4}, {2400, 1, 976}, {2400, 4, 976}, false, 0, 0, 32, 513UL,
    "4 976 2400 32 75 12 3 3 31 8 "
  },
  {
    "BatchMatMulV3_910D1_test_batchmatmultomul_fp32_02", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {2400, 1, 976}, {2400, 1, 8}, {2400, 976, 8}, false, 0, 0, 32, 513UL,
    "976 8 2400 32 75 7 5 5 31 8 "
  },
  {
    "BatchMatMulV3_910D1_test_batchmatmultomul_N1", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {2400, 1, 20}, {2400, 1, 1}, {2400, 20, 1}, false, 0, 0, 32, 273UL,
    "20 1 1 2400 75 16 1 32 16 16 0 0 "
  },
  // {
  //   "BatchMatMulV3_910D1_test_basiciterbatch_02", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
  //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true, {88, 40, 151}, {88, 40, 1917}, {88, 151, 1917}, false, 0, 0, 32, 529UL,
  //   "151 1917 40 88 1 1 1 160 192 48 ", DT_FLOAT16, DT_FLOAT16
  // },
  // {
  //   "BatchMatMulV3_910D1_test_basiciterbatch_03", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
  //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, true, {5653, 255, 64}, {5653, 128, 255}, {5653, 64, 128}, false, 0, 0, 32, 593UL,
  //   "64 128 255 5653 2 1 1 64 64 256 ", DT_FLOAT16, DT_FLOAT16
  // },
  // {
  //   "BatchMatMulV3_910D1_test_basiciterbatch_05", "BatchMatMulV3", R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
  //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
  //     "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
  //     "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
  //     "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
  //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, true, {512, 150, 150}, {512, 150, 37}, {512, 150, 37}, false, 0, 0, 32, 513UL,
  //   "150 37 150 512 1 1 1 48 48 160 "
  // },
};


TEST_F(BatchMatMulV3TilingRuntime, fail_case) {
  gert::StorageShape x1_shape = {{2, 15, 16}, {2, 15, 16}};
  gert::StorageShape x2_shape = {{2, 16, 16}, {2, 1, 1, 16, 16}};
  gert::StorageShape bias_shape = {{16,}, {16,}};
  std::vector<gert::StorageShape> output_shapes(1, {{2, 15, 16}, {2, 15, 16}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":false, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":8},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8, "socVersion": "Ascend310P" },
      "format_a":"FORMAT_ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})";
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

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();

  auto tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);

  x1_shape = {{2, 16, 16}, {2, 16, 16}};
  x2_shape = {{2, 16, 15}, {2, 1, 1, 16, 16}};
  std::vector<gert::StorageShape> output_shapes_two(1, {{2, 16, 15}, {2, 16, 15}});
  std::vector<void *> output_shapes_ref_two(1);
  for (size_t i = 0; i < output_shapes_two.size(); ++i) {
    output_shapes_ref_two[i] = &output_shapes_two[i];
  }
  holder = gert::TilingContextFaker()
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref_two)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND)
                    .NodeInputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .TilingData(tiling_data.get())
                    .Workspace(ws_size)
                    .Build();
  tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases) {
  gert::StorageShape x1_shape = { {10, 10, 32, 32},  {10, 10, 32, 32}};
  gert::StorageShape x2_shape = { {10, 10, 32, 32},  {10, 10, 32, 32}};
  gert::StorageShape bias_shape = { {32,}, {32,}};
  std::vector<gert::StorageShape> output_shapes(1, { {10, 10, 32, 32},  {10, 10, 32, 32}});
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
  map<string, string> soc_version;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
  aicore_spec["cube_freq"] = "1800";

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
  uint64_t tiling_key = tiling_context->GetTilingKey();;
  uint32_t block_dim = tiling_context->GetBlockDim();
  string case_name = "BatchMatMulV3TilingRuntime_bias_cases";
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
  auto golden_tiling_data = GenGoldenTilingData(
    "24 32 32 32 32 32 32 32 32 32 32 124 124 1 1 1 0 0 0 4160 4096 0 1 1 1 1 62 62 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 100 100 100 1 1 1 1 1 1 10 10 10 10 10 10 0 0 32 100 0 0 ",
    case_name, tiling_key);
  cout<<"===== "<<tiling_key<<" === "<<tiling_data_result<<std::endl;
  ASSERT_EQ(tiling_key, 65536);
  ASSERT_EQ(block_dim, 24);
  ASSERT_EQ(tiling_data_result,golden_tiling_data);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_batchbias_failed_1) {
  gert::StorageShape x1_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape x2_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape bias_shape = { {2048, 1, 16}, {2048, 1, 16}};
  std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16},  {2048, 16, 16}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
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

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_batchbias_failed) {
  gert::StorageShape x1_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape x2_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape bias_shape = { {2047, 1, 16}, {2047, 1, 16}};
  std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16},  {2048, 16, 16}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
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

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_two_dim_bias_failed) {
  gert::StorageShape x1_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape x2_shape = { {2048, 16, 16},  {2048, 16, 16}};
  gert::StorageShape bias_shape = { {2, 16}, {2, 16}};
  std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16},  {2048, 16, 16}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
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

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}


TEST_F(BatchMatMulV3TilingRuntime, bias_cases_l1_iterbatchbias_success) {
  gert::StorageShape x1_shape = { {992, 28, 2},  {992, 2, 1851}};
  gert::StorageShape x2_shape = { {992, 28, 2},  {992, 2, 1851}};
  gert::StorageShape bias_shape = { {1851,}, {1851,}};
  std::vector<gert::StorageShape> output_shapes(1, {{992, 28, 1851},  {992, 28, 1851}});
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  fe::PlatFormInfos platform_info;

  platform_info.Init();
  string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
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

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
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
                    .SetOpType("BatchMatMulV3")
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
  ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

} // namespace