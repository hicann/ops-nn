/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ge;

namespace {

static string TilingData2Str(const gert::TilingData* tiling_data) {
    if (tiling_data == nullptr) {
        return "";
    }
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
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
    int32_t batch_split_factor;

    ge::DataType input_dtype = DT_FLOAT16;
    ge::DataType y_dtype = DT_FLOAT16;
    std::initializer_list<int64_t> bias_shape;
    ge::Format bias_format;
    ge::Format bias_ori_format;
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

class TransposeBatchMatMulTilingRuntime : public testing::TestWithParam<TilingTestParam> {
  virtual void SetUp() {
  }

protected:
  static void SetUpTestCase() {
    std::cout << "TransposeBatchMatMulTilingRuntime SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TransposeBatchMatMulTilingRuntime TearDown" << std::endl;
  }
};

TEST_P(TransposeBatchMatMulTilingRuntime, general_cases) {
    TilingTestParam param = GetParam();
    gert::StorageShape x1_shape = {param.x1_shape, param.x1_shape};
    gert::StorageShape x2_shape = {param.x2_shape, param.x2_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_shape, param.y_shape});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();

    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(param.compile_info.c_str()), reinterpret_cast<void*>(&platform_info)})
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
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::KernelRunContextHolder holder;
    holder = gert::TilingContextFaker()
                 .SetOpType(param.op_type.c_str())
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(param.perm_x1)},
                             {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(param.perm_x2)},
                             {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(param.perm_y)},
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
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, param.tiling_key);
    ASSERT_EQ(block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

static TilingTestParam ascend910B_cases_params[] = {
    {"TransposeBatchMatMul_basic_test01",
     "TransposeBatchMatMul",
     R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {2, 64, 128}, {2, 128, 256}, {2, 64, 256}, false, 0, 0, 24, 10000000000000000000UL,
    "24 64 256 128 128 64 256 128 64 256 64 32 8 1 1 0 0 0 0 69632 16384 0 1 1 1 1 16 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 4 2 0 0 123 123 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 64 2 0 0 1 0 ",
    {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, 1
    },
    {"TransposeBatchMatMul_basic_test02",
      "TransposeBatchMatMul",
      R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
       "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
       "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
       "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {64, 2, 128}, {2, 128, 256}, {2, 64, 256}, false, 0, 0, 24, 10000000000000000001UL,
     "24 64 256 128 128 64 256 128 64 256 64 32 8 1 1 0 0 0 0 69632 16384 0 1 1 1 1 16 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 4 2 0 0 213 123 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 64 2 0 0 1 0 ",
     {1, 0, 2}, {0, 1, 2}, {0, 1, 2}, 1
    },
    {"TransposeBatchMatMul_basic_test03",
      "TransposeBatchMatMul",
      R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0, "batch_split_factor":2},
       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
       "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
       "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
       "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {48, 128, 128}, {128, 128, 512}, {48, 128, 512}, false, 0, 0, 24, 10000000000000000001UL,
     "24 48 512 128 128 48 512 128 48 512 32 80 8 1 1 0 0 0 0 69632 16384 0 1 1 1 1 40 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 3 4 0 0 213 123 0 0 0 0 0 0 0 0 0 0 0 0 24 128 128 128 1 1 1 1 1 1 1 1 1 128 128 128 0 0 48 128 0 0 1 0 ",
     {1, 0, 2}, {0, 1, 2}, {1, 0, 2}, 1
    }
};

static TilingTestParam ascend910_95_cases_params[] = {
  {
    "TransposeBatchMatMul_910D1_test_iterbatch_1", "TransposeBatchMatMul", R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "socVersion": "Ascend910_95" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0, false, {32, 32, 128}, {32, 128, 512}, {32, 32, 512}, false, 0, 0, 32, 10000900009000090000UL,
    "32 32 512 128 128 256 256 128 256 256 64 4 4 1 1 0 0 0 0 69632 16384 0 1 1 1 1 2 2 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 4 32 32 32 1 1 1 1 1 1 1 1 1 1 32 32 32 0 1 ",
    {1, 0, 2}, {0, 1, 2}, {1, 0, 2}
  }
};

INSTANTIATE_TEST_CASE_P(TransposeBatchMatMul910B, TransposeBatchMatMulTilingRuntime, testing::ValuesIn(ascend910B_cases_params));
INSTANTIATE_TEST_CASE_P(TransposeBatchMatMul910_95, TransposeBatchMatMulTilingRuntime, testing::ValuesIn(ascend910_95_cases_params));

} // namespace