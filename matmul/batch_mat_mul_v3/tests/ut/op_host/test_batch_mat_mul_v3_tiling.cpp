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
#include <memory>
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "tiling_context_faker.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "../../../../common/op_host/op_tiling/debug_tiling.h"
#include "../../../op_host/op_tiling/arch35/batch_matmul_v3_tiling_key.h"
#include "../../../op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_tiling.h"
#include "../../../op_host/op_tiling/arch35/batch_matmul_v3_k_equal_zero_tiling.h"

using namespace std;
using namespace ge;
using namespace gert;

namespace {
string get_map_string(const std::map<string, string>& map, const string& key)
{
    auto it = map.find(key);
    if (it != map.end()) {
        return it->second;
    } else {
        return "0";
    }
}
bool IsDisplayTilingdata(const string& case_name, size_t index, uint64_t tilingKey)
{
    // 0-18 25-27 30-32 48-91 表示bmm实际用到的tilingdata(不含VectorTilingInfo)
    // 新增rowStride后原字段innerBatch的index=22， x3Batch变为index=24 遵循原先只验证到uint8_t ubDB
    // 字段，修改为index>=25 VectorTilingInfo从index
    // 92开始(sizeof(MatmulTilingData)+sizeof(MultiBatchInfo)=368字节=92个int32)
    if (index < 18 || (index >= 25 && index <= 27) || (index >= 30 && index <= 32) || (index >= 48 && index < 92)) {
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
    virtual void SetUp() {}

protected:
    static void SetUpTestCase() { std::cout << "BatchMatMulV3TilingRuntime SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchMatMulV3TilingRuntime TearDown" << std::endl; }
};

static string to_string(const std::stringstream& tiling_data)
{
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

TEST_P(BatchMatMulV3TilingRuntime, general_cases)
{
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
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(param.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> aicore_memory_rates;
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1800";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        aicore_spec["cube_freq"] = "1650";
        aicore_memory_rates["ddr_rate"] = "31";
        aicore_memory_rates["l2_rate"] = "100";
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreMemoryRates",
                                                                                                aicore_memory_rates);
    }
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

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
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), param.case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData(param.tiling_data, param.case_name, param.tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, param.tiling_key);
    ASSERT_EQ(block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

static TilingTestParam ascend910B_cases_params[] = {
    {"BatchMatMulV3_basic_test01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {2, 240, 288},
     {2, 288, 96},
     {2, 240, 96},
     false,
     0,
     0,
     24,
     65536,
     "24 240 96 288 288 240 96 288 240 128 32 8 16 1 1 0 0 0 0 163840 2048 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 1 1 2 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 240 2 0 0 "},
    {"BatchMatMulV3_multi_batch_test_02",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {10, 10, 32, 32},
     {10, 10, 32, 32},
     {10, 10, 32, 32},
     false,
     0,
     0,
     24,
     65552,
     "24 32 32 32 32 32 32 32 32 32 32 64 64 1 1 0 0 0 0 8192 4096 0 1 1 1 1 32 32 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 4 0 1 1 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 100 100 100 1 1 1 1 1 1 10 10 10 10 10 10 4 0 32 100 0 0 "},
    {"BatchMatMulV3_multi_batch_test_03",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":20, "vector_core_cnt": 40},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20, "vector_core_cnt": 40, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     true,
     0,
     false,
     {57, 3, 1434, 380},
     {57, 3, 813, 1434},
     {57, 3, 380, 813},
     false,
     0,
     0,
     20,
     65536,
     "20 380 813 1434 1434 256 128 1434 256 128 32 8 16 1 1 0 0 0 0 325632 73728 0 1 1 1 1 4 8 0 0 2 2 1 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 0 1 1 2 7 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 20 171 171 171 1 1 1 1 1 1 57 57 57 3 3 3 0 0 380 "
     "171 0 0 "},
    {"BatchMatMulV3_multi_batch_test_03",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {15, 993, 9472},
     {15, 993, 224},
     {15, 9472, 224},
     false,
     0,
     0,
     24,
     65536,
     "24 9472 224 993 993 512 64 993 512 64 16 8 64 1 1 0 0 0 0 131072 16384 0 1 1 1 1 4 32 0 0 2 2 1 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 15 1 19 4 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 15 15 15 1 1 1 1 1 1 1 1 1 15 15 15 0 0 9472 15 0 "
     "0 "},
    {"BatchMatMulV3_multi_batch_test_04",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {1024, 13, 12},
     {12, 12},
     {1024, 13, 12},
     false,
     0,
     0,
     24,
     65536,
     "24 13312 12 12 12 512 12 12 512 16 16 8 256 1 1 0 0 0 0 66560 32768 0 1 1 1 1 4 128 0 0 2 2 1 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 1 1 104 1 0 0 0 0 1 0 0 0 0 0 0 0 278 16 0 0 24 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 13 1 0 0 "},
    {"BatchMatMulV3_multi_batch_test_07",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {1500, 21, 16},
     {1500, 21, 4},
     {1500, 16, 4},
     false,
     0,
     0,
     24,
     4112,
     "24 16 4 21 21 16 4 21 16 16 32 128 128 1 1 0 0 0 0 3072 1024 0 1 1 1 1 64 64 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 62 0 1 1 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 62 0 16 1500 "
     "0 0 "},
    {"BatchMatMulV3_multi_batch_test_08",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {1500, 21, 124},
     {1500, 21, 124},
     {1500, 124, 124},
     false,
     0,
     0,
     24,
     4112,
     "24 124 124 21 21 124 124 21 128 128 32 16 16 1 1 0 0 0 0 6144 3072 0 1 1 1 1 8 8 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 14 0 1 1 1 1 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 14 0 124 "
     "1500 0 0 "},
    {"BatchMatMulV3_multi_batch_test_09",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     true,
     0,
     false,
     {1, 67664, 224},
     {1, 400, 67664},
     {1, 224, 400},
     false,
     0,
     0,
     24,
     0,
     "24 224 400 67664 67664 32 16 67664 32 16 32 64 128 1 1 0 0 0 0 180224 28672 0 1 1 1 1 32 64 0 0 2 2 1 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 0 0 0 1 3 7 9 0 0 1 1 0 1 0 0 0 0 0 0 0 0 17 1032 24 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 224 1 0 "
     "0 "},
    {"BatchMatMulV3_AL1FullLoad_boundary_test_fp32",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {256, 256},
     {48, 256, 256},
     {48, 256, 256},
     false,
     0,
     0,
     24,
     65536,
     "24 256 256 256 256 256 128 256 256 128 32 4 12 1 1 0 0 0 0 278528 16384 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 1 1 8 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 48 48 1 1 1 1 1 1 1 1 1 1 48 48 0 0 256 48 0 0 "},
    {"BatchMatMulV3_AL1FullLoad_boundary_test_fp16",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {256, 512},
     {48, 512, 256},
     {48, 256, 256},
     false,
     0,
     0,
     24,
     65536,
     "24 256 256 512 512 256 128 512 256 128 64 4 12 1 1 0 0 0 0 278528 16384 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 1 1 8 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 48 48 1 1 1 1 1 1 1 1 1 1 48 48 0 0 256 48 0 0 ",
     DT_FLOAT16,
     DT_FLOAT16},
    {"BatchMatMulV3_AL1FullLoad_general_test_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {50, 26},
     {1500, 50, 124},
     {1500, 26, 124},
     false,
     0,
     0,
     24,
     65792,
     "24 26 124 50 50 26 124 50 32 128 64 1 8 1 1 0 0 0 0 35840 16384 0 1 1 1 1 1 4 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1 1500 1500 1 1 1 1 1 1 1 1 1 1 1500 1500 0 0 26 1500 0 0 "},
    {"BatchMatMulV3_multi_batch_unaligned_test_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {49638, 1166, 2},
     {49638, 2, 1},
     {49638, 1166, 1},
     false,
     0,
     0,
     24,
     16,
     "24 1166 1 2 2 1166 1 2 400 16 16 8 256 1 1 0 0 0 0 2560 4096 0 1 1 1 1 4 128 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 6 0 1 1 10 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 24 49638 49638 49638 1 1 1 1 1 1 1 1 1 49638 49638 49638 6 0 "
     "1166 1346 0 0 "},
    {"BatchMatMulV3_BL1FullLoad_boundary_test_fp32",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {48, 256, 256},
     {256, 256},
     {48, 256, 256},
     false,
     0,
     0,
     24,
     65536,
     "24 256 256 256 256 256 128 256 256 128 32 4 12 1 1 0 0 0 0 114688 12288 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 1 1 2 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 48 1 48 1 1 1 1 1 1 1 1 1 48 1 48 0 0 256 48 0 0 "},
    {"BatchMatMulV3_BL1FullLoad_boundary_test_fp16",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524032, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {48, 512, 256},
     {512, 256},
     {48, 256, 256},
     false,
     0,
     0,
     24,
     65536,
     "24 256 256 512 512 256 128 512 256 128 64 4 12 1 1 0 0 0 0 114688 12288 0 1 1 1 1 2 6 0 0 2 2 1 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 1 1 2 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 48 1 48 1 1 1 1 1 1 1 1 1 48 1 48 0 0 256 48 0 0 ",
     DT_FLOAT16,
     DT_FLOAT16},
    {"BatchMatMulV3_BL1FullLoad_general_test_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {1500, 50, 124},
     {50, 26},
     {1500, 124, 26},
     false,
     0,
     0,
     24,
     66048,
     "24 124 26 50 50 124 26 50 128 32 64 8 1 1 1 0 0 0 0 35840 16384 0 1 1 1 1 4 1 0 0 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1 1500 1 1 1 1 1 1 1 1 1 1500 1 1500 0 0 124 1500 0 "
     "0 "},
    {"BatchMatMulV3_MultiBatch_AL1FullLoad_general_test_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {1500, 1, 512},
     {1500, 512, 128},
     {1500, 1, 128},
     false,
     0,
     0,
     24,
     65537,
     "24 1 128 512 512 1 128 512 16 128 64 64 8 1 1 0 0 0 0 65536 1024 0 1 1 1 1 32 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 0 0 1 "
     "1500 7 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"BatchMatMulV3_MultiBatch_AL1FullLoad_general_test_02",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true,"offset_x":0,"enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     false,
     {1500, 1, 128},
     {1500, 512, 128},
     {1500, 1, 512},
     false,
     0,
     0,
     24,
     65537,
     "24 1 512 128 128 1 512 128 16 64 128 16 8 1 1 0 0 0 0 24576 2048 0 1 1 1 1 8 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 1 0 1 1 1 16 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 24 1500 1500 1500 1 1 1 1 1 1 1 1 1 1500 1500 1500 0 0 1 1500 "
     "31 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "}};

static TilingTestParam ascend950_cases_params[] = {
    {"BatchMatMulV3_950_test_iterbatch_1",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     false,
     {10, 10, 320, 32},
     {10, 10, 32, 32},
     {10, 10, 320, 32},
     false,
     0,
     0,
     32,
     2UL,
     "32 320 32 32 160 32 32 160 32 32 32 1 1 1 1 0 0 33686528 0 160 1 0 100 1 "},
    // singleCoreM 256->16
    {"BatchMatMulV3_950_test_asw_1",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     true,
     {1, 1},
     {1, 1},
     {1, 1},
     false,
     0,
     0,
     32,
     2UL,
     "1 1 1 1 16 16 16 16 16 16 1 1 1 1 1 0 0 33686529 0 16 1 0 1 1"},
    // singeCoreK / baseK < 8  -> 之前stepK= 1 全载场景去掉上述判断代码以后, stepK = 2
    // bmm aFullLoad basic
    {"BatchMatMulV3_950_test_al1fullload_2",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     true,
     {1, 29, 11},
     {230, 29, 2687},
     {230, 11, 2687},
     false,
     0,
     0,
     32,
     65554UL,
     "32 11 2687 29 16 128 32 256 128 32 29 1 1 1 1 0 0 16909313 0 256 1 0 230 1 "},
    // singleCoreM 256->64
    // 拆分tiling后修复bmm b全载tilingKey和adjustTiling不匹配问题, apiLevel_未赋值导致ubDb未正确计算
    {"BatchMatMulV3_950_test_bl1fullload_1",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     true,
     {47, 3680, 64},
     {1, 3680, 16},
     {47, 64, 16},
     false,
     0,
     0,
     32,
     131090UL,
     "32 64 16 3680 64 16 96 128 256 32 3680 1 1 1 1 0 0 33686529 0 128 1 0 47 1 "},
    {"BatchMatMulV3_950_test_iterbatch_basicapi_fp32_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":false,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     false,
     0,
     true,
     {1200, 32, 32},
     {1200, 32, 32},
     {1200, 32, 32},
     false,
     0,
     0,
     32,
     257UL,
     "32 32 32 1200 32 8 1 32 32 32 0 1 0 0 "},
    {"BatchMatMulV3_950_test_iterbatch_basicapi_fixpip_opt_fp32_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":false,"adj_x2":true, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     true,
     {10, 9, 47, 8},
     {10, 9, 77, 8},
     {10, 9, 47, 77},
     false,
     0,
     0,
     32,
     2097473UL,
     "47 77 8 90 3 3 1 48 80 16 0 1 0 0 "},
    {"BatchMatMulV3_950_test_batchmatmultomul_fp32_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     true,
     {2400, 1, 4},
     {2400, 1, 976},
     {2400, 4, 976},
     false,
     0,
     0,
     64,
     513UL,
     "4 976 2400 64 38 12 2 2 47 8 "},
    {"BatchMatMulV3_950_test_batchmatmultomul_fp32_02",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     true,
     {2400, 1, 976},
     {2400, 1, 8},
     {2400, 976, 8},
     false,
     0,
     0,
     64,
     513UL,
     "976 8 2400 64 38 7 3 1 53 8 "},
    {"BatchMatMulV3_950_test_batchmatmultomul_N1",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     true,
     {2400, 1, 20},
     {2400, 1, 1},
     {2400, 20, 1},
     false,
     0,
     0,
     32,
     273UL,
     "20 1 1 2400 75 16 1 32 16 16 0 1 0 0 "},
    {"BatchMatMulV3_910D1_test_mergebatch_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,"vector_core_cnt": 64,  "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     true,
     {2400, 1, 976},
     {2400, 4, 976},
     {2400, 1, 4},
     false,
     0,
     0,
     32,
     833UL,
     "1 4 976 2400 8 8 8 256 64 1 0 1 "},
    {"BatchMatMulV3_950_test_streamk_basicapi_01",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"adj_x1":true,"adj_x2":true, "offset_x":0, "enable_hf32":true},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     true,
     0,
     true,
     {2, 217211, 133},
     {2, 204, 217211},
     {2, 133, 204},
     false,
     0,
     0,
     32,
     2101330UL,
     "32 133 204 217211 144 208 128 144 208 32 13576 1 1 1 1 0 0 16843265 0 144 1 0 2 1 "},
    {"BatchMatMulV3_950_test_swat_1",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     false,
     {82, 22, 16559, 21},
     {82, 22, 16559, 9},
     {82, 22, 21, 9},
     false,
     0,
     0,
     32,
     17UL,
     "32 21 9 16559 32 16 512 32 16 256 16559 1 1 1 1 0 0 33686528 0 32 1 1 0 1804 1 "},
    {"BatchMatMulV3_950_test_swat_2",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     false,
     {21, 4, 13851},
     {21, 1, 13851},
     {21, 4, 1},
     false,
     0,
     0,
     32,
     65UL,
     "21 4 1 13851 16 16 1024 16 16 512 13851 1 1 1 1 0 0 33686528 0 16 1 13851 0 21 1 "},
    // {
    //   "BatchMatMulV3_950_test_basiciterbatch_02", "BatchMatMulV3", R"({"_pattern": "MatMul",
    //   "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz":
    //     false, "l2_size":134217728},"binary_mode_flag":true, "block_dim":{"CORE_NUM":32, "vector_core_cnt":
    //     64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0, "hardware_info":
    //     {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false,
    //     "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false,
    //     "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    //     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64,
    //     "socVersion": "Ascend950" }, "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, true,
    //   {88, 40, 151}, {88, 40, 1917}, {88, 151, 1917}, false, 0, 0, 32, 529UL, "151 1917 40 88 1 1 1 160 192 48 ",
    //   DT_FLOAT16, DT_FLOAT16
    // },
    // {
    //   "BatchMatMulV3_950_test_basiciterbatch_03", "BatchMatMulV3", R"({"_pattern": "MatMul",
    //   "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz":
    //     false, "l2_size":134217728},"binary_mode_flag":true, "block_dim":{"CORE_NUM":32, "vector_core_cnt":
    //     64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0, "hardware_info":
    //     {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false,
    //     "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false,
    //     "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    //     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64,
    //     "socVersion": "Ascend950" }, "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, true, 0, true,
    //   {5653, 255, 64}, {5653, 128, 255}, {5653, 64, 128}, false, 0, 0, 32, 593UL, "64 128 255 5653 2 1 1 64 64 256 ",
    //   DT_FLOAT16, DT_FLOAT16
    // },
    // {
    //   "BatchMatMulV3_950_test_basiciterbatch_05", "BatchMatMulV3", R"({"_pattern": "MatMul",
    //   "attrs":{"adj_x1":true,"adj_x2":false, "offset_x":0, "enable_hf32":true},
    //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz":
    //     false, "l2_size":134217728},"binary_mode_flag":true, "block_dim":{"CORE_NUM":32, "vector_core_cnt":
    //     64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0, "hardware_info":
    //     {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": false,
    //     "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": false,
    //     "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    //     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64,
    //     "socVersion": "Ascend950" }, "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, false, false, 0,
    //   true, {512, 150, 150}, {512, 150, 37}, {512, 150, 37}, false, 0, 0, 32, 513UL, "150 37 150 512 1 1 1 48 48 160
    //   "
    // },
    {"BatchMatMulV3_950_test_bmm_iterbatch_broadcast",
     "BatchMatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "enable_hf32":1},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     false,
     {5, 256, 8, 1, 64},
     {5, 1, 8, 200, 64},
     {5, 256, 8, 1, 64},
     false,
     0,
     0,
     32,
     1858UL,
     "32 1 200 64 64 1 1 1 16 208 32 1 1 1 1 0 0 0 0 58176 13312 0 1 1 1 1 1 1 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
     "0 0 1 1 1 1 1 0 0 0 4 0 10240 40 10240 1 1 1 1 5 5 5 256 1 256 8 8 8 0 1 1 16 208 32 16843264 16 1 0 4 1 4 2 "}};

INSTANTIATE_TEST_CASE_P(BatchMatMulV3910B, BatchMatMulV3TilingRuntime, testing::ValuesIn(ascend910B_cases_params));
INSTANTIATE_TEST_CASE_P(BatchMatMulV3950, BatchMatMulV3TilingRuntime, testing::ValuesIn(ascend950_cases_params));

TEST_F(BatchMatMulV3TilingRuntime, fail_case)
{
    gert::StorageShape x1_shape = {{2, 15, 16}, {2, 15, 16}};
    gert::StorageShape x2_shape = {{2, 16, 16}, {2, 1, 1, 16, 16}};
    gert::StorageShape bias_shape = {{
                                         16,
                                     },
                                     {
                                         16,
                                     }};
    std::vector<gert::StorageShape> output_shapes(1, {{2, 15, 16}, {2, 15, 16}});
    std::vector<void*> output_shapes_ref(1);
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
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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
    std::vector<void*> output_shapes_ref_two(1);
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

TEST_F(BatchMatMulV3TilingRuntime, bias_cases)
{
    gert::StorageShape x1_shape = {{10, 10, 32, 32}, {10, 10, 32, 32}};
    gert::StorageShape x2_shape = {{10, 10, 32, 32}, {10, 10, 32, 32}};
    gert::StorageShape bias_shape = {{
                                         32,
                                     },
                                     {
                                         32,
                                     }};
    std::vector<gert::StorageShape> output_shapes(1, {{10, 10, 32, 32}, {10, 10, 32, 32}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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
    uint64_t tiling_key = tiling_context->GetTilingKey();
    ;
    uint32_t block_dim = tiling_context->GetBlockDim();
    string case_name = "BatchMatMulV3TilingRuntime_bias_cases";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData(
        "24 32 32 32 32 32 32 32 32 32 32 124 124 1 1 1 0 0 0 4160 4096 0 1 1 1 1 62 62 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 24 100 100 100 1 1 1 1 1 1 10 10 10 10 10 10 0 0 32 100 "
        "0 0 ",
        case_name, tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 65536);
    ASSERT_EQ(block_dim, 24);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_batchbias_failed_1)
{
    gert::StorageShape x1_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape x2_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape bias_shape = {{2048, 1, 16}, {2048, 1, 16}};
    std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16}, {2048, 16, 16}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_batchbias_failed)
{
    gert::StorageShape x1_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape x2_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape bias_shape = {{2047, 1, 16}, {2047, 1, 16}};
    std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16}, {2048, 16, 16}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_two_dim_bias_failed)
{
    gert::StorageShape x1_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape x2_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape bias_shape = {{2, 16}, {2, 16}};
    std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16}, {2048, 16, 16}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_l0_iterbatchbias_success)
{
    gert::StorageShape x1_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape x2_shape = {{2048, 16, 16}, {2048, 16, 16}};
    gert::StorageShape bias_shape = {{16}, {16}};
    std::vector<gert::StorageShape> output_shapes(1, {{2048, 16, 16}, {2048, 16, 16}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    ;
    uint32_t block_dim = tiling_context->GetBlockDim();
    string case_name = "BatchMatMulV3TilingRuntime_bias_cases_l0_iterbatchbias_success";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("16 16 16 2048 64 64 0 16 16 16 0 1 0 0", case_name, tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 257UL);
    ASSERT_EQ(block_dim, 32);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_F(BatchMatMulV3TilingRuntime, bias_cases_l1_iterbatchbias_success)
{
    gert::StorageShape x1_shape = {{992, 28, 2}, {992, 2, 1851}};
    gert::StorageShape x2_shape = {{992, 28, 2}, {992, 2, 1851}};
    gert::StorageShape bias_shape = {{
                                         1851,
                                     },
                                     {
                                         1851,
                                     }};
    std::vector<gert::StorageShape> output_shapes(1, {{992, 28, 1851}, {992, 28, 1851}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

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

TEST_F(BatchMatMulV3TilingRuntime, 950_transpose_non_contiguous_cases)
{
    gert::StorageShape x1_shape = {{512, 150, 150}, {512, 150, 150}};
    gert::StorageShape x2_shape = {{512, 150, 32}, {2457600}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);

    Stride x2_stride({32, 16384, 1});
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr, x2_stride, 0);

    std::vector<gert::StorageShape> output_shapes(1, {{512, 150, 32}, {512, 150, 32}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
 	       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
 	       "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
 	       "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
 	       "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1650";

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    std::vector<gert::TensorV2*> inputTensors = {&x1Tensor, &x2Tensor};
    gert::KernelRunContextHolder holder;
    holder = gert::TilingContextFaker()
                 .SetOpType("BatchMatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                 .InputTensors(inputTensors)
                 .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .CompileInfo(&compile_info)
                 .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                 .TilingData(tiling_data.get())
                 .Workspace(ws_size)
                 .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    uint32_t block_dim = tiling_context->GetBlockDim();
    string case_name = "BatchMatMulV3TilingRuntime_950_transpose_non_contiguous_cases";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("150 32 150 512 4 1 0 96 32 160 512 1 0 0", case_name, tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 257UL);
    ASSERT_EQ(block_dim, 32);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_F(BatchMatMulV3TilingRuntime, 950_transpose_non_contiguous_cases1)
{
    gert::StorageShape x1_shape = {{512, 150, 150}, {512, 150, 150}};
    gert::StorageShape x2_shape = {{512, 32, 150}, {2457600}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    Stride x2_stride({150, 76800, 1});
    x2Tensor.MutableStride() = x2_stride;
    x2Tensor.SetOffset(0);

    std::vector<gert::StorageShape> output_shapes(1, {{512, 150, 32}, {512, 150, 32}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
 	       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
 	       "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
 	       "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
 	       "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1650";

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    if (get_map_string(soc_version, "NpuArch") == "3510") {
        compile_info.aivNum = std::stoi(soc_infos["vector_core_cnt"]);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    std::vector<gert::TensorV2*> inputTensors = {&x1Tensor, &x2Tensor};
    gert::KernelRunContextHolder holder;
    holder = gert::TilingContextFaker()
                 .SetOpType("BatchMatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                 .InputTensors(inputTensors)
                 .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .CompileInfo(&compile_info)
                 .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                 .TilingData(tiling_data.get())
                 .Workspace(ws_size)
                 .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    uint32_t block_dim = tiling_context->GetBlockDim();
    string case_name = "BatchMatMulV3TilingRuntime_950_transpose_non_contiguous_cases1";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("150 32 150 512 4 1 0 96 32 160 512 1 0 0 ", case_name, tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 321UL);
    ASSERT_EQ(block_dim, 32);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_F(BatchMatMulV3TilingRuntime, 910d_transpose_non_contiguous_cases2)
{
    gert::StorageShape x1_shape = {{16, 196, 128}, {401408}};
    gert::StorageShape x2_shape = {{16, 196, 128}, {401408}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    Stride x1_stride({128, 16 * 128, 1});
    x1Tensor.MutableStride() = x1_stride;
    x1Tensor.SetOffset(0);

    Stride x2_stride({128, 16 * 128, 1});
    x2Tensor.MutableStride() = x2_stride;
    x2Tensor.SetOffset(0);

    std::vector<gert::StorageShape> output_shapes(1, {{16, 192, 192}, {16, 192, 192}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
 	       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
 	       "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
 	       "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
 	       "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})";
    optiling::MatmulV3CompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> aicore_memory_rates;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1650";
    aicore_memory_rates["ddr_rate"] = "31";
    aicore_memory_rates["l2_rate"] = "100";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreMemoryRates",
                                                                                            aicore_memory_rates);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    std::vector<gert::TensorV2*> inputTensors = {&x1Tensor, &x2Tensor};
    gert::KernelRunContextHolder holder;
    holder = gert::TilingContextFaker()
                 .SetOpType("BatchMatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                 .InputTensors(inputTensors)
                 .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .CompileInfo(&compile_info)
                 .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                 .TilingData(tiling_data.get())
                 .Workspace(ws_size)
                 .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    uint32_t block_dim = tiling_context->GetBlockDim();
    string case_name = "BatchMatMulV3TilingRuntime_910d_transpose_non_contiguous_cases2";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData(
        "32 196 196 128 208 112 128 208 112 64 128 1 1 1 1 0 0 33686528 0 208 1 128 16 16 1 ", case_name, tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 65UL);
    ASSERT_EQ(block_dim, 32);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

} // namespace

namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;
using namespace optiling::batch_matmul_v3_advanced;

gert::KernelRunContextHolder BuildTilingContext()
{
    struct TestCompileInfo {};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    TestCompileInfo compileInfo;
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    gert::StorageShape inputShape = {{8, 16}, {8, 16}};
    gert::StorageShape outputShape = {{2, 4, 8}, {2, 4, 8}};
    auto tilingData = gert::TilingData::CreateCap(64);
    auto* rawTilingData = reinterpret_cast<gert::TilingData*>(tilingData.get());

    return gert::TilingContextFaker()
        .SetOpType("BatchMatMulV3")
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1}, {1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapes({&inputShape, &inputShape, &inputShape})
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .TilingData(rawTilingData)
        .Workspace(wsSize)
        .Build();
}

class BatchMatMulV3IterBatchTilingForTest : public BatchMatMulV3IterBatchTiling {
public:
    using BatchMatMulV3IterBatchTiling::BatchMatMulV3IterBatchTiling;

    ge::graphStatus DoOpTilingPublic() { return DoOpTiling(); }

    uint64_t GetTilingKeyPublic() const { return GetTilingKey(); }

    uint64_t GetRunInfoBaseM() const { return runInfo_.baseM; }

    uint64_t GetRunInfoBaseN() const { return runInfo_.baseN; }

    uint64_t GetRunInfoBaseK() const { return runInfo_.baseK; }

    uint64_t GetRunInfoSingleCoreM() const { return runInfo_.singleCoreM; }

    uint64_t GetRunInfoSingleCoreN() const { return runInfo_.singleCoreN; }

    uint64_t GetRunInfoSingleCoreK() const { return runInfo_.singleCoreK; }

    uint64_t GetRunInfoStepM() const { return runInfo_.stepM; }

    uint64_t GetRunInfoStepN() const { return runInfo_.stepN; }

    uint64_t GetRunInfoStepKa() const { return runInfo_.stepKa; }

    uint64_t GetRunInfoStepKb() const { return runInfo_.stepKb; }

    uint64_t GetRunInfoDepthA1() const { return runInfo_.depthA1; }

    uint64_t GetRunInfoDepthB1() const { return runInfo_.depthB1; }

    uint64_t GetRunInfoIterBatch() const { return runInfo_.bmmRunInfo.iterBatch; }

    uint64_t GetRunInfoBatchOutNum() const { return runInfo_.bmmRunInfo.batchOutNum; }
};

class BatchMatMulV3IterBatchTilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        holder_ = BuildTilingContext();
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
        compileInfo_.aicNum = 32;
        compileInfo_.aivNum = 16;
        compileInfo_.l0ASize = 131072;
        compileInfo_.l0BSize = 131072;
        compileInfo_.l0CSize = 131072;
        compileInfo_.l1Size = 262144;
        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.aDtypeSize = 2;
        args_.bDtypeSize = 2;
        args_.batchInfo = &batchInfo_;
    }

    void SetShape(uint64_t m, uint64_t n, uint64_t k)
    {
        args_.mValue = m;
        args_.nValue = n;
        args_.kValue = k;
    }

    std::unique_ptr<BatchMatMulV3IterBatchTilingForTest> CreateTiling()
    {
        cfg_ = std::make_unique<MatMulTilingCfg>(false, &compileInfo_, &args_, nullptr);
        return std::make_unique<BatchMatMulV3IterBatchTilingForTest>(context_, *cfg_);
    }

    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    MatmulV3CompileInfo compileInfo_;
    MatMulV3Args args_;
    MatMulV3BatchInfo batchInfo_;
    std::unique_ptr<MatMulTilingCfg> cfg_;
};

TEST_F(BatchMatMulV3IterBatchTilingTest, DoOpTiling_EnableMultiBatch)
{
    SetShape(64, 128, 512);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);

    EXPECT_EQ(tiling->GetRunInfoSingleCoreM(), args_.mValue);
    EXPECT_EQ(tiling->GetRunInfoSingleCoreN(), args_.nValue);
    EXPECT_EQ(tiling->GetRunInfoSingleCoreK(), args_.kValue);
    EXPECT_EQ(tiling->GetRunInfoBaseM(), 64UL);  // CeilAlign(64, 16)
    EXPECT_EQ(tiling->GetRunInfoBaseN(), 128UL); // CeilAlign(128, 16)
    EXPECT_EQ(tiling->GetRunInfoBaseK(), 256UL);
    EXPECT_EQ(tiling->GetRunInfoStepM(), 1UL);
    EXPECT_EQ(tiling->GetRunInfoStepN(), 1UL);
    EXPECT_EQ(tiling->GetRunInfoStepKa(), 2UL); // CeilDiv(512, 256)
    EXPECT_EQ(tiling->GetRunInfoStepKb(), tiling->GetRunInfoStepKa());
    EXPECT_EQ(tiling->GetRunInfoDepthA1(), tiling->GetRunInfoStepKa() * tiling->GetRunInfoStepM());
    EXPECT_EQ(tiling->GetRunInfoDepthB1(), tiling->GetRunInfoStepKb() * tiling->GetRunInfoStepN());
    EXPECT_EQ(tiling->GetRunInfoIterBatch(), 0UL); // FloorAlign(1, 2)
    EXPECT_EQ(tiling->GetRunInfoBatchOutNum(), 0UL);
}

TEST_F(BatchMatMulV3IterBatchTilingTest, DoOpTiling_DisableMultiBatch)
{
    SetShape(1024, 2048, 512);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);

    EXPECT_EQ(tiling->GetRunInfoSingleCoreM(), args_.mValue);
    EXPECT_EQ(tiling->GetRunInfoSingleCoreN(), args_.nValue);
    EXPECT_EQ(tiling->GetRunInfoSingleCoreK(), args_.kValue);
    EXPECT_EQ(tiling->GetRunInfoBaseM(), 128UL); // reset base
    EXPECT_EQ(tiling->GetRunInfoBaseN(), 256UL); // reset base
    EXPECT_EQ(tiling->GetRunInfoBaseK(), 128UL);
    EXPECT_EQ(tiling->GetRunInfoStepM(), 8UL);  // CeilDiv(1024, 128)
    EXPECT_EQ(tiling->GetRunInfoStepN(), 8UL);  // CeilDiv(2048, 256)
    EXPECT_EQ(tiling->GetRunInfoStepKa(), 4UL); // CeilDiv(512, 128)
    EXPECT_EQ(tiling->GetRunInfoStepKb(), tiling->GetRunInfoStepKa());
    EXPECT_EQ(tiling->GetRunInfoDepthA1(), tiling->GetRunInfoStepKa() * tiling->GetRunInfoStepM());
    EXPECT_EQ(tiling->GetRunInfoDepthB1(), tiling->GetRunInfoStepKb() * tiling->GetRunInfoStepN());
    EXPECT_EQ(tiling->GetRunInfoIterBatch(), 0UL);
    EXPECT_EQ(tiling->GetRunInfoBatchOutNum(), 1UL);
}

TEST_F(BatchMatMulV3IterBatchTilingTest, GetTilingKey_NoTrans)
{
    args_.isATrans = false;
    args_.isBTrans = false;
    auto tiling = CreateTiling();
    uint64_t tilingKey = tiling->GetTilingKeyPublic();

    BatchMatMulV3TilingKey expectedKey;
    expectedKey.SetTrans(false, false).SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL);
    EXPECT_EQ(tilingKey, expectedKey.GetTilingKey());

    MatMulV3TilingKey keyParser;
    EXPECT_EQ(keyParser.GetModel(tilingKey), MatMulV3Model::BASIC);
    EXPECT_EQ(keyParser.GetBatchModel(tilingKey), MatMulV3BatchModel::SINGLE_BIAS_MODEL);
    EXPECT_EQ(keyParser.GetApiLevel(tilingKey), MatMulV3ApiLevel::HIGH_LEVEL);
}

TEST_F(BatchMatMulV3IterBatchTilingTest, GetTilingKey_WithTrans)
{
    args_.isATrans = true;
    args_.isBTrans = true;
    auto tiling = CreateTiling();
    uint64_t tilingKey = tiling->GetTilingKeyPublic();

    BatchMatMulV3TilingKey expectedKey;
    expectedKey.SetTrans(true, true).SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL);
    EXPECT_EQ(tilingKey, expectedKey.GetTilingKey());
}

TEST_F(BatchMatMulV3IterBatchTilingTest, GetTilingKey_AfterDoOpTiling)
{
    SetShape(64, 128, 512);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    uint64_t tilingKey = tiling->GetTilingKeyPublic();

    BatchMatMulV3TilingKey expectedKey;
    expectedKey.SetTrans(false, false).SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL);
    EXPECT_EQ(tilingKey, expectedKey.GetTilingKey());
}

class BatchMatMulV3KEqZeroTilingForTest : public BatchMatMulV3KEqZeroTiling {
public:
    using BatchMatMulV3KEqZeroTiling::BatchMatMulV3KEqZeroTiling;

    ge::graphStatus DoOpTilingPublic() { return DoOpTiling(); }

    uint64_t GetNumBlocksPublic() const { return GetNumBlocks(); }

    uint64_t GetTilingKeyPublic() const { return GetTilingKey(); }

    ge::graphStatus GetTilingDataPublic(TilingResult& tiling) const { return GetTilingData(tiling); }

    uint64_t GetRunInfoTotalDataAmount() const { return runInfo_.totalDataAmount; }

    uint64_t GetRunInfoUsedCoreNum() const { return runInfo_.usedCoreNum; }
};

class BatchMatMulV3KEqZeroTilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        holder_ = BuildTilingContext();
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
        batchInfo_.batchC = 2;
        args_.mValue = 64;
        args_.nValue = 128;
        args_.kValue = 0;
        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.batchInfo = &batchInfo_;
        compileInfo_.aivNum = 32;
    }

    std::unique_ptr<BatchMatMulV3KEqZeroTilingForTest> CreateTiling()
    {
        cfg_ = std::make_unique<MatMulTilingCfg>(false, &compileInfo_, &args_, nullptr);
        return std::make_unique<BatchMatMulV3KEqZeroTilingForTest>(context_, *cfg_);
    }

    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    MatmulV3CompileInfo compileInfo_;
    MatMulV3Args args_;
    MatMulV3BatchInfo batchInfo_;
    std::unique_ptr<MatMulTilingCfg> cfg_;
};

TEST_F(BatchMatMulV3KEqZeroTilingTest, DoOpTiling_SetRunInfo)
{
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetRunInfoUsedCoreNum(), compileInfo_.aivNum);
    EXPECT_EQ(tiling->GetRunInfoTotalDataAmount(), args_.mValue * args_.nValue * batchInfo_.batchC);
}

TEST_F(BatchMatMulV3KEqZeroTilingTest, GetNumBlocks)
{
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetNumBlocksPublic(), compileInfo_.aivNum);
}

TEST_F(BatchMatMulV3KEqZeroTilingTest, GetTilingKey)
{
    auto tiling = CreateTiling();
    uint64_t tilingKey = tiling->GetTilingKeyPublic();

    BatchMatMulV3TilingKey expectedKey;
    expectedKey.SetTrans(false, false)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::K_EQUAL_ZERO)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY);
    EXPECT_EQ(tilingKey, expectedKey.GetTilingKey());

    MatMulV3TilingKey keyParser;
    EXPECT_EQ(keyParser.GetModel(tilingKey), MatMulV3Model::K_EQUAL_ZERO);
    EXPECT_EQ(keyParser.GetBatchModel(tilingKey), MatMulV3BatchModel::BATCH_MODEL);
    EXPECT_EQ(keyParser.GetApiLevel(tilingKey), MatMulV3ApiLevel::BASIC_LEVEL);
}

TEST_F(BatchMatMulV3KEqZeroTilingTest, GetTilingData)
{
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);

    TilingResult result;
    ASSERT_EQ(tiling->GetTilingDataPublic(result), ge::GRAPH_SUCCESS);
    ASSERT_NE(result.tilingData, nullptr);
    EXPECT_EQ(result.tilingDataSize, sizeof(MatMulV3KEqZeroBasicTilingData));

    auto* data = static_cast<MatMulV3KEqZeroBasicTilingData*>(result.tilingData.get());
    EXPECT_EQ(data->totalDataAmount, args_.mValue * args_.nValue * batchInfo_.batchC);
    EXPECT_EQ(data->aivNum, compileInfo_.aivNum);
}

gert::KernelRunContextHolder BuildDebugTilingTestContext(gert::TilingData* tilingData)
{
    struct TestCompileInfo {};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    TestCompileInfo compileInfo;
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    gert::StorageShape inputShape = {{8, 16}, {8, 16}};
    gert::StorageShape outputShape = {{2, 4, 8}, {2, 4, 8}};

    return gert::TilingContextFaker()
        .SetOpType("BatchMatMulV3")
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1}, {1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_FRACTAL_NZ)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_ND)
        .InputShapes({&inputShape, &inputShape, &inputShape})
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .TilingData(tilingData)
        .Workspace(wsSize)
        .Build();
}

class DebugTilingTestForMatmulCommon : public testing::Test {};

TEST_F(DebugTilingTestForMatmulCommon, DebugTilingContext_WithInputsAndOutputs)
{
    auto tilingData = gert::TilingData::CreateCap(64);
    auto holder = BuildDebugTilingTestContext(reinterpret_cast<gert::TilingData*>(tilingData.get()));
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);

    string str = Ops::NN::DebugTilingContext(context);
    EXPECT_NE(str.find("input0"), string::npos);
    EXPECT_NE(str.find("input1"), string::npos);
    EXPECT_NE(str.find("input2"), string::npos);
    EXPECT_NE(str.find("output0"), string::npos);
    EXPECT_NE(str.find("(dtype:"), string::npos);
    EXPECT_NE(str.find("(shape:"), string::npos);
    EXPECT_NE(str.find("(ori_shape:"), string::npos);
    EXPECT_NE(str.find("(format:"), string::npos);
    EXPECT_NE(str.find("(ori_format:"), string::npos);
}

TEST_F(DebugTilingTestForMatmulCommon, DebugTilingData_WithInt32Values)
{
    auto tilingData = gert::TilingData::CreateCap(64);
    auto* rawTilingData = reinterpret_cast<gert::TilingData*>(tilingData.get());
    ASSERT_EQ(rawTilingData->Append(1), ge::GRAPH_SUCCESS);
    ASSERT_EQ(rawTilingData->Append(2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(rawTilingData->Append(3), ge::GRAPH_SUCCESS);

    auto holder = BuildDebugTilingTestContext(rawTilingData);
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);

    string str = Ops::NN::DebugTilingData(context);
    EXPECT_NE(str.find("1, "), string::npos);
    EXPECT_NE(str.find("2, "), string::npos);
    EXPECT_NE(str.find("3, "), string::npos);
}

TEST_F(DebugTilingTestForMatmulCommon, DebugTilingData_Empty)
{
    auto tilingData = gert::TilingData::CreateCap(64);
    auto holder = BuildDebugTilingTestContext(reinterpret_cast<gert::TilingData*>(tilingData.get()));
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);

    string str = Ops::NN::DebugTilingData(context);
    EXPECT_TRUE(str.empty());
}

} // namespace
