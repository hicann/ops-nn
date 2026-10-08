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
#include "../../../../op_host/op_tiling/matmul_v3_compile_info.h"
#include "../../../../op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h"
#include "../../../../op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.h"
#include "../../../../op_host/op_tiling/arch35/matmul_v3_k_equal_zero_tiling.h"

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
    // 0-18 23-27 30-32 48-57 表示mm实际用到的tilingdata
    // 新增rowStride后原字段innerBatch的index=22 按照之前设计, 非连续字段不在参数化用例中验证，单独设计ut验证即可
    // 特修改此处index >= 23UL，跳过innerBatch slice rowStride这些字段
    if (index < 18UL || (index >= 23UL && index <= 27UL) || (index >= 30UL && index <= 32UL) ||
        (index >= 48UL && index <= 57UL)) {
        return true;
    }
    // 非950tiling要检查57位以后的tilingdata
    if (case_name.find("MatMulV3_950") == string::npos && index > 57UL) {
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
    int64_t opImplMode; // 0x40 for hf32, 0x4 for enable_force_grp_acc_for_fp32.
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

    bool has_bias = false;
    ge::Format bias_format = ge::FORMAT_ND;
    ge::Format bias_ori_format = ge::FORMAT_ND;
    std::initializer_list<int64_t> bias_shape;
    std::initializer_list<int64_t> bias_orishape;
    ge::DataType bias_dtype = DT_FLOAT16;
};

class MatMulV3TilingRuntime : public testing::TestWithParam<TilingTestParam> {
    virtual void SetUp() {}
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

static void TestSlice(const TilingTestParam& param)
{
    gert::StorageShape x1_shape = {param.x1_orishape, param.x1_shape};
    gert::StorageShape x2_shape = {param.x2_orishape, param.x2_shape};
    gert::StorageShape bias_shape = {param.bias_orishape, param.bias_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_orishape, param.y_shape});
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
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1800";

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance()
                                      .GetOpImpl(param.op_type.c_str())
                                      ->gen_simplifiedkey;
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

    if (param.has_bias) {
        holder = gert::TilingContextFaker()
                     .SetOpType(param.op_type.c_str())
                     .NodeIoNum(3, 1)
                     .IrInstanceNum({1, 1, 1})
                     .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                     .OutputShapes(output_shapes_ref)
                     .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                 {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                 {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset_x)},
                                 {"opImplMode", Ops::NN::AnyValue::CreateFrom<bool>(param.opImplMode)}})
                     .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                     .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                     .NodeInputTd(2, param.bias_dtype, param.bias_ori_format, param.bias_format)
                     .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .TilingData(tiling_data.get())
                     .Workspace(ws_size)
                     .Build();
    } else {
        holder = gert::TilingContextFaker()
                     .SetOpType(param.op_type.c_str())
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1, 1})
                     .InputShapes({&x1_shape, &x2_shape})
                     .OutputShapes(output_shapes_ref)
                     .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                 {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                 {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset_x)},
                                 {"opImplMode", Ops::NN::AnyValue::CreateFrom<bool>(param.opImplMode)}})
                     .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                     .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                     .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .TilingData(tiling_data.get())
                     .Workspace(ws_size)
                     .Build();
    }

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // get (m, k, n)
    int64_t mkDims[2];
    int64_t knDims[2];
    // init tiling class
    optiling::matmul_v3_advanced::MatMulV3Tiling mmv3Tiling(tiling_context);
    // only for ut coverage
    auto checkNonContiguous = mmv3Tiling.ExtractNonContiguousDims(mkDims, knDims);
    auto checkSlice = mmv3Tiling.ExtractSliceDims(mkDims);
}

static void TestOneParamCase(const TilingTestParam& param)
{
    gert::StorageShape x1_shape = {param.x1_orishape, param.x1_shape};
    gert::StorageShape x2_shape = {param.x2_orishape, param.x2_shape};
    gert::StorageShape bias_shape = {param.bias_orishape, param.bias_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_orishape, param.y_shape});
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
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1800";

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance()
                                      .GetOpImpl(param.op_type.c_str())
                                      ->gen_simplifiedkey;
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

    if (param.has_bias) {
        holder = gert::TilingContextFaker()
                     .SetOpType(param.op_type.c_str())
                     .NodeIoNum(3, 1)
                     .IrInstanceNum({1, 1, 1})
                     .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                     .OutputShapes(output_shapes_ref)
                     .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                 {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                 {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset_x)},
                                 {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.opImplMode)}})
                     .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                     .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                     .NodeInputTd(2, param.bias_dtype, param.bias_ori_format, param.bias_format)
                     .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .TilingData(tiling_data.get())
                     .Workspace(ws_size)
                     .Build();
    } else {
        holder = gert::TilingContextFaker()
                     .SetOpType(param.op_type.c_str())
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1, 1})
                     .InputShapes({&x1_shape, &x2_shape})
                     .OutputShapes(output_shapes_ref)
                     .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_a)},
                                 {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.trans_b)},
                                 {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(param.offset_x)},
                                 {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.opImplMode)}})
                     .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                     .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                     .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                     .TilingData(tiling_data.get())
                     .Workspace(ws_size)
                     .Build();
    }

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::char_t simplifiedKey[100] = {0};
    ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    uint32_t block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), param.case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData(param.tiling_data, param.case_name, param.tiling_key);
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, param.tiling_key);
    ASSERT_EQ(block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

TEST_P(MatMulV3TilingRuntime, general_cases) { TestOneParamCase(GetParam()); }

static TilingTestParam ascend950_cases_params[] = {
    {"MatMulV3_950_basic_testNZ_streamk",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_FRACTAL_NZ,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     0,
     {4096, 8192},
     {512, 80, 16, 16},
     {4096, 1280},
     {4096, 8192},
     {1280, 8192},
     {4096, 1280},
     false,
     0,
     0,
     32,
     4162UL,
     "32 4096 1280 8192 256 256 256 256 256 64 4096 1 1 1 1 0 0 16843264 0 256 1 0 "},
    {"MatMulV3_950_perf_tmp_1280_16384_x_16384_32_fp32",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1280, 16384},
     {16384, 32},
     {1280, 32},
     {1280, 16384},
     {16384, 32},
     {1280, 32},
     false,
     0,
     0,
     27,
     24578UL,
     "27 1280 32 16384 48 32 256 48 32 128 16384 1 1 1 1 0 0 33686016 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_basic_testNZ_streamk_fp32",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_FRACTAL_NZ,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     false,
     true,
     0,
     0,
     {248, 10448},
     {1306, 3, 16, 8},
     {248, 39},
     {248, 10448},
     {39, 10448},
     {248, 39},
     false,
     0,
     0,
     32,
     4162UL,
     "32 248 39 10448 256 48 128 256 48 32 328 1 1 1 1 0 0 16843264 0 256 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_basic_testNZ_bFullLoad",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_FRACTAL_NZ,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     0,
     {89, 11665},
     {3, 6, 16, 16},
     {11665, 47},
     {89, 11665},
     {89, 47},
     {11665, 47},
     false,
     0,
     0,
     31,
     18UL,
     "31 11665 47 89 384 48 128 384 48 32 89 1 1 1 1 0 0 33686016 "},
    {"MatMulV3_950_basic_testNZ_aFullLoad",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_FRACTAL_NZ,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     true,
     0,
     0,
     {16, 32},
     {1, 525, 16, 16},
     {32, 8400},
     {16, 32},
     {8400, 16},
     {32, 8400},
     false,
     0,
     0,
     31,
     82UL,
     "31 32 8400 16 32 272 16 32 272 16 16 1 1 1 1 0 0 33686016 0 32 1 0 "},
    {"MatMulV3_950_basic_testNZ_aswt",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
      "format_a":"ND","format_b":"FRACTAL_NZ","repo_range":{},"repo_seeds":{}})",
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_FRACTAL_NZ,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     true,
     false,
     0,
     0,
     {15083, 8906},
     {1, 943, 16, 16},
     {8906, 2},
     {15083, 8906},
     {15083, 2},
     {8906, 2},
     false,
     0,
     0,
     28,
     18UL,
     "28 8906 2 15083 320 16 192 320 16 48 15083 1 1 1 1 0 0 33686016 "},
    {"MatMulV3_950_basic_test15",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
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
     0,
     {4096, 8192},
     {1280, 8192},
     {4096, 1280},
     {4096, 8192},
     {1280, 8192},
     {4096, 1280},
     false,
     0,
     0,
     32,
     4162UL,
     "32 4096 1280 8192 256 256 256 256 256 64 4096 1 1 1 1 0 0 16843264 0 256 1 0 "},
    {"MatMulV3_950_basic_test16",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
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
     0,
     {4096, 8192},
     {128, 8192},
     {4096, 128},
     {4096, 8192},
     {128, 8192},
     {4096, 128},
     false,
     0,
     0,
     32,
     4162UL,
     "32 4096 128 8192 256 128 256 256 128 64 4096 1 1 1 1 0 0 16843264 0 256 1 0 "},
    {"MatMulV3_950_streamK_fp16_test17",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
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
     0,
     {4, 8192},
     {1280, 8192},
     {4, 1280},
     {4, 8192},
     {1280, 8192},
     {4, 1280},
     false,
     0,
     0,
     32,
     4162UL,
     "32 4 1280 8192 16 256 256 16 256 64 1366 1 1 1 1 0 0 16843264 0 16 1 0 "},
    {"MatMulV3_950_streamK2aswt_fp16_test18",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 56},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 56, "socVersion": "Ascend950" },
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
     0,
     {4, 8192},
     {1280, 8192},
     {4, 1280},
     {4, 8192},
     {1280, 8192},
     {4, 1280},
     false,
     0,
     0,
     27,
     66UL,
     "27 4 1280 8192 16 48 512 16 48 256 8192 1 1 1 1 0 0 33686016 "},
    {"MatMulV3_950_basic_test20",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":true, "offset_x":0, "opImplMode":0},
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
     0,
     {3952, 224},
     {192, 3952},
     {224, 192},
     {3952, 224},
     {192, 3952},
     {224, 192},
     false,
     0,
     0,
     32,
     4178UL,
     "32 224 192 3952 224 192 256 224 192 64 124 1 1 1 1 0 0 16843264 0 224 1 0 "},
    {"MatMulV3_950_streamK_fp16_test21",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {523, 10866},
     {10866, 246},
     {523, 246},
     {523, 10866},
     {10866, 246},
     {523, 246},
     false,
     0,
     0,
     32,
     2101250UL,
     "32 523 246 10866 176 256 256 176 256 64 1087 1 1 1 1 0 0 16843264 0 176 1 0 "},
    {"MatMulV3_950_al1_full_load_22",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30 ,"vector_core_cnt": 60},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
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
     0,
     {48, 953},
     {3306, 953},
     {48, 3306},
     {48, 953},
     {3306, 953},
     {48, 3306},
     false,
     0,
     0,
     30,
     66UL,
     "30 48 3306 953 48 112 256 48 112 128 953 1 1 1 1 0 0 33686016 0 48 1 0 "},
    // {
    //   "MatMulV3_950_al1_full_load_23", "MatMulV3", R"({"_pattern": "MatMul",
    //   "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
    //     "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz":
    //     false, "l2_size":134217728},"binary_mode_flag":true, "block_dim":{"CORE_NUM":32, "vector_core_cnt":
    //     64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0, "hardware_info":
    //     {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true,
    //     "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true,
    //     "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    //     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "vector_core_cnt": 64,
    //     "socVersion": "Ascend950" }, "format_a":"ND","format_b":"ND","repo_range":{},"repo_seeds":{}})",
    //   ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, true, false, 0, 0,
    //   {1166, 48}, {1166, 673}, {48, 673}, {1166, 48}, {1166, 673}, {48, 673}, false, 0, 0, 32,
    //   10000900009001090001UL, "32 48 673 1166 1166 48 32 1166 48 32 160 8 2 1 1 0 0 0 0 409600 6144 0 1 1 1 1 8 1 0 0
    //   2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ", ge::DT_FLOAT, ge::DT_FLOAT
    // },
    {"MatMulV3_950_abl1_full_load_03",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30 ,"vector_core_cnt": 60},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
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
     0,
     {190, 16},
     {2130, 16},
     {190, 2130},
     {190, 16},
     {2130, 16},
     {190, 2130},
     false,
     0,
     0,
     27,
     66UL,
     "27 190 2130 16 64 256 16 64 256 16 16 1 1 1 1 0 0 33686016 "},
    {"MatMulV3_950_abl1_full_load_04",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
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
     0,
     {304, 112},
     {3152, 112},
     {304, 3152},
     {304, 112},
     {3152, 112},
     {304, 3152},
     false,
     0,
     0,
     32,
     66UL,
     "32 304 3152 112 160 208 128 160 208 64 112 1 1 1 1 0 0 33620480 0 160 1 0 "},
    {"MatMulV3_950_al1_full_load_05",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
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
     0,
     {4, 8192},
     {32000, 8192},
     {4, 32000},
     {4, 8192},
     {32000, 8192},
     {4, 32000},
     false,
     0,
     0,
     32,
     65602UL,
     "32 4 32000 8192 16 336 128 16 336 32 8192 1 1 1 1 0 0 33686016 2 16 1 0 "},
    {"MatMulV3_950_stream_k_black_24",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 8192},
     {2048, 8192},
     {1024, 2048},
     {1024, 8192},
     {2048, 8192},
     {1024, 2048},
     false,
     0,
     0,
     32,
     24642UL,
     "32 1024 2048 8192 256 256 128 256 256 32 8192 1 1 1 1 0 0 16843264 0 256 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_stream_k_fp16_black_25",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {5120, 821},
     {5120, 32},
     {821, 32},
     {5120, 821},
     {5120, 32},
     {821, 32},
     false,
     0,
     0,
     32,
     4114UL,
     "32 821 32 5120 208 32 128 208 32 64 640 1 1 1 1 0 0 16843264 0 208 1 0 ",
     ge::DT_FLOAT16,
     ge::DT_FLOAT16},
    {"MatMulV3_950_stream_k_fp32_white_26",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     {32, 8192},
     {8192, 64},
     {32, 64},
     {32, 8192},
     {8192, 64},
     {32, 64},
     true,
     0,
     0,
     32,
     4098UL,
     "32 32 64 8192 32 64 256 32 64 128 256 1 1 1 1 0 0 16843264 0 32 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_stream_k_dpsk_tf32_white_27",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 8192},
     {8192, 2304},
     {1024, 2304},
     {1024, 8192},
     {8192, 2304},
     {1024, 2304},
     true,
     0,
     0,
     32,
     4098UL,
     "32 1024 2304 8192 256 256 256 256 256 64 1024 1 1 1 1 0 0 16843264 0 256 1 0 ",
     ge::DT_FLOAT16,
     ge::DT_FLOAT16},
    {"MatMulV3_950_stream_k_dpsk_fp16_black_28",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 8192},
     {8192, 2048},
     {1024, 2048},
     {1024, 8192},
     {8192, 2048},
     {1024, 2048},
     true,
     0,
     0,
     32,
     24578UL,
     "32 1024 2048 8192 256 256 128 256 256 32 8192 1 1 1 1 0 0 16843264 0 256 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    // ASWT大于一轮切换基础API
    {"MatMulV3_950_stream_k_dpsk_fp16_black_29",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 8192},
     {8192, 2303},
     {1024, 2303},
     {1024, 8192},
     {8192, 2303},
     {1024, 2303},
     true,
     0,
     0,
     32,
     24578UL,
     "32 1024 2303 8192 256 256 128 256 256 32 8192 4 2 1 1 0 0 16843264 0 256 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_bl1_full_load_26",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {560, 953},
     {80, 953},
     {560, 80},
     {560, 953},
     {80, 953},
     {560, 80},
     false,
     0,
     0,
     27,
     66UL,
     "27 560 80 953 64 32 512 64 32 256 953 1 1 1 1 0 0 33686016 0 64 1 0 "},
    {"MatMulV3_950_abl1_full_load_27",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0, "opImplMode":0},
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
     0,
     {32, 640},
     {32, 480},
     {640, 480},
     {32, 640},
     {32, 480},
     {640, 480},
     false,
     0,
     0,
     20,
     18UL,
     "20 640 480 32 128 128 128 128 128 32 32 1 1 1 1 0 0 33686016 "},
    {"MatMulV3_950_abl1_full_load_28",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {512, 1024},
     {16, 1024},
     {512, 1024},
     {512, 1024},
     {16, 1024},
     {512, 1024},
     false,
     0,
     0,
     32,
     131138UL,
     "32 512 16 1024 16 16 1024 16 16 512 1024 1 1 1 1 0 0 33686528 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_abl1_full_load_29",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {300, 560},
     {300, 20},
     {560, 20},
     {300, 560},
     {300, 20},
     {560, 20},
     false,
     0,
     0,
     24,
     18UL,
     "24 560 20 300 48 16 640 48 16 160 300 1 1 1 1 0 0 33686016 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_abl1_full_load_31",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":30 ,"vector_core_cnt": 60},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
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
     0,
     {944, 48},
     {80, 48},
     {944, 80},
     {944, 48},
     {80, 48},
     {944, 80},
     false,
     0,
     0,
     30,
     66UL,
     "30 944 80 48 32 80 48 32 80 48 48 1 1 1 1 0 0 33686016 0 32 1 0 "},
    {"MatMulV3_950_bl1_full_load_32",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 1022},
     {1024, 25},
     {1022, 25},
     {1024, 1022},
     {1024, 25},
     {1022, 25},
     false,
     0,
     0,
     32,
     18UL,
     "32 1022 25 1024 32 32 1024 32 32 256 1024 1 1 1 1 0 0 33686016 0 32 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_fixpipe_opti_01",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "opImplMode":0},
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
     0,
     {1024, 128},
     {4090, 128},
     {1024, 4090},
     {1024, 128},
     {4090, 128},
     {1024, 4090},
     false,
     0,
     0,
     32,
     1048642UL,
     "32 1024 4090 128 256 256 128 256 256 64 128 1 1 1 1 0 0 16843264 0 256 1 0 "},
    {"MatMulV3_950_fixpipe_opti_02",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
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
     0,
     {10256, 32},
     {32, 720},
     {10256, 720},
     {10256, 32},
     {32, 720},
     {10256, 720},
     false,
     0,
     0,
     32,
     2097154UL,
     "32 10256 720 32 224 256 32 224 256 32 32 3 1 1 1 0 0 33620480 0 224 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_asw_big_k_01",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
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
     0,
     {160, 2080000},
     {2080000, 128},
     {160, 128},
     {160, 2080000},
     {2080000, 128},
     {160, 128},
     false,
     0,
     0,
     20,
     24578UL,
     "20 160 128 2080000 32 32 512 32 32 256 2080000 1 1 1 1 0 0 33686016 0 32 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    // ASWT大于一轮切换基础API
    {"MatMulV3_950_asw_load_balance_m",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
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
     0,
     {1188, 64},
     {64, 2524},
     {1188, 2524},
     {1188, 64},
     {64, 2524},
     {1188, 2524},
     false,
     0,
     0,
     32,
     2097154UL,
     "32 1188 2524 64 208 256 64 208 256 32 64 1 1 1 1 0 0 33620480 0 208 1 0 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_matmul_to_mul",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
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
     4,
     {1, 4096},
     {4096, 4096},
     {1, 4096},
     {1, 4096},
     {4096, 4096},
     {1, 4096},
     false,
     0,
     0,
     64,
     12289UL,
     "64 64 1 4096 4096 64 0 198 136 21 1 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_bias",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
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
     0,
     {1024, 8192},
     {1024, 8192},
     {1024, 1024},
     {1024, 8192},
     {1024, 8192},
     {1024, 1024},
     false,
     0,
     0,
     32,
     4162UL,
     "32 1024 1024 8192 256 256 128 256 256 64 4096 1 1 1 1 0 0 16843264 0 256 1 0",
     DT_FLOAT16,
     DT_FLOAT16,
     {},
     {},
     {},
     true,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     {1024},
     {1024},
     DT_FLOAT16},
    {"MatMulV3_950_AFullLoad_bias",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false, "offset_x":0, "opImplMode":0},
      "binary_attrs":{"bias_flag":true, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
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
     0,
     {1, 9398},
     {9398, 135021},
     {1, 135021},
     {1, 9398},
     {9398, 135021},
     {1, 135021},
     false,
     0,
     0,
     32,
     65538UL,
     "32 1 135021 9398 16 384 64 16 384 32 9398 1 1 1 1 0 0 33686528 0 16 1 0 ",
     DT_FLOAT16,
     DT_FLOAT16,
     {},
     {},
     {},
     true,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     {135021},
     {135021},
     DT_FLOAT16},
    {"MatMulV3_950_matmul_to_multi_mul",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
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
     4,
     {10, 4096},
     {192, 4096},
     {10, 192},
     {10, 4096},
     {192, 4096},
     {10, 192},
     false,
     0,
     0,
     64,
     16449UL,
     "64 10 192 4096 5 6 64 ",
     ge::DT_FLOAT,
     ge::DT_FLOAT},
    {"MatMulV3_950_4buffer_bias",
     "MatMulV3",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":true},
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
     4,
     {351, 8},
     {229678, 351},
     {8, 229678},
     {351, 8},
     {229678, 351},
     {8, 229678},
     false,
     0,
     0,
     32,
     65618UL,
     "32 8 229678 351 16 480 128 16 480 32 351 1 1 1 1 0 0 33686528 0 16 1 0 ",
     DT_FLOAT16,
     DT_FLOAT16,
     {},
     {},
     {},
     true,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     {229678},
     {229678},
     DT_FLOAT}};
INSTANTIATE_TEST_CASE_P(MatMulV3Ascend950, MatMulV3TilingRuntime, testing::ValuesIn(ascend950_cases_params));

static std::string GetPlatformConfigKey(const std::string& compile_info)
{
    try {
        auto j = nlohmann::json::parse(compile_info);
        std::string key;
        if (j.contains("hardware_info")) {
            key += j["hardware_info"].dump();
        }
        if (j.contains("block_dim")) {
            key += j["block_dim"].dump();
        }
        return key;
    } catch (...) {
        return compile_info;
    }
}

static void TestMultiThread(const TilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    if (thread_num == 0) {
        return;
    }
    std::map<std::string, std::vector<size_t>> config_groups;
    for (size_t i = 0; i < testcase_num; ++i) {
        config_groups[GetPlatformConfigKey(params[i].compile_info)].push_back(i);
    }

    for (const auto& kv : config_groups) {
        const auto& indices = kv.second;
        std::vector<std::thread> threads;
        threads.reserve(thread_num);
        for (size_t t = 0; t < thread_num; ++t) {
            threads.emplace_back([&indices, params, t, thread_num]() {
                for (size_t i = t; i < indices.size(); i += thread_num) {
                    TestOneParamCase(params[indices[i]]);
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

TEST_F(MatMulV3TilingRuntime, ascend950_thread_cases)
{
    TestMultiThread(ascend950_cases_params, sizeof(ascend950_cases_params) / sizeof(TilingTestParam), 3);
}

class MatMulV3BiasTilingRuntime : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MatMulV3BiasTilingRuntime SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MatMulV3BiasTilingRuntime TearDown" << std::endl; }
};

TEST_F(MatMulV3BiasTilingRuntime, big_axis_cases)
{
    gert::StorageShape x1_shape = {{2, INT32_MAX + 1L}, {2, INT32_MAX + 1L}};
    gert::StorageShape x2_shape = {{INT32_MAX + 1L, 2}, {INT32_MAX + 1L, 2}};
    gert::StorageShape bias_shape = {{
                                         320,
                                     },
                                     {
                                         320,
                                     }};
    std::vector<gert::StorageShape> output_shapes(1, {{2, 2}, {2, 2}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(3, 1)
                 .IrInstanceNum({1, 1, 1})
                 .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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

TEST_F(MatMulV3BiasTilingRuntime, abnormal_bias_dim_value_case)
{
    gert::StorageShape x1_shape = {{127, 640}, {127, 640}};
    gert::StorageShape x2_shape = {{640, 320}, {640, 320}};
    gert::StorageShape bias_shape = {{321}, {321}};
    std::vector<gert::StorageShape> output_shapes(1, {{127, 320}, {127, 320}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(3, 1)
                 .IrInstanceNum({1, 1, 1})
                 .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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

TEST_F(MatMulV3BiasTilingRuntime, abnormal_wrong_k_case)
{
    gert::StorageShape x1_shape = {{640, 127}, {640, 127}};
    gert::StorageShape x2_shape = {{640, 320}, {640, 320}};
    gert::StorageShape bias_shape = {{320}, {320}};
    std::vector<gert::StorageShape> output_shapes(1, {{127, 320}, {127, 320}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string = R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(3, 1)
                 .IrInstanceNum({1, 1, 1})
                 .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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

TEST_F(MatMulV3TilingRuntime, splitK_nkm_fp32)
{
    gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_fp32:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_indivisibleM)
{
    gert::StorageShape x1_shape = {{62081, 1536}, {62081, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{62081, 384}, {62081, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_indivisibleM:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_indivisibleN)
{
    gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
    gert::StorageShape x2_shape = {{1536, 62081}, {1536, 62081}};
    std::vector<gert::StorageShape> output_shapes(1, {{384, 62081}, {384, 62081}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_indivisibleN:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddK)
{
    gert::StorageShape x1_shape = {{62080, 768}, {62080, 768}};
    gert::StorageShape x2_shape = {{768, 384}, {768, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_oddK:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddM)
{
    gert::StorageShape x1_shape = {{512, 1536}, {512, 1536}};
    gert::StorageShape x2_shape = {{1536, 65536}, {1536, 65536}};
    std::vector<gert::StorageShape> output_shapes(1, {{512, 65536}, {512, 65536}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_oddM:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_oddN)
{
    gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
    gert::StorageShape x2_shape = {{1536, 256}, {1536, 256}};
    std::vector<gert::StorageShape> output_shapes(1, {{62080, 256}, {62080, 256}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_oddN:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_invalidRatio)
{
    gert::StorageShape x1_shape = {{2560, 1536}, {2560, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{2560, 384}, {2560, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_invalidRatio:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_bigMN)
{
    gert::StorageShape x1_shape = {{98304, 1536}, {98304, 1536}};
    gert::StorageShape x2_shape = {{1536, 768}, {1536, 768}};
    std::vector<gert::StorageShape> output_shapes(1, {{98304, 768}, {98304, 768}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_bigMN: " << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_NE(tiling_key, 65568);
    ASSERT_NE(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_targetShape)
{
    gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_targetShape:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_randomShape1)
{
    gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
    gert::StorageShape x2_shape = {{1536, 49152}, {1536, 49152}};
    std::vector<gert::StorageShape> output_shapes(1, {{384, 49152}, {384, 49152}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_randomShape1:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_nkm_randomShape2)
{
    gert::StorageShape x1_shape = {{50176, 1536}, {50176, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{50176, 384}, {50176, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_nkm_randomShape2:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_mkn_bigK)
{
    gert::StorageShape x1_shape = {{3840, 30720}, {3840, 30720}};
    gert::StorageShape x2_shape = {{30720, 3840}, {30720, 3840}};
    std::vector<gert::StorageShape> output_shapes(1, {{3840, 3840}, {3840, 3840}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_shift_mkn_bigK:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65632);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_mkn_bigN)
{
    gert::StorageShape x1_shape = {{384, 1536}, {384, 1536}};
    gert::StorageShape x2_shape = {{1536, 50816}, {1536, 50816}};
    std::vector<gert::StorageShape> output_shapes(1, {{384, 50816}, {384, 50816}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_shift_mkn_bigN:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_TRUE(tiling_key == 65568 || tiling_key == 65632);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_nkm_bigM)
{
    gert::StorageShape x1_shape = {{49152, 1536}, {49152, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{49152, 384}, {49152, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_shift_nkm_bigM:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, splitK_shift_nkm_targetShape)
{
    gert::StorageShape x1_shape = {{62080, 1536}, {62080, 1536}};
    gert::StorageShape x2_shape = {{1536, 384}, {1536, 384}};
    std::vector<gert::StorageShape> output_shapes(1, {{62080, 384}, {62080, 384}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":24, "vector_core_cnt": 48},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM":24, "vector_core_cnt": 48, "socVersion": "Ascend910B" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    cout << "===== splitK_shift_nkm_targetShape:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 65616);
}

TEST_F(MatMulV3TilingRuntime, 950_slice_non_contiguous_case)
{
    gert::StorageShape x1_shape = {{5, 2, 7}, {70}};
    gert::StorageShape x2_shape = {{7, 4}, {7, 4}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);

    std::vector<gert::StorageShape> output_shapes(1, {{10, 4}, {10, 4}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;

    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 235952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM":32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .InputTensors(inputTensors)
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    string case_name = "950_slice_non_contiguous_case";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("1 10 4 7 16 16 16 16 16 16 7 1 1 1 1 0 0 33686016 0 2 0 0",
                                                  case_name, tiling_key);
    cout << "===== 950_slice_non_contiguous_case:" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 20482);
    ASSERT_EQ(block_dim, 1);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

// ========== rowStride 验证测试用例 ==========

// Case 1: 连续 + 转置 (rowStride应该等于1)
TEST_F(MatMulV3TilingRuntime, 950_rowstride_continuous_transpose)
{
    gert::StorageShape x1_shape = {{128, 256}, {128, 256}};
    gert::StorageShape x2_shape = {{128, 64}, {128, 64}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);

    std::vector<gert::StorageShape> output_shapes(1, {{256, 64}, {256, 64}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":true,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 235952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM":32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .InputTensors(inputTensors)
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    string case_name = "950_rowstride_continuous_transpose";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("4 256 64 128 64 64 512 64 64 128 128 1 1 1 1 0 0 33686016",
                                                  case_name, tiling_key);
    cout << "===== " << case_name << ":" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 18UL);
    ASSERT_EQ(block_dim, 4);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

// Case 2: 连续 + 不转置 (rowStride应该等于k=256)
TEST_F(MatMulV3TilingRuntime, 950_rowstride_continuous_no_transpose)
{
    gert::StorageShape x1_shape = {{128, 256}, {128, 256}};
    gert::StorageShape x2_shape = {{256, 64}, {256, 64}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);

    std::vector<gert::StorageShape> output_shapes(1, {{128, 64}, {128, 64}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 235952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM":32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .InputTensors(inputTensors)
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    string case_name = "950_rowstride_continuous_no_transpose";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("8 128 64 256 16 64 256 16 64 256 256 1 1 1 1 0 0 33686016",
                                                  case_name, tiling_key);
    cout << "===== " << case_name << ":" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 2UL);
    ASSERT_EQ(block_dim, 8);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

// Case 3: 非连续 2D slice (rowStride应该等于stride[0]=66)
TEST_F(MatMulV3TilingRuntime, 950_rowstride_noncontiguous_2d_slice)
{
    // view shape: [128, 64], storage shape: [128, 66]
    // stride: [66, 1]
    gert::StorageShape x1_shape = {{128, 64}, {128, 66}};
    gert::StorageShape x2_shape = {{64, 32}, {64, 32}};

    gert::TensorV2 x1Tensor(x1_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);
    gert::TensorV2 x2Tensor(x2_shape, {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, TensorPlacement::kOnHost,
                            ge::DT_FLOAT16, nullptr, nullptr);

    std::vector<gert::StorageShape> output_shapes(1, {{128, 32}, {128, 32}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    string compile_info_string =
        R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"opImplMode":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32, "vector_core_cnt": 64},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 235952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM":32, "vector_core_cnt": 64, "socVersion": "Ascend950" },
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

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->gen_simplifiedkey;
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
                 .SetOpType("MatMulV3")
                 .IrInstanceNum({1, 1}, {1})
                 .InputTensors(inputTensors)
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    string case_name = "950_rowstride_noncontiguous_2d_slice";
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData(), case_name, tiling_key);
    auto golden_tiling_data = GenGoldenTilingData("8 128 32 64 16 32 64 16 32 64 64 1 1 1 1 0 0 33686016", case_name,
                                                  tiling_key);
    cout << "===== " << case_name << ":" << tiling_key << " === \n" << tiling_data_result << std::endl;
    ASSERT_EQ(tiling_key, 2UL);
    ASSERT_EQ(block_dim, 8);
    ASSERT_EQ(tiling_data_result, golden_tiling_data);
}

// =================================================================================================
// MatMulV3BasicStreamKTiling 白盒直调用例（append-only 新增段）
// 覆盖目标（op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp）：
//   L26-29   CheckStreamKDPSKTilingDefault  非3510架构兜底，恒返回false
//   L55-58   CheckStreamKSKTilingDefault    非3510架构兜底，恒返回false
//   L91      GetL0C2OutFlagDefault          非3510架构兜底，返回ON_THE_FLY
//   L201-206 baseM==baseN && depthB1==2*depthA1 调平分支（depthA1翻倍/depthB1减半/stepKa/stepKb重算）
//   L207-210 totalMNCnt_>aicNum && hasBias     bias预留分支（stepKa/stepKb=3）
// 触达方式：MatMulV3BasicStreamKTiling 仅注册于 DAV_3510（BASIC_STREAM_K），经注册表的全量tiling流程
//   只会在 npuArch==DAV_3510 时选中该类，Default 兜底在生产路由下不可达；故沿用本仓
//   batch_mat_mul_v3 直调范式：TilingContextFaker 构造上下文 + ForTest 派生类暴露 protected 入口，
//   以非3510 compileInfo（DAV_2201/DAV_2002）直调 IsCapable/DoOpTiling。
// =================================================================================================
class MatMulV3BasicStreamKTilingForTest : public optiling::matmul_v3_advanced::MatMulV3BasicStreamKTiling {
public:
    using MatMulV3BasicStreamKTiling::MatMulV3BasicStreamKTiling;

    bool IsCapablePublic() { return IsCapable(); }

    ge::graphStatus DoOpTilingPublic() { return DoOpTiling(); }

    uint64_t GetTilingKeyPublic() const { return GetTilingKey(); }

    uint64_t GetRunInfoBaseM() const { return runInfo_.baseM; }

    uint64_t GetRunInfoBaseN() const { return runInfo_.baseN; }

    uint64_t GetRunInfoBaseK() const { return runInfo_.baseK; }

    uint64_t GetRunInfoSingleCoreK() const { return runInfo_.singleCoreK; }

    uint64_t GetRunInfoStepKa() const { return runInfo_.stepKa; }

    uint64_t GetRunInfoStepKb() const { return runInfo_.stepKb; }

    uint64_t GetRunInfoDepthA1() const { return runInfo_.depthA1; }

    uint64_t GetRunInfoDepthB1() const { return runInfo_.depthB1; }

    uint64_t GetRunInfoKCnt() const { return runInfo_.tailInfo.kCnt; }
};

class MatMulV3StreamKTilingDirectTest : public testing::Test {
protected:
    void SetUp() override
    {
        platformInfo_.Init();
        compileInfo_.aicNum = 8UL;
        compileInfo_.aivNum = 16UL; // streamk模板要求 aivNum == aicNum * 2
        compileInfo_.l1Size = 524288UL;
        compileInfo_.l0ASize = 65536UL;
        compileInfo_.l0BSize = 65536UL;
        compileInfo_.l0CSize = 262144UL;
        compileInfo_.l2Size = 134217728UL;
        compileInfo_.ubSize = 253952UL;
        compileInfo_.btSize = 4096UL;
        compileInfo_.npuArch = NpuArch::DAV_2201;
        args_.opName = "MatMulV3StreamKDirectUT";
        args_.aType = ge::DT_FLOAT16;
        args_.bType = ge::DT_FLOAT16;
        args_.cType = ge::DT_FLOAT16;
        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.outFormat = ge::FORMAT_ND;
        args_.aDtypeSize = 2UL;
        args_.bDtypeSize = 2UL;
        SetShape(256UL, 256UL, 4096UL);
        RebuildContext();
    }

    void SetShape(uint64_t m, uint64_t n, uint64_t k)
    {
        args_.mValue = m;
        args_.nValue = n;
        args_.kValue = k;
    }

    // 构造带确定性等级0、合法输入shape的TilingContext，保证IsCapable中
    // GetDeterministicLevel/aFormat/IsSelfNonContiguous/aivNum等守卫可通过
    void RebuildContext()
    {
        aShape_ = gert::StorageShape({static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.kValue)},
                                     {static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.kValue)});
        bShape_ = gert::StorageShape({static_cast<int64_t>(args_.kValue), static_cast<int64_t>(args_.nValue)},
                                     {static_cast<int64_t>(args_.kValue), static_cast<int64_t>(args_.nValue)});
        yShape_ = gert::StorageShape({static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.nValue)},
                                     {static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.nValue)});
        tilingDataBuf_ = gert::TilingData::CreateCap(2048);
        workspaceBuf_ = gert::ContinuousVector::Create<size_t>(4096);
        holder_ = gert::TilingContextFaker()
                      .SetOpType("MatMulV3")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&aShape_, &bShape_})
                      .OutputShapes({&yShape_})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicLevelInfo(0)
                      .CompileInfo(&compileInfo_)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                      .TilingData(reinterpret_cast<gert::TilingData*>(tilingDataBuf_.get()))
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspaceBuf_.get()))
                      .Build();
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
    }

    std::unique_ptr<MatMulV3BasicStreamKTilingForTest> CreateTiling()
    {
        cfg_ = std::make_unique<optiling::MatMulTilingCfg>(false, &compileInfo_, &args_, nullptr);
        return std::make_unique<MatMulV3BasicStreamKTilingForTest>(context_, *cfg_);
    }

    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    fe::PlatFormInfos platformInfo_;
    optiling::MatmulV3CompileInfo compileInfo_;
    optiling::matmul_v3_advanced::MatMulV3Args args_;
    std::unique_ptr<optiling::MatMulTilingCfg> cfg_;
    gert::StorageShape aShape_;
    gert::StorageShape bShape_;
    gert::StorageShape yShape_;
    std::unique_ptr<uint8_t[]> tilingDataBuf_;
    std::unique_ptr<uint8_t[]> workspaceBuf_;
};

// L55-58/L26-29：DAV_2201未注册CheckStreamK*函数（map仅注册DAV_3510），IsCapable经
// CheckStreamKSKTiling/CheckStreamKDPSKTiling先后回落Default（均返回false），整体返回false
TEST_F(MatMulV3StreamKTilingDirectTest, IsCapable_Non3510Arch_FallsBackToDefault)
{
    compileInfo_.npuArch = NpuArch::DAV_2201;
    auto tiling = CreateTiling();
    EXPECT_FALSE(tiling->IsCapablePublic());
}

// 正向对照：同入参（m=n=256,k=4096,aicNum=8）下3510走CheckStreamKSKTilingDav3510返回true，
// 证明上一用例的false确由非3510回落Default导致，而非IsCapable前置守卫拦截
TEST_F(MatMulV3StreamKTilingDirectTest, IsCapable_Dav3510_SktEnablePositiveControl)
{
    compileInfo_.npuArch = NpuArch::DAV_3510;
    auto tiling = CreateTiling();
    EXPECT_TRUE(tiling->IsCapablePublic());
}

// L91：DAV_2201下GetL0C2OutFlag回落Default返回ON_THE_FLY（经totalMNCnt_=1 <= aicNum/2分支触发）；
// 同时baseM(128)!=baseN(256)，L201调平分支与L207 bias分支均不进入（对照路径）
TEST_F(MatMulV3StreamKTilingDirectTest, DoOpTiling_Non3510Arch_GetL0C2OutDefault_LevelingNotTaken)
{
    compileInfo_.npuArch = NpuArch::DAV_2201;
    SetShape(128UL, 256UL, 512UL);
    RebuildContext();
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetRunInfoBaseM(), 128UL);
    EXPECT_EQ(tiling->GetRunInfoBaseN(), 256UL);
    EXPECT_NE(tiling->GetRunInfoBaseM(), tiling->GetRunInfoBaseN()); // 调平分支条件不成立
    EXPECT_EQ(tiling->GetRunInfoKCnt(), 8UL);                        // FloorDiv(aicNum=8, totalMNCnt_=1)
    EXPECT_EQ(tiling->GetRunInfoSingleCoreK(), 64UL);                // CeilDiv(k=512, kCnt=8)
    EXPECT_EQ(tiling->GetRunInfoBaseK(), 64UL);                      // min(singleCoreK=64, kValueMax=128)
    // CalL1TilingDefault产生的depthA1==depthB1且非0，调平未生效
    EXPECT_EQ(tiling->GetRunInfoDepthA1(), 8UL);
    EXPECT_EQ(tiling->GetRunInfoDepthB1(), 8UL);
    EXPECT_EQ(tiling->GetRunInfoStepKa(), 4UL);
    EXPECT_EQ(tiling->GetRunInfoStepKb(), 4UL);
    // tilingkey的L0C2Out域来自GetL0C2OutFlagDefault返回的ON_THE_FLY（模型STREAM_K、张量API级）
    uint64_t expectedKey = optiling::matmul_v3_advanced::MatMulV3TilingKey()
                               .SetTrans(false, false)
                               .SetModel(MatMulV3Model::STREAM_K)
                               .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
                               .SetApiLevel(MatMulV3ApiLevel::TENSOR_LEVEL)
                               .GetTilingKey();
    EXPECT_EQ(tiling->GetTilingKeyPublic(), expectedKey);
}

// L201-206：DAV_2002走CalL1Tiling310P，NZ/NZ下ka全载失败(depthA1=16)而kb全载成功(depthB1=32)，
// 且m==n使baseM==baseN==128，命中调平分支：depthA1翻倍32/depthB1减半16/stepKa=16/stepKb=8
TEST_F(MatMulV3StreamKTilingDirectTest, DoOpTiling_DepthLeveling_WhenBaseMEqBaseNAndDepthB1TwiceDepthA1)
{
    compileInfo_.npuArch = NpuArch::DAV_2002;
    compileInfo_.l1Size = 2097152UL; // 2MB：使128*2048*4恰好不小于l1Size/2（ka全载失败），fp16侧成功
    args_.aType = ge::DT_FLOAT;
    args_.aDtypeSize = 4UL; // a为fp32、b为fp16，制造ka/kb全载条件不对称
    args_.aFormat = ge::FORMAT_FRACTAL_NZ;
    args_.bFormat = ge::FORMAT_FRACTAL_NZ;
    SetShape(128UL, 128UL, 2048UL);
    RebuildContext();
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetRunInfoBaseM(), 128UL);
    EXPECT_EQ(tiling->GetRunInfoBaseN(), 128UL);       // m==n且totalMNCnt_=1<=aicNum/2，重算后baseM==baseN
    EXPECT_EQ(tiling->GetRunInfoSingleCoreK(), 256UL); // CeilAlign(CeilDiv(2048, 8), 16)
    EXPECT_EQ(tiling->GetRunInfoBaseK(), 64UL);        // min(256, kValueMax=FloorAlign(64,32))
    // 调平分支调整后的取值
    EXPECT_EQ(tiling->GetRunInfoDepthA1(), 32UL); // 16 * 2
    EXPECT_EQ(tiling->GetRunInfoDepthB1(), 16UL); // 32 / 2
    EXPECT_EQ(tiling->GetRunInfoStepKa(), 16UL);  // 32 / DB_SIZE
    EXPECT_EQ(tiling->GetRunInfoStepKb(), 8UL);   // 16 / DB_SIZE
}

// L207-210：DAV_2201下m=256,n=1280使totalMNCnt_=10>aicNum=8（且10%8!=0保证else分支可除），hasBias
// 时stepKa/stepKb被预置为3（为bias预留L1空间），depthA1/depthB1保持CalL1Tiling结果不变
TEST_F(MatMulV3StreamKTilingDirectTest, DoOpTiling_BiasReserve_SetsStepKaStepKb3)
{
    compileInfo_.npuArch = NpuArch::DAV_2201;
    compileInfo_.l1Size = 33554432UL; // 32MB，保证CalL1TilingDefault非0退出
    args_.hasBias = true;
    SetShape(256UL, 1280UL, 1024UL);
    RebuildContext();
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetRunInfoKCnt(), 4UL);          // aicNum/(10%8)=4, CeilDiv(1024,256)=4
    EXPECT_EQ(tiling->GetRunInfoSingleCoreK(), 256UL); // CeilDiv(k=1024, kCnt=4)
    EXPECT_EQ(tiling->GetRunInfoBaseK(), 64UL);        // min(256, kValueMax=FloorAlign(65536/2/2/256,64)=64)
    EXPECT_EQ(tiling->GetRunInfoStepKa(), 3UL);        // NUM_THREE，bias预留
    EXPECT_EQ(tiling->GetRunInfoStepKb(), 3UL);        // NUM_THREE，bias预留
    EXPECT_EQ(tiling->GetRunInfoDepthA1(), 8UL);       // 2*stepKa(CalL1Tiling)=8，未被bias分支修改
    EXPECT_EQ(tiling->GetRunInfoDepthB1(), 8UL);       // 2*stepKb(CalL1Tiling)=8，未被bias分支修改
}

// =================================================================================================
// MatMulV3KEqZeroTiling 白盒直调用例（append-only 新增段）
// 覆盖目标（op_host/op_tiling/arch35/matmul_v3_k_equal_zero_tiling.cpp）：
//   L27-37   IsCapable      hasBias拦截分支/kValue!=0拦截分支/通过路径
//   L39-44   DoOpTiling     totalDataAmount=m*n、usedCoreNum=aivNum
//   L46      GetNumBlocks   返回compileInfo_.aivNum
//   L48-59   GetTilingKey   tilingKeyObj为空走tmp分支，构造含K_EQUAL_ZERO model的key
//   L61-64   GetTilingData  经GetTilingDataImpl<MatMulV3KEqZeroBasicTilingData>填充tilingData
// 触达方式：该类仅注册于DAV_3510（MATMUL_INPUT_K_EQUAL_ZERO），生产路由下仅k==0时被选中，
//   全量tiling流程对非k0输入不会进入其DoOpTiling；故沿用本仓 batch_mat_mul_v3 直调范式：
//   TilingContextFaker 构造上下文 + ForTest 派生类暴露 protected 入口，以 kValue=0/非0 的
//   args 直调各入口并断言精确值。
// =================================================================================================
class MatMulV3KEqZeroTilingForTest : public optiling::matmul_v3_advanced::MatMulV3KEqZeroTiling {
public:
    using MatMulV3KEqZeroTiling::MatMulV3KEqZeroTiling;

    bool IsCapablePublic() { return IsCapable(); }

    ge::graphStatus DoOpTilingPublic() { return DoOpTiling(); }

    uint64_t GetTilingKeyPublic() const { return GetTilingKey(); }

    uint64_t GetNumBlocksPublic() const { return GetNumBlocks(); }

    ge::graphStatus GetTilingDataPublic(optiling::TilingResult& tiling) const { return GetTilingData(tiling); }

    uint64_t GetRunInfoTotalDataAmount() const { return runInfo_.totalDataAmount; }

    uint64_t GetRunInfoUsedCoreNum() const { return runInfo_.usedCoreNum; }
};

class MatMulV3KEqZeroTilingDirectTest : public testing::Test {
protected:
    void SetUp() override
    {
        platformInfo_.Init();
        compileInfo_.aicNum = 8UL;
        compileInfo_.aivNum = 16UL;
        compileInfo_.l1Size = 524288UL;
        compileInfo_.l0ASize = 65536UL;
        compileInfo_.l0BSize = 65536UL;
        compileInfo_.l0CSize = 262144UL;
        compileInfo_.l2Size = 134217728UL;
        compileInfo_.ubSize = 253952UL;
        compileInfo_.btSize = 4096UL;
        compileInfo_.npuArch = NpuArch::DAV_3510;
        args_.opName = "MatMulV3KEqZeroDirectUT";
        args_.aType = ge::DT_FLOAT16;
        args_.bType = ge::DT_FLOAT16;
        args_.cType = ge::DT_FLOAT16;
        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.outFormat = ge::FORMAT_ND;
        args_.aDtypeSize = 2UL;
        args_.bDtypeSize = 2UL;
        SetShape(64UL, 128UL, 0UL); // k=0：KEqZero模板的通过条件
        RebuildContext();
    }

    void SetShape(uint64_t m, uint64_t n, uint64_t k)
    {
        args_.mValue = m;
        args_.nValue = n;
        args_.kValue = k;
    }

    // 构造带确定性等级0、合法输入shape的TilingContext，保证context_非空且各守卫可通过
    void RebuildContext()
    {
        aShape_ = gert::StorageShape({static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.kValue)},
                                     {static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.kValue)});
        bShape_ = gert::StorageShape({static_cast<int64_t>(args_.kValue), static_cast<int64_t>(args_.nValue)},
                                     {static_cast<int64_t>(args_.kValue), static_cast<int64_t>(args_.nValue)});
        yShape_ = gert::StorageShape({static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.nValue)},
                                     {static_cast<int64_t>(args_.mValue), static_cast<int64_t>(args_.nValue)});
        tilingDataBuf_ = gert::TilingData::CreateCap(2048);
        workspaceBuf_ = gert::ContinuousVector::Create<size_t>(4096);
        holder_ = gert::TilingContextFaker()
                      .SetOpType("MatMulV3")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&aShape_, &bShape_})
                      .OutputShapes({&yShape_})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicLevelInfo(0)
                      .CompileInfo(&compileInfo_)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                      .TilingData(reinterpret_cast<gert::TilingData*>(tilingDataBuf_.get()))
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspaceBuf_.get()))
                      .Build();
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
    }

    std::unique_ptr<MatMulV3KEqZeroTilingForTest> CreateTiling()
    {
        cfg_ = std::make_unique<optiling::MatMulTilingCfg>(false, &compileInfo_, &args_, nullptr);
        return std::make_unique<MatMulV3KEqZeroTilingForTest>(context_, *cfg_);
    }

    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    fe::PlatFormInfos platformInfo_;
    optiling::MatmulV3CompileInfo compileInfo_;
    optiling::matmul_v3_advanced::MatMulV3Args args_;
    std::unique_ptr<optiling::MatMulTilingCfg> cfg_;
    gert::StorageShape aShape_;
    gert::StorageShape bShape_;
    gert::StorageShape yShape_;
    std::unique_ptr<uint8_t[]> tilingDataBuf_;
    std::unique_ptr<uint8_t[]> workspaceBuf_;
};

// L29-31：hasBias=true时IsCapable在kValue判断前即拦截，返回false
TEST_F(MatMulV3KEqZeroTilingDirectTest, IsCapable_HasBias_ReturnFalse)
{
    args_.hasBias = true;
    auto tiling = CreateTiling();
    EXPECT_FALSE(tiling->IsCapablePublic());
}

// L33-35：hasBias=false但kValue!=0时拦截，返回false
TEST_F(MatMulV3KEqZeroTilingDirectTest, IsCapable_KValueNonZero_ReturnFalse)
{
    SetShape(64UL, 128UL, 512UL);
    auto tiling = CreateTiling();
    EXPECT_FALSE(tiling->IsCapablePublic());
}

// L33-36：hasBias=false且kValue=0时通过，返回true
TEST_F(MatMulV3KEqZeroTilingDirectTest, IsCapable_KZeroNoBias_ReturnTrue)
{
    auto tiling = CreateTiling();
    EXPECT_TRUE(tiling->IsCapablePublic());
}

// L39-44：k=0通过前置后直调DoOpTiling，totalDataAmount=m*n=64*128=8192、usedCoreNum=aivNum=16
TEST_F(MatMulV3KEqZeroTilingDirectTest, DoOpTiling_KZero_SetsTotalDataAmountAndUsedCoreNum)
{
    auto tiling = CreateTiling();
    ASSERT_TRUE(tiling->IsCapablePublic());
    EXPECT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetRunInfoTotalDataAmount(), 8192UL); // 64 * 128
    EXPECT_EQ(tiling->GetRunInfoUsedCoreNum(), 16UL);       // compileInfo_.aivNum
}

// L46：GetNumBlocks直接返回compileInfo_.aivNum
TEST_F(MatMulV3KEqZeroTilingDirectTest, GetNumBlocks_ReturnsAivNum)
{
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetNumBlocksPublic(), 16UL); // compileInfo_.aivNum
}

// L48-59：tilingKeyObj为空走tmp分支，返回key与手工构造的K_EQUAL_ZERO组合一致；
// 并经静态位段解析核对model/batchModel/apiLevel三个关键域
TEST_F(MatMulV3KEqZeroTilingDirectTest, GetTilingKey_TilingKeyObjNull_TmpBranchKEqualZeroModel)
{
    auto tiling = CreateTiling();
    uint64_t expectedKey = optiling::matmul_v3_advanced::MatMulV3TilingKey()
                               .SetTrans(false, false)
                               .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
                               .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
                               .SetModel(MatMulV3Model::K_EQUAL_ZERO)
                               .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
                               .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
                               .GetTilingKey();
    EXPECT_EQ(tiling->GetTilingKeyPublic(), expectedKey);
    EXPECT_EQ(optiling::matmul_v3_advanced::MatMulV3TilingKey::GetModel(tiling->GetTilingKeyPublic()),
              MatMulV3Model::K_EQUAL_ZERO);
    EXPECT_EQ(optiling::matmul_v3_advanced::MatMulV3TilingKey::GetBatchModel(tiling->GetTilingKeyPublic()),
              MatMulV3BatchModel::BATCH_MODEL);
    EXPECT_EQ(optiling::matmul_v3_advanced::MatMulV3TilingKey::GetApiLevel(tiling->GetTilingKeyPublic()),
              MatMulV3ApiLevel::BASIC_LEVEL);
}

// L51：tilingKeyObj非空走*tilingKeyObj分支（三目另一臂），结果与tmp分支构造的key一致
TEST_F(MatMulV3KEqZeroTilingDirectTest, GetTilingKey_TilingKeyObjNonNull_UsesProvidedObj)
{
    optiling::matmul_v3_advanced::MatMulV3TilingKey keyObj;
    cfg_ = std::make_unique<optiling::MatMulTilingCfg>(false, &compileInfo_, &args_, &keyObj);
    auto tiling = std::make_unique<MatMulV3KEqZeroTilingForTest>(context_, *cfg_);
    uint64_t expectedKey = optiling::matmul_v3_advanced::MatMulV3TilingKey()
                               .SetTrans(false, false)
                               .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
                               .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
                               .SetModel(MatMulV3Model::K_EQUAL_ZERO)
                               .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
                               .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
                               .GetTilingKey();
    EXPECT_EQ(tiling->GetTilingKeyPublic(), expectedKey);
}

// L61-64：GetTilingData经GetTilingDataImpl<MatMulV3KEqZeroBasicTilingData>填充tilingData，
// 字段取自runInfo_（totalDataAmount=m*n、aivNum=usedCoreNum）
TEST_F(MatMulV3KEqZeroTilingDirectTest, GetTilingData_FillsKEqZeroBasicTilingData)
{
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->DoOpTilingPublic(), ge::GRAPH_SUCCESS);
    optiling::TilingResult tilingResult{};
    ASSERT_EQ(tiling->GetTilingDataPublic(tilingResult), ge::GRAPH_SUCCESS);
    ASSERT_NE(tilingResult.tilingData, nullptr);
    EXPECT_EQ(tilingResult.tilingDataSize, sizeof(MatMulV3KEqZeroBasicTilingData));
    auto* kEqZeroData = static_cast<MatMulV3KEqZeroBasicTilingData*>(tilingResult.tilingData.get());
    EXPECT_EQ(kEqZeroData->totalDataAmount, 8192U); // runInfo_.totalDataAmount = 64 * 128
    EXPECT_EQ(kEqZeroData->aivNum, 16U);            // runInfo_.usedCoreNum = compileInfo_.aivNum
}

// =================================================================================================
// MatMulV3Tiling 校验/提取阶段白盒直调用例（append-only 新增段）
// 覆盖目标（op_host/op_tiling/arch35/matmul_v3_tiling_advanced.cpp）：
//   L44-74    InvalidDtypeErrorMsg          hasBias 两臂（经 ValidateDtype L342 校验失败触达）
//   L76-99    InvalidDtypeErrorMsgForResv   hasBias 两臂（经 ValidateDtype L333 校验失败触达）
//   L231-241  ValidateFormat                DAV_RESV 架构存在非ND格式报错分支
//   L243-252  ValidateFormat                非DAV_RESV 时 a/out 为 FRACTAL_NZ 报错分支
//   L267-274  ValidateShape                 k=0 且带 bias 报错分支
//   L277-287  ValidateShape                 m/n/k 维度值越界报错分支
//   L290-295  ValidateShape                 输出c维度数<2报错分支
//   L308-314  ValidateBias                  bias末维与c末维不一致报错分支
//   L325-334  ValidateDtype                 DAV_RESV分支（匹配成功臂329-331 + 失败臂333）
//   L490-497  ExtractSliceDims              transposeA=true报错分支（经view输入路由自然触达）
//   L516-523  ExtractTransposeDims          维度数!=3防御分支（ExtractNonContiguousDims两个调用点
//                                           均有oriDimNum==3守卫，生产路由不可达，按本文件
//                                           TestSlice中L269-270直调先例覆盖）
//   L584-591  ExtractNormalDims             ND格式storage维度数<2报错分支
//   L593-600  ExtractNormalDims             FRACTAL_NZ格式storage维度数<4报错分支
//   L603-611  ExtractNormalDims             FRACTAL_NZ对齐校验失败报错分支
// 触达方式：沿用本仓既有范式——TilingContextFaker 构造上下文 + ForTest 派生类暴露 protected 入口
//   直调 GetShapeAttrsInfo（InitContext→CheckArgs→GetArgs 完整前置流程）及各提取/校验阶段；
//   私有 Extract* 函数对路由可达者经 ExtractMKN→ExtractNonContiguousDims 自然触达，
//   防御分支与点态验证按 L269-270 先例直调。
// =================================================================================================
class MatMulV3TilingAdvancedForTest : public optiling::matmul_v3_advanced::MatMulV3Tiling {
public:
    using MatMulV3Tiling::MatMulV3Tiling;

    ge::graphStatus InitContextPublic() { return InitContext(); }

    ge::graphStatus CheckArgsPublic() { return CheckArgs(); }

    ge::graphStatus GetShapeAttrsInfoPublic() { return GetShapeAttrsInfo(); }

    ge::graphStatus GetShapePublic() { return GetShape(); }

    ge::graphStatus ValidateFormatPublic() { return ValidateFormat(); }

    ge::graphStatus ValidateShapePublic() { return ValidateShape(); }

    ge::graphStatus ValidateBiasPublic() { return ValidateBias(); }

    ge::graphStatus ValidateDtypePublic() { return ValidateDtype(); }

    ge::graphStatus ExtractTransposePublic() { return ExtractTranspose(); }

    ge::graphStatus ExtractMKNPublic() { return ExtractMKN(); }

    void ExtractFormatPublic() { ExtractFormat(); }

    void ExtractDtypePublic() { ExtractDtype(); }

    bool GetIsSelfSlice() const { return isSelfSlice_; }

    uint64_t GetMValue() const { return args_.mValue; }

    uint64_t GetKValue() const { return args_.kValue; }

    uint64_t GetNValue() const { return args_.nValue; }

    int64_t GetKBValue() const { return kBValue_; }
};

class MatMulV3TilingAdvancedDirectTest : public testing::Test {
protected:
    void SetUp() override
    {
        platformInfo_.Init();
        compileInfo_.aicNum = 8UL;
        compileInfo_.aivNum = 16UL;
        compileInfo_.l1Size = 524288UL;
        compileInfo_.l0ASize = 65536UL;
        compileInfo_.l0BSize = 65536UL;
        compileInfo_.l0CSize = 262144UL;
        compileInfo_.l2Size = 134217728UL;
        compileInfo_.ubSize = 253952UL;
        compileInfo_.btSize = 4096UL;
        compileInfo_.npuArch = NpuArch::DAV_3510;
    }

    void SetNpuArch(NpuArch arch) { compileInfo_.npuArch = arch; }

    // 构造普通（非view）输入的TilingContext：a/b/bias/y 均经 InputShapes + NodeInputTd 提供
    void BuildContext(const gert::StorageShape& aShape, ge::DataType aDtype, ge::Format aOriFmt, ge::Format aFmt,
                      const gert::StorageShape& bShape, ge::DataType bDtype, ge::Format bOriFmt, ge::Format bFmt,
                      const gert::StorageShape& yShape, ge::DataType yDtype,
                      const gert::StorageShape* biasShape = nullptr, ge::DataType biasDtype = ge::DT_FLOAT16,
                      bool transA = false, bool transB = false)
    {
        aShape_ = aShape;
        bShape_ = bShape;
        yShape_ = yShape;
        yShapes_ = {yShape_};
        yRefs_ = {&yShapes_[0]};
        auto attrs = [transA, transB]() {
            return std::vector<std::pair<string, Ops::NN::AnyValue>>{
                {"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}};
        };
        if (biasShape != nullptr) {
            biasShape_ = *biasShape;
            holder_ = gert::TilingContextFaker()
                          .SetOpType("MatMulV3")
                          .NodeIoNum(3, 1)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&aShape_, &bShape_, &biasShape_})
                          .OutputShapes(yRefs_)
                          .NodeAttrs(attrs())
                          .NodeInputTd(0, aDtype, aOriFmt, aFmt)
                          .NodeInputTd(1, bDtype, bOriFmt, bFmt)
                          .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .CompileInfo(&compileInfo_)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                          .Build();
        } else {
            holder_ = gert::TilingContextFaker()
                          .SetOpType("MatMulV3")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes({&aShape_, &bShape_})
                          .OutputShapes(yRefs_)
                          .NodeAttrs(attrs())
                          .NodeInputTd(0, aDtype, aOriFmt, aFmt)
                          .NodeInputTd(1, bDtype, bOriFmt, bFmt)
                          .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .CompileInfo(&compileInfo_)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                          .Build();
        }
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
    }

    // 构造a为view（origin多维、storage一维压平）且b为普通2-D张量的TilingContext
    void BuildViewContext(std::initializer_list<int64_t> aOri, std::initializer_list<int64_t> aStorage,
                          std::initializer_list<int64_t> bShape, std::initializer_list<int64_t> yShape,
                          bool transA = false, bool transB = false)
    {
        aShape_ = gert::StorageShape(aOri, aStorage);
        bShape_ = gert::StorageShape(bShape, bShape);
        yShapes_ = {gert::StorageShape(yShape, yShape)};
        yRefs_ = {&yShapes_[0]};
        aTensor_ = std::make_unique<gert::TensorV2>(aShape_,
                                                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()),
                                                    TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr, nullptr);
        bTensor_ = std::make_unique<gert::TensorV2>(bShape_,
                                                    gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()),
                                                    TensorPlacement::kOnHost, ge::DT_FLOAT16, nullptr, nullptr);
        std::vector<gert::TensorV2*> inputTensors = {aTensor_.get(), bTensor_.get()};
        holder_ = gert::TilingContextFaker()
                      .SetOpType("MatMulV3")
                      .IrInstanceNum({1, 1}, {1})
                      .InputTensors(inputTensors)
                      .OutputShapes(yRefs_)
                      .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                                  {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"opImplMode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo_)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                      .Build();
        context_ = holder_.GetContext<gert::TilingContext>();
        ASSERT_NE(context_, nullptr);
    }

    std::unique_ptr<MatMulV3TilingAdvancedForTest> CreateTiling()
    {
        return std::make_unique<MatMulV3TilingAdvancedForTest>(context_);
    }

    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    fe::PlatFormInfos platformInfo_;
    optiling::MatmulV3CompileInfo compileInfo_;
    gert::StorageShape aShape_ = {{2, 3}, {2, 3}};
    gert::StorageShape bShape_ = {{3, 4}, {3, 4}};
    gert::StorageShape biasShape_ = {{4}, {4}};
    gert::StorageShape yShape_ = {{2, 4}, {2, 4}};
    std::vector<gert::StorageShape> yShapes_;
    std::vector<void*> yRefs_;
    std::unique_ptr<gert::TensorV2> aTensor_;
    std::unique_ptr<gert::TensorV2> bTensor_;
};

// 全部前置校验通过的对照用例：fp16/ND/2-D/无bias，GetShapeAttrsInfo 完整链路返回SUCCESS
TEST_F(MatMulV3TilingAdvancedDirectTest, GetShapeAttrsInfo_AllValid_Success)
{
    BuildContext(aShape_, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, bShape_, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND,
                 yShape_, DT_FLOAT16);
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateFormatPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateShapePublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateBiasPublic(), ge::GRAPH_SUCCESS); // 无bias直接通过臂
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_SUCCESS);
}

// L44-59（经L342触达）：非RESV架构下INT8不匹配支持列表且hasBias，走带bias报错臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_NonResv_InvalidDtypeWithBias_Fail)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND, y, DT_INT8, &bias,
                 DT_INT8);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    // 精确断言：失败点在dtype校验（前一阶段format/shape/bias均已通过）
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_FAILED);
}

// L60-73（经L342触达）：非RESV架构下INT8不匹配支持列表且无bias，走无bias报错臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_NonResv_InvalidDtypeNoBias_Fail)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND, y, DT_INT8);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_FAILED);
}

// L76-88（经L333触达）：RESV架构下FP32不满足全fp16要求且hasBias，走带bias报错臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_Resv_InvalidDtypeWithBias_Fail)
{
    SetNpuArch(NpuArch::DAV_RESV);
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, y, DT_FLOAT,
                 &bias, DT_FLOAT);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_FAILED);
}

// L89-97（经L333触达）：RESV架构下FP32不满足全fp16要求且无bias，走无bias报错臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_Resv_InvalidDtypeNoBias_Fail)
{
    SetNpuArch(NpuArch::DAV_RESV);
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND, y, DT_FLOAT);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_FAILED);
}

// L325-331：RESV架构全fp16（无bias）命中RESV专属支持列表匹配成功臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_Resv_AllFp16NoBias_Success)
{
    SetNpuArch(NpuArch::DAV_RESV);
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_SUCCESS);
}

// L325-331：RESV架构a/b/c/bias全fp16（4元组与RESV列表逐项相等）匹配成功臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_Resv_AllFp16WithBias_Success)
{
    SetNpuArch(NpuArch::DAV_RESV);
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16, &bias, DT_FLOAT16);
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_SUCCESS);
}

// L336-341对照：非RESV架构命中混合支持列表项（fp16,fp16,fp32,fp32）通过臂
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateDtype_NonResv_MixedSupportList_Success)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y, DT_FLOAT,
                 &bias, DT_FLOAT);
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateDtypePublic(), ge::GRAPH_SUCCESS);
}

// L231-241：RESV架构仅支持ND，a为FRACTAL_NZ时报错返回GRAPH_FAILED
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateFormat_ResvNonNd_Fail)
{
    SetNpuArch(NpuArch::DAV_RESV);
    gert::StorageShape a = {{2, 32}, {1, 16, 1, 32}};
    gert::StorageShape b = {{32, 4}, {32, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    // 精确断言：args_已提取完成，单独重放ValidateFormat
    EXPECT_EQ(tiling->ValidateFormatPublic(), ge::GRAPH_FAILED);
}

// L243-252：非RESV架构下a不允许FRACTAL_NZ，报错返回GRAPH_FAILED（b为NZ不在此校验范围）
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateFormat_NonResv_ANz_Fail)
{
    gert::StorageShape a = {{2, 32}, {1, 16, 1, 32}};
    gert::StorageShape b = {{32, 4}, {32, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateFormatPublic(), ge::GRAPH_FAILED);
}

// L267-274：k=0且带bias时报错返回GRAPH_FAILED（k两侧相等，通过第一道K一致性校验）
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateShape_KZeroWithBias_Fail)
{
    gert::StorageShape a = {{2, 0}, {2, 0}};
    gert::StorageShape b = {{0, 4}, {0, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16, &bias, DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateShapePublic(), ge::GRAPH_FAILED);
}

// L277-287：m=0不满足(0, INT32_MAX]时报错返回GRAPH_FAILED（k=0无bias可通过前道校验形成对照）
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateShape_InvalidDimValue_Fail)
{
    gert::StorageShape a = {{0, 3}, {0, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{0, 4}, {0, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateShapePublic(), ge::GRAPH_FAILED);
}

// L290-295：输出c维度数为1（<2）时报错返回GRAPH_FAILED
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateShape_CDimLessThanTwo_Fail)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{8}, {8}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateShapePublic(), ge::GRAPH_FAILED);
}

// L308-314：bias末维(5)与c末维(4)不一致时报错返回GRAPH_FAILED
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateBias_LastDimMismatch_Fail)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{5}, {5}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16, &bias, DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->ValidateBiasPublic(), ge::GRAPH_FAILED);
}

// L299-315对照：bias末维与c末维一致时ValidateBias通过
TEST_F(MatMulV3TilingAdvancedDirectTest, ValidateBias_LastDimMatch_Success)
{
    gert::StorageShape a = {{2, 3}, {2, 3}};
    gert::StorageShape b = {{3, 4}, {3, 4}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    gert::StorageShape bias = {{4}, {4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16, &bias, DT_FLOAT16);
    auto tiling = CreateTiling();
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ValidateBiasPublic(), ge::GRAPH_SUCCESS);
}

// L490-497：a为非连续slice view且transposeA=true时ExtractSliceDims报错；
// 路由链：ExtractMKN→ExtractNonContiguousDims（isASliceNonContiguous）→ExtractSliceDims
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractSliceDims_TransAForbidden_Fail)
{
    BuildViewContext({5, 7}, {70}, {7, 4}, {5, 4}, true, false);
    ASSERT_TRUE(context_->InputIsView(0)); // 路由前提：a为view tensor
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling->ExtractTransposePublic(), ge::GRAPH_SUCCESS); // args_.isATrans=true
    // 直调私有函数（沿用本文件TestSlice中L269-270先例），精确命中L490-497报错臂
    int64_t dims[2] = {0, 0};
    EXPECT_EQ(tiling->ExtractSliceDims(dims), ge::GRAPH_FAILED);
    // 完整前置流程同样在ExtractMKN处失败
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
}

// L498-507：a为非连续slice view且transposeA=false时，2-D origin走L503-504取m/k臂，
// 置isSelfSlice_=true并成功；后续校验全通过形成对照
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractSliceDims_TwoDimView_Success)
{
    BuildViewContext({5, 7}, {70}, {7, 4}, {5, 4}, false, false);
    ASSERT_TRUE(context_->InputIsView(0));
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling->ExtractTransposePublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    ASSERT_EQ(tiling->ExtractSliceDims(dims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dims[0], 5); // m = selfShape[0]
    EXPECT_EQ(dims[1], 7); // sliceK = selfShape[1]
    EXPECT_TRUE(tiling->GetIsSelfSlice());
    ASSERT_EQ(tiling->ExtractMKNPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetMValue(), 5UL);
    EXPECT_EQ(tiling->GetKValue(), 7UL);
    EXPECT_EQ(tiling->GetKBValue(), 7);
    EXPECT_EQ(tiling->GetNValue(), 4UL);
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
}

// L516-523：ExtractTransposeDims维度数!=3防御分支。生产路由下两个调用点均有oriDimNum==3
// 守卫（isATransposeNonContiguous要求selfDimNum==3、isBTransposeNonContiguous要求mat2DimNum==3），
// 该报错臂经路由不可达，故按本文件TestSlice中L269-270直调先例覆盖
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractTransposeDims_DimNotThree_Fail)
{
    BuildContext(aShape_, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, bShape_, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND,
                 yShape_, DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    EXPECT_EQ(tiling->ExtractTransposeDims(dims, 1), ge::GRAPH_FAILED); // b为2维
    EXPECT_EQ(tiling->ExtractTransposeDims(dims, 0), ge::GRAPH_FAILED); // a为2维
}

// L511-527对照：origin为3维时ExtractTransposeDims取末两维成功
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractTransposeDims_ThreeDim_Success)
{
    gert::StorageShape a = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape b = {{3, 4, 5}, {3, 4, 5}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, yShape_,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    ASSERT_EQ(tiling->ExtractTransposeDims(dims, 0), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dims[0], 3);
    EXPECT_EQ(dims[1], 4);
    ASSERT_EQ(tiling->ExtractTransposeDims(dims, 1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dims[0], 4);
    EXPECT_EQ(dims[1], 5);
}

// L584-591：ND格式下storage维度数<2时报错。origin保持2维避免L581-582读越界，
// 仅storage压平为1维且非view（路由前提ASSERT_FALSE(InputIsView)）
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractNormalDims_NdStorageDimLessThanTwo_Fail)
{
    gert::StorageShape a = {{5, 7}, {35}};
    gert::StorageShape b = {{7, 4}, {7, 4}};
    gert::StorageShape y = {{5, 4}, {5, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, y,
                 DT_FLOAT16);
    ASSERT_FALSE(context_->InputIsView(0)); // 非view：路由进ExtractNormalDims
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    EXPECT_EQ(tiling->ExtractNormalDims(context_->GetInputShape(0)->GetStorageShape(),
                                        context_->GetInputShape(0)->GetOriginShape(), 2UL, ge::FORMAT_ND, dims, "a"),
              ge::GRAPH_FAILED);
    // 经ExtractMKN的集成路由同样失败（同时覆盖ExtractNonContiguousDims L555-556失败日志臂）
    ASSERT_EQ(tiling->ExtractTransposePublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ExtractMKNPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
}

// L593-600：FRACTAL_NZ格式下storage维度数<4时报错（b为NZ不触发ValidateFormat拦截，
// 经ExtractMKN自然路由至b侧ExtractNormalDims）
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractNormalDims_NzStorageDimLessThanFour_Fail)
{
    gert::StorageShape a = {{2, 32}, {2, 32}};
    gert::StorageShape b = {{32, 4}, {16, 16, 2}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    EXPECT_EQ(
        tiling->ExtractNormalDims(context_->GetInputShape(1)->GetStorageShape(),
                                  context_->GetInputShape(1)->GetOriginShape(), 2UL, ge::FORMAT_FRACTAL_NZ, dims, "b"),
        ge::GRAPH_FAILED);
    // 按GetArgs阶段顺序先提取format/dtype（bFormat=NZ、bDtypeSize=2），再经ExtractMKN集成路由
    tiling->ExtractFormatPublic();
    tiling->ExtractDtypePublic();
    ASSERT_EQ(tiling->ExtractTransposePublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ExtractMKNPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
}

// L603-611：FRACTAL_NZ格式下oriShape对齐值与storageShape不一致时报错
// （storage[1]*storage[2]=256 != CeilAlign(m=32,16)=32）
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractNormalDims_NzAlignMismatch_Fail)
{
    gert::StorageShape a = {{2, 32}, {2, 32}};
    gert::StorageShape b = {{32, 4}, {1, 16, 16, 32}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    EXPECT_EQ(
        tiling->ExtractNormalDims(context_->GetInputShape(1)->GetStorageShape(),
                                  context_->GetInputShape(1)->GetOriginShape(), 2UL, ge::FORMAT_FRACTAL_NZ, dims, "b"),
        ge::GRAPH_FAILED);
    // 按GetArgs阶段顺序先提取format/dtype（bFormat=NZ、bDtypeSize=2），再经ExtractMKN集成路由
    tiling->ExtractFormatPublic();
    tiling->ExtractDtypePublic();
    ASSERT_EQ(tiling->ExtractTransposePublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->ExtractMKNPublic(), ge::GRAPH_FAILED);
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_FAILED);
}

// L601-613对照：NZ格式下对齐一致（storage[1]*storage[2]=32=CeilAlign(k=32,16)、
// storage[0]*storage[3]=16=CeilAlign(n=4,16)）时ExtractNormalDims通过，且全流程SUCCESS
TEST_F(MatMulV3TilingAdvancedDirectTest, ExtractNormalDims_NzAligned_Success)
{
    gert::StorageShape a = {{2, 32}, {2, 32}};
    gert::StorageShape b = {{32, 4}, {1, 16, 2, 16}};
    gert::StorageShape y = {{2, 4}, {2, 4}};
    BuildContext(a, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND, b, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, y,
                 DT_FLOAT16);
    auto tiling = CreateTiling();
    ASSERT_EQ(tiling->InitContextPublic(), ge::GRAPH_SUCCESS);
    int64_t dims[2] = {0, 0};
    ASSERT_EQ(
        tiling->ExtractNormalDims(context_->GetInputShape(1)->GetStorageShape(),
                                  context_->GetInputShape(1)->GetOriginShape(), 2UL, ge::FORMAT_FRACTAL_NZ, dims, "b"),
        ge::GRAPH_SUCCESS);
    EXPECT_EQ(dims[0], 32); // k
    EXPECT_EQ(dims[1], 4);  // n
    EXPECT_EQ(tiling->GetShapeAttrsInfoPublic(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->GetKValue(), 32UL);
    EXPECT_EQ(tiling->GetKBValue(), 32);
    EXPECT_EQ(tiling->GetNValue(), 4UL);
}

} // namespace
