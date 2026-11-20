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
#include <sstream>
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
#include "../../../op_host/op_tiling/arch35/transpose_batch_mat_mul_tiling_advanced.h"
#include "../../../op_host/op_tiling/arch35/transpose_batch_mat_mul_iterbatch_tiling.h"
#include "../../../op_kernel/arch35/transpose_batch_mat_mul_tiling_data.h"
#include "../../../op_host/op_tiling/transpose_batch_mat_mul_einsum_tiling.h"
#include "../../../op_host/op_tiling/transpose_batch_mat_mul_base_tiling.h"
#include "../../../op_host/op_tiling/transpose_batch_mat_mul_simplifiedkey.h"

using namespace std;
using namespace ge;

namespace {

bool IsDav3510CompileInfo(const string& compileInfo)
{
    return compileInfo.find(R"("npuArch": "dav3510")") != string::npos ||
           compileInfo.find(R"("npuArch":"dav3510")") != string::npos;
}

void GetPlatFormInfosDavAware(const string& compileInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                              map<string, string>& intrinsics, map<string, string>& socVersion)
{
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    if (IsDav3510CompileInfo(compileInfo)) {
        socVersion.clear();
        socVersion["NpuArch"] = "3510";
    }
}

static string TilingData2Str(const gert::TilingData* tiling_data)
{
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
    int32_t batch_split_factor = 1;

    ge::DataType input_dtype = DT_FLOAT16;
    ge::DataType y_dtype = DT_FLOAT16;
    std::initializer_list<int64_t> bias_shape;
    ge::Format bias_format;
    ge::Format bias_ori_format;
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

class TransposeBatchMatMulTilingRuntime : public testing::TestWithParam<TilingTestParam> {
    virtual void SetUp() {}

protected:
    static void SetUpTestCase() { std::cout << "TransposeBatchMatMulTilingRuntime SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TransposeBatchMatMulTilingRuntime TearDown" << std::endl; }
};

TEST_P(TransposeBatchMatMulTilingRuntime, general_cases)
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
    GetPlatFormInfosDavAware(param.compile_info, soc_infos, aicore_spec, intrinsics, soc_version);
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
                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(param.enable_hf32)},
                             {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(param.batch_split_factor)}})
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
    if (IsDav3510CompileInfo(param.compile_info)) {
        ASSERT_EQ(tiling_context->GetRawTilingData()->GetDataSize(), sizeof(TransposeBatchMatMulBasicTilingData));
        const auto* td = reinterpret_cast<const TransposeBatchMatMulBasicTilingData*>(
            tiling_context->GetRawTilingData()->GetData());
        const auto& bmm = td->matMulTilingData;
        const auto& mm = bmm.matMulTilingData;
        // arch35 golden记录基础结构中kernel使用的shape、分块、调度和计算模式。
        std::ostringstream basicFields;
        basicFields << mm.m << " " << mm.n << " " << mm.k << " " << mm.baseM << " " << mm.baseN << " " << mm.baseK
                    << " " << mm.mL1 << " " << mm.nL1 << " " << mm.kL1 << " " << bmm.batchDimAll << " "
                    << td->batchSplitFactor << " " << mm.mTailCnt << " " << mm.nTailCnt << " " << mm.mBaseTailSplitCnt
                    << " " << mm.nBaseTailSplitCnt << " " << mm.mTailMain << " " << mm.nTailMain << " "
                    << static_cast<uint32_t>(mm.mmadParam) << " " << static_cast<uint32_t>(mm.l2CacheDisable) << " "
                    << mm.sliceM << " " << mm.srcNdStride << " " << mm.innerBatch << " " << mm.splitSingleCoreKBase
                    << " " << static_cast<uint32_t>(mm.aswWindowLen) << " " << static_cast<uint32_t>(mm.l0cDB) << " ";
        EXPECT_TRUE(mm.l1BufferNum == 2U || mm.l1BufferNum == 4U);
        tiling_data_result = basicFields.str();
    }
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
     {2, 64, 128},
     {2, 128, 256},
     {2, 64, 256},
     false,
     0,
     0,
     24,
     0,
     "24 64 256 128 128 64 256 128 64 256 64 32 8 1 1 0 0 0 0 69632 16384 0 1 1 1 1 16 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 1 1 4 2 0 0 123 123 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 64 2 0 0 0 0 0 0 "
     "0 0 1 0 ",
     {0, 1, 2},
     {0, 1, 2},
     {0, 1, 2},
     1},
    {"TransposeBatchMatMul_basic_test02",
     "TransposeBatchMatMul",
     R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0},
       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
       "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
       "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
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
     {64, 2, 128},
     {2, 128, 256},
     {2, 64, 256},
     false,
     0,
     0,
     24,
     32,
     "24 64 256 128 128 64 256 128 64 256 64 32 8 1 1 0 0 0 0 69632 16384 0 1 1 1 1 16 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 1 1 4 2 0 0 213 123 0 0 0 0 0 0 0 0 0 0 0 0 24 2 2 2 1 1 1 1 1 1 1 1 1 2 2 2 0 0 64 2 0 0 0 0 0 0 "
     "0 0 1 0 ",
     {1, 0, 2},
     {0, 1, 2},
     {0, 1, 2},
     1},
    {"TransposeBatchMatMul_basic_test03",
     "TransposeBatchMatMul",
     R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0, "batch_split_factor":2},
       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
       "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
       "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
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
     {48, 128, 144},
     {128, 144, 512},
     {48, 128, 512},
     false,
     0,
     0,
     24,
     32,
     "24 48 512 144 144 48 512 144 48 512 32 80 8 1 1 0 0 0 0 104448 16384 0 1 1 1 1 40 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 1 1 3 2 0 0 213 123 1 0 0 0 0 0 0 0 16 144 0 0 24 128 128 128 1 1 1 1 1 1 1 1 1 128 128 128 0 0 "
     "48 128 0 0 0 0 0 0 0 0 1 0 ",
     {1, 0, 2},
     {0, 1, 2},
     {1, 0, 2},
     1},
    {"TransposeBatchMatMul_base_load_balance_01",
     "TransposeBatchMatMul",
     R"({"_pattern": "BatchMatMul", "attrs":{"transpose_a":false,"transpose_b":false,"offset_x":0,"enable_hf32":0, "batch_split_factor":2},
       "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":33554432},"binary_mode_flag":true,
       "block_dim":{"CORE_NUM":24},"corerect_range_flag":null,"dynamic_mode":"dynamic_mknb", "fused_double_operand_num": 0,
       "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 196608, "L2_SIZE": 201326592, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24, "socVersion": "Ascend910B" },
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
     {1, 8, 4096},
     {8, 4096, 1024},
     {1, 8, 1024},
     false,
     0,
     0,
     24,
     32,
     "24 1 1024 4096 4096 1 384 4096 16 384 32 256 8 1 1 0 0 0 0 393216 16384 0 1 1 1 1 128 4 0 0 2 2 1 0 0 0 0 0 0 0 "
     "0 0 0 0 0 0 0 0 0 0 1 1 1 8 0 0 213 123 0 0 0 0 0 0 0 0 0 0 0 0 24 8 8 8 1 1 1 1 1 1 1 1 1 8 8 8 0 0 1 8 0 0 0 0 "
     "0 0 0 0 1 "
     "0 ",
     {1, 0, 2},
     {0, 1, 2},
     {1, 0, 2},
     1}};

static TilingTestParam dav3510_cases_params[] = {
    {"TransposeBatchMatMul_dav3510_test_iterbatch_1",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 128, 512},
     {32, 32, 512},
     false,
     0,
     0,
     32,
     5251073UL,
     "32 512 128 32 512 32 32 512 128 32 1 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {1, 0, 2},
     {0, 1, 2},
     {1, 0, 2},
     1},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_2",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 512, 128},
     {32, 32, 512},
     false,
     0,
     0,
     32,
     5251089UL,
     "32 512 128 32 512 32 32 512 128 32 1 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {1, 0, 2},
     {0, 2, 1},
     {1, 0, 2},
     1},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_3",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 128, 512},
     {4, 32, 4096},
     false,
     0,
     0,
     32,
     5251329UL,
     "32 512 128 32 512 32 32 512 128 32 4 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {1, 0, 2},
     {0, 1, 2},
     {1, 0, 2},
     4},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_4",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 512, 128},
     {32, 32, 4096},
     false,
     0,
     0,
     32,
     5251345UL,
     "32 512 128 32 512 32 32 512 128 32 4 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {1, 0, 2},
     {0, 2, 1},
     {1, 0, 2},
     4},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_5",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 128, 512},
     {32, 32, 512},
     false,
     0,
     0,
     32,
     4202496UL,
     "32 512 128 32 512 32 32 512 128 32 1 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {0, 1, 2},
     {0, 1, 2},
     {1, 0, 2},
     1},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_6",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 512, 128},
     {32, 32, 512},
     false,
     0,
     0,
     32,
     4202512UL,
     "32 512 128 32 512 32 32 512 128 32 1 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {0, 1, 2},
     {0, 2, 1},
     {1, 0, 2},
     1},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_7",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 128, 512},
     {32, 32, 4096},
     false,
     0,
     0,
     32,
     4202752UL,
     "32 512 128 32 512 32 32 512 128 32 4 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {0, 1, 2},
     {0, 1, 2},
     {1, 0, 2},
     4},
    {"TransposeBatchMatMul_dav3510_test_iterbatch_8",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {32, 32, 128},
     {32, 512, 128},
     {32, 32, 4096},
     false,
     0,
     0,
     32,
     4202768UL,
     "32 512 128 32 512 32 32 512 128 32 4 1 1 1 1 0 0 0 0 32 1 0 1 4 2 ",
     {0, 1, 2},
     {0, 2, 1},
     {1, 0, 2},
     4},
    {"TransposeBatchMatMul_dav3510_base_load_balance_1",
     "TransposeBatchMatMul",
     R"({"_pattern": "MatMul", "attrs":{"transpose_a":false,"transpose_b":false, "offset_x":0, "enable_hf32":0},
      "binary_attrs":{"bias_flag":false, "nd_flag":true, "split_k_flag":false, "zero_flag":false, "weight_nz": false, "l2_size":134217728},"binary_mode_flag":true,
      "block_dim":{"CORE_NUM":32},"corerect_range_flag":null,"dynamic_mode":"dynamic_mkn", "fused_double_operand_num": 0,
      "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown", "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false, "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_l12bt": true, "Intrinsic_data_move_out2l1_nd2nz": true, "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32, "npuArch": "dav3510" },
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
     {1, 4, 4096},
     {4, 4096, 1024},
     {1, 4, 1024},
     false,
     0,
     0,
     32,
     5251073UL,
     "1 1024 4096 16 128 128 16 128 256 4 1 1 1 1 1 0 0 0 0 16 1 0 1 4 2 ",
     {1, 0, 2},
     {0, 1, 2},
     {1, 0, 2},
     1}};

INSTANTIATE_TEST_CASE_P(TransposeBatchMatMul910B, TransposeBatchMatMulTilingRuntime,
                        testing::ValuesIn(ascend910B_cases_params));
INSTANTIATE_TEST_CASE_P(TransposeBatchMatMulDav3510, TransposeBatchMatMulTilingRuntime,
                        testing::ValuesIn(dav3510_cases_params));

// ================== CheckScale unit tests ==================
// CheckScale is a protected method of TransposeBatchMatMulTiling, made accessible
// via the `protected public` macro in kernel_run_context_facker.h.
using optiling::transpose_batch_mat_mul_advanced::TransposeBatchMatMulTiling;

namespace {

// Build a TilingContext for TransposeBatchMatMul with given shapes, dtypes and batchSplitFactor.
// Returns the context via holder (holder must outlive the context).
struct CheckScaleContext {
    gert::KernelRunContextHolder holder;
    gert::TilingContext* ctx = nullptr;
    std::vector<gert::StorageShape> outShapes;
    optiling::MatmulV3CompileInfo compileInfo;
    std::unique_ptr<unsigned char[]> tilingData;
    std::unique_ptr<unsigned char[]> wsSize;
    fe::PlatFormInfos platformInfo;

    CheckScaleContext(const std::initializer_list<int64_t>& x1Shape, const std::initializer_list<int64_t>& x2Shape,
                      const std::initializer_list<int64_t>& yShape, ge::DataType dtype = DT_FLOAT16,
                      int64_t batchSplitFactor = 1)
        : tilingData(gert::TilingData::CreateCap(2048)), wsSize(gert::ContinuousVector::Create<size_t>(4096))
    {
        gert::StorageShape x1ShapeSs = {x1Shape, x1Shape};
        gert::StorageShape x2ShapeSs = {x2Shape, x2Shape};
        outShapes.emplace_back(yShape, yShape);
        std::vector<void*> outShapeRefs = {static_cast<void*>(&outShapes[0])};

        platformInfo.Init();

        holder = gert::TilingContextFaker()
                     .SetOpType("TransposeBatchMatMul")
                     .NodeIoNum(2, 1)
                     .IrInstanceNum({1, 1})
                     .InputShapes({&x1ShapeSs, &x2ShapeSs})
                     .OutputShapes(outShapeRefs)
                     .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 2})},
                                 {"perm_x2", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 2})},
                                 {"perm_y", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 2})},
                                 {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                 {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(batchSplitFactor)}})
                     .NodeInputTd(0, dtype, FORMAT_ND, FORMAT_ND)
                     .NodeInputTd(1, dtype, FORMAT_ND, FORMAT_ND)
                     .NodeOutputTd(0, dtype, FORMAT_ND, FORMAT_ND)
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                     .TilingData(tilingData.get())
                     .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsSize.get()))
                     .Build();
        ctx = holder.GetContext<gert::TilingContext>();
    }
};

} // namespace

TEST(TransposeBatchMatMulCheckScaleTest, ScalePassFp16)
{
    CheckScaleContext c({2, 128, 256}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16, 1);
    TransposeBatchMatMulTiling tiling(c.ctx);
    tiling.args_.opName = "TransposeBatchMatMul";
    // bPerm = [0,1,2], so b = bShape[0] = 2, n = bShape[2] = 256, b*n = 512
    gert::Shape scaleShape{512, 1};
    EXPECT_EQ(tiling.CheckScale(scaleShape), ge::GRAPH_SUCCESS);
}

TEST(TransposeBatchMatMulCheckScaleTest, ScaleBatchSplitFactorNotOne)
{
    CheckScaleContext c({2, 128, 256}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16, 2);
    TransposeBatchMatMulTiling tiling(c.ctx);
    tiling.args_.opName = "TransposeBatchMatMul";
    gert::Shape scaleShape{512, 1};
    EXPECT_EQ(tiling.CheckScale(scaleShape), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulCheckScaleTest, ScaleShapeMismatch)
{
    CheckScaleContext c({2, 128, 256}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16, 1);
    TransposeBatchMatMulTiling tiling(c.ctx);
    tiling.args_.opName = "TransposeBatchMatMul";
    // b*n = 512, but scale dim0 = 256 -> mismatch
    gert::Shape scaleShape{256, 1};
    EXPECT_EQ(tiling.CheckScale(scaleShape), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulCheckScaleTest, ScaleNonFp16)
{
    CheckScaleContext c({2, 128, 256}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT, 1);
    TransposeBatchMatMulTiling tiling(c.ctx);
    tiling.args_.opName = "TransposeBatchMatMul";
    gert::Shape scaleShape{512, 1};
    EXPECT_EQ(tiling.CheckScale(scaleShape), ge::GRAPH_FAILED);
}

// ================== EinsumTiling destructor test ==================
TEST(TransposeBatchMatMulEinsumTilingTest, Destructor)
{
    {
        auto* tiling = new optiling::transpose_batch_mat_mul::TransposeBatchMatMulEinsumTiling(nullptr);
        delete tiling;
    }
    {
        optiling::transpose_batch_mat_mul::TransposeBatchMatMulEinsumTiling tiling(nullptr);
    }
}

// ================== GenSimplifiedKey test ==================
TEST(TransposeBatchMatMulGenSimplifiedKeyTest, ContextNull)
{
    char key[128] = {0};
    auto ret = optiling::transpose_batch_matmul::GenSimplifiedKey(nullptr, key);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulGenSimplifiedKeyTest, KeyNull)
{
    auto tilingData = gert::TilingData::CreateCap(64);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    gert::StorageShape inputShape = {{2, 64, 128}, {2, 64, 128}};
    gert::StorageShape outputShape = {{2, 64, 64}, {2, 64, 64}};
    gert::KernelRunContextHolder holder;
    optiling::MatmulV3CompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    holder = gert::TilingContextFaker()
                 .SetOpType("TransposeBatchMatMul")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&inputShape, &inputShape})
                 .OutputShapes({&outputShape})
                 .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .CompileInfo(&compileInfo)
                 .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                 .TilingData(tilingData.get())
                 .Workspace(wsSize)
                 .Build();
    auto ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx, nullptr);
    auto ret = optiling::transpose_batch_matmul::GenSimplifiedKey(ctx, nullptr);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// ================== Advanced tiling failure path tests ==================

namespace {

// Build a TilingContext for advanced tiling tests with custom perm / dtype / optional inputs.
// Returns the context via holder (holder must outlive the context).
struct AdvancedContext {
    gert::KernelRunContextHolder holder;
    gert::TilingContext* ctx = nullptr;
    std::vector<gert::StorageShape> outShapes;
    optiling::MatmulV3CompileInfo compileInfo;
    std::unique_ptr<unsigned char[]> tilingData;
    std::unique_ptr<unsigned char[]> wsSize;
    fe::PlatFormInfos platformInfo;
    // Store optional shapes as members to ensure they outlive the faker build
    gert::StorageShape x1ShapeSs;
    gert::StorageShape x2ShapeSs;
    gert::StorageShape biasShape_ = {{0}, {0}};
    gert::StorageShape scaleSs_ = {{0}, {0}};

    AdvancedContext(const std::initializer_list<int64_t>& x1Shape, const std::initializer_list<int64_t>& x2Shape,
                    const std::initializer_list<int64_t>& yShape, ge::DataType dtype = DT_FLOAT16,
                    const std::initializer_list<int64_t>& perm_x1 = {1, 0, 2},
                    const std::initializer_list<int64_t>& perm_x2 = {0, 1, 2},
                    const std::initializer_list<int64_t>& perm_y = {1, 0, 2}, int64_t batchSplitFactor = 1,
                    bool hasBias = false, bool hasScale = false, const std::initializer_list<int64_t>& scaleShape = {0},
                    const std::initializer_list<int64_t>& biasShape = {}, ge::DataType biasType = DT_FLOAT,
                    ge::Format bFormat = FORMAT_ND)
        : tilingData(gert::TilingData::CreateCap(2048)), wsSize(gert::ContinuousVector::Create<size_t>(4096))
    {
        x1ShapeSs = {x1Shape, x1Shape};
        x2ShapeSs = {x2Shape, x2Shape};
        if (bFormat == FORMAT_FRACTAL_NZ) {
            const std::vector<int64_t> dims(x2Shape);
            x2ShapeSs = {x2Shape, {dims[0], (dims[2] + 15) / 16, (dims[1] + 15) / 16, 16, 16}};
        }
        outShapes.emplace_back(yShape, yShape);
        std::vector<void*> outShapeRefs = {static_cast<void*>(&outShapes[0])};

        platformInfo.Init();

        // Build input shapes: x1, x2, [bias], [scale]
        std::vector<gert::StorageShape*> inputShapes = {&x1ShapeSs, &x2ShapeSs};
        int inputCount = 2;
        if (hasBias) {
            biasShape_ = biasShape.size() == 0 ? gert::StorageShape{yShape, yShape} :
                                                 gert::StorageShape{biasShape, biasShape};
            inputShapes.push_back(&biasShape_);
            inputCount++;
        }
        if (hasScale) {
            scaleSs_ = {{scaleShape}, {scaleShape}};
            inputShapes.push_back(&scaleSs_);
            inputCount++;
        }
        std::vector<uint32_t> instanceNum = {1, 1, hasBias ? 1U : 0U, hasScale ? 1U : 0U};

        auto faker = gert::TilingContextFaker();
        faker.SetOpType("TransposeBatchMatMul")
            .NodeIoNum(inputCount, 1)
            .IrInstanceNum(instanceNum)
            .InputShapes(inputShapes)
            .OutputShapes(outShapeRefs)
            .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_x1)},
                        {"perm_x2", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_x2)},
                        {"perm_y", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_y)},
                        {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                        {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(batchSplitFactor)}})
            .NodeInputTd(0, dtype, FORMAT_ND, FORMAT_ND)
            .NodeInputTd(1, dtype, FORMAT_ND, bFormat)
            .NodeOutputTd(0, hasScale ? DT_INT8 : dtype, FORMAT_ND, FORMAT_ND)
            .CompileInfo(&compileInfo)
            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
            .TilingData(tilingData.get())
            .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsSize.get()));
        if (hasBias) {
            faker.NodeInputTd(2, biasType, FORMAT_ND, FORMAT_ND);
        }
        if (hasScale) {
            faker.NodeInputTd(inputCount - 1, DT_INT64, FORMAT_ND, FORMAT_ND);
        }
        holder = faker.Build();
        ctx = holder.GetContext<gert::TilingContext>();
    }
};

} // namespace

// 覆盖 advanced tiling 221,225: invalid permX1
TEST(TransposeBatchMatMulAdvancedTilingTest, InvalidPermX1)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{0, 2, 1}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2});
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 231,235: invalid permX2
TEST(TransposeBatchMatMulAdvancedTilingTest, InvalidPermX2)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{1, 0, 2}, /*perm_y=*/{1, 0, 2});
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 240,244: invalid permY
TEST(TransposeBatchMatMulAdvancedTilingTest, InvalidPermY)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{2, 1, 0});
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 132,141: invalid dtype (not in supported list)
TEST(TransposeBatchMatMulAdvancedTilingTest, InvalidDtype)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_INT32);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 335,337: bias exists
TEST(TransposeBatchMatMulAdvancedTilingTest, BiasExists)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                      /*batchSplitFactor=*/1, /*hasBias=*/true);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 340,342: scale exists with CheckScale returning failure
TEST(TransposeBatchMatMulAdvancedTilingTest, ScaleExistsInvalid)
{
    // scale shape {64} does not match batch*n = 2*256 = 512
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                      /*batchSplitFactor=*/1, /*hasBias=*/false, /*hasScale=*/true,
                      /*scaleShape=*/{64});
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 256,261: batchSplitFactor not divisible by batch
TEST(TransposeBatchMatMulAdvancedTilingTest, InvalidBatchSplitFactor)
{
    // batch = 2 (from perm_x1={1,0,2}, x1[0]=2), batchSplitFactor=3, 2%3 != 0
    AdvancedContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                      /*batchSplitFactor=*/3);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// 覆盖 advanced tiling 395,400: batchA != batchB (via GetBatchInfo in DoTiling)
TEST(TransposeBatchMatMulAdvancedTilingTest, BatchMismatch)
{
    // x1 batch = 64 (from perm_x1={1,0,2}, aShape[1]=64), x2 batch = 4 (from perm_x2={0,1,2}, bShape[0]=4)
    // GetBatchInfo detects batchA(64) != batchB(4) and returns GRAPH_FAILED
    AdvancedContext c({2, 64, 128}, {4, 128, 256}, {2, 64, 256}, DT_FLOAT16);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// ================== Base tiling failure path tests ==================
// These test error paths in transpose_batch_mat_mul_base_tiling.cpp
// by directly instantiating TransposeBatchMatMulBaseTiling and calling
// protected methods (made public via `#define protected public`).

namespace {

// Minimal context builder for base tiling tests
struct BaseTilingContext {
    gert::KernelRunContextHolder holder;
    gert::TilingContext* ctx = nullptr;
    std::vector<gert::StorageShape> outShapes;
    optiling::MatmulV3CompileInfo compileInfo;
    std::unique_ptr<unsigned char[]> tilingData;
    std::unique_ptr<unsigned char[]> wsSize;
    gert::StorageShape biasShape_ = {{0}, {0}};
    gert::StorageShape scaleSs_ = {{0}, {0}};
    fe::PlatFormInfos platformInfo;

    BaseTilingContext(const std::initializer_list<int64_t>& x1Shape, const std::initializer_list<int64_t>& x2Shape,
                      const std::initializer_list<int64_t>& yShape, ge::DataType dtype = DT_FLOAT16,
                      const std::initializer_list<int64_t>& perm_x1 = {1, 0, 2},
                      const std::initializer_list<int64_t>& perm_x2 = {0, 1, 2},
                      const std::initializer_list<int64_t>& perm_y = {1, 0, 2}, int64_t batchSplitFactor = 1,
                      bool hasBias = false, bool hasScale = false,
                      const std::initializer_list<int64_t>& scaleShape = {0})
        : tilingData(gert::TilingData::CreateCap(2048)), wsSize(gert::ContinuousVector::Create<size_t>(4096))
    {
        gert::StorageShape x1ShapeSs = {x1Shape, x1Shape};
        gert::StorageShape x2ShapeSs = {x2Shape, x2Shape};
        outShapes.emplace_back(yShape, yShape);
        std::vector<void*> outShapeRefs = {static_cast<void*>(&outShapes[0])};

        platformInfo.Init();

        std::vector<gert::StorageShape*> inputShapes = {&x1ShapeSs, &x2ShapeSs};
        int inputCount = 2;
        if (hasBias) {
            biasShape_ = {{yShape}, {yShape}};
            inputShapes.push_back(&biasShape_);
            inputCount++;
        }
        if (hasScale) {
            scaleSs_ = {{scaleShape}, {scaleShape}};
            inputShapes.push_back(&scaleSs_);
            inputCount++;
        }
        std::vector<uint32_t> instanceNum(inputCount, 1);

        holder = gert::TilingContextFaker()
                     .SetOpType("TransposeBatchMatMul")
                     .NodeIoNum(inputCount, 1)
                     .IrInstanceNum(instanceNum)
                     .InputShapes(inputShapes)
                     .OutputShapes(outShapeRefs)
                     .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_x1)},
                                 {"perm_x2", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_x2)},
                                 {"perm_y", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(perm_y)},
                                 {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                 {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(batchSplitFactor)}})
                     .NodeInputTd(0, dtype, FORMAT_ND, FORMAT_ND)
                     .NodeInputTd(1, dtype, FORMAT_ND, FORMAT_ND)
                     .NodeOutputTd(0, dtype, FORMAT_ND, FORMAT_ND)
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                     .TilingData(tilingData.get())
                     .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsSize.get()))
                     .Build();
        ctx = holder.GetContext<gert::TilingContext>();
    }
};

} // namespace

// 覆盖 base_tiling 619-620: unsupported perm (transA_ not 213 or 123)
TEST(TransposeBatchMatMulBaseTilingTest, InvalidPerm)
{
    BaseTilingContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                        /*perm_x1=*/{2, 0, 1}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2});
    // transA_ = (2+1)(0+1)(1+1) = 312, not 213 or 123
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 522-523: illegal m/k/n values (m=0)
TEST(TransposeBatchMatMulBaseTilingTest, InvalidDim)
{
    BaseTilingContext c({0, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 491: unsupported dtype
TEST(TransposeBatchMatMulBaseTilingTest, InvalidDtype)
{
    BaseTilingContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_INT32);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 422-423: scale with batch*n >= 65536 (kSupportedInnerAxis)
TEST(TransposeBatchMatMulBaseTilingTest, ScaleTooLarge)
{
    // batch*n = 64*256 = 16384, but scale shape dim0 = 65536 >= 65536 → fails
    BaseTilingContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                        /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                        /*batchSplitFactor=*/1, /*hasBias=*/false, /*hasScale=*/true,
                        /*scaleShape=*/{65536});
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 559-563: bias shape n does not match nOriValue
TEST(TransposeBatchMatMulBaseTilingTest, BiasNNotMatch)
{
    // biasValue = biasShape[lastDim-1] = 64 (from biasShape={2,64,256}), nOriValue = 256
    // 64 != 256 → GRAPH_FAILED via GetShapeBias
    BaseTilingContext c({2, 64, 128}, {2, 128, 256}, {2, 64, 256}, DT_FLOAT16,
                        /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                        /*batchSplitFactor=*/1, /*hasBias=*/true);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 496-520: batch 一致时无 bias 的非法 dtype 组合
TEST(TransposeBatchMatMulBaseTilingTest, InvalidDtypeWithoutBias)
{
    BaseTilingContext c({2, 64, 128}, {64, 128, 256}, {2, 64, 256}, DT_INT32);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 500-502: batch 一致时有 bias 的非法 dtype 组合
TEST(TransposeBatchMatMulBaseTilingTest, InvalidDtypeWithBias)
{
    BaseTilingContext c({2, 64, 128}, {64, 128, 256}, {2, 64, 256}, DT_INT32,
                        /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                        /*batchSplitFactor=*/1, /*hasBias=*/true);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// 覆盖 base_tiling 590-592: batch 一致时 bias 最后一维不等于 n
TEST(TransposeBatchMatMulBaseTilingTest, BiasNNotMatchWithMatchedBatch)
{
    BaseTilingContext c({2, 64, 128}, {64, 128, 256}, {2, 64, 128}, DT_FLOAT16,
                        /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2},
                        /*batchSplitFactor=*/1, /*hasBias=*/true);
    using namespace optiling::transpose_batch_mat_mul;
    TransposeBatchMatMulBaseTiling tiling(c.ctx);
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
}

// ================== Einsum mode tiling test ==================
TEST(TransposeBatchMatMulEinsumTest, DoTilingEinsumMode)
{
    // Use Ascend910B (non-advanced) platform to trigger base tiling dispatcher
    TilingTestParam param = ascend910B_cases_params[0];
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
    GetPlatFormInfosDavAware(param.compile_info, soc_infos, aicore_spec, intrinsics, soc_version);
    aicore_spec["cube_freq"] = "1800";

    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);
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
    // Use perm_x2={0,2,1} to trigger IsPpMatmulEinsumMode (PermDecode returns 132)
    holder = gert::TilingContextFaker()
                 .SetOpType(param.op_type.c_str())
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&x1_shape, &x2_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                             {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2, 1})},
                             {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                             {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                             {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                 .NodeInputTd(0, param.input_dtype, param.x1_ori_format, param.x1_format)
                 .NodeInputTd(1, param.input_dtype, param.x2_ori_format, param.x2_format)
                 .NodeOutputTd(0, param.y_dtype, param.y_ori_format, param.y_format)
                 .CompileInfo(&compile_info)
                 .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                 .TilingData(tiling_data.get())
                 .Workspace(ws_size)
                 .Build();

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(param.op_type.c_str())->tiling;
    ASSERT_NE(tiling_func, nullptr);
    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    uint64_t tiling_key = tiling_context->GetTilingKey();
    // Expected: BATCH_SPLIT=0(2b), PP_MAT_MUL_EINSUM_MODE=1(2b), PERM_X1=2(4b), PERM_X2=1(4b)
    // tiling_key = 0 | (1<<2) | (2<<4) | (1<<8) = 292
    EXPECT_EQ(tiling_key, 292UL);
    EXPECT_GT(tiling_context->GetBlockDim(), static_cast<uint32_t>(0));
}

TEST(TransposeBatchMatMulGenSimplifiedKeyTest, ValidKey)
{
    auto tilingData = gert::TilingData::CreateCap(64);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    gert::StorageShape inputShape = {{2, 64, 128}, {2, 64, 128}};
    gert::StorageShape outputShape = {{2, 64, 64}, {2, 64, 64}};
    gert::KernelRunContextHolder holder;
    optiling::MatmulV3CompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    holder = gert::TilingContextFaker()
                 .SetOpType("TransposeBatchMatMul")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&inputShape, &inputShape})
                 .OutputShapes({&outputShape})
                 .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                 .CompileInfo(&compileInfo)
                 .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                 .TilingData(tilingData.get())
                 .Workspace(wsSize)
                 .Build();
    auto ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx, nullptr);
    char key[128] = {0};
    auto ret = optiling::transpose_batch_matmul::GenSimplifiedKey(ctx, key);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// ================== IterBatch tiling strategy tests ==================
// 参照batch_matmul_v3 iterbatch准入: 等batch + batchC>aicNum + 单batch容量 + L0装载
namespace {
using TBMMKeyBuilder = optiling::transpose_batch_mat_mul_advanced::TBMMTilingKey;

void SetIterBatchCompileInfo(optiling::MatmulV3CompileInfo& compileInfo)
{
    compileInfo.npuArch = NpuArch::DAV_3510;
    compileInfo.aicNum = 20UL;
    compileInfo.aivNum = 40UL;
    compileInfo.ubSize = 196608UL;
    compileInfo.l1Size = 524288UL;
    compileInfo.l2Size = 201326592UL;
    compileInfo.l0CSize = 131072UL;
    compileInfo.l0ASize = 65536UL;
    compileInfo.l0BSize = 65536UL;
    compileInfo.btSize = 65536UL;
    compileInfo.cubeFreq = 1800;
}

const BatchMatMulV3IterBatchBasicTilingData* GetTbmmIterBatchTilingData(gert::TilingContext* ctx)
{
    return reinterpret_cast<const BatchMatMulV3IterBatchBasicTilingData*>(ctx->GetRawTilingData()->GetData());
}

const TransposeBatchMatMulBasicTilingData* GetTbmmBasicTilingData(gert::TilingContext* ctx)
{
    return reinterpret_cast<const TransposeBatchMatMulBasicTilingData*>(ctx->GetRawTilingData()->GetData());
}
} // namespace

// 等batch(b=64>aicNum=20) + permY=[0,1,2](bmn) + 连续输入: 准入iterbatch, 校验iterBatchL1/L0与baseM/N/K
TEST(TransposeBatchMatMulIterBatchTilingTest, ContiguousBmnAdmitted)
{
    AdvancedContext c({64, 64, 128}, {64, 128, 96}, {64, 64, 96}, DT_FLOAT16,
                      /*perm_x1=*/{0, 1, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t expectedKey = TBMMKeyBuilder()
                               .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                               .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                               .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                               .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                               .SetModel(TBMMModel::TBMM_ITER_BATCH)
                               .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                               .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    auto* td = GetTbmmIterBatchTilingData(c.ctx);
    ASSERT_EQ(c.ctx->GetRawTilingData()->GetDataSize(), sizeof(BatchMatMulV3IterBatchBasicTilingData));
    // 单batch 40KB(fp16), iterBatchL1=min(FloorDiv(L1/2,40KB)=6, CeilDiv(64,20)=4)=4;
    // iterBatchL0=min(L0A=2,L0B=1,L0C=2)=1
    EXPECT_EQ(td->iterBatchL1, 4U);
    EXPECT_EQ(td->iterBatchL0, 1U);
    EXPECT_EQ(td->baseM, 64U);
    EXPECT_EQ(td->baseN, 96U);
    EXPECT_EQ(td->baseK, 128U);
    EXPECT_EQ(c.ctx->GetBlockDim(), 20U);
}

// MBK A沿用BMM分块计算，由Kernel适配GM布局。
TEST(TransposeBatchMatMulIterBatchTilingTest, MbkABmnIterBatchAdmitted)
{
    AdvancedContext c({32, 64, 128}, {64, 128, 96}, {64, 32, 96}, DT_FLOAT16,
                      /*perm_x1=*/{1, 0, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t expectedKey = TBMMKeyBuilder()
                               .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                               .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                               .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                               .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                               .SetModel(TBMMModel::TBMM_ITER_BATCH)
                               .SetNonContiguousType(TBMMNonContiguousType::A_NON_CONTIGUOUS)
                               .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
}

// 连续MBN输出也参与iterbatch策略选择。
TEST(TransposeBatchMatMulIterBatchTilingTest, MbnOutputAdmitted)
{
    AdvancedContext c({64, 64, 128}, {64, 128, 96}, {64, 64, 96}, DT_FLOAT16,
                      /*perm_x1=*/{0, 1, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{1, 0, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t expectedKey = TBMMKeyBuilder()
                               .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                               .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                               .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                               .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                               .SetModel(TBMMModel::TBMM_ITER_BATCH)
                               .SetNonContiguousType(TBMMNonContiguousType::C_TRANSPOSE)
                               .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, MbnWeightNzFallsBackToBasic)
{
    AdvancedContext c({16, 64, 32}, {64, 129, 32}, {2, 16, 4128}, DT_BF16, {1, 0, 2}, {0, 2, 1}, {1, 0, 2}, 2, false,
                      false, {0}, {1, 129}, DT_FLOAT, FORMAT_FRACTAL_NZ);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(c.ctx->GetTilingKey(), TBMMKeyBuilder()
                                         .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                                         .SetPermX2(TBMMPermX2::PERM_X2_0_2_1)
                                         .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_TRUE)
                                         .SetModel(TBMMModel::TBMM_BASIC)
                                         .SetNonContiguousType(TBMMNonContiguousType::AC_TRANSPOSE)
                                         .GetTilingKey());
    const auto* td = GetTbmmBasicTilingData(c.ctx);
    EXPECT_EQ(td->batchSplitFactor, 2U);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, MbnNdSplitWithoutBiasUsesBasic)
{
    AdvancedContext c({64, 16, 32}, {64, 129, 32}, {2, 16, 4128}, DT_BF16, {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, 2, false,
                      false, {0}, {1, 129}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX2(TBMMPermX2::PERM_X2_0_2_1)
                                 .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_TRUE)
                                 .SetModel(TBMMModel::TBMM_BASIC)
                                 .SetNonContiguousType(TBMMNonContiguousType::C_TRANSPOSE)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    const auto* td = GetTbmmBasicTilingData(c.ctx);
    EXPECT_EQ(td->batchSplitFactor, 2U);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, MbkMbnSplitBiasRejected)
{
    AdvancedContext c({16, 64, 32}, {64, 129, 32}, {2, 16, 4128}, DT_BF16, {1, 0, 2}, {0, 2, 1}, {1, 0, 2}, 2, true,
                      false, {0}, {1, 129}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

// batch(2) <= aicNum(20): iterbatch拒绝, 回退BASE
TEST(TransposeBatchMatMulIterBatchTilingTest, SmallBatchFallsBackToBase)
{
    AdvancedContext c({2, 64, 128}, {2, 128, 96}, {2, 64, 96}, DT_FLOAT16,
                      /*perm_x1=*/{0, 1, 2}, /*perm_x2=*/{0, 1, 2}, /*perm_y=*/{0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t baseKey = TBMMKeyBuilder()
                           .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                           .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                           .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                           .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                           .SetModel(TBMMModel::TBMM_BASIC)
                           .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                           .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), baseKey);
}

// x2 kbn(permX2=[1,0,2], 存储形状(k,b,n)连续) + permY=[0,1,2](bmn): 准入iterbatch,
// KBN由PERM_X2和NON_CONTIGUOUS_TYPE编码；L1分组须整除batch。
TEST(TransposeBatchMatMulIterBatchTilingTest, KbnBBmnIterbatchAdmitted)
{
    AdvancedContext c({64, 64, 128}, {128, 64, 96}, {64, 64, 96}, DT_FLOAT16,
                      /*perm_x1=*/{0, 1, 2}, /*perm_x2=*/{1, 0, 2}, /*perm_y=*/{0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t expectedKey = TBMMKeyBuilder()
                               .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                               .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                               .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                               .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                               .SetModel(TBMMModel::TBMM_ITER_BATCH)
                               .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                               .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    auto* td = GetTbmmIterBatchTilingData(c.ctx);
    EXPECT_EQ(td->iterBatchL1, 4U);
    EXPECT_EQ(td->iterBatchL0, 1U);
    EXPECT_EQ(td->baseM, 64U);
    EXPECT_EQ(td->baseN, 96U);
    EXPECT_EQ(td->baseK, 128U);
    EXPECT_EQ(c.ctx->GetBlockDim(), 20U);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, NbkBBmnIterbatchAdmitted)
{
    AdvancedContext c({64, 64, 128}, {96, 64, 128}, {64, 64, 96}, DT_FLOAT16,
                      /*perm_x1=*/{0, 1, 2}, /*perm_x2=*/{1, 2, 0}, /*perm_y=*/{0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    uint64_t expectedKey = TBMMKeyBuilder()
                               .SetApiLevel(TBMMApiLevel::TBMM_TENSOR_LEVEL)
                               .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                               .SetPermX2(TBMMPermX2::PERM_X2_1_2_0)
                               .SetBatchSplitMode(TBMMBatchSplit::BATCH_SPLIT_FALSE)
                               .SetModel(TBMMModel::TBMM_ITER_BATCH)
                               .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                               .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    auto* td = GetTbmmIterBatchTilingData(c.ctx);
    EXPECT_EQ(td->iterBatchL1, 4U);
    EXPECT_EQ(td->iterBatchL0, 1U);
    EXPECT_EQ(td->baseM, 64U);
    EXPECT_EQ(td->baseN, 96U);
    EXPECT_EQ(td->baseK, 128U);
    EXPECT_EQ(c.ctx->GetBlockDim(), 20U);
}

// KBN is a BMN-only layout and must fail before strategy selection for MBN.
TEST(TransposeBatchMatMulIterBatchTilingTest, KbnMbnRejected)
{
    AdvancedContext c({64, 32, 64}, {64, 64, 16}, {32, 64, 16}, DT_FLOAT16, {0, 1, 2}, {1, 0, 2}, {1, 0, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, MbkKbnWithFp32BiasIterBatchAdmitted)
{
    AdvancedContext c({32, 64, 64}, {64, 64, 16}, {64, 32, 16}, DT_BF16, {1, 0, 2}, {1, 0, 2}, {0, 1, 2}, 1, true,
                      false, {0}, {1, 16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    auto expectedKey = TBMMKeyBuilder()
                           .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                           .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                           .SetModel(TBMMModel::TBMM_ITER_BATCH)
                           .SetNonContiguousType(TBMMNonContiguousType::AB_NON_CONTIGUOUS)
                           .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, MbkNbkWithFp32BiasIterBatchAdmitted)
{
    AdvancedContext c({32, 64, 64}, {16, 64, 64}, {64, 32, 16}, DT_BF16, {1, 0, 2}, {1, 2, 0}, {0, 1, 2}, 1, true,
                      false, {0}, {1, 16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    auto expectedKey = TBMMKeyBuilder()
                           .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                           .SetPermX2(TBMMPermX2::PERM_X2_1_2_0)
                           .SetModel(TBMMModel::TBMM_ITER_BATCH)
                           .SetNonContiguousType(TBMMNonContiguousType::AB_NON_CONTIGUOUS)
                           .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, TransposedABWithFp32Bias)
{
    AdvancedContext c({64, 64, 32}, {64, 16, 64}, {64, 32, 16}, DT_BF16, {0, 2, 1}, {0, 2, 1}, {0, 1, 2}, 1, true,
                      false, {0}, {1, 16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    auto expectedKey = TBMMKeyBuilder()
                           .SetPermX1(TBMMPermX1::PERM_X1_0_2_1)
                           .SetPermX2(TBMMPermX2::PERM_X2_0_2_1)
                           .SetModel(TBMMModel::TBMM_ITER_BATCH)
                           .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                           .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GE(GetTbmmIterBatchTilingData(c.ctx)->iterBatchL1, 1U);
    EXPECT_GE(GetTbmmIterBatchTilingData(c.ctx)->iterBatchL0, 1U);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, TransposedAKbnWithFp32Bias)
{
    AdvancedContext c({64, 64, 32}, {64, 64, 16}, {64, 32, 16}, DT_BF16, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, 1, true,
                      false, {0}, {1, 16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    auto expectedKey = TBMMKeyBuilder()
                           .SetPermX1(TBMMPermX1::PERM_X1_0_2_1)
                           .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                           .SetModel(TBMMModel::TBMM_ITER_BATCH)
                           .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                           .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GE(GetTbmmIterBatchTilingData(c.ctx)->iterBatchL1, 1U);
    EXPECT_GE(GetTbmmIterBatchTilingData(c.ctx)->iterBatchL0, 1U);
}

TEST(TransposeBatchMatMulIterBatchTilingTest, WeightNzTransposedAWithBiasFallsBackToBasic)
{
    AdvancedContext c({64, 64, 32}, {64, 16, 64}, {64, 32, 16}, DT_FLOAT16, {0, 2, 1}, {0, 2, 1}, {0, 1, 2}, 1, true,
                      false, {0}, {16}, DT_FLOAT, FORMAT_FRACTAL_NZ);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(c.ctx->GetTilingKey(), TBMMKeyBuilder()
                                         .SetPermX1(TBMMPermX1::PERM_X1_0_2_1)
                                         .SetPermX2(TBMMPermX2::PERM_X2_0_2_1)
                                         .SetModel(TBMMModel::TBMM_BASIC)
                                         .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                                         .GetTilingKey());
}

TEST(TransposeBatchMatMulIterBatchTilingTest, L1CapacityFallsBackToBasic)
{
    AdvancedContext c({64, 512, 512}, {64, 512, 128}, {64, 512, 128}, DT_FLOAT16, {0, 1, 2}, {0, 1, 2}, {0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(c.ctx->GetTilingKey(), TBMMKeyBuilder()
                                         .SetModel(TBMMModel::TBMM_BASIC)
                                         .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                                         .GetTilingKey());
}

TEST(TransposeBatchMatMulIterBatchTilingTest, KbnFixpipeSelection)
{
    for (auto dtype : {DT_FLOAT16, DT_BF16, DT_FLOAT}) {
        for (int64_t n : {64, 65, 128, 129, 256}) {
            AdvancedContext c({64, 16, 32}, {32, 64, n}, {64, 16, n}, dtype, {0, 1, 2}, {1, 0, 2}, {0, 1, 2});
            SetIterBatchCompileInfo(c.compileInfo);
            TransposeBatchMatMulTiling tiling(c.ctx);
            ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
            const auto l0C2Out = n == 129 || (dtype == DT_FLOAT && n == 65) ? TBMML0C2Out::ND_FIXPIPE_1_2 :
                                                                              TBMML0C2Out::ON_THE_FLY;
            EXPECT_EQ(c.ctx->GetTilingKey(), TBMMKeyBuilder()
                                                 .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                                                 .SetModel(TBMMModel::TBMM_ITER_BATCH)
                                                 .SetL0C2Out(l0C2Out)
                                                 .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                                                 .GetTilingKey());
        }
    }
}

TEST(TransposeBatchMatMulIterBatchTilingTest, FixpipeRequiresTwoAivPerAic)
{
    AdvancedContext c({64, 16, 32}, {32, 64, 129}, {64, 16, 129}, DT_FLOAT16, {0, 1, 2}, {1, 0, 2}, {0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    c.compileInfo.aivNum = c.compileInfo.aicNum;
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(c.ctx->GetTilingKey(), TBMMKeyBuilder()
                                         .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                                         .SetModel(TBMMModel::TBMM_ITER_BATCH)
                                         .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                                         .GetTilingKey());
}

TEST(TransposeBatchMatMulAdvancedTilingTest, BiasWithScaleRejected)
{
    AdvancedContext c({32, 6, 64}, {6, 64, 16}, {32, 1, 96}, DT_FLOAT16, {1, 0, 2}, {0, 1, 2}, {1, 0, 2}, 1, true, true,
                      {96}, {16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    EXPECT_EQ(tiling.DoTiling(), ge::GRAPH_FAILED);
}

TEST(TransposeBatchMatMulAdvancedTilingTest, Int8ScaleL1Capacity)
{
    for (auto bFormat : {FORMAT_ND, FORMAT_FRACTAL_NZ}) {
        AdvancedContext c({128, 32, 512}, {32, 512, 128}, {1, 128, 4096}, DT_FLOAT16, {1, 0, 2}, {0, 1, 2}, {1, 0, 2},
                          1, false, true, {4096}, {}, DT_FLOAT, bFormat);
        SetIterBatchCompileInfo(c.compileInfo);
        c.compileInfo.aicNum = 32UL;
        c.compileInfo.aivNum = 64UL;
        TransposeBatchMatMulTiling tiling(c.ctx);
        ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
        const auto* td = GetTbmmBasicTilingData(c.ctx);
        ASSERT_EQ(c.ctx->GetRawTilingData()->GetDataSize(), sizeof(TransposeBatchMatMulBasicTilingData));
        const auto& mm = td->matMulTilingData.matMulTilingData;
        ASSERT_GT(mm.l1BufferNum, 0U);
        // 按kernel的末槽地址计算scale结束位置，检查ND/NZ量化路径均不越界。
        const uint64_t lastSlot = c.compileInfo.l1Size / mm.l1BufferNum * (mm.l1BufferNum - 1);
        const uint64_t aL1Size = static_cast<uint64_t>(mm.mL1) * mm.kL1 * sizeof(uint16_t);
        const uint64_t bL1Size = static_cast<uint64_t>(mm.nL1) * mm.kL1 * sizeof(uint16_t);
        const uint64_t scaleEnd = lastSlot + aL1Size + bL1Size + mm.nL1 * (sizeof(uint16_t) + sizeof(uint64_t));
        EXPECT_LE(scaleEnd, c.compileInfo.l1Size);
    }
}

TEST(TransposeBatchMatMulAdvancedTilingTest, NbkBmnBasic)
{
    AdvancedContext c({2, 32, 64}, {16, 2, 64}, {2, 32, 16}, DT_FLOAT16, {0, 1, 2}, {1, 2, 0}, {0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX2(TBMMPermX2::PERM_X2_1_2_0)
                                 .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
}

TEST(TransposeBatchMatMulAdvancedTilingTest, BmnSplitK)
{
    AdvancedContext c({2, 32, 1024}, {2, 1024, 16}, {2, 32, 16}, DT_FLOAT, {0, 1, 2}, {0, 1, 2}, {0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX1(TBMMPermX1::PERM_X1_0_1_2)
                                 .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                                 .SetModel(TBMMModel::TBMM_BASIC_SPLIT_K)
                                 .SetNonContiguousType(TBMMNonContiguousType::CONTIGUOUS)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GT(GetTbmmBasicTilingData(c.ctx)->matMulTilingData.matMulTilingData.splitSingleCoreKBase, 0U);
}

TEST(TransposeBatchMatMulAdvancedTilingTest, MbkBmnSplitK)
{
    AdvancedContext c({32, 2, 1024}, {2, 1024, 16}, {2, 32, 16}, DT_FLOAT, {1, 0, 2}, {0, 1, 2}, {0, 1, 2});
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                                 .SetPermX2(TBMMPermX2::PERM_X2_0_1_2)
                                 .SetModel(TBMMModel::TBMM_BASIC_SPLIT_K)
                                 .SetNonContiguousType(TBMMNonContiguousType::A_NON_CONTIGUOUS)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GT(GetTbmmBasicTilingData(c.ctx)->matMulTilingData.matMulTilingData.splitSingleCoreKBase, 0U);
}

TEST(TransposeBatchMatMulAdvancedTilingTest, TransposedANbkBmnSplitKWithBias)
{
    AdvancedContext c({2, 1024, 32}, {16, 2, 1024}, {2, 32, 16}, DT_FLOAT, {0, 2, 1}, {1, 2, 0}, {0, 1, 2}, 1, true,
                      false, {0}, {16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX1(TBMMPermX1::PERM_X1_0_2_1)
                                 .SetPermX2(TBMMPermX2::PERM_X2_1_2_0)
                                 .SetModel(TBMMModel::TBMM_BASIC_SPLIT_K)
                                 .SetNonContiguousType(TBMMNonContiguousType::B_NON_CONTIGUOUS)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GT(GetTbmmBasicTilingData(c.ctx)->matMulTilingData.matMulTilingData.splitSingleCoreKBase, 0U);
}

TEST(TransposeBatchMatMulAdvancedTilingTest, MbkKbnBmnSplitKWithBias)
{
    AdvancedContext c({32, 2, 1024}, {1024, 2, 16}, {2, 32, 16}, DT_FLOAT, {1, 0, 2}, {1, 0, 2}, {0, 1, 2}, 1, true,
                      false, {0}, {16}, DT_FLOAT);
    SetIterBatchCompileInfo(c.compileInfo);
    TransposeBatchMatMulTiling tiling(c.ctx);
    ASSERT_EQ(tiling.DoTiling(), ge::GRAPH_SUCCESS);
    const auto expectedKey = TBMMKeyBuilder()
                                 .SetPermX1(TBMMPermX1::PERM_X1_1_0_2)
                                 .SetPermX2(TBMMPermX2::PERM_X2_1_0_2)
                                 .SetModel(TBMMModel::TBMM_BASIC_SPLIT_K)
                                 .SetNonContiguousType(TBMMNonContiguousType::AB_NON_CONTIGUOUS)
                                 .GetTilingKey();
    EXPECT_EQ(c.ctx->GetTilingKey(), expectedKey);
    EXPECT_GT(GetTbmmBasicTilingData(c.ctx)->matMulTilingData.matMulTilingData.splitSingleCoreKBase, 0U);
}

} // namespace
