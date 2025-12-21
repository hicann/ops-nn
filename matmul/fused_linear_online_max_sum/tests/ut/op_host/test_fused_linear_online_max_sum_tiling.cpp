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
 * \file test_fused_linear_online_max_sum_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../op_host/fused_linear_online_max_sum_tiling.h"

using namespace std;
using namespace ge;

struct FusedLinearOnlineMaxSumTilingTestParam {
    string case_name;
    ge::DataType fp_dtype = DT_FLOAT16;
    ge::DataType int_dtype = DT_INT32;

    std::initializer_list<int64_t> input_shape;
    std::initializer_list<int64_t> weight_shape;
    std::initializer_list<int64_t> target_shape;
    std::initializer_list<int64_t> logits_max_local_out_shape;
    std::initializer_list<int64_t> sum_exp_logits_local_out_shape;
    std::initializer_list<int64_t> predicted_logits_local_out_shape;
    std::initializer_list<int64_t> target_mask_out_shape;
    std::initializer_list<int64_t> masked_target_out_shape;
    std::initializer_list<int64_t> vocab_parallel_logits_out_optional_shape;
    int64_t vocab_start_index;
    int64_t vocab_end_index;
    bool vocab_parallel_logits_out_flag;
    ge::graphStatus status;
    uint32_t expected_block_dim;
    uint64_t expected_tiling_key;
    string expected_tiling_data;
};


class FusedLinearOnlineMaxSumTilingTest : public testing::TestWithParam<FusedLinearOnlineMaxSumTilingTestParam> {
    protected:
        static void SetUpTestCase() {
            std::cout << "FusedLinearOnlineMaxSumTilingTest SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "FusedLinearOnlineMaxSumTilingTest TearDown" << std::endl;
        }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();

    stringstream ss;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(uint32_t)) {
        ss << std::to_string((reinterpret_cast<const uint32_t*>(tiling_data->GetData())[i / sizeof(uint32_t)])) << " ";
    }

    return ss.str();
}

TEST_P(FusedLinearOnlineMaxSumTilingTest, test_case_fused_linear_online_max_sum_tiling)
{
    FusedLinearOnlineMaxSumTilingTestParam param = GetParam();

    gert::StorageShape input_shape = {param.input_shape, param.input_shape};
    gert::StorageShape weight_shape = {param.weight_shape, param.weight_shape};
    gert::StorageShape target_shape = {param.target_shape, param.target_shape};
    gert::StorageShape logits_max_local_out_shape = {param.logits_max_local_out_shape, param.logits_max_local_out_shape};
    gert::StorageShape sum_exp_logits_local_out_shape = {param.sum_exp_logits_local_out_shape,
        param.sum_exp_logits_local_out_shape};
    gert::StorageShape predicted_logits_local_out_shape = {param.predicted_logits_local_out_shape,
        param.predicted_logits_local_out_shape};
    gert::StorageShape target_mask_out_shape = {param.target_mask_out_shape, param.target_mask_out_shape};
    gert::StorageShape masked_target_out_shape = {param.masked_target_out_shape, param.masked_target_out_shape};
    gert::StorageShape vocab_parallel_logits_out_optional_shape = {param.vocab_parallel_logits_out_optional_shape,
        param.vocab_parallel_logits_out_optional_shape};

    string op_type("FusedLinearOnlineMaxSum");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                      "Intrinsic_fix_pipe_l0c2out": false,
                                                      "Intrinsic_data_move_l12ub": true,
                                                      "Intrinsic_data_move_l0c2ub": true,
                                                      "Intrinsic_data_move_out2l1_nd2nz": false,
                                                      "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                      "CORE_NUM": 40}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::FusedLinearOnlineMaxSumCompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 6)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &weight_shape, &target_shape})
                      .OutputShapes({&logits_max_local_out_shape, &sum_exp_logits_local_out_shape,
                          &predicted_logits_local_out_shape, &target_mask_out_shape, &masked_target_out_shape,
                          &vocab_parallel_logits_out_optional_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.int_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, param.int_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(5, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"vocab_start_index", Ops::NN::AnyValue::CreateFrom<int64_t>(param.vocab_start_index)},
                          {"vocab_end_index", Ops::NN::AnyValue::CreateFrom<int64_t>(param.vocab_end_index)},
                          {"vocab_parallel_logits_out_flag",
                              Ops::NN::AnyValue::CreateFrom<bool>(param.vocab_parallel_logits_out_flag)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), param.status);
    if (param.status == ge::GRAPH_SUCCESS) {
        // check tiling result
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, param.expected_tiling_key);
        auto block_dim = tiling_context->GetBlockDim();
        ASSERT_EQ(block_dim, param.expected_block_dim);
        auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
        ASSERT_EQ(tiling_data_result, param.expected_tiling_data);
    }
}

static FusedLinearOnlineMaxSumTilingTestParam cases[] = {
    {"test_case_fp16_int32_hp_success", ge::DT_FLOAT16, ge::DT_INT32,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_SUCCESS, 40, 1,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 1960 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_fp16_int64_hp_success", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_SUCCESS, 40, 1,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 980 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_bf16_int32_hp_success", ge::DT_BF16, ge::DT_INT32,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_SUCCESS, 40, 1,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 1960 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_bf16_int64_hp_success", ge::DT_BF16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_SUCCESS, 40, 1,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 980 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_fp16_int32_lm_success", ge::DT_FLOAT16, ge::DT_INT32,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, false,
     ge::GRAPH_SUCCESS, 40, 0,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 1960 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_fp16_int64_lm_success", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, false,
     ge::GRAPH_SUCCESS, 40, 0,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 980 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_bf16_int32_lm_success", ge::DT_BF16, ge::DT_INT32,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, false,
     ge::GRAPH_SUCCESS, 40, 0,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 1960 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_bf16_int64_lm_success", ge::DT_BF16, ge::DT_INT32,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, false,
     ge::GRAPH_SUCCESS, 40, 0,
     "8192 0 4096 0 19392 0 188416 0 40 0 40 0 25 0 24 0 1960 0 0 1124073472 4096 0 0 0 0 0 1 8192 19392 4096 4096 128 256 4096 128 256 64 4 8 1 1 0 0 0 0 327680 131072 0 1 1 1 1 2 4 0 0 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "},
    {"test_case_input_dim_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096, 1}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
    {"test_case_weight_dim_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096, 1}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
    {"test_case_weight_shape_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 1}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
    {"test_case_target_dim_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192, 1},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
    {"test_case_target_shape_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {1},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
     {"test_case_vocab_start_index_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     -1, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
     {"test_case_vocab_end_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     128, 19, true,
     ge::GRAPH_FAILED, 40, 1, ""},
     {"test_case_input_dim_one_size_invalid", ge::DT_FLOAT16, ge::DT_INT64,
     {8192, 4096, 1}, {19392, 4096}, {8192},
     {8192}, {8192}, {8192}, {1024}, {8192}, {8192, 19392},
     0, 128, true,
     ge::GRAPH_FAILED, 40, 1, ""},
};

INSTANTIATE_TEST_CASE_P(FusedLinearOnlineMaxSum, FusedLinearOnlineMaxSumTilingTest, testing::ValuesIn(cases));
