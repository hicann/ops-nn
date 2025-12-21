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
 * \file test_apply_top_k_top_p_with_sorted.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "apply_top_k_top_p_with_sorted_compile_info.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;

struct ApplyTopKTopPWithSortedTilingTestParam {
    string case_name;
    ge::DataType fp_dtype = DT_FLOAT16;
    ge::DataType int_dtype = DT_FLOAT16;

    std::initializer_list<int64_t> sorted_value_shape;
    std::initializer_list<int64_t> sorted_indices_shape;
    std::initializer_list<int64_t> p_shape;
    std::initializer_list<int64_t> k_shape;
    std::initializer_list<int64_t> out_shape;
    ge::graphStatus status;
    uint32_t expected_block_dim;
    uint64_t expected_tiling_key;
    string expected_tiling_data;
};


class ApplyTopKTopPWithSortedTilingTest : public testing::TestWithParam<ApplyTopKTopPWithSortedTilingTestParam> {
    protected:
        static void SetUpTestCase() {
            std::cout << "ApplyTopKTopPWithSortedTilingTest SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "ApplyTopKTopPWithSortedTilingTest TearDown" << std::endl;
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

TEST_P(ApplyTopKTopPWithSortedTilingTest, test_case_apply_top_k_top_p_with_sorted_tiling)
{
    ApplyTopKTopPWithSortedTilingTestParam param = GetParam();

    gert::StorageShape sorted_value_shape = {param.sorted_value_shape, param.sorted_value_shape};
    gert::StorageShape sorted_indices_shape = {param.sorted_indices_shape, param.sorted_indices_shape};
    gert::StorageShape p_shape = {param.p_shape, param.p_shape};
    gert::StorageShape k_shape = {param.k_shape, param.k_shape};
    gert::StorageShape out_shape = {param.out_shape, param.out_shape};

    string op_type("ApplyTopKTopPWithSorted");
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
    optiling::TilingForApplyTopKTopPWithSortedCompileInfo compile_info;
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
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&sorted_value_shape, &sorted_indices_shape, &p_shape, &k_shape})
                      .OutputShapes({&out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.int_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, param.int_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.fp_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
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

static ApplyTopKTopPWithSortedTilingTestParam cases[] = {
    {"test_case_float32_succ",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_SUCCESS,
     40,
     0,
     "4 152064 0 4 40 1024 1024 1024 1024 512 512 151488 18 0 "},
    {"test_case_float16_succ",
     ge::DT_FLOAT16,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_SUCCESS,
     40,
     0,
     "4 152064 0 4 40 1024 1024 1024 1024 512 512 157632 18 0 "},
    {"test_case_bfloat16_succ",
     ge::DT_BF16,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_SUCCESS,
     40,
     0,
     "4 152064 0 4 40 1024 1024 1024 1024 512 512 157632 18 0 "},
    {"test_case_sorted_value_one_dim_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4},
     {4, 152064},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_sorted_indices_one_dim_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_p_two_dim_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4, 1},
     {4},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_k_two_dim_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4},
     {4, 1},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_sorted_value_shape_not_equal_sorted_indices_shape_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152061},
     {4, 152064},
     {4},
     {4},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_p_shape_invalid_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {3},
     {4},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
    {"test_case_k_shape_invalid_failed",
     ge::DT_FLOAT,
     ge::DT_INT32,
     {4, 152064},
     {4, 152064},
     {4},
     {3},
     {4, 152064},
     ge::GRAPH_FAILED,
     40,
     0,
     ""},
};

INSTANTIATE_TEST_CASE_P(ApplyTopKTopPWithSorted, ApplyTopKTopPWithSortedTilingTest, testing::ValuesIn(cases));
