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
#include <gtest/gtest.h>
#include "log/log.h"

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

struct AdaptiveAvgPool3dTilingTestParam {
    string case_name;

    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> y_shape;
    std::initializer_list<int64_t> output_size;
    string data_format;

    ge::DataType data_type;

    uint32_t expected_block_dim;
    uint64_t expected_tiling_key;
    string expected_tiling_data;
};

struct AdaptiveAvgPool3dCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

class AdaptiveAvgPool3dTilingTest : public testing::TestWithParam<AdaptiveAvgPool3dTilingTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3dTilingTest TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();

    stringstream ss;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        ss << std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)])) << " ";
    }

    return ss.str();
}

TEST_P(AdaptiveAvgPool3dTilingTest, test_case_adaptive_avg_pool3d_tiling)
{
    AdaptiveAvgPool3dTilingTestParam param = GetParam();

    gert::StorageShape x_shape = {param.x_shape, param.x_shape};
    gert::StorageShape y_shape = {param.y_shape, param.y_shape};

    vector<int64_t> output_size = param.output_size;

    string op_type("AdaptiveAvgPool3d");
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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    AdaptiveAvgPool3dCompileInfo compile_info;
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, param.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(output_size)},
                                   {"data_format", param.data_format}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, param.expected_tiling_key);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, param.expected_block_dim);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    ASSERT_EQ(tiling_data_result, param.expected_tiling_data);
}

static AdaptiveAvgPool3dTilingTestParam cases[] = {
    {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint32_float32",
     {19, 46, 50, 33, 5},
     {19, 157, 163, 172, 5},
     {157, 163, 172},
     "NDHWC",
     ge::DT_FLOAT,
     64,
     3,
     "19 5 46 50 33 157 163 172 "},
    {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint32_float16",
     {19, 46, 50, 33, 5},
     {19, 157, 163, 172, 5},
     {157, 163, 172},
     "NDHWC",
     ge::DT_FLOAT16,
     64,
     3,
     "19 5 46 50 33 157 163 172 "},
    {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint32_bfloat16",
     {19, 46, 50, 33, 5},
     {19, 157, 163, 172, 5},
     {157, 163, 172},
     "NDHWC",
     ge::DT_BF16,
     64,
     3,
     "19 5 46 50 33 157 163 172 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint32_float32",
     {1, 14, 31, 152, 119},
     {1, 14, 30, 45, 49},
     {30, 45, 49},
     "NCDHW",
     ge::DT_FLOAT,
     64,
     35,
     "1 14 31 152 119 30 45 49 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint32_float16",
     {1, 14, 31, 152, 119},
     {1, 14, 30, 45, 49},
     {30, 45, 49},
     "NCDHW",
     ge::DT_FLOAT16,
     64,
     35,
     "1 14 31 152 119 30 45 49 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint32_bfloat16",
     {1, 14, 31, 152, 119},
     {1, 14, 30, 45, 49},
     {30, 45, 49},
     "NCDHW",
     ge::DT_BF16,
     64,
     35,
     "1 14 31 152 119 30 45 49 "},
     {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint64_float32",
     {5, 673, 615, 763, 3},
     {5, 47, 345, 496, 3},
     {47, 345, 496},
     "NDHWC",
     ge::DT_FLOAT,
     64,
     9,
     "5 3 673 615 763 47 345 496 "},
    {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint64_float16",
     {5, 673, 615, 763, 3},
     {5, 47, 345, 496, 3},
     {47, 345, 496},
     "NDHWC",
     ge::DT_FLOAT16,
     64,
     9,
     "5 3 673 615 763 47 345 496 "},
    {"test_case_adaptive_avg_pool3d_simt_ndhwc_uint64_bfloat16",
     {5, 673, 615, 763, 3},
     {5, 47, 345, 496, 3},
     {47, 345, 496},
     "NDHWC",
     ge::DT_BF16,
     64,
     9,
     "5 3 673 615 763 47 345 496 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint64_float32",
     {85, 1601, 63, 25, 32},
     {85, 1601, 51, 19, 18},
     {51, 19, 18},
     "NCDHW",
     ge::DT_FLOAT,
     64,
     41,
     "85 1601 63 25 32 51 19 18 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint64_float16",
     {85, 1601, 63, 25, 32},
     {85, 1601, 51, 19, 18},
     {51, 19, 18},
     "NCDHW",
     ge::DT_FLOAT16,
     64,
     41,
     "85 1601 63 25 32 51 19 18 "},
    {"test_case_adaptive_avg_pool3d_simt_ncdhw_uint64_bfloat16",
     {85, 1601, 63, 25, 32},
     {85, 1601, 51, 19, 18},
     {51, 19, 18},
     "NCDHW",
     ge::DT_BF16,
     64,
     41,
     "85 1601 63 25 32 51 19 18 "},
};

INSTANTIATE_TEST_CASE_P(AdaptiveAvgPool3d, AdaptiveAvgPool3dTilingTest, testing::ValuesIn(cases));
