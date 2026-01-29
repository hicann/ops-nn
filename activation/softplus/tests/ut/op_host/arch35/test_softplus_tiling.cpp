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
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/softplus_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftplusTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftplusTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftplusTilingTest TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

void DoTilingTest(
    gert::StorageShape input_shape, gert::StorageShape output_shape, ge::DataType input_dataType,
    ge::DataType output_dataType, ge::graphStatus expect_status, int expect_tilingKey, string expect_tiling_data)
{
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    std::string compile_info_string = R"({
 	       "hardware_info": {
 	         "BT_SIZE": 0, "load3d_constraints": "1",
 	         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
 	         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
 	         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
 	         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
 	       }
 	     })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    std::string op_type("Softplus");

    Ops::Base::ElewiseCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(1, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&input_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, input_dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, output_dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), expect_status);
    if(expect_status != ge::GRAPH_SUCCESS){
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expect_tilingKey);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    ASSERT_EQ(tiling_data_result, expect_tiling_data);
}

TEST_F(SoftplusTilingTest, test_tiling_fp16_01)
{
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    int expect_tilingKey = 101;
    string expect_tiling_data = "8192 32985348833284 2048 4 1 1 2048 2048 7680 1 ";
    DoTilingTest(Shape, Shape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, expect_tilingKey, expect_tiling_data);
}

TEST_F(SoftplusTilingTest, test_tiling_bf16_02)
{
    gert::StorageShape Shape = {{4, 5, 6}, {4, 5, 6}};
    int expect_tilingKey = 101;
    string expect_tiling_data = "120 32985348833281 512 1 1 1 512 120 7680 1 ";
    DoTilingTest(Shape, Shape, ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, expect_tilingKey, expect_tiling_data);
}

TEST_F(SoftplusTilingTest, test_tiling_fp32_03)
{
    gert::StorageShape Shape = {{1, 1024}, {1, 1024}};
    int expect_tilingKey = 101;
    string expect_tiling_data = "1024 37658273251329 1024 1 1 1 1024 1024 8768 1 ";
    DoTilingTest(Shape, Shape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, expect_tilingKey, expect_tiling_data);
}

TEST_F(SoftplusTilingTest, test_tiling_failed_diff_shape_04) {
    gert::StorageShape input_shape = {{1, 1024}, {1, 1024}};
    gert::StorageShape output_shape = {{2, 1024}, {2, 1024}};
    int expect_tilingKey = 0;
    string expect_tiling_data = "";
    DoTilingTest(input_shape, output_shape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, expect_tilingKey, expect_tiling_data);
}

TEST_F(SoftplusTilingTest, test_tiling_failed_diff_dtype_05) {
    gert::StorageShape Shape = {{1, 1024}, {1, 1024}};
    int expect_tilingKey = 0;
    string expect_tiling_data = "";
    DoTilingTest(Shape, Shape, ge::DT_FLOAT, ge::DT_FLOAT16, ge::GRAPH_FAILED, expect_tilingKey, expect_tiling_data);
}
