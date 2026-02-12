/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_log_sigmoid_tiling.cpp
 * \brief
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
#include "../../../../op_host/arch35/log_sigmoid_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class LogSigmoidRegbaseTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LogSigmoidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LogSigmoidTiling TearDown" << std::endl;
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

static void ExecuteTestCase(
    ge::DataType xDtype, ge::DataType yDtype, gert::StorageShape xShape, gert::StorageShape yShape, int tilingKeyValue,
    string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, 
                        "Intrinsic_data_move_l12ub": true, 
                        "Intrinsic_data_move_l0c2ub": true, 
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, socversions);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    Ops::Base::ElewiseCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    std::string op_type("LogSigmoid");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(1024 * 1024);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024 * 1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)      
                      .IrInstanceNum({1})    
                      .InputShapes({&xShape}) 
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND) 
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    EXPECT_EQ(tiling_key, tilingKeyValue);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(LogSigmoidRegbaseTiling, test_log_sigmoid_tiling_fp16_001)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_FLOAT16;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    int tilingKeyValue = 3;
    string expectTilingData = "8192 45079976738820 2048 4 1 1 2048 2048 10496 1 ";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(LogSigmoidRegbaseTiling, test_log_sigmoid_tiling_bf16_002)
{
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType yDtype = ge::DT_BF16;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    int tilingKeyValue = 5;
    string expectTilingData = "8192 45079976738820 2048 4 1 1 2048 2048 10496 1 ";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(LogSigmoidRegbaseTiling, test_log_sigmoid_tiling_fp32_003)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    int tilingKeyValue = 7;
    string expectTilingData = "8192 68169720922120 1024 8 1 1 1024 1024 15872 1 ";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(LogSigmoidRegbaseTiling, test_tiling_failed_dtype_input_output_diff_004)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_BF16;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    int tilingKeyValue = 0;
    string expectTilingData = "";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(LogSigmoidRegbaseTiling, test_tiling_failed_shape_input_output_diff_005)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    int tilingKeyValue = 0;
    string expectTilingData = "";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(LogSigmoidRegbaseTiling, test_tiling_failed_empty_tensor_006)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT;
    gert::StorageShape xShape = {{1, 0, 2, 64}, {1, 0, 2, 64}};
    gert::StorageShape yShape = {{1, 0, 2, 64}, {1, 0, 2, 64}};
    int tilingKeyValue = 0;
    string expectTilingData = "";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(LogSigmoidRegbaseTiling, test_tiling_failed_unsupport_input_007)
{
    ge::DataType xDtype = ge::DT_DOUBLE;
    ge::DataType yDtype = ge::DT_FLOAT;
    gert::StorageShape xShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::StorageShape yShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    int tilingKeyValue = 0;
    string expectTilingData = "";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(LogSigmoidRegbaseTiling, test_tiling_failed_unsupport_input_008)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_DOUBLE;
    gert::StorageShape xShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    gert::StorageShape yShape = {{1, 64, 2, 32}, {1, 64, 2, 32}};
    int tilingKeyValue = 0;
    string expectTilingData = "";
    ExecuteTestCase(xDtype, yDtype, xShape, yShape, tilingKeyValue, expectTilingData, ge::GRAPH_FAILED);
}