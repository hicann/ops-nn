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
 * \file test_sync_batch_norm_backward_elemt_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/sync_batch_norm_backward_elemt_tiling_arch35.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include "log/log.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SyncBatchNormBackwardElemtTilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SyncBatchNormBackwardElemtTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SyncBatchNormBackwardElemtTilingTest TearDown" << std::endl;
    }
};

void SyncBatchNormBackwardElemtTilingRun(
    gert::StorageShape grad_output_shape_in, gert::StorageShape mean_shape_in, ge::DataType grad_input_type,
    ge::DataType mean_type, ge::graphStatus expectRes, uint64_t expectTilingKey)
{
    gert::StorageShape grad_output_shape = grad_output_shape_in;
    gert::StorageShape save_input_shape = grad_output_shape_in;
    gert::StorageShape mean_shape = mean_shape_in;
    gert::StorageShape invstd_shape = mean_shape_in;
    gert::StorageShape weight_shape = mean_shape_in;
    gert::StorageShape mean_dy_shape = mean_shape_in;
    gert::StorageShape mean_dy_xmu_shape = mean_shape_in;
    gert::StorageShape grad_input_shape = grad_output_shape_in;

    string compile_info_string = R"({
        "_pattern":"ElemWise", 
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    Ops::Base::ElewiseCompileInfo compile_info;

    string op_type("SyncBatchNormBackwardElemt");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    compile_info_string = R"({})";
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&grad_output_shape, &save_input_shape, &mean_shape, &invstd_shape, &weight_shape,
                           &mean_dy_shape, &mean_dy_xmu_shape})
                      .OutputShapes({&grad_input_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, grad_input_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, grad_input_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, mean_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, mean_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, mean_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, mean_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, mean_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, grad_input_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), expectRes);
    if (expectRes == ge::GRAPH_SUCCESS) {
        auto tilingData = tiling_context->GetRawTilingData();
        ASSERT_NE(tilingData, nullptr);
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, expectTilingKey);
    }
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_fp16)
{
    gert::StorageShape grad_output_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    SyncBatchNormBackwardElemtTilingRun(
        grad_output_shape, mean_shape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS, 0);
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_bf16)
{
    gert::StorageShape grad_output_shape = {{1000}, {1000}};
    gert::StorageShape mean_shape = {{1000}, {1000}};
    SyncBatchNormBackwardElemtTilingRun(grad_output_shape, mean_shape, ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS, 0);
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_fp32)
{
    gert::StorageShape grad_output_shape = {{4096}, {4096}};
    gert::StorageShape mean_shape = {{4096}, {4096}};
    SyncBatchNormBackwardElemtTilingRun(
        grad_output_shape, mean_shape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_fp16_fp32)
{
    gert::StorageShape grad_output_shape = {{1024}, {1024}};
    gert::StorageShape mean_shape = {{1024}, {1024}};
    SyncBatchNormBackwardElemtTilingRun(
        grad_output_shape, mean_shape, ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_SUCCESS, 0);
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_shape_not_equal)
{
    gert::StorageShape grad_output_shape = {{2046}, {2046}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    SyncBatchNormBackwardElemtTilingRun(grad_output_shape, mean_shape, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
}

TEST_F(SyncBatchNormBackwardElemtTilingTest, test_ascend910d_dtype_not_equal)
{
    gert::StorageShape grad_output_shape = {{1024}, {1024}};
    gert::StorageShape mean_shape = {{1024}, {1024}};
    SyncBatchNormBackwardElemtTilingRun(
        grad_output_shape, mean_shape, ge::DT_FLOAT, ge::DT_FLOAT16, ge::GRAPH_FAILED, 0);
}
