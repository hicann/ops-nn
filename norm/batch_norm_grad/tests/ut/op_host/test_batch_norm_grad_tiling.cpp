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
 * \file test_batch_norm_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "norm/batch_norm_grad/op_host/batch_norm_grad_tiling.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class BatchNormGradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormGradTiling TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

// Training, Full load, fp32
TEST_F(BatchNormGradTiling, batch_norm_grad_RARfull_load_fp32)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 225 0 904 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 16 0 2 0 16 0 3 0 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, Full load, fp16
TEST_F(BatchNormGradTiling, batch_norm_grad_RARfull_load_fp16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 225 0 912 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 22 0 2 0 10 0 2 0 11 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, Full load, bf16
TEST_F(BatchNormGradTiling, batch_norm_grad_RARfull_load_bf16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 225 0 912 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 22 0 2 0 10 0 2 0 11 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r1, fp32
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r1_fp32)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 2 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

TEST_F(BatchNormGradTiling, batch_norm_grad_RARfull_load)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape scale_shape = {{2}, {2}};
    gert::StorageShape mean_shape = {{2}, {2}};
    gert::StorageShape var_rstd_shape = {{2}, {2}};
    gert::StorageShape reserve_in_shape = {{2}, {2}};
    // output
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    gert::StorageShape reserve_out1_shape = {{2}, {2}};
    gert::StorageShape reserve_out2_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "5 0 2 0 12 0 64 0 1 0 0 0 2 0 3 0 32 0 0 0 0 0 230 0 1 0 2 0 1 0 3 0 ");
    dlog_setlevel(0, 3, 0);
}

TEST_F(BatchNormGradTiling, batch_norm_grad_RAfull_load)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 1, 1, 1, 470}, {5, 1, 1, 1, 470}};
    gert::StorageShape x_shape = {{5, 1, 1, 1, 470}, {5, 1, 1, 1, 470}};
    gert::StorageShape scale_shape = {{470}, {470}};
    gert::StorageShape mean_shape = {{470}, {470}};
    gert::StorageShape var_rstd_shape = {{470}, {470}};
    gert::StorageShape reserve_in_shape = {{470}, {470}};
    // output
    gert::StorageShape dx_shape = {{5, 1, 1, 1, 470}, {5, 1, 1, 1, 470}};
    gert::StorageShape dscale_shape = {{470}, {470}};
    gert::StorageShape dbias_shape = {{470}, {470}};
    gert::StorageShape reserve_out1_shape = {{470}, {470}};
    gert::StorageShape reserve_out2_shape = {{470}, {470}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "5 0 470 0 1 0 0 0 64 0 22 0 7 0 8 0 0 0 0 0 0 0 30 0 16 0 6 0 29 0 1 0 16 0 16 0 6 0 16 0 0 0 4 0 1 0 0 0 1 0 2 0 0 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r1, fp16
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r1_fp16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "8 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 4 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r1, bf16
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r1_bf16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "8 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 4 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r0, fp32
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r0_fp32)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape x_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 89280 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 14880 0 14880 0 4 0 29760 0 7440 0 "
        "7440 0 4 0 0 0 7440 0 7440 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r0, fp16
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r0_fp16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape x_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 2048 0 89280 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 14880 0 14880 0 4 0 29760 0 7440 0 "
        "7440 0 4 0 0 0 7440 0 7440 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, recompute, split, r0, bf16
TEST_F(BatchNormGradTiling, batch_norm_grad_recompute_split_r0_bf16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "8 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 4 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, split core, r1, fp32
TEST_F(BatchNormGradTiling, batch_norm_grad_split_core_r1_fp32)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape x_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape scale_shape = {{4}, {4}};
    gert::StorageShape mean_shape = {{4}, {4}};
    gert::StorageShape var_rstd_shape = {{4}, {4}};
    gert::StorageShape reserve_in_shape = {{4}, {4}};
    // output
    gert::StorageShape dx_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape dscale_shape = {{4}, {4}};
    gert::StorageShape dbias_shape = {{4}, {4}};
    gert::StorageShape reserve_out1_shape = {{4}, {4}};
    gert::StorageShape reserve_out2_shape = {{4}, {4}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 400 0 4 0 8 0 225000 0 58 0 7 0 1 0 12168 0 19 0 5976 0 5976 0 1 0 7 0 1 0 1 "
        "0 1 0 1 0 8 0 4 0 1 0 8 0 1 0 4 0 10224 0 23 0 72 0 72 0 1 0 7 0 1 0 1 0 1 0 1 0 8 0 4 0 1 0 128 0 5 0 8 0 7 "
        "0 16 0 3 0 4 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, split core, r1, fp16, FAILED
TEST_F(BatchNormGradTiling, batch_norm_grad_split_core_r1_fp16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape x_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape scale_shape = {{2}, {2}};
    gert::StorageShape mean_shape = {{2}, {2}};
    gert::StorageShape var_rstd_shape = {{2}, {2}};
    gert::StorageShape reserve_in_shape = {{2}, {2}};
    // output
    gert::StorageShape dx_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape dscale_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    gert::StorageShape reserve_out1_shape = {{2}, {2}};
    gert::StorageShape reserve_out2_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86000 0 2 0 8 0 225 0 64 0 1344 0 1328 0 240 0 1 0 225 0 240 0 63 0 22 0 21 0 "
        "22 0 5 0 1 0 8 0 2 0 1 0 8 0 1 0 2 0 240 0 1 0 225 0 240 0 42 0 32 0 42 0 32 0 26 0 2 0 8 0 1 0 2 0 16 0 6 0 "
        "5 0 4 0 16 0 6 0 4 0 ");
    dlog_setlevel(0, 3, 0);
}

// Training, split core, r1, bf16
TEST_F(BatchNormGradTiling, batch_norm_grad_split_core_r1_bf16)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape x_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape scale_shape = {{68}, {68}};
    gert::StorageShape mean_shape = {{68}, {68}};
    gert::StorageShape var_rstd_shape = {{68}, {68}};
    gert::StorageShape reserve_in_shape = {{68}, {68}};
    // output
    gert::StorageShape dx_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape dscale_shape = {{68}, {68}};
    gert::StorageShape dbias_shape = {{68}, {68}};
    gert::StorageShape reserve_out1_shape = {{68}, {68}};
    gert::StorageShape reserve_out2_shape = {{68}, {68}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 50000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "144000 0 68 0 64 0 64 0 2250 0 2250 0 36 0 32 0 10 0 36 0 32 0 10 0 68 0 80 0 68 0 1 0 252 0 234 0 9 0 252 0 "
        "234 0 9 0 6 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, ra
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_ra_0)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape scale_shape = {{4}, {4}};
    gert::StorageShape mean_shape = {{4}, {4}};
    gert::StorageShape var_rstd_shape = {{4}, {4}};
    gert::StorageShape reserve_in_shape = {{4}, {4}};
    // output
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{4}, {4}};
    gert::StorageShape dbias_shape = {{4}, {4}};
    gert::StorageShape reserve_out1_shape = {{4}, {4}};
    gert::StorageShape reserve_out2_shape = {{4}, {4}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 900000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "1 0 1 0 1 0 4 0 1 0 1 0 128 0 4 0 124 0 30 0 30 0 925353388 0 128 "
        "0 0 0 1 0 30 0 0 0 0 0 1 0 0 0 4 0 1 0 16 0 4 0 1 0 1 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, ra
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_ra_1)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 1, 2, 3, 4}, {5, 1, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 1, 2, 3, 4}, {5, 1, 2, 3, 4}};
    gert::StorageShape scale_shape = {{4}, {4}};
    gert::StorageShape mean_shape = {{4}, {4}};
    gert::StorageShape var_rstd_shape = {{4}, {4}};
    gert::StorageShape reserve_in_shape = {{4}, {4}};
    // output
    gert::StorageShape dx_shape = {{5, 1, 2, 3, 4}, {5, 1, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{4}, {4}};
    gert::StorageShape dbias_shape = {{4}, {4}};
    gert::StorageShape reserve_out1_shape = {{4}, {4}};
    gert::StorageShape reserve_out2_shape = {{4}, {4}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 900000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "1 0 1 0 1 0 4 0 1 0 1 0 128 0 4 0 124 0 30 0 30 0 925353388 0 128 "
        "0 0 0 1 0 30 0 0 0 0 0 1 0 0 0 4 0 1 0 16 0 4 0 1 0 1 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, rar
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_rar_splitR1_0)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape scale_shape = {{2}, {2}};
    gert::StorageShape mean_shape = {{2}, {2}};
    gert::StorageShape var_rstd_shape = {{2}, {2}};
    gert::StorageShape reserve_in_shape = {{2}, {2}};
    // output
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    gert::StorageShape reserve_out1_shape = {{2}, {2}};
    gert::StorageShape reserve_out2_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910001);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "1 0 1 0 1 0 5 0 2 0 12 0 1 0 1 0 1 0 5 0 5 0 2 0 2 0 128 0 12 0 0 0 925353388 0 32 0 0 0 0 0 1 0 0 0 0 0 2 0 "
        "0 0 1 0 2 0 60 0 64 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, rar
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_rar_splitR1_1)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{1, 456, 1, 173}, {1, 456, 1, 173}};
    gert::StorageShape x_shape = {{1, 456, 1, 173}, {1, 456, 1, 173}};
    gert::StorageShape scale_shape = {{456}, {456}};
    gert::StorageShape mean_shape = {{456}, {456}};
    gert::StorageShape var_rstd_shape = {{456}, {456}};
    gert::StorageShape reserve_in_shape = {{456}, {456}};
    // output
    gert::StorageShape dx_shape = {{1, 456, 1, 173}, {1, 456, 1, 173}};
    gert::StorageShape dscale_shape = {{456}, {456}};
    gert::StorageShape dbias_shape = {{456}, {456}};
    gert::StorageShape reserve_out1_shape = {{456}, {456}};
    gert::StorageShape reserve_out2_shape = {{456}, {456}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910001);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "4 0 1 0 4 0 1 0 456 0 173 0 1 0 4 0 1 0 1 0 1 0 119 0 99 0 256 0 173 0 0 0 925353388 0 128 0 0 0 2 0 1 0 0 0 0 0 64 0 "
        "8 0 7 0 8 0 173 0 176 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, rar
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_rar_splitR0)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape x_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape scale_shape = {{6}, {6}};
    gert::StorageShape mean_shape = {{6}, {6}};
    gert::StorageShape var_rstd_shape = {{6}, {6}};
    gert::StorageShape reserve_in_shape = {{6}, {6}};
    // output
    gert::StorageShape dx_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape dscale_shape = {{6}, {6}};
    gert::StorageShape dbias_shape = {{6}, {6}};
    gert::StorageShape reserve_out1_shape = {{6}, {6}};
    gert::StorageShape reserve_out2_shape = {{6}, {6}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "270 0 5 0 54 0 1 0 6 0 1346832 0 1 0 6 0 45 0 1 0 1 0 1 0 1 0 30464 0 6416 0 0 0 925353388 0 8192 0 1 0 64 0 "
        "4096 0 0 0 64 0 6 0 "
        "0 0 1 0 2 0 14880 0 14880 0 64 0 394512 0 7440 0 7440 0 52 0 7632 0 7440 0 7440 0 2 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, rar
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_rar_splitR0_1)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape x_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape scale_shape = {{6}, {6}};
    gert::StorageShape mean_shape = {{6}, {6}};
    gert::StorageShape var_rstd_shape = {{6}, {6}};
    gert::StorageShape reserve_in_shape = {{6}, {6}};
    // output
    gert::StorageShape dx_shape = {{1, 6, 199, 47, 144}, {1, 6, 199, 47, 144}};
    gert::StorageShape dscale_shape = {{6}, {6}};
    gert::StorageShape dbias_shape = {{6}, {6}};
    gert::StorageShape reserve_out1_shape = {{6}, {6}};
    gert::StorageShape reserve_out2_shape = {{6}, {6}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "270 0 5 0 54 0 1 0 6 0 1346832 0 1 0 6 0 45 0 1 0 1 0 1 0 1 0 30464 0 6416 0 0 0 925353388 0 8192 0 1 0 64 0 "
        "4096 0 0 0 64 0 6 0 "
        "0 0 1 0 2 0 14880 0 14880 0 64 0 394512 0 7440 0 7440 0 52 0 7632 0 7440 0 7440 0 2 0 1 1 1 0 ");
    dlog_setlevel(0, 3, 0);
}

// Infer, rar channel miss match
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_ChannelMissMatch)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape scale_shape = {{2}, {2}};
    gert::StorageShape mean_shape = {{2}, {2}};
    gert::StorageShape var_rstd_shape = {{2}, {2}};
    gert::StorageShape reserve_in_shape = {{2}, {2}};
    // output
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    gert::StorageShape reserve_out1_shape = {{2}, {2}};
    gert::StorageShape reserve_out2_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

// Training, channel miss match
TEST_F(BatchNormGradTiling, batch_norm_grad_train_ChannelMissMatch)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape scale_shape = {{2048}, {2048}};
    gert::StorageShape mean_shape = {{2048}, {2048}};
    gert::StorageShape var_rstd_shape = {{2048}, {2048}};
    gert::StorageShape reserve_in_shape = {{2048}, {2048}};
    // output
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dscale_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out1_shape = {{2048}, {2048}};
    gert::StorageShape reserve_out2_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NCHW")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

// Infer, rar dtype miss match
TEST_F(BatchNormGradTiling, batch_norm_grad_infer_DtypeMixMatch)
{
    dlog_setlevel(0, 0, 0);
    // Input
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape scale_shape = {{2}, {2}};
    gert::StorageShape mean_shape = {{2}, {2}};
    gert::StorageShape var_rstd_shape = {{2}, {2}};
    gert::StorageShape reserve_in_shape = {{2}, {2}};
    // output
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dscale_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    gert::StorageShape reserve_out1_shape = {{2}, {2}};
    gert::StorageShape reserve_out2_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend910_95"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradCompileInfo compile_info;

    std::string op_type("BatchNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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
                      .NodeIoNum(6, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &scale_shape, &mean_shape, &var_rstd_shape, &reserve_in_shape})
                      .OutputShapes({&dx_shape, &dscale_shape, &dbias_shape, &reserve_out1_shape, &reserve_out2_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NHWC")},
                           {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(is_training)},
                           {"output_mask", Ops::NN::AnyValue::CreateFrom<vector<bool>>({true, true, true})}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}