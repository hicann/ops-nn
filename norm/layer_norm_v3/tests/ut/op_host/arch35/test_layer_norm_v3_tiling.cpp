/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file test_layer_norm_v3_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../op_host/arch35/layer_norm_v3_tiling_arch35.h"
using namespace ut_util;
using namespace std;
using namespace ge;

class LayerNormV3TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LayerNormV3Tiling TearDown" << std::endl;
    }
};

TEST_F(LayerNormV3TilingTest, layer_norm_v3_tiling_regbase_two_pass)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape gamma_shape = {{4096}, {4096}};
    gert::StorageShape beta_shape = {{4096}, {4096}};
    gert::StorageShape y_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape mean_shape = {{64}, {64}};
    gert::StorageShape rstd_shape = {{64}, {64}};
    float eps = 1e-5;

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
    optiling::LayerNormV3OpInfo compile_info;

    std::string op_type("LayerNormV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .SetOpType(op_type)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"begin_norm_axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"begin_params_axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    // ASSERT_EQ(tiling_key, 300);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "4096 0 4096 0 64 0 3 0 1 0 64 0 2048 0 0 0 32 0 4096 0 128 0 0 0 0 0 925353388 0 ");
    dlog_setlevel(0, 3, 0);
}
