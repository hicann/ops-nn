/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_dense_grad_v2_tiling.cpp
 * \brief
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
#include "../../../op_host/gather_nd_tiling.h"

using namespace std;

class GatherNdTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GatherNdTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GatherNdTiling TearDown" << std::endl;
    }
};

using namespace ge;

static void runCase(gert::StorageShape x, gert::StorageShape indices, gert::StorageShape y)
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
    std::string op_type("GatherNd");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::GatherNdCompileInfo compile_info;
    compile_info.ub_size = 262144;
    compile_info.core_num = 32;
    compile_info.l1_size = 2097152;
    compile_info.indices_dsize = 4;
    compile_info.params_dsize = 2;
    compile_info.is_tik = true;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherNd")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherNd")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GatherNd")->gen_simplifiedkey;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &indices})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(GatherNdTiling, gather_nd_tiling_0)
{
    gert::StorageShape x = {{87552}, {87552}};
    gert::StorageShape indices = {{174, 1}, {174, 1}};
    gert::StorageShape y = {{174}, {174}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_1)
{
    gert::StorageShape x = {{5, 10, 13, 31}, {5, 10, 13, 31}};
    gert::StorageShape indices = {{2, 4}, {2, 4}};
    gert::StorageShape y = {{2}, {2}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_2)
{
    gert::StorageShape x = {{7, 6, 81, 6, 32}, {7, 6, 81, 6, 32}};
    gert::StorageShape indices = {{2, 6, 3}, {2, 6, 3}};
    gert::StorageShape y = {{2, 6, 6, 32}, {2, 6, 6, 32}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_3)
{
    gert::StorageShape x = {{81, 6, 32}, {81, 6, 32}};
    gert::StorageShape indices = {{1}, {1}};
    gert::StorageShape y = {{6, 32}, {6, 32}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_4)
{
    gert::StorageShape x = {{81, 600, 310}, {81, 600, 310}};
    gert::StorageShape indices = {{1}, {1}};
    gert::StorageShape y = {{600, 310}, {600, 310}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_5)
{
    gert::StorageShape x = {{800, 600, 320}, {800, 600, 320}};
    gert::StorageShape indices = {{4, 2}, {4, 2}};
    gert::StorageShape y = {{4, 320}, {4, 320}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_6)
{
    gert::StorageShape x = {{800, 611, 1111}, {800, 611, 1111}};
    gert::StorageShape indices = {{2, 2}, {2, 2}};
    gert::StorageShape y = {{2, 1111}, {2, 1111}};

    runCase(x, indices, y);
}

TEST_F(GatherNdTiling, gather_nd_tiling_7)
{
    gert::StorageShape x = {{800, 611, 1111}, {800, 611, 1111}};
    gert::StorageShape indices = {{2, 0}, {2, 0}};
    gert::StorageShape y = {{2, 0}, {2, 0}};

    runCase(x, indices, y);
}
