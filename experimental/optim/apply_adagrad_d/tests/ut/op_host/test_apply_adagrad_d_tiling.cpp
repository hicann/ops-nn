/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

using namespace ut_util;
using namespace std;
using namespace ge;

class ApplyAdagradDTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdagradDTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyAdagradDTiling TearDown" << std::endl;
    }
};

static string GetCompileInfoString()
{
    return R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 48}
                        })";
}

static void RunTilingTest(ge::DataType dtype, bool expected_success = true)
{
    // inputs: var, accum, lr (scalar shape {1}), grad
    gert::StorageShape var_shape   = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape accum_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape lr_shape    = {{1}, {1}};
    gert::StorageShape grad_shape  = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    // outputs: var, accum
    gert::StorageShape var_out_shape   = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape accum_out_shape = {{1, 2, 8, 16}, {1, 2, 8, 16}};

    string compile_info_string = GetCompileInfoString();
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct ApplyAdagradDTilingCompileInfo {};
    ApplyAdagradDTilingCompileInfo compile_info;

    std::string op_type("ApplyAdagradD");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                        .SetOpType("ApplyAdagradD")
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&var_shape, &accum_shape, &lr_shape, &grad_shape})
                        .OutputShapes({&var_out_shape, &accum_out_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    if (expected_success) {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    } else {
        EXPECT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(ApplyAdagradDTiling, apply_adagrad_d_float32_success)
{
    RunTilingTest(ge::DT_FLOAT);
}

TEST_F(ApplyAdagradDTiling, apply_adagrad_d_float16_success)
{
    RunTilingTest(ge::DT_FLOAT16);
}

TEST_F(ApplyAdagradDTiling, apply_adagrad_d_bf16_success)
{
    RunTilingTest(ge::DT_BF16);
}

TEST_F(ApplyAdagradDTiling, apply_adagrad_d_empty_input)
{
    // empty shape: totalElems == 0, tiling should still return success (sets all to 0)
    gert::StorageShape var_shape   = {{0}, {0}};
    gert::StorageShape accum_shape = {{0}, {0}};
    gert::StorageShape lr_shape    = {{1}, {1}};
    gert::StorageShape grad_shape  = {{0}, {0}};
    gert::StorageShape var_out_shape   = {{0}, {0}};
    gert::StorageShape accum_out_shape = {{0}, {0}};

    string compile_info_string = GetCompileInfoString();
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct ApplyAdagradDTilingCompileInfo {};
    ApplyAdagradDTilingCompileInfo compile_info;

    std::string op_type("ApplyAdagradD");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                        .SetOpType("ApplyAdagradD")
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&var_shape, &accum_shape, &lr_shape, &grad_shape})
                        .OutputShapes({&var_out_shape, &accum_out_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
