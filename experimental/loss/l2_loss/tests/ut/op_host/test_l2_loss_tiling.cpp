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

class L2LossTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "L2LossTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "L2LossTiling TearDown" << endl;
    }
};

static const string kCompileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 48}
                        })";

struct L2LossTilingCompileInfo {};

static ge::graphStatus RunL2LossTiling(ge::DataType dtype)
{
    gert::StorageShape x_shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape y_shape = {{1}, {1}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(kCompileInfoString.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    L2LossTilingCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("L2Loss")->tiling;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(1, 1)
            .Inputs({const_cast<char*>(kCompileInfoString.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    auto holder = gert::TilingContextFaker()
                        .SetOpType("L2Loss")
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1})
                        .InputShapes({&x_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return tiling_func(tiling_context);
}

TEST_F(L2LossTiling, l2_loss_tiling_fp32_success)
{
    EXPECT_EQ(RunL2LossTiling(ge::DT_FLOAT), ge::GRAPH_SUCCESS);
}

TEST_F(L2LossTiling, l2_loss_tiling_fp16_success)
{
    EXPECT_EQ(RunL2LossTiling(ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
}

TEST_F(L2LossTiling, l2_loss_tiling_bf16_success)
{
    EXPECT_EQ(RunL2LossTiling(ge::DT_BF16), ge::GRAPH_SUCCESS);
}
