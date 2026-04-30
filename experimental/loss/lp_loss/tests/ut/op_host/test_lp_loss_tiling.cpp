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
#include <map>
#include <string>
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

namespace {

const char* kCompileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 48}
                        })";

struct LpLossTilingCompileInfo {
};

void InitPlatForm(
    fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    GetPlatFormInfos(kCompileInfoString, socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

void DoLpLossTilingCase(DataType inputDtype, const std::string& reduction)
{
    gert::StorageShape predictShape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape labelShape = {{1, 2, 8, 16}, {1, 2, 8, 16}};
    gert::StorageShape outputShape =
        (reduction == "none") ? gert::StorageShape{{1, 2, 8, 16}, {1, 2, 8, 16}} : gert::StorageShape{{}, {}};

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    fe::PlatFormInfos platformInfo;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    LpLossTilingCompileInfo compileInfo;

    std::string opType("LpLoss");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(kCompileInfoString), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("LpLoss")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&predictShape, &labelShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"p", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
}

} // namespace

class LpLossTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "LpLossTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "LpLossTiling TearDown" << endl;
    }
};

TEST_F(LpLossTiling, lp_loss_float32_none_success)
{
    DoLpLossTilingCase(ge::DT_FLOAT, "none");
}

TEST_F(LpLossTiling, lp_loss_float16_mean_success)
{
    DoLpLossTilingCase(ge::DT_FLOAT16, "mean");
}

TEST_F(LpLossTiling, lp_loss_bf16_sum_success)
{
    DoLpLossTilingCase(ge::DT_BF16, "sum");
}