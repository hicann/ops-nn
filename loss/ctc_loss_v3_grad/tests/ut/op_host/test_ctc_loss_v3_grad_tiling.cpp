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
 * \file test_ctc_loss_v3_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ctc_loss_v3_grad_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"
#include "ut_op_util.h"

using namespace std;
using namespace ge;

class CTCLossV3GradTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CTCLossV3GradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CTCLossV3GradTiling TearDown" << std::endl;
    }
};

void TestCTCLossV3GradTiling(
    gert::StorageShape& gradOutShape, gert::StorageShape& logProbsShape, gert::StorageShape& targetsShape,
    gert::StorageShape& inputLengthsShape, gert::StorageShape& targetLengthsShape, gert::StorageShape& lossShape,
    gert::StorageShape& logAlphaShape, gert::StorageShape& gradShape,
    std::vector<std::pair<std::string, Ops::NN::AnyValue>>& AttrList, ge::DataType gradDataType,
    ge::DataType indicesDataType, uint64_t expectTilingKey)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    optiling::CTCLossV3GradCompileInfo compileInfo;

    std::string op_type("CTCLossV3Grad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // TilingParseFunc simulate
    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING_910B.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3Grad")
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradOutShape, &logProbsShape, &targetsShape, &inputLengthsShape, &targetLengthsShape,
                           &lossShape, &logAlphaShape})
                      .OutputShapes({&gradShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                           {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto realTilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(realTilingKey, expectTilingKey);
}

TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tilingkey_0)
{
    std::cout << "run case: " << "ctc_loss_v3_grad_tilingkey_0" << std::endl;
    gert::StorageShape gradOutShape = {{1}, {1}};
    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape inputLengthsShape = {{1}, {1}};
    gert::StorageShape targetLengthsShape = {{1}, {1}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};
    gert::StorageShape gradShape = {{32, 1, 1024}, {32, 1, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {};
    TestCTCLossV3GradTiling(
        gradOutShape, logProbsShape, targetsShape, inputLengthsShape, targetLengthsShape, lossShape, logAlphaShape,
        gradShape, attrList, ge::DT_FLOAT, ge::DT_INT64, 0);
}