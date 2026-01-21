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
 * \file test_thnn_fused_lstm_cell_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct ThnnFusedLstmCellGradCompileInfo {
    uint32_t aicCoreNum{0};
    int64_t sysWorkspaceSize{0};
    int64_t ubSizePlatForm{0};
};

class ThnnFusedLstmCellGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ThnnFusedLstmCellGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ThnnFusedLstmCellGradTiling TearDown" << std::endl;
    }
};

void TestThnnFusedLstmCellGradTiling(
    int64_t batch, int64_t hiddenSize, ge::DataType dataType,
    uint64_t expectTilingKey)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape hShape = {{batch, hiddenSize}, {batch, hiddenSize}};
    gert::StorageShape gatesShape = {{batch, 4 * hiddenSize}, {batch, 4 * hiddenSize}};
    gert::StorageShape bShape = {{4 * hiddenSize}, {4 * hiddenSize}};

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
    ThnnFusedLstmCellGradCompileInfo compileInfo;

    std::string op_type("ThnnFusedLstmCellGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ThnnFusedLstmCellGrad")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&hShape, &hShape, &hShape, &hShape, &gatesShape})
                      .OutputShapes({&gatesShape, &hShape, &bShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"has_bias", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(ThnnFusedLstmCellGradTiling, thnn_fused_lstm_cell_grad_tilingkey_0)
{
    std::cout << "run case: " << "thnn_fused_lstm_cell_grad_tilingkey_0" << std::endl;
    TestThnnFusedLstmCellGradTiling(40, 8, ge::DT_FLOAT, 0);
}
