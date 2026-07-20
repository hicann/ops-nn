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
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_graph/inplace_apply_adagrad_da_proto.h"

using namespace ge;
using namespace ut_util;

class InplaceApplyAdagradDATilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "InplaceApplyAdagradDATilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "InplaceApplyAdagradDATilingTest TearDown" << std::endl;
    }
};

static void DoTilingTest(ge::DataType varDtype, ge::DataType gsDtype,
                         gert::StorageShape &varShape, gert::StorageShape &scalarShape, bool useLocking) {
    std::string opType("InplaceApplyAdagradDA");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}
    })";
    map<string, string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector *>(wsHolder.get());
    ge::Format fmt = ge::FORMAT_ND;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &varShape, &varShape, &varShape,
                                    &scalarShape, &scalarShape, &scalarShape, &scalarShape})
                      .OutputShapes({&varShape, &varShape, &varShape})
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .NodeInputTd(0, varDtype, fmt, fmt)
                      .NodeInputTd(1, varDtype, fmt, fmt)
                      .NodeInputTd(2, varDtype, fmt, fmt)
                      .NodeInputTd(3, varDtype, fmt, fmt)
                      .NodeInputTd(4, varDtype, fmt, fmt)
                      .NodeInputTd(5, varDtype, fmt, fmt)
                      .NodeInputTd(6, varDtype, fmt, fmt)
                      .NodeInputTd(7, gsDtype, fmt, fmt)
                      .NodeOutputTd(0, varDtype, fmt, fmt)
                      .NodeOutputTd(1, varDtype, fmt, fmt)
                      .NodeOutputTd(2, varDtype, fmt, fmt)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(useLocking)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext *ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx->GetPlatformInfo(), nullptr);
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(ctx), ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyAdagradDATilingTest, tiling_fp32_int32_1d) {
    gert::StorageShape varShape = {{1024}, {1024}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, ge::DT_INT32, varShape, scalarShape, false);
}

TEST_F(InplaceApplyAdagradDATilingTest, tiling_fp16_int64_2d) {
    gert::StorageShape varShape = {{8, 128}, {8, 128}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT16, ge::DT_INT64, varShape, scalarShape, false);
}

TEST_F(InplaceApplyAdagradDATilingTest, tiling_fp16_int32_3d) {
    gert::StorageShape varShape = {{4, 3, 4}, {4, 3, 4}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT16, ge::DT_INT32, varShape, scalarShape, false);
}

TEST_F(InplaceApplyAdagradDATilingTest, tiling_fp32_use_locking) {
    gert::StorageShape varShape = {{256}, {256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, ge::DT_INT32, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradDATilingTest, tiling_empty_tensor) {
    gert::StorageShape varShape = {{0}, {0}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, ge::DT_INT32, varShape, scalarShape, false);
}
