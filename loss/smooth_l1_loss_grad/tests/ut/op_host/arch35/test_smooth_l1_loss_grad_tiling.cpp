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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

using namespace ge;
using namespace ut_util;

namespace optiling {
struct SmoothL1LossGradCompileInfo {};
} // namespace optiling

class SmoothL1LossGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SmoothL1LossGradTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SmoothL1LossGradTilingTest TearDown" << std::endl; }
};

static void DoTilingTest(std::initializer_list<int64_t> predictDims, std::initializer_list<int64_t> labelDims,
                         std::initializer_list<int64_t> doutDims, ge::DataType predictDtype, float sigma,
                         ge::graphStatus expectedStatus, ge::DataType labelDtype = ge::DT_UNDEFINED,
                         ge::DataType doutDtype = ge::DT_UNDEFINED, ge::Format predictFormat = ge::FORMAT_ND,
                         ge::Format labelFormat = ge::FORMAT_ND, ge::Format doutFormat = ge::FORMAT_ND,
                         ge::Format outputFormat = ge::FORMAT_ND)
{
    std::string opType("SmoothL1LossGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    std::string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}
    })";
    std::map<std::string, std::string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SmoothL1LossGradCompileInfo compileInfo;
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    gert::StorageShape predictShape = {predictDims, predictDims};
    gert::StorageShape labelShape = {labelDims, labelDims};
    gert::StorageShape doutShape = {doutDims, doutDims};
    ge::DataType effectiveLabelDtype = labelDtype == ge::DT_UNDEFINED ? predictDtype : labelDtype;
    ge::DataType effectiveDoutDtype = doutDtype == ge::DT_UNDEFINED ? predictDtype : doutDtype;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .OutputShapes({&predictShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, predictDtype, predictFormat, predictFormat)
                      .NodeInputTd(1, effectiveLabelDtype, labelFormat, labelFormat)
                      .NodeInputTd(2, effectiveDoutDtype, doutFormat, doutFormat)
                      .NodeOutputTd(0, predictDtype, outputFormat, outputFormat)
                      .NodeAttrs({{"sigma", Ops::NN::AnyValue::CreateFrom<float>(sigma)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(ctx->GetPlatformInfo(), nullptr);
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(ctx), expectedStatus);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_fp32_accepted)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_INT8, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dtype_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_FLOAT16);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_exact_shape_mismatch_rejected)
{
    DoTilingTest({2, 6}, {3, 4}, {2, 6}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_sigma_zero_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 0.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_unsupported_format_rejected)
{
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_rank_9_rejected)
{
    DoTilingTest({1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}, ge::DT_FLOAT16,
                 1.0f, ge::GRAPH_FAILED);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_label_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_INT8);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_unsupported_dtype_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_INT8);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_dtype_mismatch_rejected)
{
    DoTilingTest({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED, ge::DT_FLOAT16);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_label_unsupported_format_rejected)
{
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_unsupported_format_rejected)
{
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_output_unsupported_format_rejected)
{
    DoTilingTest({1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED, ge::DT_UNDEFINED,
                 ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCHW);
}

TEST_F(SmoothL1LossGradTilingTest, tiling_dout_shape_mismatch_rejected)
{
    DoTilingTest({4}, {4}, {2, 2}, ge::DT_FLOAT, 1.0f, ge::GRAPH_FAILED);
}
