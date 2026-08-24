/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bn3_d_training_update_infershape.cpp
 * \brief InferShape UT for BN3DTrainingUpdate operator.
 *
 * Schema (aligned to op_host/bn3_d_training_update_infershape.cpp):
 *   Inputs  (7): x, sum, square_sum, scale, offset, mean, variance
 *   Outputs (5): y, mean, variance, batch_mean, batch_variance
 * InferShape contract:
 *   - y (output 0) follows x (input 0) shape
 *   - mean / variance / batch_mean / batch_variance (outputs 1..4)
 *     follow sum (input 1) shape
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/bn3_d_training_update_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {

// Common InferShape runner.
// xShape / statShape: shapes; xDtype / xFormat: for x (input 0) and y (output 0);
// stat tensors (1..6 / 1..4) use DT_FLOAT / FORMAT_ND throughout.
static void RunInferShape(const gert::Shape& xShapeRef, const gert::Shape& statShapeRef, ge::DataType xDtype,
                          ge::Format xFormat)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingUpdate"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingUpdate")->infer_shape;

    // Caller-owned shapes must be non-const for InferShapeContextFaker.
    gert::Shape xShape = xShapeRef;
    gert::Shape statShape = statShapeRef;
    gert::Shape yShape, meanOutShape, varianceOutShape, batchMeanShape, batchVarianceShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &statShape, &statShape, &statShape, &statShape, &statShape, &statShape})
                      .OutputShapes({&yShape, &meanOutShape, &varianceOutShape, &batchMeanShape, &batchVarianceShape})
                      .NodeInputTd(0, xDtype, xFormat, xFormat)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, xFormat, xFormat)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* inferCtx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(inferCtx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*inferCtx->GetOutputShape(0), *inferCtx->GetInputShape(0));
    EXPECT_EQ(*inferCtx->GetOutputShape(1), *inferCtx->GetInputShape(1));
    EXPECT_EQ(*inferCtx->GetOutputShape(2), *inferCtx->GetInputShape(1));
    EXPECT_EQ(*inferCtx->GetOutputShape(3), *inferCtx->GetInputShape(1));
    EXPECT_EQ(*inferCtx->GetOutputShape(4), *inferCtx->GetInputShape(1));
}

} // namespace

class BN3DTrainingUpdateInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BN3DTrainingUpdate Infershape Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "BN3DTrainingUpdate Infershape Test TearDown" << std::endl; }
};

// RANK=4 NCHW: x {2,3,4,5}, stats (C,) = {3}
TEST_F(BN3DTrainingUpdateInfershapeTest, rank4_nchw_infershape_test)
{
    RunInferShape({2, 3, 4, 5}, {3}, ge::DT_FLOAT16, ge::FORMAT_NCHW);
}

// RANK=5 NCDHW: x {2,3,4,5,6}, stats (C,) = {3}
TEST_F(BN3DTrainingUpdateInfershapeTest, rank5_ncdhw_infershape_test)
{
    RunInferShape({2, 3, 4, 5, 6}, {3}, ge::DT_BF16, ge::FORMAT_NCDHW);
}

// RANK=4 FLOAT32: x {1,4,8,8}, stats (C,) = {4}
TEST_F(BN3DTrainingUpdateInfershapeTest, rank4_float32_infershape_test)
{
    RunInferShape({1, 4, 8, 8}, {4}, ge::DT_FLOAT, ge::FORMAT_NCHW);
}

// x UNKNOWN_RANK (-2): y propagates unknown rank (dim_num=1, dim[0]=-2), stats follow sum.
TEST_F(BN3DTrainingUpdateInfershapeTest, rank_unknown_minus2_infershape_test)
{
    RunInferShape({-2}, {3}, ge::DT_FLOAT16, ge::FORMAT_NCHW);
}

// x with dynamic dim (-1): dims propagate through plain copy.
TEST_F(BN3DTrainingUpdateInfershapeTest, dynamic_dim_minus1_infershape_test)
{
    RunInferShape({-1, 3, 4, 5}, {3}, ge::DT_FLOAT, ge::FORMAT_NCHW);
}

// sum UNKNOWN_RANK (-2): statistics outputs (1..4) propagate unknown rank, y follows x.
TEST_F(BN3DTrainingUpdateInfershapeTest, stat_unknown_minus2_infershape_test)
{
    RunInferShape({2, 3, 4, 5}, {-2}, ge::DT_FLOAT, ge::FORMAT_NCHW);
}
