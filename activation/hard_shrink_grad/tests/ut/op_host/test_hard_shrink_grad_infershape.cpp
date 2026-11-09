/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

// ----------------HardShrinkGrad--------------
class HardShrinkGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HardShrinkGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "HardShrinkGradInfershapeTest TearDown" << std::endl; }
};

TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_positive)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {3, 4, 5};
    gert::Shape selfShape = {3, 4, 5};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 3U);
    EXPECT_EQ(inferredShape->GetDim(0), 3);
    EXPECT_EQ(inferredShape->GetDim(1), 4);
    EXPECT_EQ(inferredShape->GetDim(2), 5);
}

// The output shape must mirror input 0 regardless of the declared output placeholder.
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_ignores_placeholder)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {7, 2};
    gert::Shape selfShape = {7, 2};
    gert::Shape outputShape = {1};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 2U);
    EXPECT_EQ(inferredShape->GetDim(0), 7);
    EXPECT_EQ(inferredShape->GetDim(1), 2);
}

// Negative: gradients and features with different ranks must be rejected (cross-input check).
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_rank_mismatch_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {1, 2, 3, 4, 5, 6, 7};
    gert::Shape selfShape = {1, 2, 3};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Negative: same rank but a per-dimension mismatch must be rejected (elementwise
// same-shape contract, aligned with the tiling check).
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_dim_mismatch_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {2, 3};
    gert::Shape selfShape = {3, 2};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// Scalar and (1,) are compared with the same normalization as the tiling guard
// (rank-0 is treated as {1}), so this pair stays legal on the graph path.
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_scalar_vs_one_accepted)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {};
    gert::Shape selfShape = {1};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 0U);
}

// Legal double-empty tensors keep succeeding (empty early-return contract).
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_double_empty_accepted)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {0};
    gert::Shape selfShape = {0};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 1U);
    EXPECT_EQ(inferredShape->GetDim(0), 0);
}

// An unknown-rank (-2) input bypasses the cross-input checks and propagates input 0 as-is.
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_unknown_rank_bypass_accepted)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {-1, -1, -1, -1, -1, -1};
    gert::Shape selfShape = {-2};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 6U);
}

// Mixed unknown-dim (-1) markers skip only the unknowable dimension; known dims still compare.
TEST_F(HardShrinkGradInfershapeTest, hard_shrink_grad_infershape_mixed_marker_dim_skip_accepted)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HardShrinkGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {-1, 5};
    gert::Shape selfShape = {13777067, 5};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradShape, &selfShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(inferredShape->GetDimNum(), 2U);
    EXPECT_EQ(inferredShape->GetDim(1), 5);
}
