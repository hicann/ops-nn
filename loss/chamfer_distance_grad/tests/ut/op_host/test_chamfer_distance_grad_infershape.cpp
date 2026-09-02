/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * @file test_reducemax_d_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "../../../op_graph/chamfer_distance_grad_proto.h"

class ChamferDistanceGradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ChamferDistanceGradTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ChamferDistanceGradTest TearDown" << std::endl; }
};
TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_fp32)
{
    ge::op::ChamferDistanceGrad op;
    op.UpdateInputDesc("xyz1",
                       create_desc_with_ori({20, 15, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {20, 15, 2}, ge::FORMAT_ND));
    op.UpdateInputDesc("xyz2",
                       create_desc_with_ori({20, 15, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {20, 15, 2}, ge::FORMAT_ND));
    op.UpdateInputDesc("idx1", create_desc_with_ori({20, 15}, ge::DT_INT32, ge::FORMAT_ND, {20, 15}, ge::FORMAT_ND));
    op.UpdateInputDesc("idx2", create_desc_with_ori({20, 15}, ge::DT_INT32, ge::FORMAT_ND, {20, 15}, ge::FORMAT_ND));
    op.UpdateInputDesc("grad_dist1",
                       create_desc_with_ori({20, 15}, ge::DT_FLOAT, ge::FORMAT_ND, {20, 15}, ge::FORMAT_ND));
    op.UpdateInputDesc("grad_dist2",
                       create_desc_with_ori({20, 15}, ge::DT_FLOAT, ge::FORMAT_ND, {20, 15}, ge::FORMAT_ND));
    std::vector<int64_t> expected_output_shape = {20, 15, 2};
    Runtime2TestParam chamfer_distance_grad_param{{}};
    EXPECT_EQ(InferShapeTest(op, chamfer_distance_grad_param), ge::GRAPH_SUCCESS);
    auto output0_desc = op.GetOutputDesc(0);
    auto output1_desc = op.GetOutputDesc(1);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_shape;
    gert::Shape xyz1Shape = {-1, -1, -1};
    gert::Shape xyz2Shape = {-1, -1, -1};
    gert::Shape idx1Shape = {-1, -1};
    gert::Shape idx2Shape = {-1, -1};
    gert::Shape gradDist1Shape = {-1, -1};
    gert::Shape gradDist2Shape = {-1, -1};
    gert::Shape gradXyz1Shape = {0, 0, 0};
    gert::Shape gradXyz2Shape = {0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 2)
                      .InputShapes({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape})
                      .OutputShapes({&gradXyz1Shape, &gradXyz2Shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto out0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto out1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expectedOutputShape = {-1, -1, -1};
    ASSERT_EQ(Ops::Base::ToString(*out0), Ops::Base::ToString(expectedOutputShape));
    ASSERT_EQ(Ops::Base::ToString(*out1), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_partial_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_shape;
    gert::Shape xyz1Shape = {4, -1, 2};
    gert::Shape xyz2Shape = {4, -1, 2};
    gert::Shape idx1Shape = {4, -1};
    gert::Shape idx2Shape = {4, -1};
    gert::Shape gradDist1Shape = {4, -1};
    gert::Shape gradDist2Shape = {4, -1};
    gert::Shape gradXyz1Shape = {0, 0, 0};
    gert::Shape gradXyz2Shape = {0, 0, 0};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 2)
                      .InputShapes({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape})
                      .OutputShapes({&gradXyz1Shape, &gradXyz2Shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto out0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto out1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expectedOutputShape = {4, -1, 2};
    ASSERT_EQ(Ops::Base::ToString(*out0), Ops::Base::ToString(expectedOutputShape));
    ASSERT_EQ(Ops::Base::ToString(*out1), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_mixed_partial_unknown_and_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_shape;
    gert::Shape xyz1Shape = {4, -1, 2};
    gert::Shape xyz2Shape = {4, -1, 2};
    gert::Shape idx1Shape = {4, -1};
    gert::Shape idx2Shape = {4, -1};
    gert::Shape gradDist1Shape = {-2};
    gert::Shape gradDist2Shape = {4, -1};
    gert::Shape gradXyz1Shape = {0};
    gert::Shape gradXyz2Shape = {0};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 2)
                      .InputShapes({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape})
                      .OutputShapes({&gradXyz1Shape, &gradXyz2Shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto out0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto out1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*out0), Ops::Base::ToString(expectedOutputShape));
    ASSERT_EQ(Ops::Base::ToString(*out1), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_shape;
    gert::Shape xyz1Shape = {-2};
    gert::Shape xyz2Shape = {-2};
    gert::Shape idx1Shape = {-2};
    gert::Shape idx2Shape = {-2};
    gert::Shape gradDist1Shape = {-2};
    gert::Shape gradDist2Shape = {-2};
    gert::Shape gradXyz1Shape = {0};
    gert::Shape gradXyz2Shape = {0};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 2)
                      .InputShapes({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape})
                      .OutputShapes({&gradXyz1Shape, &gradXyz2Shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto out0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto out1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*out0), Ops::Base::ToString(expectedOutputShape));
    ASSERT_EQ(Ops::Base::ToString(*out1), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_dtype_fp32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_datatype;
    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_FLOAT;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(6, 2)
                                 .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes(
                                     {&inputRef1, &inputRef1, &inputRef2, &inputRef2, &inputRef1, &inputRef1})
                                 .OutputDataTypes({&outputRef, &outputRef})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
        EXPECT_EQ(context->GetOutputDataType(1), outputRef);
    }
}
