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

namespace {
// 负向用例公共构造器：inputShapes/outputShapes 列表中首个 nullptr 表示该输入/输出实例缺失，
// 其自身及之后的实例数全部置 0（防止后续实例前移错位），nullptr 之前的输入保持实例数 1；
// 通过双参数 IrInstanceNum 使 GetInputShape/GetOutputShape 按下标越界规则返回空指针
ge::graphStatus RunChamferDistanceGradInferShapeNeg(std::initializer_list<gert::Shape*> inputShapes,
                                                    std::initializer_list<gert::Shape*> outputShapes)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ChamferDistanceGrad")->infer_shape;
    std::vector<uint32_t> inputInstanceNum;
    std::vector<gert::Shape*> inputPtrs;
    for (auto* shape : inputShapes) {
        if (shape == nullptr) {
            break;
        }
        inputInstanceNum.push_back(1U);
        inputPtrs.push_back(shape);
    }
    while (inputInstanceNum.size() < 6U) {
        inputInstanceNum.push_back(0U);
    }
    std::vector<uint32_t> outputInstanceNum;
    for (auto* shape : outputShapes) {
        if (shape == nullptr) {
            break;
        }
        outputInstanceNum.push_back(1U);
    }
    while (outputInstanceNum.size() < 2U) {
        outputInstanceNum.push_back(0U);
    }

    gert::InferShapeContextFaker faker;
    faker.SetOpType("ChamferDistanceGrad").IrInstanceNum(inputInstanceNum, outputInstanceNum);
    faker.InputShapes(inputPtrs);
    static const std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32,
                                                          ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT};
    for (size_t i = 0; i < inputPtrs.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), inputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    return inferShapeFunc(holder.GetContext<gert::InferShapeContext>());
}
} // namespace

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_xyz1)
{
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {nullptr, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_xyz2)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, nullptr, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_idx1)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, nullptr, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_idx2)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, nullptr, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_grad_dist1)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(
        RunChamferDistanceGradInferShapeNeg({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, nullptr, &gradDist2Shape},
                                            {&gradXyz1Shape, &gradXyz2Shape}),
        ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_grad_dist2)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(
        RunChamferDistanceGradInferShapeNeg({&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, nullptr},
                                            {&gradXyz1Shape, &gradXyz2Shape}),
        ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_grad_xyz1)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {nullptr, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_null_grad_xyz2)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, nullptr}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_xyz_rank_invalid)
{
    gert::Shape xyz1Shape = {20, 15};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_pointwise_rank_invalid)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15, 1};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_coord_dim_invalid)
{
    gert::Shape xyz1Shape = {20, 15, 3};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 3};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_batch_dim_mismatch)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {21, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {21, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {21, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {21, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_point_dim_mismatch)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 16, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 16};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 16};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 16, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_idx1_incompatible)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 16};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_idx2_incompatible)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {21, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_grad_dist1_incompatible)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {21, 15};
    gert::Shape gradDist2Shape = {20, 15};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceGradTest, chamfer_distance_grad_infer_shape_grad_dist2_incompatible)
{
    gert::Shape xyz1Shape = {20, 15, 2};
    gert::Shape xyz2Shape = {20, 15, 2};
    gert::Shape idx1Shape = {20, 15};
    gert::Shape idx2Shape = {20, 15};
    gert::Shape gradDist1Shape = {20, 15};
    gert::Shape gradDist2Shape = {20, 16};
    gert::Shape gradXyz1Shape = {20, 15, 2};
    gert::Shape gradXyz2Shape = {20, 15, 2};
    ASSERT_EQ(RunChamferDistanceGradInferShapeNeg(
                  {&xyz1Shape, &xyz2Shape, &idx1Shape, &idx2Shape, &gradDist1Shape, &gradDist2Shape},
                  {&gradXyz1Shape, &gradXyz2Shape}),
              ge::GRAPH_FAILED);
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
