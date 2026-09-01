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

#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "ut_op_util.h"
#include "../../../op_graph/bn3_d_training_reduce_grad_proto.h"

using namespace std;
using namespace ge;

class BN3DTrainingReduceGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BN3DTrainingReduceGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BN3DTrainingReduceGradInfershapeTest TearDown" << std::endl; }
};

// ============================================================================
// InferShape — 常规 5D 正例：y.shape = grads.shape
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, y_shape_equals_grads)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto* outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(outputDesc->GetDimNum(), 5u);
    ASSERT_EQ(outputDesc->GetDim(0), 2);
    ASSERT_EQ(outputDesc->GetDim(1), 3);
    ASSERT_EQ(outputDesc->GetDim(2), 4);
    ASSERT_EQ(outputDesc->GetDim(3), 5);
    ASSERT_EQ(outputDesc->GetDim(4), 6);
}

// ============================================================================
// InferShape — grads 与 x shape 不一致：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, grads_x_shape_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape xShape = {{2, 3, 4, 5, 7}, {2, 3, 4, 5, 7}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — 参数张量长度 != C：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, param_len_not_equal_channel)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape paramShape = {{4}, {4}}; // C = 3（dim1）或 6（dim4），4 均不匹配
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — rank 非法（grads 4D）：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, invalid_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape xShape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — 空 tensor（任一维为 0）：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, zero_dim_input)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 0, 4, 5, 6}, {2, 0, 4, 5, 6}};
    gert::StorageShape xShape = {{2, 0, 4, 5, 6}, {2, 0, 4, 5, 6}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — 空 tensor 逐轴枚举（0 依次置于 dim0..dim4）：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, zero_dim_each_axis)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    for (int axis = 0; axis < 5; ++axis) {
        int64_t dims[5] = {2, 3, 4, 5, 6};
        dims[axis] = 0;
        gert::StorageShape gradsShape = {{dims[0], dims[1], dims[2], dims[3], dims[4]},
                                         {dims[0], dims[1], dims[2], dims[3], dims[4]}};
        gert::StorageShape xShape = gradsShape;
        gert::StorageShape paramShape = {{3}, {3}};
        gert::StorageShape yShape = {{}, {}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(7, 1)
                          .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                          .InputShapes(
                              {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                          .OutputShapes({&yShape})
                          .Build();

        ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED) << "axis=" << axis;
    }
}

// ============================================================================
// InferShape — 多轴同时为 0（空 tensor 任一维为 0 即拒）：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, zero_dim_multi_axis)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{0, 0, 4, 5, 6}, {0, 0, 4, 5, 6}};
    gert::StorageShape xShape = {{0, 0, 4, 5, 6}, {0, 0, 4, 5, 6}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — 参数张量 (0,)（长度 0 ≠ C）：GRAPH_FAILED
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, param_shape_zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape paramShape = {{0}, {0}}; // 参数张量为空，长度 0 != C
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ============================================================================
// InferShape — 未知秩输入：输出传播 UNKNOWN_RANK
// ============================================================================
TEST_F(BN3DTrainingReduceGradInfershapeTest, unknown_rank_propagate)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BN3DTrainingReduceGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradsShape = {{-2}, {-2}}; // UNKNOWN_RANK：dim_num==1 且 dim0==UNKNOWN_DIM_NUM
    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape paramShape = {{3}, {3}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&gradsShape, &xShape, &paramShape, &paramShape, &paramShape, &paramShape, &paramShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto* outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(outputDesc->GetDimNum(), 1u);
    ASSERT_EQ(outputDesc->GetDim(0), -2);
}
