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
 * \file test_fake_quant_with_min_max_vars_per_channel_infershape.cpp
 * \brief op_proto UT for FakeQuantWithMinMaxVarsPerChannel.
 *
 * Covers:
 *   - InferShape: y.shape == x.shape on various ranks / C values
 *   - InferShape on a small last-axis (empty-like / minimal C=1) case
 *   - InferDataType: y.dtype == float
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/fake_quant_with_min_max_vars_per_channel_proto.h"

class FakeQuantWithMinMaxVarsPerChannelInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantWithMinMaxVarsPerChannel infershape SetUp" << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannel infershape TearDown" << std::endl;
    }
};

TEST_F(FakeQuantWithMinMaxVarsPerChannelInferShapeTest, infer_shape_rank2)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannel op;
    op.UpdateInputDesc("x", create_desc({8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({16}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto out_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected = {8, 16};
    EXPECT_EQ(out_y.GetShape().GetDims(), expected);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelInferShapeTest, infer_shape_rank4)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannel op;
    op.UpdateInputDesc("x", create_desc({2, 3, 4, 5}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({5}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({5}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto out_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected = {2, 3, 4, 5};
    EXPECT_EQ(out_y.GetShape().GetDims(), expected);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelInferShapeTest, infer_shape_rank1_min_c)
{
    // Smallest channel count.
    ge::op::FakeQuantWithMinMaxVarsPerChannel op;
    op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto out_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected = {1};
    EXPECT_EQ(out_y.GetShape().GetDims(), expected);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelInferShapeTest, infer_shape_large_c)
{
    ge::op::FakeQuantWithMinMaxVarsPerChannel op;
    op.UpdateInputDesc("x", create_desc({4, 1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("min", create_desc({1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({1024}, ge::DT_FLOAT));

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto out_y = op.GetOutputDescByName("y");
    std::vector<int64_t> expected = {4, 1024};
    EXPECT_EQ(out_y.GetShape().GetDims(), expected);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelInferShapeTest, infer_dtype_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxVarsPerChannel"), nullptr);
    auto infer_dtype_func = gert::OpImplRegistry::GetInstance()
                                .GetOpImpl("FakeQuantWithMinMaxVarsPerChannel")
                                ->infer_datatype;
    ASSERT_NE(infer_dtype_func, nullptr);

    ge::DataType x_ref = ge::DT_FLOAT;
    ge::DataType min_ref = ge::DT_FLOAT;
    ge::DataType max_ref = ge::DT_FLOAT;
    ge::DataType y_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&x_ref, &min_ref, &max_ref})
                              .OutputDataTypes({&y_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(infer_dtype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
