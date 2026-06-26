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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../../op_graph/swiglu_group_proto.h"

namespace {
const Runtime2TestParam kRuntimeParam{{"clamp_limit"}};

void UpdateInputX(ge::op::SwigluGroup& op, const std::vector<int64_t>& dims, ge::DataType dtype)
{
    ge::TensorDesc xDesc;
    ge::Shape xShape(dims);
    xDesc.SetDataType(dtype);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);
}

class SwigluGroupInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluGroupInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluGroupInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(SwigluGroupInferShapeTest, infer_shape_fp16)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {8, 128, 8192}, ge::DT_FLOAT16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({8, 128, 4096}));
}

TEST_F(SwigluGroupInferShapeTest, infer_shape_bf16)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {4, 64, 2048}, ge::DT_BF16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 64, 1024}));
}

TEST_F(SwigluGroupInferShapeTest, infer_shape_fp32)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {4, 512}, ge::DT_FLOAT);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 256}));
}

TEST_F(SwigluGroupInferShapeTest, infer_shape_unknown_rank)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {-2}, ge::DT_FLOAT16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({-2}));
}

TEST_F(SwigluGroupInferShapeTest, infer_shape_unknown_last_dim)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {4, 64, -1}, ge::DT_BF16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 64, -1}));
}

TEST_F(SwigluGroupInferShapeTest, infer_shape_error_invalid_last_dim)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {4, 64, 1023}, ge::DT_FLOAT16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_FAILED);
}

TEST_F(SwigluGroupInferShapeTest, infer_dtype_fp16)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_FLOAT16);

    EXPECT_EQ(InferDataTypeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
}

TEST_F(SwigluGroupInferShapeTest, infer_dtype_bf16)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_BF16);

    EXPECT_EQ(InferDataTypeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetDataType(), ge::DT_BF16);
}

TEST_F(SwigluGroupInferShapeTest, infer_dtype_fp32)
{
    ge::op::SwigluGroup op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_FLOAT);

    EXPECT_EQ(InferDataTypeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetDataType(), ge::DT_FLOAT);
}
} // namespace
