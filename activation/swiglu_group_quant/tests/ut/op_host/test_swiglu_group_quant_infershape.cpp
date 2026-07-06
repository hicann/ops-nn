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
#include "../../../op_graph/swiglu_group_quant_proto.h"

namespace {
const Runtime2TestParam kRuntimeParam{
    {"dst_type", "quant_mode", "block_size", "round_scale", "clamp_limit", "dst_type_max", "output_origin"}};

void UpdateInputX(ge::op::SwigluGroupQuant& op, const std::vector<int64_t>& dims, ge::DataType dtype)
{
    ge::TensorDesc xDesc;
    ge::Shape xShape(dims);
    xDesc.SetDataType(dtype);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);
}

class SwigluGroupQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluGroupQuantInferShapeTest TearDown" << std::endl; }
};

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_block_fp8)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {8, 128, 8192}, ge::DT_FLOAT16);
    op.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN));
    op.SetAttr("quant_mode", static_cast<int64_t>(0));

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({8, 128, 4096}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({8, 128, 32}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({8, 128, 4096}));
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_mx_fp8)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {4, 64, 2048}, ge::DT_BF16);
    op.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT8_E5M2));
    op.SetAttr("quant_mode", static_cast<int64_t>(1));
    op.SetAttr("round_scale", true);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 64, 1024}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({4, 64, 16, 2}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({4, 64, 1024}));
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_mx_fp4_packed)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_FLOAT16);
    op.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT4_E2M1));
    op.SetAttr("quant_mode", static_cast<int64_t>(1));
    op.SetAttr("round_scale", true);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 8, 256}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 8, 8, 2}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({2, 8, 512}));
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_unknown_rank)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {-2}, ge::DT_FLOAT16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({-2}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({-2}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({-2}));
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_unknown_last_dim)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {4, 64, -1}, ge::DT_BF16);
    op.SetAttr("quant_mode", static_cast<int64_t>(1));
    op.SetAttr("round_scale", true);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 64, -1}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({4, 64, -1}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({4, 64, -1}));
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_shape_error_invalid_last_dim)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {4, 64, 1023}, ge::DT_FLOAT16);

    EXPECT_EQ(InferShapeTest(op, kRuntimeParam), ge::GRAPH_FAILED);
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_dtype_mx_fp4)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_BF16);
    op.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT4_E1M2));
    op.SetAttr("quant_mode", static_cast<int64_t>(1));
    op.SetAttr("round_scale", true);

    EXPECT_EQ(InferDataTypeTest(op, kRuntimeParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetDataType(), ge::DT_FLOAT4_E1M2);
    EXPECT_EQ(op.GetOutputDesc(1).GetDataType(), ge::DT_FLOAT8_E8M0);
    EXPECT_EQ(op.GetOutputDesc(2).GetDataType(), ge::DT_BF16);
}

TEST_F(SwigluGroupQuantInferShapeTest, infer_dtype_error_invalid_dst_type)
{
    ge::op::SwigluGroupQuant op;
    UpdateInputX(op, {2, 8, 1024}, ge::DT_FLOAT16);
    op.SetAttr("dst_type", static_cast<int64_t>(ge::DT_FLOAT));

    EXPECT_EQ(InferDataTypeTest(op, kRuntimeParam), ge::GRAPH_FAILED);
}
} // namespace
