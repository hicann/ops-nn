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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/swiglu_mx_quant_proto.h"

namespace {
class SwigluMxQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluMxQuantTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SwigluMxQuantTest TearDown" << std::endl; }
};

TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_case_0_fp16)
{
    ge::op::SwigluMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8, 128, 8192});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias",
                             "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {8, 128, 4096};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_case_1_bf16)
{
    ge::op::SwigluMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 2048});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias",
                             "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {4, 64, 1024};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_case_dynamic_shape)
{
    ge::op::SwigluMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({-2});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias",
                             "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {-2};
    std::vector<int64_t> expectedScaleShape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_error_invalid_dim)
{
    ge::op::SwigluMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 1023});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias",
                             "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_error_invalid_axis)
{
    ge::op::SwigluMxQuant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 2048});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);
    op.SetAttr("axis", 3);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias",
                             "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

namespace {
void UpdateInputXAndGroup(ge::op::SwigluMxQuant& op, const std::vector<int64_t>& xDims,
                          const std::vector<int64_t>& groupDims, ge::DataType xDtype = ge::DT_FLOAT16)
{
    ge::TensorDesc xDesc;
    ge::Shape xShape(xDims);
    xDesc.SetDataType(xDtype);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    ge::TensorDesc groupDesc;
    ge::Shape groupShape(groupDims);
    groupDesc.SetDataType(ge::DT_INT32);
    groupDesc.SetShape(groupShape);
    groupDesc.SetOriginShape(groupShape);
    op.UpdateInputDesc("group_index", groupDesc);
}

const Runtime2TestParam kGroupParam{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha",
                                     "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg",
                                     "max_dtype_value"}};
} // namespace

// group_index 存在，但 activate_dim 与 axis 均为尾轴（-1）时，x 允许大于 2 维（修复前会误报错）
TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_group_rank3_tail_axis_success)
{
    ge::op::SwigluMxQuant op;
    UpdateInputXAndGroup(op, {4, 64, 2048}, {8});
    op.SetAttr("activate_dim", static_cast<int64_t>(-1));
    op.SetAttr("axis", static_cast<int64_t>(-1));

    EXPECT_EQ(InferShapeTest(op, kGroupParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, 64, 1024}));
}

// group_index 存在，且 axis 为非尾轴（-2）时，x 必须为 2 维，rank=3 应报错
TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_group_rank3_nontail_axis_failed)
{
    ge::op::SwigluMxQuant op;
    UpdateInputXAndGroup(op, {4, 64, 2048}, {8});
    op.SetAttr("activate_dim", static_cast<int64_t>(-1));
    op.SetAttr("axis", static_cast<int64_t>(-2));

    EXPECT_EQ(InferShapeTest(op, kGroupParam), ge::GRAPH_FAILED);
}

// group_index 存在，axis 为非尾轴（-2），x 为 2 维时应通过
TEST_F(SwigluMxQuantTest, SwigluMxQuant_infershape_group_rank2_nontail_axis_success)
{
    ge::op::SwigluMxQuant op;
    UpdateInputXAndGroup(op, {64, 2048}, {8});
    op.SetAttr("activate_dim", static_cast<int64_t>(-1));
    op.SetAttr("axis", static_cast<int64_t>(-2));

    EXPECT_EQ(InferShapeTest(op, kGroupParam), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({64, 1024}));
}

} // namespace
