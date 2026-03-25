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
    static void SetUpTestCase()
    {
        std::cout << "SwigluMxQuantTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluMxQuantTest TearDown" << std::endl;
    }
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

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
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

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
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

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
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

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
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
    op.SetAttr("axis", 0);

    Runtime2TestParam param{{"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "group_mode", "axis", "dst_type", "round_mode", "scale_alg", "max_dtype_value"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

}  // namespace
