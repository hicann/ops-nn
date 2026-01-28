/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_mx_quant_with_dual_axis_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/dynamic_mx_quant_with_dual_axis_proto.h"

using namespace ge;

class DynamicMxQuantWithDualAxisTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_mx_quant_with_dual_axis test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_mx_quant_with_dual_axis test TearDown" << std::endl;
    }
};

TEST_F(DynamicMxQuantWithDualAxisTest, dynamic_mx_quant_with_dual_axis_case_1) {
    ge::op::DynamicMxQuantWithDualAxis quant_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({128, 128});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);
    quant_op.UpdateInputDesc("x", XDesc);

    Runtime2TestParam param{{"round_mode", "dst_type", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(quant_op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = quant_op.GetOutputDesc(0);
    auto output_scale1_desc = quant_op.GetOutputDesc(1);
    auto output_y2_desc = quant_op.GetOutputDesc(2);
    auto output_scale2_desc = quant_op.GetOutputDesc(3);
    std::vector<int64_t> expected_y1_shape = {128, 128};
    std::vector<int64_t> expected_scale1_shape = {128, 2, 2};
    std::vector<int64_t> expected_y2_shape = {128, 128};
    std::vector<int64_t> expected_scale2_shape = {2, 128, 2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale2_shape);
}