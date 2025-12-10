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
 * \file test_multi_scale_deformable_attn_function_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/multi_scale_deformable_attn_function_proto.h"

class MultiScaleDeformableAttnFunProto : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MultiScaleDeformableAttnFunProto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MultiScaleDeformableAttnFunProto Test TearDown" << std::endl;
    }
};

TEST_F(MultiScaleDeformableAttnFunProto, multi_scale_deformable_attn_function_fp32) {
    ge::op::MultiScaleDeformableAttnFunction op;
    op.UpdateInputDesc("value", create_desc({8, 2, 1, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sampling_locations", create_desc({5, 70, 1, 1, 6, 2}, ge::DT_FLOAT));

    std::vector<int64_t> expectedOutShape = {8, 70, 4};
    Runtime2TestParam param {};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputTensor = op.GetOutputDesc("output");
    EXPECT_EQ(outputTensor.GetShape().GetDims(), expectedOutShape);
}