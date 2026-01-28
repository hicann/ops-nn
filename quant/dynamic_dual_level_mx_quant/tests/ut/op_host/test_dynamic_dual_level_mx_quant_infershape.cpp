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
 * \file test_dynamic_dual_level_mx_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/dynamic_dual_level_mx_quant_proto.h"

using namespace ge;

class DynamicDualLevelMxQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_dual_level_mx_quant test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_dual_level_mx_quant test TearDown" << std::endl;
    }
};

TEST_F(DynamicDualLevelMxQuantTest, dynamic_dual_level_mx_quant_case_1) {
    ge::op::DynamicDualLevelMxQuant quant_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({128, 512});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);
    quant_op.UpdateInputDesc("x", XDesc);
	quant_op.SetAttr("round_mode", "rint");
	quant_op.SetAttr("level0_block_size", 512);
	quant_op.SetAttr("level1_block_size", 32);

    Runtime2TestParam param{{"round_mode", "level0_block_size", "level1_block_size"}};
    EXPECT_EQ(InferShapeTest(quant_op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = quant_op.GetOutputDesc(0);
    auto output_level0_scale_desc = quant_op.GetOutputDesc(1);
    auto output_level1_scale_desc = quant_op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {128, 512};
    std::vector<int64_t> expected_level0_scale_shape = {128, 1};
    std::vector<int64_t> expected_level1_scale_shape = {128, 8, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_level0_scale_desc.GetShape().GetDims(), expected_level0_scale_shape);
    EXPECT_EQ(output_level1_scale_desc.GetShape().GetDims(), expected_level1_scale_shape);
}