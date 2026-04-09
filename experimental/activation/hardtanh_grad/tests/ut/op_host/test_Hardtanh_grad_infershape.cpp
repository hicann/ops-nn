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
 * \file test_HardtanhGrad_proto.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/hardtanh_grad_proto.h"

class HardtanhGradInfershapeTest:public testing::Test{
    protected:
        static void SetUpTestCase(){
            std::cout<<"HardtanhGradInfershapeTest SetUp"<<std::endl;
        }

        static void TearDownTestCase(){
            std::cout<<"HardtanhGradInfershapeTest TearDown"<<std::endl;
        }
};


TEST_F(HardtanhGradInfershapeTest, hardtanh_grad_infershape_diff_test){
    ge::op::HardtanhGrad op;
    auto tensor_desc = create_desc_shape_range({-1, 8, 375},
                                               ge::DT_FLOAT, ge::FORMAT_ND,
                                               {16, 8, 375},
                                               ge::FORMAT_ND, {{16, 16}, {8, 8}, {375, 375}});

    op.UpdateInputDesc("result", tensor_desc);
    op.UpdateInputDesc("grad", tensor_desc);
    float min_val = -1.0;
    float max_val = 1.0;
    op.SetAttr("min_val", min_val);
    op.SetAttr("max_val", max_val);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {-1, 8, 375};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
    std::vector<std::pair<int64_t, int64_t>> output_shape_range;
    EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(HardtanhGradInfershapeTest, hardtanh_grad_infershape_diff_test2){
    ge::op::HardtanhGrad op;
    auto tensor_desc1 = create_desc_shape_range({-1, 8, 35},
                                                ge::DT_FLOAT16, ge::FORMAT_ND,
                                                {16, 8, 35},
                                                ge::FORMAT_ND, {{16, 16}, {8, 8}, {35, 35}});
    auto tensor_desc2 = create_desc_shape_range({-1, 8, 35},
                                                ge::DT_FLOAT, ge::FORMAT_ND,
                                                {16, 8, 35},
                                                ge::FORMAT_ND, {{16, 16}, {8, 8}, {35, 35}});
    op.UpdateInputDesc("result", tensor_desc1);
    op.UpdateInputDesc("grad", tensor_desc2);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}
