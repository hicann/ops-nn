/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/group_norm_grad_proto.h"

class GroupNormGradProto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupNormGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupNormGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(GroupNormGradProto, group_norm_grad_infershape_test)
{
    ge::op::GroupNormGrad op;
    op.UpdateInputDesc("dy", create_desc({24, 1152, 64, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({24, 32}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({24, 32}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({24, 1152, 64, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1152}, ge::DT_FLOAT16));
    op.SetAttr("num_groups", 32);
    Runtime2TestParam param{{"num_groups"}};

    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc1 = op.GetOutputDescByName("dx");
    auto output_desc2 = op.GetOutputDescByName("dgamma");
    auto output_desc3 = op.GetOutputDescByName("dbeta");
    std::vector<int64_t> expected_output_shape1 = {24, 1152, 64, 64};
    std::vector<int64_t> expected_output_shape2 = {1152};
    std::vector<int64_t> expected_output_shape3 = {1152};
    EXPECT_EQ(output_desc1.GetShape().GetDims(), expected_output_shape1);
    EXPECT_EQ(output_desc2.GetShape().GetDims(), expected_output_shape2);
    EXPECT_EQ(output_desc3.GetShape().GetDims(), expected_output_shape3);
}

TEST_F(GroupNormGradProto, group_norm_grad_inferdtype_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(5, 3)
                                  .IrInstanceNum({1, 1, 1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
