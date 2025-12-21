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
 * \file test_fake_quant_affine_cachemask_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>

#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/fake_quant_affine_cachemask_proto.h"

class FakeQuantAffineCachemaskTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "fake_quant_affine_cachemask test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "fake_quant_affine_cachemask test TearDown" << std::endl;
    }
};

TEST_F(FakeQuantAffineCachemaskTest, fake_quant_affine_cachemask_test_infer_shape)
{
    ge::op::FakeQuantAffineCachemask op;
    op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));

    Runtime2TestParam param;
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_y = op.GetOutputDescByName("y");
    auto output_mask = op.GetOutputDescByName("mask");
    std::vector<int64_t> expected_output_shape = {4, 3, 4};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_mask.GetShape().GetDims(), expected_output_shape);
}

TEST_F(FakeQuantAffineCachemaskTest, fake_quant_affine_cachemask_test_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantAffineCachemask"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantAffineCachemask")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT;
        ge::DataType output_mask_ref = ge::DT_BOOL;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(3, 2)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &zero_point_ref})
                                  .OutputDataTypes({&output_y_ref, &output_mask_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), zero_point_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_mask_ref);
    }
}
