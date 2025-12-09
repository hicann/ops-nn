/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_AddRmsNormDynamicQuant_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/add_rms_norm_dynamic_quant_proto.h"

class AddRmsNormDynamicQuant : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddRmsNormDynamicQuant Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddRmsNormDynamicQuant Proto Test TearDown" << std::endl;
    }
};

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_dynamic)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_shape;

    if (infer_shape_func != nullptr) {

        gert::StorageShape input_shape = {{1, 1, 16}, {1, 1, 16}};
        gert::StorageShape gamma_shape = {{16,}, {16,}};
        gert::StorageShape out_shape = {{1, 1, 16}, {1, 1, 16}};
        gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(5, 5)
                          .IrInstanceNum({1, 1, 1, 1, 1})
                          .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                          .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                          .NodeAttrs({
                              {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                          })
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        uint64_t dim_zero = 0;
        uint64_t dim_one = 1;
        uint64_t dim_two = 2;
        uint64_t shape_size_fir = 1;
        uint64_t shape_size_sec = 16;
        EXPECT_EQ(context->GetInputShape(0)->GetDim(dim_zero), shape_size_fir);
        EXPECT_EQ(context->GetInputShape(0)->GetDim(dim_one), shape_size_fir);
        EXPECT_EQ(context->GetInputShape(0)->GetDim(dim_two), shape_size_sec);
  }
}
