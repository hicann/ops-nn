/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_AddLayerNormQuant_proto.cpp
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
#include "../../../op_graph/add_layer_norm_quant_proto.h"

class AddLayerNormQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLayerNormQuant Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLayerNormQuant Proto Test TearDown" << std::endl;
    }
};

TEST_F(AddLayerNormQuant, AddLayerNormQuant_infershape_case_dynamic)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape input_shape = {{24, 1, 11264}, {24, 1, 11264}};
        gert::StorageShape gamma_shape = {
            {
                11264,
            },
            {
                11264,
            }};
        gert::StorageShape out_shape = {{24, 1, 11264}, {24, 1, 11264}};
        gert::StorageShape reduce_shape = {{24, 1, 1}, {24, 1, 1}};

        auto holder =
            gert::InferShapeContextFaker()
                .NodeIoNum(7, 5)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .InputShapes(
                    {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .NodeAttrs(
                    {{"quant_mode", ge::AnyValue::CreateFrom<string>("dynamic")},
                     {"epsilon", ge::AnyValue::CreateFrom<float>(0.01)},
                     {"additional_output", ge::AnyValue::CreateFrom<bool>(true)}})
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetInputShape(0)->GetDim(0), 24);
        EXPECT_EQ(context->GetInputShape(0)->GetDim(1), 1);
        EXPECT_EQ(context->GetInputShape(0)->GetDim(2), 11264);

        EXPECT_EQ(context->GetOutputShape(3)->GetDim(0), 24);
        EXPECT_EQ(context->GetOutputShape(3)->GetDim(1), 1);

        EXPECT_EQ(context->GetOutputShape(3)->GetDimNum(), 2);
    }
}
