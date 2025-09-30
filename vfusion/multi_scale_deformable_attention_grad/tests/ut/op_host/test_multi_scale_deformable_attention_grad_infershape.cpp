/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_multi_scale_deformable_attention_grad_proto.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/multi_scale_deformable_attention_grad_proto.h"

class MultiScaleDeformableAttentionGradProto : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MultiScaleDeformableAttentionGradProto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MultiScaleDeformableAttentionGradProto Test TearDown" << std::endl;
    }
};

TEST_F(MultiScaleDeformableAttentionGradProto, multi_scale_deformable_attention_grad_fp32) {
    ge::op::MultiScaleDeformableAttentionGrad op;
    op.UpdateInputDesc("value", create_desc({8, 2, 3, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sampling_locations", create_desc({1, 2, 5, 6, 7, 70}, ge::DT_FLOAT));
    op.UpdateInputDesc("attention_weights", create_desc({1, 2, 3, 4, 5, 6}, ge::DT_FLOAT));

    std::vector<int64_t> expectedValueShape = {8, 2, 3, 4};
    std::vector<int64_t> expectedSampleShape = {1, 2, 5, 6, 7, 70};
    std::vector<int64_t> expectedAttnShape = {1, 2, 5, 6, 70};
    Runtime2TestParam param {};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto gradValue = op.GetOutputDesc("grad_value");
    auto gradSampling = op.GetOutputDesc("grad_sampling_locations");
    auto gradAttn = op.GetOutputDesc("grad_attention_weights");
    EXPECT_EQ(gradValue.GetShape().GetDims(), expectedValueShape);
    EXPECT_EQ(gradSampling.GetShape().GetDims(), expectedSampleShape);
    EXPECT_EQ(gradAttn.GetShape().GetDims(), expectedAttnShape);
}

TEST_F(MultiScaleDeformableAttentionGradProto, multi_scale_deformable_attention_grad_dtype) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MultiScaleDeformableAttentionGrad"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MultiScaleDeformableAttentionGrad")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_FLOAT;
        auto contextHolder = gert::InferDataTypeContextFaker()
            .NodeIoNum(6, 3)
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&inputRef1, &inputRef2, &inputRef2, &inputRef1, &inputRef1, &inputRef1})
            .OutputDataTypes({&outputRef, &outputRef, &outputRef})
            .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}