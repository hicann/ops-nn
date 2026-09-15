/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "infershape_test_util.h"
#include "ut_op_common.h"

class MaskedScatterInferShapeTest : public testing::Test {};

TEST_F(MaskedScatterInferShapeTest, outputShapeFollowsInput)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaskedScatter");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::Shape inputShape = {2, 3, 4};
    gert::Shape maskShape = {2, 3, 4};
    gert::Shape updatesShape = {7};
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&inputShape, &maskShape, &updatesShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);
    auto inferredShape = context->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    ASSERT_EQ(inferredShape->GetDimNum(), 3U);
    EXPECT_EQ(inferredShape->GetDim(0), 2);
    EXPECT_EQ(inferredShape->GetDim(1), 3);
    EXPECT_EQ(inferredShape->GetDim(2), 4);
}

TEST_F(MaskedScatterInferShapeTest, outputDataTypeFollowsInput)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaskedScatter");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    for (auto inputDataType : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32}) {
        SCOPED_TRACE(static_cast<int>(inputDataType));
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(3, 1)
                          .IrInstanceNum({1, 1, 1}, {1})
                          .NodeInputTd(0, inputDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, inputDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), inputDataType);
    }
}
