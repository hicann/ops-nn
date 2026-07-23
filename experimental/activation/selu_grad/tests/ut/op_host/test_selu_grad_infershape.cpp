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

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

class SeluGradInferShapeTest : public testing::Test {};

TEST_F(SeluGradInferShapeTest, same_shape_success)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape gradientsShape = {2, 3, 4, 5};
    gert::Shape outputsShape = {2, 3, 4, 5};
    gert::Shape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* inferredShape = context->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    ASSERT_EQ(inferredShape->GetDimNum(), gradientsShape.GetDimNum());
    for (size_t i = 0; i < gradientsShape.GetDimNum(); ++i) {
        EXPECT_EQ(inferredShape->GetDim(i), gradientsShape.GetDim(i));
    }
}

TEST_F(SeluGradInferShapeTest, shape_mismatch_failed)
{
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SeluGrad");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape gradientsShape = {2, 3, 4};
    gert::Shape outputsShape = {2, 3, 5};
    gert::Shape yShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gradientsShape, &outputsShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
