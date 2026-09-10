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

#include <vector>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

namespace {
using AnyValue = Ops::NN::AnyValue;

auto BuildShapeContext(gert::Shape* inputShape, gert::Shape* outputShape)
{
    return gert::InferShapeContextFaker()
        .NodeIoNum(1, 1)
        .IrInputNum({1})
        .InputShapes({inputShape})
        .OutputShapes({outputShape})
        .NodeAttrs({{"boundaries", AnyValue::CreateFrom<std::vector<float>>({1.0F, 3.0F, 5.0F})},
                    {"dtype", AnyValue::CreateFrom<int64_t>(ge::DT_INT32)},
                    {"right", AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
}

auto BuildDataTypeContext(int64_t outputDtype)
{
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType outputDataType = ge::DT_UNDEFINED;
    return gert::InferDataTypeContextFaker()
        .NodeIoNum(1, 1)
        .IrInputNum({1})
        .InputDataTypes({&inputDtype})
        .OutputDataTypes({&outputDataType})
        .NodeAttrs({{"boundaries", AnyValue::CreateFrom<std::vector<float>>({1.0F, 3.0F, 5.0F})},
                    {"dtype", AnyValue::CreateFrom<int64_t>(outputDtype)},
                    {"right", AnyValue::CreateFrom<bool>(false)}})
        .Build();
}

void RunInferShapeCase(gert::Shape inputShape)
{
    gert::Shape outputShape = {};
    auto holder = BuildShapeContext(&inputShape, &outputShape);

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Bucketize");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_shape(context), ge::GRAPH_SUCCESS);

    const auto* inferredShape = context->GetOutputShape(0);
    ASSERT_NE(inferredShape, nullptr);
    EXPECT_EQ(*inferredShape, inputShape);
}

void RunInferDataTypeCase(int64_t outputDtype)
{
    auto holder = BuildDataTypeContext(outputDtype);

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Bucketize");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), static_cast<ge::DataType>(outputDtype));
}

void RunInferDataTypeRejectCase(int64_t outputDtype)
{
    auto holder = BuildDataTypeContext(outputDtype);

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Bucketize");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);

    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(context), ge::GRAPH_FAILED);
}

} // namespace

class BucketizeInfershapeTest : public testing::Test {};

TEST_F(BucketizeInfershapeTest, TensorShape) { RunInferShapeCase({2, 3, 4}); }

TEST_F(BucketizeInfershapeTest, ScalarShape) { RunInferShapeCase({}); }

TEST_F(BucketizeInfershapeTest, EmptyShape) { RunInferShapeCase({0, 4}); }

TEST_F(BucketizeInfershapeTest, UnknownRank) { RunInferShapeCase({-2}); }

TEST_F(BucketizeInfershapeTest, UnknownDim) { RunInferShapeCase({-1, 4}); }

TEST_F(BucketizeInfershapeTest, InferDataType)
{
    RunInferDataTypeCase(ge::DT_INT32);
    RunInferDataTypeCase(ge::DT_INT64);
}

TEST_F(BucketizeInfershapeTest, InferDataTypeRejectsNullContext)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Bucketize");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    EXPECT_EQ(opImpl->infer_datatype(nullptr), ge::GRAPH_FAILED);
}

TEST_F(BucketizeInfershapeTest, InferDataTypeRejectsUnsupportedDtype) { RunInferDataTypeRejectCase(ge::DT_FLOAT); }
