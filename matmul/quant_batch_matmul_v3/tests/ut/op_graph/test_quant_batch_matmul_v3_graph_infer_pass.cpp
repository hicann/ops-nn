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
 * \file test_quant_batch_matmul_v3_graph_infer_pass.cpp
 * \brief QuantBatchMatmulV3 InferDataType UT (op_graph)
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "op_impl_registry.h"

namespace ops {
ge::graphStatus InferDataTypeForQuantBatchMatmulV3(gert::InferDataTypeContext* context);
ge::graphStatus SetQuantBatchMatmulV3OutputDtype(gert::InferDataTypeContext* context, int64_t dtypeValue);
} // namespace ops

namespace {
constexpr const char* kOpType = "QuantBatchMatmulV3";
constexpr ge::DataType kInitialOutputDtype = ge::DT_INT16;

const auto* GetOpImpl() { return gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType); }

gert::ContextHolder<gert::InferDataTypeContext> BuildContext(const int64_t* dtype = nullptr)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(kOpType).OpName("qbmmv3_infer_dtype");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    if (dtype != nullptr) {
        builder.AppendAttr(*dtype);
    }
    return builder.Build();
}

TEST(QuantBatchMatmulV3GraphInfer, RegistersShapeInference)
{
    const auto* opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_shape, nullptr);
}

TEST(QuantBatchMatmulV3GraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulV3(nullptr), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV3GraphInfer, RejectsMissingDtypeAttribute)
{
    auto holder = BuildContext();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulV3(context), ge::GRAPH_FAILED);
    EXPECT_EQ(context->GetOutputDataType(0), kInitialOutputDtype);
}

TEST(QuantBatchMatmulV3GraphInfer, AcceptsEverySupportedOutputDtype)
{
    for (ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32, ge::DT_BF16}) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        auto holder = BuildContext();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ops::SetQuantBatchMatmulV3OutputDtype(context, static_cast<int64_t>(dtype)), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}

TEST(QuantBatchMatmulV3GraphInfer, RejectsUnsupportedDtypeValues)
{
    for (int64_t dtype :
         {static_cast<int64_t>(ge::DT_UINT8), static_cast<int64_t>(ge::DT_DOUBLE), static_cast<int64_t>(999999)}) {
        SCOPED_TRACE(dtype);
        auto holder = BuildContext();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ops::SetQuantBatchMatmulV3OutputDtype(context, dtype), ge::GRAPH_FAILED);
        EXPECT_EQ(context->GetOutputDataType(0), kInitialOutputDtype);
    }
}

TEST(QuantBatchMatmulV3GraphInfer, RejectsValuesOutside32BitRange)
{
    for (int64_t dtype : {int64_t{4294967297}, int64_t{-4294967295}}) {
        SCOPED_TRACE(dtype);
        auto holder = BuildContext();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ops::SetQuantBatchMatmulV3OutputDtype(context, dtype), ge::GRAPH_FAILED);
        EXPECT_EQ(context->GetOutputDataType(0), kInitialOutputDtype);
    }
}

TEST(QuantBatchMatmulV3GraphInfer, ReadsDtypeAttribute)
{
    for (int64_t dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT32, ge::DT_BF16}) {
        SCOPED_TRACE(dtype);
        auto holder = BuildContext(&dtype);
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulV3(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), static_cast<ge::DataType>(dtype));
    }
}

TEST(QuantBatchMatmulV3GraphInfer, CallbackRejectsInvalidAttributeWithoutChangingOutput)
{
    for (int64_t dtype : {int64_t{-1}, int64_t{ge::DT_DOUBLE}, int64_t{4294967297}}) {
        SCOPED_TRACE(dtype);
        auto holder = BuildContext(&dtype);
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, kInitialOutputDtype), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulV3(context), ge::GRAPH_FAILED);
        EXPECT_EQ(context->GetOutputDataType(0), kInitialOutputDtype);
    }
}

TEST(QuantBatchMatmulV3GraphInfer, PropagatesOutputSetterFailure)
{
    const int64_t dtype = ge::DT_BF16;
    auto holder = BuildContext(&dtype);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    auto* rawContext = reinterpret_cast<gert::KernelContext*>(context)->GetContext();
    const auto outputSize = rawContext->output_size;
    rawContext->output_size = 0; // malformed execution context: output 0 cannot be written.
    EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulV3(context), ge::GRAPH_FAILED);
    rawContext->output_size = outputSize;
}
} // namespace
