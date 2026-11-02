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
 * \file test_quant_batch_matmul_inplace_add_graph_infer_pass.cpp
 * \brief QuantBatchMatmulInplaceAdd InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "op_impl_registry.h"

namespace ops {
ge::graphStatus InferDataTypeForQuantBatchMatmulInplaceAdd(gert::InferDataTypeContext* context);
}

namespace {
constexpr const char* kOpType = "QuantBatchMatmulInplaceAdd";

const auto* GetOpImpl() { return gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType); }

TEST(QuantBatchMatmulInplaceAddGraphInfer, RegistersShapeInference)
{
    const auto* opImpl = GetOpImpl();
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_shape, nullptr);
}

TEST(QuantBatchMatmulInplaceAddGraphInfer, SetsOutputDataTypeToFloat32)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(kOpType).OpName("qbmmia_infer_dtype");
    builder.IONum(5, 1);
    builder.IOInstanceNum({1, 1, 1, 1, 1}, {1});
    builder.InputTensorDesc(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(4, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);

    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_INT16), ge::GRAPH_SUCCESS);

    ASSERT_EQ(ops::InferDataTypeForQuantBatchMatmulInplaceAdd(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(QuantBatchMatmulInplaceAddGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForQuantBatchMatmulInplaceAdd(nullptr), ge::GRAPH_FAILED);
}
} // namespace
