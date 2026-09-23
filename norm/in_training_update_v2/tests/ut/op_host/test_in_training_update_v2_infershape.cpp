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

#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/in_training_update_v2_proto.h"

namespace {

struct InferResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    gert::Shape y;
    gert::Shape batchMean;
    gert::Shape batchVariance;
};

InferResult RunInferShape(const std::vector<int64_t>& xDims, const std::vector<int64_t>& sumDims, ge::DataType xDtype,
                          ge::Format format)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateV2");
    EXPECT_NE(impl, nullptr);
    EXPECT_NE(impl == nullptr ? nullptr : impl->infer_shape, nullptr);
    InferResult result;
    if (impl == nullptr || impl->infer_shape == nullptr) {
        return result;
    }
    gert::Shape xShape = CreateShape(xDims);
    gert::Shape sumShape = CreateShape(sumDims);
    gert::Shape squareShape = CreateShape(sumDims);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("INTrainingUpdateV2")
                      .NodeIoNum(3, 3)
                      .IrInputNum(7)
                      .IrInstanceNum({1, 1, 1, 0, 0, 0, 0}, {1, 1, 1})
                      .InputShapes({&xShape, &sumShape, &squareShape})
                      .OutputShapes({&result.y, &result.batchMean, &result.batchVariance})
                      .NodeInputTd(0, xDtype, format, format)
                      .NodeInputTd(1, ge::DT_FLOAT, format, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, format, ge::FORMAT_ND)
                      .NodeOutputTd(0, xDtype, format, format)
                      .NodeOutputTd(1, ge::DT_FLOAT, format, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, format, ge::FORMAT_ND)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    result.status = impl->infer_shape(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.y = *context->GetOutputShape(0);
        result.batchMean = *context->GetOutputShape(1);
        result.batchVariance = *context->GetOutputShape(2);
    }
    return result;
}

TEST(INTrainingUpdateV2InferShape, CopiesNchwInputAndStatisticsShapes)
{
    const auto result = RunInferShape({2, 3, 4, 5}, {2, 3, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.y, CreateShape({2, 3, 4, 5}));
    EXPECT_EQ(result.batchMean, CreateShape({2, 3, 1, 1}));
    EXPECT_EQ(result.batchVariance, CreateShape({2, 3, 1, 1}));
}

TEST(INTrainingUpdateV2InferShape, CopiesNhwcDynamicDimensionsWithoutInventingValues)
{
    const auto result = RunInferShape({-1, 8, 8, -1}, {-1, 1, 1, -1}, ge::DT_FLOAT, ge::FORMAT_NHWC);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.y, CreateShape({-1, 8, 8, -1}));
    EXPECT_EQ(result.batchMean, CreateShape({-1, 1, 1, -1}));
}

TEST(INTrainingUpdateV2InferShape, RejectsUnknownRank)
{
    EXPECT_EQ(RunInferShape({-2}, {-2}, ge::DT_FLOAT, ge::FORMAT_NCHW).status, ge::GRAPH_FAILED);
}

TEST(INTrainingUpdateV2InferShape, RejectsNonFourDimensionalSum)
{
    EXPECT_EQ(RunInferShape({2, 3, 4, 5}, {2, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW).status, ge::GRAPH_FAILED);
}

TEST(INTrainingUpdateV2InferDataType, YFollowsXAndStatisticsStayFloat)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);

    ge::DataType xType = ge::DT_FLOAT16;
    ge::DataType statType = ge::DT_FLOAT;
    ge::DataType undefinedType = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(7)
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1, 0, 0, 0, 0}, {1, 1, 1})
                      .NodeInputTd(0, xType, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, statType, ge::FORMAT_NCHW, ge::FORMAT_ND)
                      .NodeInputTd(2, statType, ge::FORMAT_NCHW, ge::FORMAT_ND)
                      .NodeOutputTd(0, undefinedType, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, undefinedType, ge::FORMAT_NCHW, ge::FORMAT_ND)
                      .NodeOutputTd(2, undefinedType, ge::FORMAT_NCHW, ge::FORMAT_ND)
                      .InputDataTypes({&xType, &statType, &statType})
                      .OutputDataTypes({&undefinedType, &undefinedType, &undefinedType})
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

} // namespace
