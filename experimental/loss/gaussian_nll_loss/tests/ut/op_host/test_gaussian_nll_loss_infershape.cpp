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
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

namespace {
ge::graphStatus RunInferShape(const std::initializer_list<int64_t>& inputDims,
                              const std::initializer_list<int64_t>& targetDims,
                              const std::initializer_list<int64_t>& varDims, float eps, const std::string& reduction,
                              gert::Shape& outputShape)
{
    const auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GaussianNllLoss");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape inputShape(inputDims);
    gert::Shape targetShape(targetDims);
    gert::Shape varShape(varDims);
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GaussianNllLoss")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &targetShape, &varShape})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({{"full", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        outputShape = *context->GetOutputShape(0);
    }
    return status;
}

TEST(GaussianNllLossInferShape, NoneCopiesInputForAllBroadcastForms)
{
    const std::initializer_list<std::initializer_list<int64_t>> varShapes = {{2, 3, 4}, {2, 3, 1}, {2, 3}, {}};
    for (const auto& varShape : varShapes) {
        gert::Shape output;
        ASSERT_EQ(RunInferShape({2, 3, 4}, {2, 1, 4}, varShape, 1e-6f, "none", output), ge::GRAPH_SUCCESS);
        ASSERT_EQ(output.GetDimNum(), 3);
        EXPECT_EQ(output.GetDim(0), 2);
        EXPECT_EQ(output.GetDim(1), 3);
        EXPECT_EQ(output.GetDim(2), 4);
    }
}

TEST(GaussianNllLossInferShape, SumAndMeanProduceSingleElement)
{
    for (const std::string reduction : {"sum", "mean"}) {
        gert::Shape output;
        ASSERT_EQ(RunInferShape({7, 5}, {7, 5}, {}, 1e-6f, reduction, output), ge::GRAPH_SUCCESS);
        ASSERT_EQ(output.GetDimNum(), 1);
        EXPECT_EQ(output.GetDim(0), 1);
    }
}

TEST(GaussianNllLossInferShape, SupportsDynamicShape)
{
    gert::Shape output;
    ASSERT_EQ(RunInferShape({-1, 3}, {-1, 1}, {-1}, 1e-6f, "none", output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output.GetDim(0), -1);
    EXPECT_EQ(output.GetDim(1), 3);
}

TEST(GaussianNllLossInferShape, RejectsInvalidContract)
{
    gert::Shape output;
    EXPECT_EQ(RunInferShape({2, 3, 4}, {1, 1, 4}, {}, 1e-6f, "none", output), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({2, 3, 4}, {2, 3}, {}, 1e-6f, "none", output), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({2, 3, 4}, {2, 3, 4}, {2, 4}, 1e-6f, "none", output), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({2, 3}, {2, 3}, {}, 0.0f, "none", output), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape({2, 3}, {2, 3}, {}, 1e-6f, "invalid", output), ge::GRAPH_FAILED);
}
} // namespace
