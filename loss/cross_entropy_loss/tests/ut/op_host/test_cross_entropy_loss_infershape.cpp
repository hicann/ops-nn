/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "ut_op_util.h"

using ut_util::ToVector;
class CrossEntropyLoss : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossEntropyLoss Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossEntropyLoss Proto Test TearDown" << std::endl;
    }
};

TEST_F(CrossEntropyLoss, cross_entropy_loss_infershape_bf16_mean_case)
{
    gert::StorageShape inputShape = {{49, 112595}, {49, 112595}};
    gert::StorageShape targetShape = {{49}, {49}};
    gert::StorageShape weightShape = {{112595}, {112595}};
    gert::StorageShape lossShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CrossEntropyLoss")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &targetShape, &weightShape})
                      .OutputShapes({&lossShape, &inputShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", ge::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", ge::AnyValue::CreateFrom<int64_t>(-100)},
                           {"label_smoothing", ge::AnyValue::CreateFrom<float>(0.0f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLoss")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedLossShape = {1};
    std::vector<int64_t> expectedLogProbShape = {49, 112595};
    auto lossOutShape = context->GetOutputShape(0);
    auto logProbOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*lossOutShape), expectedLossShape);
    EXPECT_EQ(ToVector(*logProbOutShape), expectedLogProbShape);
}

TEST_F(CrossEntropyLoss, cross_entropy_loss_infershape_fp32_mean_case)
{
    gert::StorageShape inputShape = {{49, 112595}, {49, 112595}};
    gert::StorageShape targetShape = {{49}, {49}};
    gert::StorageShape weightShape = {{112595}, {112595}};
    gert::StorageShape lossShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CrossEntropyLoss")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &targetShape, &weightShape})
                      .OutputShapes({&lossShape, &inputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", ge::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", ge::AnyValue::CreateFrom<int64_t>(-100)},
                           {"label_smoothing", ge::AnyValue::CreateFrom<float>(0.0f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLoss")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedLossShape = {1};
    std::vector<int64_t> expectedLogProbShape = {49, 112595};
    auto lossOutShape = context->GetOutputShape(0);
    auto logProbOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*lossOutShape), expectedLossShape);
    EXPECT_EQ(ToVector(*logProbOutShape), expectedLogProbShape);
}

TEST_F(CrossEntropyLoss, cross_entropy_loss_infershape_fp16_mean_case)
{
    gert::StorageShape inputShape = {{49, 112595}, {49, 112595}};
    gert::StorageShape targetShape = {{49}, {49}};
    gert::StorageShape weightShape = {{112595}, {112595}};
    gert::StorageShape lossShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("CrossEntropyLoss")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &targetShape, &weightShape})
                      .OutputShapes({&lossShape, &inputShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", ge::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", ge::AnyValue::CreateFrom<int64_t>(-100)},
                           {"label_smoothing", ge::AnyValue::CreateFrom<float>(0.0f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLoss")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedLossShape = {1};
    std::vector<int64_t> expectedLogProbShape = {49, 112595};
    auto lossOutShape = context->GetOutputShape(0);
    auto logProbOutShape = context->GetOutputShape(1);
    EXPECT_EQ(ToVector(*lossOutShape), expectedLossShape);
    EXPECT_EQ(ToVector(*logProbOutShape), expectedLogProbShape);
}