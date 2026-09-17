/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"
#include "register/op_impl_registry.h"

namespace {

struct InferCase {
    gert::Shape xShape = {2, 3, 4, 5};
    gert::Shape gammaShape = {6};
    gert::Shape betaShape = {6};
    gert::Shape meanShape = {2, 3};
    gert::Shape varianceShape = {2, 3};
    bool hasGamma = true;
    bool hasBeta = true;
    bool hasMean = true;
    bool hasVariance = true;
    bool hasBatchMean = true;
    bool hasBatchVariance = true;
};

struct InferResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    gert::Shape yShape;
    gert::Shape batchMeanShape;
    gert::Shape batchVarianceShape;
    bool hasBatchMean = false;
    bool hasBatchVariance = false;
};

InferResult RunInferShape(InferCase testCase)
{
    std::vector<uint32_t> inputInstanceNum = {1U, testCase.hasGamma ? 1U : 0U, testCase.hasBeta ? 1U : 0U,
                                              testCase.hasMean ? 1U : 0U, testCase.hasVariance ? 1U : 0U};
    std::vector<gert::Shape*> inputShapes = {&testCase.xShape};
    if (testCase.hasGamma) {
        inputShapes.push_back(&testCase.gammaShape);
    }
    if (testCase.hasBeta) {
        inputShapes.push_back(&testCase.betaShape);
    }
    if (testCase.hasMean) {
        inputShapes.push_back(&testCase.meanShape);
    }
    if (testCase.hasVariance) {
        inputShapes.push_back(&testCase.varianceShape);
    }

    std::vector<uint32_t> outputInstanceNum = {1U, testCase.hasBatchMean ? 1U : 0U,
                                               testCase.hasBatchVariance ? 1U : 0U};
    // Production contexts may omit trailing zero-instance optional-output
    // entries entirely, making GetIrOutputInstanceInfo() return nullptr.
    while (outputInstanceNum.size() > 1U && outputInstanceNum.back() == 0U) {
        outputInstanceNum.pop_back();
    }
    const size_t outputNum = 1U + static_cast<size_t>(testCase.hasBatchMean) +
                             static_cast<size_t>(testCase.hasBatchVariance);
    gert::InferShapeContextFaker faker;
    faker.SetOpType("INInferV2")
        .NodeIoNum(inputShapes.size(), outputNum)
        .IrInstanceNum(inputInstanceNum, outputInstanceNum)
        .InputShapes(inputShapes);
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), i == 0 ? ge::DT_FLOAT16 : ge::DT_FLOAT, ge::FORMAT_ND,
                          ge::FORMAT_ND);
    }
    faker.NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    for (size_t outputIndex = 1; outputIndex < outputNum; ++outputIndex) {
        faker.NodeOutputTd(static_cast<int32_t>(outputIndex), ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    InferResult result;
    if (context == nullptr) {
        return result;
    }
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("INInferV2");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return result;
    }
    result.status = opImpl->infer_shape(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        const auto* yInstanceInfo = context->GetIrOutputInstanceInfo(0);
        if (yInstanceInfo != nullptr && yInstanceInfo->GetInstanceNum() == 1) {
            result.yShape = *context->GetOutputShape(yInstanceInfo->GetInstanceStart());
        }
        const auto* batchMeanInstanceInfo = context->GetIrOutputInstanceInfo(1);
        if (batchMeanInstanceInfo != nullptr && batchMeanInstanceInfo->GetInstanceNum() == 1) {
            result.hasBatchMean = true;
            result.batchMeanShape = *context->GetOutputShape(batchMeanInstanceInfo->GetInstanceStart());
        }
        const auto* batchVarianceInstanceInfo = context->GetIrOutputInstanceInfo(2);
        if (batchVarianceInstanceInfo != nullptr && batchVarianceInstanceInfo->GetInstanceNum() == 1) {
            result.hasBatchVariance = true;
            result.batchVarianceShape = *context->GetOutputShape(batchVarianceInstanceInfo->GetInstanceStart());
        }
    }
    return result;
}

void ExpectShapeEq(const gert::Shape& actual, const gert::Shape& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.GetDimNum());
    for (size_t i = 0; i < expected.GetDimNum(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected.GetDim(i));
    }
}

} // namespace

class INInferV2InferShapeTest : public testing::Test {};

TEST_F(INInferV2InferShapeTest, copiesAllStaticShapes)
{
    InferCase testCase;
    const InferResult result = RunInferShape(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShapeEq(result.yShape, testCase.xShape);
    ExpectShapeEq(result.batchMeanShape, testCase.meanShape);
    ExpectShapeEq(result.batchVarianceShape, testCase.varianceShape);
}

TEST_F(INInferV2InferShapeTest, mapsStatisticsWhenGammaAndBetaAreAbsent)
{
    InferCase testCase;
    testCase.hasGamma = false;
    testCase.hasBeta = false;
    const InferResult result = RunInferShape(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShapeEq(result.yShape, testCase.xShape);
    ExpectShapeEq(result.batchMeanShape, testCase.meanShape);
    ExpectShapeEq(result.batchVarianceShape, testCase.varianceShape);
}

TEST_F(INInferV2InferShapeTest, preservesUnknownRankForEveryOutput)
{
    InferCase testCase;
    testCase.xShape = {-2};
    testCase.meanShape = {-2};
    testCase.varianceShape = {-2};
    const InferResult result = RunInferShape(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShapeEq(result.yShape, gert::Shape({-2}));
    ExpectShapeEq(result.batchMeanShape, gert::Shape({-2}));
    ExpectShapeEq(result.batchVarianceShape, gert::Shape({-2}));
}

TEST_F(INInferV2InferShapeTest, optionalOutputsUseIrInstanceIndexes)
{
    for (const auto& outputPresence :
         {std::pair<bool, bool>{false, false}, {true, false}, {false, true}, {true, true}}) {
        InferCase testCase;
        testCase.hasBatchMean = outputPresence.first;
        testCase.hasBatchVariance = outputPresence.second;
        const InferResult result = RunInferShape(testCase);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS)
            << "hasBatchMean=" << outputPresence.first << ", hasBatchVariance=" << outputPresence.second;
        ExpectShapeEq(result.yShape, testCase.xShape);
        EXPECT_EQ(result.hasBatchMean, outputPresence.first);
        EXPECT_EQ(result.hasBatchVariance, outputPresence.second);
        if (result.hasBatchMean) {
            ExpectShapeEq(result.batchMeanShape, testCase.meanShape);
        }
        if (result.hasBatchVariance) {
            ExpectShapeEq(result.batchVarianceShape, testCase.varianceShape);
        }
    }
}

TEST_F(INInferV2InferShapeTest, copiesDynamicAndEmptyShapes)
{
    InferCase testCase;
    testCase.xShape = {0, 3, -1, 5};
    testCase.meanShape = {0, 3};
    testCase.varianceShape = {0, 3};
    const InferResult result = RunInferShape(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShapeEq(result.yShape, testCase.xShape);
    ExpectShapeEq(result.batchMeanShape, testCase.meanShape);
    ExpectShapeEq(result.batchVarianceShape, testCase.varianceShape);
}

TEST_F(INInferV2InferShapeTest, rejectsMissingMean)
{
    InferCase testCase;
    testCase.hasMean = false;
    EXPECT_EQ(RunInferShape(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(INInferV2InferShapeTest, rejectsMissingVariance)
{
    InferCase testCase;
    testCase.hasVariance = false;
    EXPECT_EQ(RunInferShape(testCase).status, ge::GRAPH_FAILED);
}
