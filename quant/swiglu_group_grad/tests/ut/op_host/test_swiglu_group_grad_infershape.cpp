/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/swiglu_group_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {

struct InferShapeCase {
    gert::Shape gradYShape = {4, 16};
    gert::Shape xShape = {4, 32};
    ge::DataType dtype = ge::DT_FLOAT;
    bool hasWeight = false;
    bool hasYOrigin = false;
    bool hasGroupIndex = false;
    gert::Shape weightShape = {};
    gert::Shape groupIndexShape = {2};
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
};

class SwigluGroupGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupGrad InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluGroupGrad InferShape TearDown" << std::endl; }
};

void ExecuteInferShapeCase(const InferShapeCase& testCase)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optionalCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SwigluGroupGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape gradYShape = testCase.gradYShape;
    gert::Shape xShape = testCase.xShape;
    gert::Shape weightShape = testCase.weightShape;
    if (weightShape.GetDimNum() == 0) {
        weightShape = gradYShape;
        weightShape.SetDim(weightShape.GetDimNum() - 1, 1);
    }
    gert::Shape yOriginShape = gradYShape;
    gert::Shape groupIndexShape = testCase.groupIndexShape;
    gert::Shape gradXShape = {};
    gert::Shape gradWeightShape = {};

    std::vector<uint32_t> inputInstanceNum = {
        1, 1, testCase.hasWeight ? 1U : 0U, testCase.hasYOrigin ? 1U : 0U, testCase.hasGroupIndex ? 1U : 0U,
    };
    std::vector<uint32_t> outputInstanceNum = {1, 1};
    std::vector<void*> inputShapes = {&gradYShape, &xShape};
    if (testCase.hasWeight) {
        inputShapes.emplace_back(&weightShape);
    }
    if (testCase.hasYOrigin) {
        inputShapes.emplace_back(&yOriginShape);
    }
    if (testCase.hasGroupIndex) {
        inputShapes.emplace_back(&groupIndexShape);
    }

    gert::InferShapeContextFaker contextFaker;
    contextFaker.NodeIoNum(inputShapes.size(), 2)
        .IrInstanceNum(inputInstanceNum, outputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&gradXShape, &gradWeightShape})
        .NodeInputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    int32_t optionalInputIndex = 2;
    if (testCase.hasWeight) {
        contextFaker.NodeInputTd(optionalInputIndex++, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasYOrigin) {
        contextFaker.NodeInputTd(optionalInputIndex++, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasGroupIndex) {
        contextFaker.NodeInputTd(optionalInputIndex, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = contextFaker.Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), testCase.expectedStatus);

    if (testCase.expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    auto* outputGradXShape = context->GetOutputShape(0);
    auto* outputGradWeightShape = context->GetOutputShape(1);
    ASSERT_NE(outputGradXShape, nullptr);
    ASSERT_NE(outputGradWeightShape, nullptr);
    ASSERT_EQ(outputGradXShape->GetDimNum(), xShape.GetDimNum());
    for (size_t dim = 0; dim < xShape.GetDimNum(); ++dim) {
        EXPECT_EQ(outputGradXShape->GetDim(dim), xShape.GetDim(dim));
    }
    if (testCase.hasWeight) {
        EXPECT_EQ(outputGradWeightShape->GetDimNum(), weightShape.GetDimNum());
        for (size_t dim = 0; dim < weightShape.GetDimNum(); ++dim) {
            EXPECT_EQ(outputGradWeightShape->GetDim(dim), weightShape.GetDim(dim));
        }
    } else {
        EXPECT_EQ(outputGradWeightShape->GetDimNum(), 0);
    }
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_fp32_basic) { ExecuteInferShapeCase({}); }

TEST_F(SwigluGroupGradInferShapeTest, infershape_fp16_basic)
{
    InferShapeCase testCase;
    testCase.dtype = ge::DT_FLOAT16;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_bf16_basic)
{
    InferShapeCase testCase;
    testCase.dtype = ge::DT_BF16;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_all_optional_inputs)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    testCase.weightShape = {4};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_group_index_only)
{
    InferShapeCase testCase;
    testCase.hasGroupIndex = true;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_rejects_empty_group_index)
{
    InferShapeCase testCase;
    testCase.hasGroupIndex = true;
    testCase.groupIndexShape = {0};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_empty_tensor)
{
    InferShapeCase testCase;
    testCase.gradYShape = {0, 16};
    testCase.xShape = {0, 32};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_rejects_zero_hidden_size)
{
    InferShapeCase testCase;
    testCase.gradYShape = {4, 0};
    testCase.xShape = {4, 0};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_rejects_4d_input)
{
    InferShapeCase testCase;
    testCase.gradYShape = {2, 4, 16, 32};
    testCase.xShape = {2, 4, 32, 64};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_accepts_3d_input)
{
    InferShapeCase testCase;
    testCase.gradYShape = {2, 4, 16};
    testCase.xShape = {2, 4, 32};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_rejects_unpaired_weight)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupGradInferShapeTest, infershape_rejects_unpaired_y_origin)
{
    InferShapeCase testCase;
    testCase.hasYOrigin = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

} // namespace
