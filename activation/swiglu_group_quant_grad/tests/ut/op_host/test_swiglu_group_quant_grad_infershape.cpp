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
#include "../../../op_graph/swiglu_group_quant_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {

constexpr size_t INPUT_NUM = 5;
constexpr size_t OUTPUT_NUM = 2;

void SetUpPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optionalCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalCompilationInfo);
}

// ---------------------------------------------------------------------------
// InferShape
// ---------------------------------------------------------------------------
struct InferShapeCase {
    gert::Shape gradYShape = {4, 16};
    gert::Shape xShape = {4, 32};
    ge::DataType dtype = ge::DT_FLOAT;
    bool hasWeight = false;
    bool hasYOrigin = false;
    bool hasGroupIndex = false;
    gert::Shape weightShape = {};  // 空 => 默认取 gradYShape 且末维置 1
    gert::Shape yOriginShape = {}; // 空 => 默认取 gradYShape
    gert::Shape groupIndexShape = {2};
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
};

class SwigluGroupQuantGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantGrad InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluGroupQuantGrad InferShape TearDown" << std::endl; }
};

void ExecuteInferShapeCase(const InferShapeCase& testCase)
{
    SetUpPlatform();

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SwigluGroupQuantGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape gradYShape = testCase.gradYShape;
    gert::Shape xShape = testCase.xShape;
    gert::Shape weightShape = testCase.weightShape;
    if (weightShape.GetDimNum() == 0) {
        weightShape = gradYShape;
        weightShape.SetDim(weightShape.GetDimNum() - 1, 1);
    }
    gert::Shape yOriginShape = testCase.yOriginShape;
    if (yOriginShape.GetDimNum() == 0) {
        yOriginShape = gradYShape;
    }
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
    contextFaker.NodeIoNum(inputShapes.size(), OUTPUT_NUM)
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
    ASSERT_NE(outputGradXShape, nullptr);
    ASSERT_EQ(outputGradXShape->GetDimNum(), xShape.GetDimNum());
    for (size_t dim = 0; dim < xShape.GetDimNum(); ++dim) {
        EXPECT_EQ(outputGradXShape->GetDim(dim), xShape.GetDim(dim));
    }
    if (testCase.hasWeight) {
        auto* outputGradWeightShape = context->GetOutputShape(1);
        ASSERT_NE(outputGradWeightShape, nullptr);
        EXPECT_EQ(outputGradWeightShape->GetDimNum(), weightShape.GetDimNum());
        for (size_t dim = 0; dim < weightShape.GetDimNum(); ++dim) {
            EXPECT_EQ(outputGradWeightShape->GetDim(dim), weightShape.GetDim(dim));
        }
    }
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_fp32_basic) { ExecuteInferShapeCase({}); }

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_fp16_basic)
{
    InferShapeCase testCase;
    testCase.dtype = ge::DT_FLOAT16;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_bf16_basic)
{
    InferShapeCase testCase;
    testCase.dtype = ge::DT_BF16;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_all_optional_inputs)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    testCase.weightShape = {4, 1};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_group_index_only)
{
    InferShapeCase testCase;
    testCase.hasGroupIndex = true;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_accepts_3d_input)
{
    InferShapeCase testCase;
    testCase.gradYShape = {2, 4, 16};
    testCase.xShape = {2, 4, 32};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_empty_tensor)
{
    InferShapeCase testCase;
    testCase.gradYShape = {0, 16};
    testCase.xShape = {0, 32};
    ExecuteInferShapeCase(testCase);
}

// 动态 shape：末维未知，跳过末维相关校验
TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_accepts_dynamic_last_dim)
{
    InferShapeCase testCase;
    testCase.gradYShape = {4, -1};
    testCase.xShape = {4, -1};
    ExecuteInferShapeCase(testCase);
}

// 动态 shape：前导维未知，末维已知且满足 2 倍关系
TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_accepts_dynamic_outer_dim)
{
    InferShapeCase testCase;
    testCase.gradYShape = {-1, 16};
    testCase.xShape = {-1, 32};
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_zero_hidden_size)
{
    InferShapeCase testCase;
    testCase.gradYShape = {4, 0};
    testCase.xShape = {4, 0};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

// quant_grad 比基础版更严：grad_y 与 x 的 rank 必须相等
TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_rank_mismatch)
{
    InferShapeCase testCase;
    testCase.gradYShape = {2, 4, 16, 32};
    testCase.xShape = {128, 64};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_mismatched_leading_dim)
{
    InferShapeCase testCase;
    testCase.gradYShape = {2, 4, 16};
    testCase.xShape = {2, 8, 32};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_x_last_not_divisible)
{
    InferShapeCase testCase;
    testCase.gradYShape = {4, 16};
    testCase.xShape = {4, 33};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_last_dim_not_half)
{
    InferShapeCase testCase;
    testCase.gradYShape = {4, 16};
    testCase.xShape = {4, 34};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_empty_group_index)
{
    InferShapeCase testCase;
    testCase.hasGroupIndex = true;
    testCase.groupIndexShape = {0};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_group_index_rank)
{
    InferShapeCase testCase;
    testCase.hasGroupIndex = true;
    testCase.groupIndexShape = {2, 2};
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_unpaired_weight)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_unpaired_y_origin)
{
    InferShapeCase testCase;
    testCase.hasYOrigin = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_bad_weight_numel)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.weightShape = {3, 1}; // numel=3 != grad_y 前导维乘积 4
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferShapeTest, infershape_rejects_y_origin_shape_mismatch)
{
    InferShapeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.weightShape = {4, 1};
    testCase.yOriginShape = {4, 8}; // 与 grad_y={4,16} 不一致
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferShapeCase(testCase);
}

// ---------------------------------------------------------------------------
// InferDataType
// ---------------------------------------------------------------------------
struct InferDataTypeCase {
    ge::DataType gradYDtype = ge::DT_FLOAT;
    ge::DataType xDtype = ge::DT_FLOAT;
    bool hasWeight = false;
    bool hasYOrigin = false;
    bool hasGroupIndex = false;
    ge::DataType weightDtype = ge::DT_FLOAT;
    ge::DataType yOriginDtype = ge::DT_FLOAT;
    ge::DataType groupIndexDtype = ge::DT_INT64;
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
};

class SwigluGroupQuantGradInferDataTypeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluGroupQuantGrad InferDataType SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluGroupQuantGrad InferDataType TearDown" << std::endl; }
};

void ExecuteInferDataTypeCase(const InferDataTypeCase& testCase)
{
    SetUpPlatform();

    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SwigluGroupQuantGrad")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType gradYDtype = testCase.gradYDtype;
    ge::DataType xDtype = testCase.xDtype;
    ge::DataType weightDtype = testCase.weightDtype;
    ge::DataType yOriginDtype = testCase.yOriginDtype;
    ge::DataType groupIndexDtype = testCase.groupIndexDtype;
    ge::DataType gradXDtype = ge::DT_UNDEFINED;
    ge::DataType gradWeightDtype = ge::DT_UNDEFINED;

    std::vector<uint32_t> inputInstanceNum = {
        1, 1, testCase.hasWeight ? 1U : 0U, testCase.hasYOrigin ? 1U : 0U, testCase.hasGroupIndex ? 1U : 0U,
    };
    std::vector<uint32_t> outputInstanceNum = {1, testCase.hasWeight ? 1U : 0U};
    std::vector<void*> inputDataTypes = {&gradYDtype, &xDtype};
    if (testCase.hasWeight) {
        inputDataTypes.emplace_back(&weightDtype);
    }
    if (testCase.hasYOrigin) {
        inputDataTypes.emplace_back(&yOriginDtype);
    }
    if (testCase.hasGroupIndex) {
        inputDataTypes.emplace_back(&groupIndexDtype);
    }
    std::vector<void*> outputDataTypes = {&gradXDtype};
    if (testCase.hasWeight) {
        outputDataTypes.emplace_back(&gradWeightDtype);
    }

    gert::InferDataTypeContextFaker contextFaker;
    contextFaker.NodeIoNum(inputDataTypes.size(), outputDataTypes.size())
        .IrInstanceNum(inputInstanceNum, outputInstanceNum)
        .NodeInputTd(0, testCase.gradYDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, testCase.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, testCase.gradYDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes(inputDataTypes)
        .OutputDataTypes(outputDataTypes);
    int32_t optionalInputIndex = 2;
    if (testCase.hasWeight) {
        contextFaker.NodeInputTd(optionalInputIndex++, testCase.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        contextFaker.NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasYOrigin) {
        contextFaker.NodeInputTd(optionalInputIndex++, testCase.yOriginDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (testCase.hasGroupIndex) {
        contextFaker.NodeInputTd(optionalInputIndex, testCase.groupIndexDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = contextFaker.Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDataTypeFunc(context), testCase.expectedStatus);

    if (testCase.expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    EXPECT_EQ(context->GetOutputDataType(0), testCase.gradYDtype);
    if (testCase.hasWeight) {
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    }
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_fp32_basic) { ExecuteInferDataTypeCase({}); }

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_fp16_basic)
{
    InferDataTypeCase testCase;
    testCase.gradYDtype = ge::DT_FLOAT16;
    testCase.xDtype = ge::DT_FLOAT16;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_bf16_basic)
{
    InferDataTypeCase testCase;
    testCase.gradYDtype = ge::DT_BF16;
    testCase.xDtype = ge::DT_BF16;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_rejects_invalid_grad_y)
{
    InferDataTypeCase testCase;
    testCase.gradYDtype = ge::DT_INT32;
    testCase.xDtype = ge::DT_INT32;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_rejects_x_mismatch)
{
    InferDataTypeCase testCase;
    testCase.gradYDtype = ge::DT_FLOAT;
    testCase.xDtype = ge::DT_FLOAT16;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_all_optional)
{
    InferDataTypeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.hasGroupIndex = true;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_rejects_weight_not_fp32)
{
    InferDataTypeCase testCase;
    testCase.hasWeight = true;
    testCase.hasYOrigin = true;
    testCase.weightDtype = ge::DT_FLOAT16;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_rejects_group_index_not_int64)
{
    InferDataTypeCase testCase;
    testCase.hasGroupIndex = true;
    testCase.groupIndexDtype = ge::DT_INT32;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferDataTypeCase(testCase);
}

TEST_F(SwigluGroupQuantGradInferDataTypeTest, inferdatatype_rejects_unpaired_weight)
{
    InferDataTypeCase testCase;
    testCase.hasWeight = true;
    testCase.expectedStatus = ge::GRAPH_FAILED;
    ExecuteInferDataTypeCase(testCase);
}

} // namespace
