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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "test_cube_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "ut_string_utils.h"

using namespace ut_str;

namespace {

struct QuantMatmulActivationQuantInferShapeParam {
    std::string socVersion;
    std::string caseName;
    size_t inputNum = 0UL;
    gert::Shape x1;
    gert::Shape x2;
    gert::Shape x1Scale;
    gert::Shape x2Scale;
    gert::Shape bias;
    ge::DataType x1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType x2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType x1ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType x2ScaleDtype = ge::DT_FLOAT8_E8M0;
    ge::DataType biasDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType yScaleDtype = ge::DT_FLOAT8_E8M0;
    bool x2IsNz = true;
    bool transposeX1 = false;
    bool transposeX2 = false;
    int64_t groupSize = 0;
    std::string activationType = "gelu_tanh";
    std::string quantMode = "mx";
    std::string roundMode = "rint";
    int64_t scaleAlg = 0;
    float dstTypeMax = 6.0f;
    graphStatus expectStatus = ge::GRAPH_SUCCESS;
    gert::Shape expectY;
    gert::Shape expectYScale;
};

static std::vector<std::string> g_csvLoadErrors;

static std::vector<QuantMatmulActivationQuantInferShapeParam> LoadParams(const std::string& socVersion)
{
    std::vector<QuantMatmulActivationQuantInferShapeParam> params;
    std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    std::string casePath(rootPath + "matmul/quant_matmul_activation_quant/tests/ut/op_host/"
                                    "test_quant_matmul_activation_quant_infershape.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        g_csvLoadErrors.push_back("cannot open case file: " + casePath);
        return params;
    }

    std::string line;
    bool skipHeader = true;
    while (std::getline(csvData, line)) {
        if (Trim(line).empty()) {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }
        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() != 27UL) {
            g_csvLoadErrors.push_back("invalid CSV column count: " + line);
            continue;
        }

        try {
            QuantMatmulActivationQuantInferShapeParam param;
            size_t idx = 0UL;
            param.socVersion = Trim(cols[idx++]);
            param.caseName = Trim(cols[idx++]);
            param.inputNum = static_cast<size_t>(ParseInt64OrDefault(cols[idx++], 0));
            param.x1 = ParseShape(cols[idx++]);
            param.x2 = ParseShape(cols[idx++]);
            param.x1Scale = ParseShape(cols[idx++]);
            param.x2Scale = ParseShape(cols[idx++]);
            param.bias = ParseShape(cols[idx++]);
            param.x1Dtype = ParseDtype(cols[idx++]);
            param.x2Dtype = ParseDtype(cols[idx++]);
            param.x1ScaleDtype = ParseDtype(cols[idx++]);
            param.x2ScaleDtype = ParseDtype(cols[idx++]);
            param.biasDtype = ParseDtype(cols[idx++]);
            param.yDtype = ParseDtype(cols[idx++]);
            param.yScaleDtype = ParseDtype(cols[idx++]);
            param.x2IsNz = (Trim(cols[idx++]) == "NZ");
            param.transposeX1 = ParseBool(cols[idx++]);
            param.transposeX2 = ParseBool(cols[idx++]);
            param.groupSize = ParseInt64OrDefault(cols[idx++], 0);
            param.activationType = Trim(cols[idx++]);
            param.quantMode = Trim(cols[idx++]);
            param.roundMode = Trim(cols[idx++]);
            param.scaleAlg = ParseInt64OrDefault(cols[idx++], 0);
            param.dstTypeMax = static_cast<float>(std::stod(Trim(cols[idx++])));
            param.expectStatus = ParseGraphStatus(cols[idx++], ge::GRAPH_SUCCESS);
            std::string expectYStr = Trim(cols[idx++]);
            std::string expectYScaleStr = Trim(cols[idx++]);
            if (!expectYStr.empty()) {
                param.expectY = ParseShape(expectYStr);
            }
            if (!expectYScaleStr.empty()) {
                param.expectYScale = ParseShape(expectYScaleStr);
            }

            if (param.socVersion != socVersion) {
                continue;
            }

            params.push_back(param);
        } catch (const std::exception& error) {
            g_csvLoadErrors.push_back("invalid CSV row: " + line + ": " + error.what());
        }
    }

    return params;
}

class TestQuantMatmulActivationQuantInferShape
    : public testing::TestWithParam<QuantMatmulActivationQuantInferShapeParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

TEST_P(TestQuantMatmulActivationQuantInferShape, InferShapeFromCsv)
{
    const auto& param = GetParam();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulActivationQuant"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulActivationQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape;
    x1Shape.MutableOriginShape() = param.x1;
    x1Shape.MutableStorageShape() = param.x1;

    gert::StorageShape x2Shape;
    x2Shape.MutableOriginShape() = param.x2;
    if (param.x2IsNz && param.x2.GetDimNum() >= 2UL) {
        int64_t dim0 = param.x2.GetDim(param.x2.GetDimNum() - 2UL);
        int64_t dim1 = param.x2.GetDim(param.x2.GetDimNum() - 1UL);
        x2Shape.MutableStorageShape() = gert::Shape({dim0 / 16, dim1 / 16, 16, 16});
        for (size_t i = 0UL; i < param.x2.GetDimNum() - 2UL; ++i) {
            x2Shape.MutableStorageShape().SetDim(i, param.x2.GetDim(i));
        }
    } else {
        x2Shape.MutableStorageShape() = param.x2;
    }

    gert::StorageShape x1ScaleShape;
    x1ScaleShape.MutableOriginShape() = param.x1Scale;
    x1ScaleShape.MutableStorageShape() = param.x1Scale;

    gert::StorageShape x2ScaleShape;
    x2ScaleShape.MutableOriginShape() = param.x2Scale;
    x2ScaleShape.MutableStorageShape() = param.x2Scale;

    gert::StorageShape biasShape;
    biasShape.MutableOriginShape() = param.bias;
    biasShape.MutableStorageShape() = param.bias;

    gert::StorageShape yShape;
    gert::StorageShape yScaleShape;

    // Input order: x1, x2, bias, x1Scale, x2Scale.
    bool hasBias = (param.bias.GetDimNum() > 0UL);
    std::vector<gert::StorageShape*> inputShapes = {&x1Shape, &x2Shape, hasBias ? &biasShape : nullptr, &x1ScaleShape,
                                                    &x2ScaleShape};

    std::vector<uint32_t> irInstanceNum(inputShapes.size(), 1U);
    auto contextHolder = gert::InferShapeContextFaker()
                             .NodeIoNum(inputShapes.size(), 2)
                             .IrInstanceNum(irInstanceNum)
                             .InputShapes(inputShapes)
                             .OutputShapes({&yShape, &yScaleShape})
                             .NodeInputTd(0, param.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, param.x2Dtype, ge::FORMAT_ND,
                                          param.x2IsNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)
                             .NodeInputTd(2, param.biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, param.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, param.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, param.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, param.yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeAttrs(
                                 {{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groupSize)},
                                  {"activation_type", Ops::NN::AnyValue::CreateFrom<string>(param.activationType)},
                                  {"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.yDtype))},
                                  {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>(param.quantMode)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(param.roundMode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(param.scaleAlg)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(param.dstTypeMax)}})
                             .Build();

    auto ret = inferShapeFunc(contextHolder.GetContext<gert::InferShapeContext>());
    ASSERT_EQ(ret, param.expectStatus) << "caseName=" << param.caseName;
    if (ret == ge::GRAPH_SUCCESS && param.expectY.GetDimNum() > 0UL) {
        auto outputY = contextHolder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*outputY), Ops::Base::ToString(param.expectY)) << "caseName=" << param.caseName;
    }
    if (ret == ge::GRAPH_SUCCESS && param.expectYScale.GetDimNum() > 0UL) {
        auto outputYScale = contextHolder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
        ASSERT_EQ(Ops::Base::ToString(*outputYScale), Ops::Base::ToString(param.expectYScale))
            << "caseName=" << param.caseName;
    }
}

static const std::vector<QuantMatmulActivationQuantInferShapeParam> kCasesParams950 = LoadParams("Ascend950");

TEST(QuantMatmulActivationQuantInferDataType, SwiGluOutputMatchesX1)
{
    const auto* implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulActivationQuant");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->infer_datatype, nullptr);
    for (const auto x1TypeValue : {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
        ge::DataType x1Type = x1TypeValue;
        ge::DataType x2Type = ge::DT_FLOAT8_E5M2;
        ge::DataType biasType = ge::DT_FLOAT;
        ge::DataType scaleType = ge::DT_FLOAT8_E8M0;
        ge::DataType yType = ge::DT_UNDEFINED;
        ge::DataType yScaleType = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(5, 2)
                          .IrInstanceNum({1, 1, 1, 1, 1})
                          .InputDataTypes({&x1Type, &x2Type, &biasType, &scaleType, &scaleType})
                          .OutputDataTypes({&yType, &yScaleType})
                          .NodeAttrs(
                              {{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                               {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                               {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                               {"activation_type", Ops::NN::AnyValue::CreateFrom<std::string>("swiglu")},
                               {"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(x1TypeValue))},
                               {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("mx")},
                               {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                               {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                               {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0f)}})
                          .Build();
        auto* context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(implementation->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), x1TypeValue);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
    }
}

TEST(QuantMatmulActivationQuantInferDataType, SwiGluRejectsOutputDifferentFromX1)
{
    const auto* implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantMatmulActivationQuant");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->infer_datatype, nullptr);
    ge::DataType x1Type = ge::DT_FLOAT8_E4M3FN;
    ge::DataType x2Type = ge::DT_FLOAT8_E5M2;
    ge::DataType biasType = ge::DT_FLOAT;
    ge::DataType scaleType = ge::DT_FLOAT8_E8M0;
    ge::DataType yType = ge::DT_UNDEFINED;
    ge::DataType yScaleType = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputDataTypes({&x1Type, &x2Type, &biasType, &scaleType, &scaleType})
                      .OutputDataTypes({&yType, &yScaleType})
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"activation_type", Ops::NN::AnyValue::CreateFrom<std::string>("swiglu")},
                                  {"y_dtype",
                                   Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E5M2))},
                                  {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("mx")},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0f)}})
                      .Build();
    EXPECT_EQ(implementation->infer_datatype(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

TEST(QuantMatmulActivationQuantInferShapeCsv, ShouldLoadValidCases)
{
    for (const auto& error : g_csvLoadErrors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(kCasesParams950.empty());
}

INSTANTIATE_TEST_CASE_P(QuantMatmulActivationQuantInferShapeCsv, TestQuantMatmulActivationQuantInferShape,
                        testing::ValuesIn(kCasesParams950));

} // namespace
