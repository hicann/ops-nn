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

#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "runtime/infer_shape_range_context.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"

using namespace ut_str;

namespace ge {
namespace {

struct QuantBatchMatmulInplaceAddInferShapeParam {
    std::string caseGroup;
    std::string caseName;
    std::vector<int64_t> x1Shape;
    std::vector<int64_t> x2Shape;
    std::vector<int64_t> x2ScaleShape;
    std::vector<int64_t> yShape;
    std::vector<int64_t> x1ScaleShape;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    ge::graphStatus expectRet;
    std::vector<int64_t> expectOutShape;
};

struct QuantBatchMatmulInplaceAddInferShapeCsvLoadResult {
    std::vector<QuantBatchMatmulInplaceAddInferShapeParam> params;
    std::vector<std::string> errors;
};

static QuantBatchMatmulInplaceAddInferShapeCsvLoadResult LoadParams()
{
    QuantBatchMatmulInplaceAddInferShapeCsvLoadResult result;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(
        rootPath +
        "matmul/quant_batch_matmul_inplace_add/tests/ut/op_host/test_quant_batch_matmul_inplace_add_infershape.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        result.errors.push_back("cannot open case file: " + casePath);
        return result;
    }

    std::string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 12UL;
    size_t lineNo = 0UL;
    while (std::getline(csvData, line)) {
        ++lineNo;
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }
        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kExpectedCols) {
            result.errors.push_back("skip invalid csv line " + std::to_string(lineNo) + " in " + casePath +
                                    ": expected at least " + std::to_string(kExpectedCols) + " columns, got " +
                                    std::to_string(cols.size()));
            continue;
        }

        size_t idx = 0UL;
        QuantBatchMatmulInplaceAddInferShapeParam param;
        param.caseGroup = Trim(cols[idx++]);
        param.caseName = Trim(cols[idx++]);
        param.x1Shape = ParseInt64Vec(cols[idx++]);
        param.x2Shape = ParseInt64Vec(cols[idx++]);
        param.x2ScaleShape = ParseInt64Vec(cols[idx++]);
        param.yShape = ParseInt64Vec(cols[idx++]);
        param.x1ScaleShape = ParseInt64Vec(cols[idx++]);
        param.transposeX1 = ParseBool(cols[idx++]);
        param.transposeX2 = ParseBool(cols[idx++]);
        param.groupSize = ParseInt64OrDefault(cols[idx++], 0);
        param.expectRet = ParseGraphStatus(cols[idx++]);
        param.expectOutShape = ParseInt64Vec(cols[idx++]);
        result.params.push_back(param);
    }
    if (result.params.empty()) {
        result.errors.push_back("no valid infershape cases loaded from: " + casePath);
    }
    return result;
}

static const QuantBatchMatmulInplaceAddInferShapeCsvLoadResult& GetParamsLoadResult()
{
    static const QuantBatchMatmulInplaceAddInferShapeCsvLoadResult result = LoadParams();
    return result;
}

static std::vector<QuantBatchMatmulInplaceAddInferShapeParam> GetParams() { return GetParamsLoadResult().params; }

static std::string SanitizeCaseName(const testing::TestParamInfo<QuantBatchMatmulInplaceAddInferShapeParam>& info)
{
    std::string name = info.param.caseName;
    for (char& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return name;
}

class TestQuantBatchMatmulInplaceAddInferShapeCsv
    : public testing::TestWithParam<QuantBatchMatmulInplaceAddInferShapeParam> {
protected:
    static void SetUpTestCase() { std::cout << "TestQuantBatchMatmulInplaceAddInferShapeCsv SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestQuantBatchMatmulInplaceAddInferShapeCsv TearDown" << std::endl; }
};

TEST(QuantBatchMatmulInplaceAddInferShapeCsvLoad, ShouldLoadValidCases)
{
    const auto& loadResult = GetParamsLoadResult();
    for (const auto& error : loadResult.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(loadResult.params.empty());
}

TEST_P(TestQuantBatchMatmulInplaceAddInferShapeCsv, runCase)
{
    const auto& param = GetParam();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulInplaceAdd"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulInplaceAdd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1;
    x1.MutableStorageShape() = ToShape(param.x1Shape);
    x1.MutableOriginShape() = ToShape(param.x1Shape);
    gert::StorageShape x2;
    x2.MutableStorageShape() = ToShape(param.x2Shape);
    x2.MutableOriginShape() = ToShape(param.x2Shape);
    gert::StorageShape x2Scale;
    x2Scale.MutableStorageShape() = ToShape(param.x2ScaleShape);
    x2Scale.MutableOriginShape() = ToShape(param.x2ScaleShape);
    gert::StorageShape y;
    y.MutableStorageShape() = ToShape(param.yShape);
    y.MutableOriginShape() = ToShape(param.yShape);
    gert::StorageShape x1Scale;
    x1Scale.MutableStorageShape() = ToShape(param.x1ScaleShape);
    x1Scale.MutableOriginShape() = ToShape(param.x1ScaleShape);
    gert::StorageShape outputShape = {{}, {}};

    auto contextHolder = gert::InferShapeContextFaker()
                             .NodeIoNum(5, 1)
                             .IrInstanceNum({1, 1, 1, 1, 1})
                             .InputShapes({&x1, &x2, &x2Scale, &y, &x1Scale})
                             .OutputShapes({&outputShape})
                             .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                                         {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)},
                                         {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groupSize)}})
                             .Build();
    ASSERT_EQ(inferShapeFunc(contextHolder.GetContext<gert::InferShapeContext>()), param.expectRet)
        << "case=" << param.caseName;
    if (param.expectRet == ge::GRAPH_SUCCESS) {
        auto output = contextHolder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(ToShape(param.expectOutShape)))
            << "case=" << param.caseName;
    }
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulInplaceAddInferShape, TestQuantBatchMatmulInplaceAddInferShapeCsv,
                         testing::ValuesIn(GetParams()), SanitizeCaseName);

} // namespace
} // namespace ge
