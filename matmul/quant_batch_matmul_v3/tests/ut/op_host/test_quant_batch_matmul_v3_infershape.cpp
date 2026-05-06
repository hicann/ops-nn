/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

namespace ge {
namespace {

struct QuantBatchMatmulV3InferShapeParam {
    std::string caseName;
    size_t inputNum = 0UL;
    gert::Shape x1;
    gert::Shape x2;
    gert::Shape scale;
    gert::Shape offset;
    gert::Shape bias;
    ge::DataType dtype = ge::DT_FLOAT16;
    bool transposeX1 = false;
    bool transposeX2 = false;
    graphStatus expectStatus = ge::GRAPH_SUCCESS;
    gert::Shape expectOutput;
};


static std::vector<QuantBatchMatmulV3InferShapeParam> GetParams()
{
    std::vector<QuantBatchMatmulV3InferShapeParam> params;
    std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    std::string casePath(rootPath +
                         "matmul/quant_batch_matmul_v3/tests/ut/op_host/test_quant_batch_matmul_v3_infershape.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
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
        if (cols.size() < 12UL) {
            continue;
        }

        QuantBatchMatmulV3InferShapeParam param;
        size_t idx = 0UL;
        param.caseName = Trim(cols[idx++]);
        param.inputNum = static_cast<size_t>(ParseInt64OrDefault(cols[idx++], 0));
        param.x1 = ParseShape(cols[idx++]);
        param.x2 = ParseShape(cols[idx++]);
        param.scale = ParseShape(cols[idx++]);
        param.offset = ParseShape(cols[idx++]);
        param.bias = ParseShape(cols[idx++]);
        param.dtype = ParseDtype(cols[idx++]);
        param.transposeX1 = ParseBool(cols[idx++]);
        param.transposeX2 = ParseBool(cols[idx++]);
        param.expectStatus = ParseGraphStatus(cols[idx++], ge::GRAPH_SUCCESS);
        param.expectOutput = ParseShape(cols[idx++]);
        params.push_back(param);
    }

    return params;
}

class TestQuantBatchMatmulV3InferShape : public testing::TestWithParam<QuantBatchMatmulV3InferShapeParam> {
protected:
    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
    }
};

TEST_P(TestQuantBatchMatmulV3InferShape, InferShapeFromCsv)
{
    const auto &param = GetParam();
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x1 = param.x1;
    gert::Shape x2 = param.x2;
    gert::Shape scale = param.scale;
    gert::Shape offset = param.offset;
    gert::Shape bias = param.bias;
    gert::Shape outputShape;
    std::vector<gert::Shape *> inputShapes = {&x1, &x2};
    if (param.inputNum >= 3UL) {
        inputShapes.push_back(&scale);
    }
    if (param.inputNum >= 4UL) {
        inputShapes.push_back(&offset);
    }
    if (param.inputNum >= 5UL) {
        inputShapes.push_back(&bias);
    }
    ASSERT_EQ(inputShapes.size(), param.inputNum);

    std::vector<uint32_t> irInstanceNum(param.inputNum, 1U);
    int64_t dtype = static_cast<int64_t>(param.dtype);
    auto contextHolder = gert::InferShapeContextFaker()
        .NodeIoNum(param.inputNum, 1)
        .IrInstanceNum(irInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)}})
        .Build();

    auto ret = inferShapeFunc(contextHolder.GetContext<gert::InferShapeContext>());
    ASSERT_EQ(ret, param.expectStatus) << "caseName=" << param.caseName;
    if (ret == ge::GRAPH_SUCCESS) {
        auto output = contextHolder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(param.expectOutput))
            << "caseName=" << param.caseName;
    }
}

INSTANTIATE_TEST_CASE_P(QuantBatchMatmulV3InferShapeCsv, TestQuantBatchMatmulV3InferShape,
                        testing::ValuesIn(GetParams()));

}  // namespace
}  // namespace ge
