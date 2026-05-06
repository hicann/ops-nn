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

#include <fstream>
#include <iostream>
#include <vector>

#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "test_cube_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "ut_string_utils.h"

using namespace ut_str;

namespace ge {
namespace {
struct Param {
    std::string caseType;
    std::string caseName;
    int nodeIoNum;
    std::vector<int64_t> x1Shape;
    std::vector<int64_t> x2Shape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> x1ScaleShape;
    std::vector<int64_t> x2ScaleShape;
    std::vector<int64_t> offsetShape;
    bool in2Null;
    bool in3Null;
    bool in4Null;
    bool in5Null;
    bool transposeX1;
    bool transposeX2;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType dtype;
    ge::graphStatus expectRet;
    std::vector<int64_t> expectOutShape;
    std::vector<int64_t> x1RangeMin;
    std::vector<int64_t> x1RangeMax;
    std::vector<int64_t> x2RangeMin;
    std::vector<int64_t> x2RangeMax;
    std::vector<int64_t> expectRangeMin;
    std::vector<int64_t> expectRangeMax;
};

static std::vector<Param> GetParams()
{
    std::vector<Param> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(
        rootPath + "matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_infershape.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    bool skipHeader = true;
    while (std::getline(csvData, line)) {
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
        if (cols.size() < 26UL) {
            cols.resize(26UL);
        }

        Param param;
        param.caseType = Trim(cols[0]);
        param.caseName = Trim(cols[1]);
        param.nodeIoNum = ParseIntOrDefault(cols[2], 0);
        param.x1Shape = ParseInt64Vec(cols[3]);
        param.x2Shape = ParseInt64Vec(cols[4]);
        param.biasShape = ParseInt64Vec(cols[5]);
        param.x1ScaleShape = ParseInt64Vec(cols[6]);
        param.x2ScaleShape = ParseInt64Vec(cols[7]);
        param.offsetShape = ParseInt64Vec(cols[8]);
        param.in2Null = ParseBool(cols[9]);
        param.in3Null = ParseBool(cols[10]);
        param.in4Null = ParseBool(cols[11]);
        param.in5Null = ParseBool(cols[12]);
        param.transposeX1 = ParseBool(cols[13]);
        param.transposeX2 = ParseBool(cols[14]);
        param.x1Dtype = ParseDtype(cols[15]);
        param.x2Dtype = ParseDtype(cols[16]);
        param.dtype = ParseDtype(cols[17]);
        param.expectRet = ParseGraphStatus(cols[18]);
        param.expectOutShape = ParseInt64Vec(cols[19]);
        param.x1RangeMin = ParseInt64Vec(cols[20]);
        param.x1RangeMax = ParseInt64Vec(cols[21]);
        param.x2RangeMin = ParseInt64Vec(cols[22]);
        param.x2RangeMax = ParseInt64Vec(cols[23]);
        param.expectRangeMin = ParseInt64Vec(cols[24]);
        param.expectRangeMax = ParseInt64Vec(cols[25]);
        params.push_back(param);
    }
    return params;
}

class TestQuantBatchMatmulV4InferShapeCsv : public testing::TestWithParam<Param> {};

TEST_P(TestQuantBatchMatmulV4InferShapeCsv, runCase)
{
    const auto &param = GetParam();
    auto *opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4");
    ASSERT_NE(opImpl, nullptr);
    if (param.caseType == "infer_shape_range") {
        auto inferShapeRangeFunc = opImpl->infer_shape_range;
        ASSERT_NE(inferShapeRangeFunc, nullptr);
        gert::Shape x1Min = ToShape(param.x1RangeMin);
        gert::Shape x1Max = ToShape(param.x1RangeMax);
        gert::Shape x2Min = ToShape(param.x2RangeMin);
        gert::Shape x2Max = ToShape(param.x2RangeMax);
        gert::Shape outMin;
        gert::Shape outMax;
        gert::Range<gert::Shape> x1Range(&x1Min, &x1Max);
        gert::Range<gert::Shape> x2Range(&x2Min, &x2Max);
        gert::Range<gert::Shape> outRange(&outMin, &outMax);
        auto holder = gert::InferShapeRangeContextFaker()
            .NodeIoNum(2, 1)
            .IrInputNum(2)
            .NodeInputTd(0, param.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, param.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputShapeRanges({&x1Range, &x2Range})
            .OutputShapeRanges({&outRange})
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.dtype))},
                        {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                        {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                        {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)}})
            .Build();
        ASSERT_EQ(inferShapeRangeFunc(holder.GetContext<gert::InferShapeRangeContext>()), param.expectRet)
            << "case=" << param.caseName;
        if (param.expectRet == ge::GRAPH_SUCCESS) {
            gert::Shape targetMin = ToShape(param.expectRangeMin);
            gert::Shape targetMax = ToShape(param.expectRangeMax);
            gert::Range<gert::Shape> targetRange(&targetMin, &targetMax);
            EXPECT_EQ(*(holder.GetContext<gert::InferShapeRangeContext>()->GetOutputShapeRange(0)), targetRange)
                << "case=" << param.caseName;
        }
        return;
    }

    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape x1 = ToShape(param.x1Shape);
    gert::Shape x2 = ToShape(param.x2Shape);
    gert::Shape bias = ToShape(param.biasShape);
    gert::Shape x1Scale = ToShape(param.x1ScaleShape);
    gert::Shape x2Scale = ToShape(param.x2ScaleShape);
    gert::Shape offset = ToShape(param.offsetShape);
    gert::Shape outputShape = {};
    if (param.nodeIoNum == 2) {
        auto holder = gert::InferShapeContextFaker()
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&x1, &x2})
            .OutputShapes({&outputShape})
            .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.dtype))},
                        {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                        {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                        {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)}})
            .NodeInputTd(1, param.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();
        ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), param.expectRet)
            << "case=" << param.caseName;
        if (param.expectRet == ge::GRAPH_SUCCESS) {
            auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
            ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(ToShape(param.expectOutShape)))
                << "case=" << param.caseName;
        }
        return;
    }
    auto holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, param.in2Null ? nullptr : &bias, param.in3Null ? nullptr : &x1Scale,
                      param.in4Null ? nullptr : &x2Scale, param.in5Null ? nullptr : &offset})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.dtype))},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX1)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transposeX2)}})
        .NodeInputTd(1, param.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), param.expectRet) << "case=" << param.caseName;
    if (param.expectRet == ge::GRAPH_SUCCESS) {
        auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(ToShape(param.expectOutShape)))
            << "case=" << param.caseName;
    }
}

INSTANTIATE_TEST_CASE_P(MMInferShape, TestQuantBatchMatmulV4InferShapeCsv, testing::ValuesIn(GetParams()));
} // namespace
} // namespace ge
