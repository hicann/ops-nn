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
#include <vector>
#include <gtest/gtest.h>

#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "ut_op_util.h"
#include "../../../op_graph/rms_norm_dynamic_quant_proto.h"

using namespace std;
using namespace ge;

class RmsNormDynamicQuantInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormDynamicQuantInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormDynamicQuantInfershapeTest TearDown" << std::endl;
    }
};

// ============================================================================
// InferShape — 输出 y 的 shape 应与 x 一致
// ============================================================================

TEST_F(RmsNormDynamicQuantInfershapeTest, y_shape_equals_x)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicQuant"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gammaShape = {{16}, {16}};
    gert::StorageShape smoothShape = {{16}, {16}};
    gert::StorageShape yShape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scaleShape = {{1, 1}, {1, 1}};

    // 4 inputs: x, gamma, smooth_scales, beta
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xShape, &gammaShape, &smoothShape, &gammaShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(3, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeOutputTd(0, DT_INT8, FORMAT_ND, FORMAT_ND)
                      .NodeOutputTd(1, DT_FLOAT, FORMAT_ND, FORMAT_ND)
                      .Build();

    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(inferShapeFunc(ctx), ge::GRAPH_SUCCESS);

    vector<int64_t> expectedYShape = {1, 1, 16};
    vector<int64_t> outputYShape;
    auto* y = ctx->GetOutputShape(0);
    for (size_t i = 0; i < y->GetDimNum(); ++i) {
        outputYShape.push_back(y->GetDim(i));
    }
    EXPECT_EQ(outputYShape, expectedYShape);
}

// ============================================================================
// InferDtype — fp16, dst_type = INT8
// ============================================================================

TEST_F(RmsNormDynamicQuantInfershapeTest, dtype_fp16_int8)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicQuant"), nullptr);
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicQuant")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ge::DataType inRef = DT_FLOAT16;
    ge::DataType outRef = DT_INT8;
    ge::DataType scaleRef = DT_FLOAT;

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(4)
                      .NodeIoNum(4, 2)
                      .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeInputTd(3, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                      .NodeOutputTd(0, DT_INT8, FORMAT_ND, FORMAT_ND)
                      .NodeOutputTd(1, scaleRef, FORMAT_ND, FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(DT_INT8)}})
                      .InputDataTypes({&inRef, &inRef, &inRef, &inRef})
                      .OutputDataTypes({&outRef, &scaleRef})
                      .Build();

    auto* ctx = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDtypeFunc(ctx), ge::GRAPH_SUCCESS);

    EXPECT_EQ(ctx->GetOutputDataType(0), DT_INT8);
    EXPECT_EQ(ctx->GetOutputDataType(1), DT_FLOAT);
}
