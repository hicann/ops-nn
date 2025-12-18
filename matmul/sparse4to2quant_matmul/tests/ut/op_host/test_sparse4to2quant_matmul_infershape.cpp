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
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "test_cube_util.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"


namespace ge {

class TestSparse4to2QuantMatmulInferShape : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "TestSparse4to2QuantMatmulInferShape Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "TestSparse4to2QuantMatmulInferShape Proto Test TearDown" << std::endl;
        }
};


TEST_F(TestSparse4to2QuantMatmulInferShape, Sparse4to2QuantMatmul_InferShapeCheck) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {64, 512};
    gert::Shape x2 = {256, 256};
    gert::Shape outputShape = {64, 256};
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&x1, &x2})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(TestSparse4to2QuantMatmulInferShape, Sparse4to2QuantMatmul_WrongDimCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {64, 512};
    gert::Shape x2 = {16, 8, 16, 32};
    gert::Shape outputShape = {64, 256};
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&x1, &x2})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestSparse4to2QuantMatmulInferShape, Sparse4to2QuantMatmul_UknownDimCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Sparse4to2QuantMatmul")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {64, 512};
    gert::Shape x2 = {-2};
    gert::Shape outputShape = {-2};
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&x1, &x2})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

}