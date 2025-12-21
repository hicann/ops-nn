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

class TestQuantBatchMatmulV3InferShape : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "TestQuantBatchMatmulV3InferShape Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "TestQuantBatchMatmulV3InferShape Proto Test TearDown" << std::endl;
        }
};


TEST_F(TestQuantBatchMatmulV3InferShape, QuantBatchMatmulV3_InferShapeCheck) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x1 = {{4, 4}, {4, 4}};
    gert::StorageShape x2 = {{4, 4}, {4, 4}};
    gert::StorageShape scale = {{4}, {4}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&x1, &x2, &scale})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, 4]");
}

TEST_F(TestQuantBatchMatmulV3InferShape, NoBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape scale = {5};
    gert::Shape offset = {};
    gert::Shape bias = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 5};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &scale, &offset, &bias})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV3InferShape, Transposex2Case) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {1, 4};
    gert::Shape scale = {2};
    gert::Shape offset = {};
    gert::Shape bias = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 1};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &scale, &offset, &bias})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV3InferShape, WithBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape scale = {5};
    gert::Shape offset = {};
    gert::Shape bias = {5};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 5};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &scale, &offset, &bias})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV3InferShape, WithBatchAndBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::Shape x1 = {2, 2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape scale = {5};
    gert::Shape offset = {};
    gert::Shape bias = {2, 1, 5};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 2, 5};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &scale, &offset, &bias})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV3InferShape, NoScaleCase) {
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;

    gert::Shape x1 = {-1, 13, 16};
    gert::Shape x2 = {-1, 16, 13};
    gert::Shape output_shape = {};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(2, 1)
        .IrInstanceNum({1, 1})
        .InputShapes({&x1, &x2})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestQuantBatchMatmulV3InferShape, ErrorDimNumCase) {
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3")->infer_shape;
    gert::Shape x1 = {2};
    gert::Shape x2 = {4, 5};
    gert::Shape scale = {5};
    gert::Shape offset = {};
    gert::Shape bias = {2, 1, 5};
    gert::Shape output_shape = {};
    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &scale, &offset, &bias})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})

        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
}