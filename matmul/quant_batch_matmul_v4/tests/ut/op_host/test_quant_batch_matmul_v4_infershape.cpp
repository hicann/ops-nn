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
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

namespace ge {

class TestQuantBatchMatmulV4InferShape : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "TestQuantBatchMatmulV4InferShape Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "TestQuantBatchMatmulV4InferShape Proto Test TearDown" << std::endl;
        }
};

TEST_F(TestQuantBatchMatmulV4InferShape, QuantBatchMatmulV4_InferShapeCheck) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::StorageShape x1 = {{2, 64}, {2, 64}};
    gert::StorageShape x2 = {{128, 8}, {128, 8}};
    gert::StorageShape outputShape = {{}, {}};
    ge::DataType x2Dtype = ge::DT_FLOAT;
    int64_t dtype = static_cast<int64_t>(ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
                                .NodeIoNum(2, 1)
                                .IrInstanceNum({1, 1})
                                .InputShapes({&x1, &x2})
                                .OutputShapes({&outputShape})
                                .NodeAttrs(
                                    {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                                .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 128]");
}

TEST_F(TestQuantBatchMatmulV4InferShape, NoBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape bias = {};
    gert::Shape x1_scale = {};
    gert::Shape x2_scale = {5};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 5};
    ge::DataType x2Dtype = ge::DT_INT8;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &bias, &x1_scale, &x2_scale, &offset})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV4InferShape, Transposex2Case) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {1, 4};
    gert::Shape bias = {};
    gert::Shape x1_scale = {};
    gert::Shape x2_scale = {2};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 1};
    ge::DataType x2Dtype = ge::DT_INT8;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, nullptr, &x2_scale, nullptr, nullptr})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV4InferShape, WithBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::Shape x1 = {2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape bias = {5};
    gert::Shape x1_scale = {};
    gert::Shape x2_scale = {5};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 5};
    ge::DataType x2Dtype = ge::DT_INT8;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &bias, nullptr, &x2_scale, nullptr})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV4InferShape, WithBatchAndBiasCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::Shape x1 = {2, 2, 4};
    gert::Shape x2 = {4, 5};
    gert::Shape bias = {2, 1, 5};
    gert::Shape x1_scale = {};
    gert::Shape x2_scale = {5};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {2, 2, 5};
    ge::DataType x2Dtype = ge::DT_INT8;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &bias, nullptr, &x2_scale, nullptr})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV4InferShape, DtypeCase) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::Shape x1 = {16, 64};
    gert::Shape x2 = {8, 128};
    gert::Shape bias = {};
    gert::Shape x1_scale = {16, 2};
    gert::Shape x2_scale = {128, 2};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {16, 128};
    ge::DataType x2Dtype = ge::DT_FLOAT;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &bias, &x1_scale, &x2_scale, &offset})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestQuantBatchMatmulV4InferShape, ErrorDimNumCase) {
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape;
    gert::Shape x1 = {16};
    gert::Shape x2 = {8, 128};
    gert::Shape bias = {};
    gert::Shape x1_scale = {16, 2};
    gert::Shape x2_scale = {128, 2};
    gert::Shape offset = {};
    gert::Shape output_shape = {};
    gert::Shape expect_output_shape = {16, 128};
    ge::DataType x2Dtype = ge::DT_FLOAT;
    int64_t dtype = static_cast<int64_t> (ge::DT_BF16);
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(6, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &bias, &x1_scale, &x2_scale, &offset})
        .OutputShapes({&output_shape})
        .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                    {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .Build();
     ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

void CoutShape(const char* name, gert::Shape& shape) {
    std::cout << name << " shape: " << shape.GetDim(0) << ", " << shape.GetDim(1) << std::endl;
}

TEST_F(TestQuantBatchMatmulV4InferShape, ChatGLM2) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4")->infer_shape_range;
    gert::Shape x1_shape_min = {0, 0};
    gert::Shape x1_shape_max = {-1, -1};
    gert::Shape x2_shape_min = {4096, 4096};
    gert::Shape x2_shape_max = {4096, 4096};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);
    int64_t dtype = static_cast<int64_t>(ge::DT_BF16);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({
            {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
            {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
            {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}
        })
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 4096};
    gert::Shape target_shape_max = {-1, 4096};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    //EXPECT_EQ(out_shape_range, target_out_shape_range);
}

}
