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
#include <iostream>
#include <memory>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/broadcast_gradient_args_proto.h"

using namespace ge;

class BroadcastGradientArgsOpUT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BroadcastGradientArgs Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BroadcastGradientArgs Proto Test TearDown" << std::endl; }
};

namespace {
template <typename T>
gert::Tensor* CreateConstTensor(const gert::Shape& shape, const std::vector<T>& data, ge::DataType dataType,
                                std::unique_ptr<uint8_t[]>& tensorHolder)
{
    size_t totalSize = 0;
    tensorHolder = gert::Tensor::CreateFollowing(static_cast<int64_t>(data.size()), dataType, totalSize);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        auto dim = shape.GetDim(i);
        tensor->MutableStorageShape().AppendDim(dim);
        tensor->MutableOriginShape().AppendDim(dim);
    }
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), data.data(), data.size() * sizeof(T));
    return tensor;
}

gert::InferShapeContext* BuildContext(const gert::Shape& x1Shape, const std::vector<int32_t>& x1Data,
                                      const gert::Shape& x2Shape, const std::vector<int32_t>& x2Data,
                                      std::unique_ptr<uint8_t[]>& x1TensorHolder,
                                      std::unique_ptr<uint8_t[]>& x2TensorHolder, gert::KernelRunContextHolder& holder)
{
    auto x1Tensor = CreateConstTensor(x1Shape, x1Data, ge::DT_INT32, x1TensorHolder);
    auto x2Tensor = CreateConstTensor(x2Shape, x2Data, ge::DT_INT32, x2TensorHolder);
    auto x1OutputShape = gert::Shape{};
    auto x2OutputShape = gert::Shape{};

    holder = gert::InferShapeContextFaker()
                 .SetOpType("BroadcastGradientArgs")
                 .NodeIoNum(2, 2)
                 .IrInstanceNum({1, 1})
                 .InputShapes(std::vector<void*>{static_cast<void*>(x1Tensor), static_cast<void*>(x2Tensor)})
                 .OutputShapes({&x1OutputShape, &x2OutputShape})
                 .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                 .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                 .Build();
    return holder.GetContext<gert::InferShapeContext>();
}
} // namespace

TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_rank_one_inputs)
{
    auto x1_shape = gert::Shape{2};
    auto x2_shape = gert::Shape{2};
    std::vector<int32_t> x1_value = {2, 1};
    std::vector<int32_t> x2_value = {1, 2};
    std::unique_ptr<uint8_t[]> x1_tensor_holder;
    std::unique_ptr<uint8_t[]> x2_tensor_holder;
    gert::KernelRunContextHolder holder;

    auto context = BuildContext(x1_shape, x1_value, x2_shape, x2_value, x1_tensor_holder, x2_tensor_holder, holder);
    ASSERT_NE(context, nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[1]");
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)), "[1]");
}

TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_x1_rank_not_one)
{
    auto x1_shape = gert::Shape{2, 1};
    auto x2_shape = gert::Shape{2};
    std::vector<int32_t> x1_value = {2, 1};
    std::vector<int32_t> x2_value = {1, 2};
    std::unique_ptr<uint8_t[]> x1_tensor_holder;
    std::unique_ptr<uint8_t[]> x2_tensor_holder;
    gert::KernelRunContextHolder holder;

    auto context = BuildContext(x1_shape, x1_value, x2_shape, x2_value, x1_tensor_holder, x2_tensor_holder, holder);
    ASSERT_NE(context, nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_x2_rank_not_one)
{
    auto x1_shape = gert::Shape{2};
    auto x2_shape = gert::Shape{2, 1};
    std::vector<int32_t> x1_value = {2, 1};
    std::vector<int32_t> x2_value = {1, 2};
    std::unique_ptr<uint8_t[]> x1_tensor_holder;
    std::unique_ptr<uint8_t[]> x2_tensor_holder;
    gert::KernelRunContextHolder holder;

    auto context = BuildContext(x1_shape, x1_value, x2_shape, x2_value, x1_tensor_holder, x2_tensor_holder, holder);
    ASSERT_NE(context, nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_zero_shape) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({0});
//   auto x1_dtype = DT_INT64;
//   auto x2_shape = std::vector<int64_t>({0});
//   auto x2_dtype = DT_INT64;
//   std::vector<int64_t> x1_value = {};
//   std::vector<int64_t> x2_value = {};
//   // expect result info
//   std::vector<int64_t> expected_y1_shape = {0};
//   std::vector<int64_t> expected_y2_shape = {0};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");

//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_SUCCESS);
//   auto output0_desc1 = test_op.GetOutputDesc(0);
//   EXPECT_EQ(output0_desc1.GetShape().GetDims(), expected_y1_shape);
//   auto output0_desc2 = test_op.GetOutputDesc(1);
//   EXPECT_EQ(output0_desc2.GetShape().GetDims(), expected_y2_shape);
// }

TEST_F(BroadcastGradientArgsOpUT, rt2_shaperange_test)
{
    gert::Shape x1_in_min{
        2,
    };
    gert::Shape x1_in_max{
        2,
    };
    gert::Shape x2_in_min{
        5,
    };
    gert::Shape x2_in_max{
        5,
    };
    gert::Shape null1{};
    gert::Shape null2{};

    gert::Range<gert::Shape> x1_range(&x1_in_min, &x1_in_max);
    gert::Range<gert::Shape> x2_range(&x2_in_min, &x2_in_max);
    gert::Range<gert::Shape> out_shape_range(&null1, &null2);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs"), nullptr);
    auto shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs")->infer_shape_range;
    ASSERT_NE(shape_range_func, nullptr);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(2)
                              .NodeIoNum(2, 2)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&x1_range, &x2_range})
                              .OutputShapeRanges({&out_shape_range, &out_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    EXPECT_EQ(shape_range_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(1)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), 5);
    EXPECT_EQ(context->GetOutputShapeRange(1)->GetMax()->GetDim(0), 5);
}

TEST_F(BroadcastGradientArgsOpUT, rt2_dtype_test)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BroadcastGradientArgs")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_INT64;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(2, 2)
                                  .IrInstanceNum({1, 1})
                                  .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
    }
}

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_int32_normal) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({8});
//   auto x1_dtype = DT_INT32;
//   auto x2_shape = std::vector<int64_t>({8});
//   auto x2_dtype = DT_INT32;
//   std::vector<int32_t> x1_value = {1, 2, 3, 4, 5, 6, 7, 8};
//   std::vector<int32_t> x2_value = {1, 2, 1, 4, 1, 6, 1, 8};
//   // expect result info
//   std::vector<int64_t> expected_y1_shape = {1};
//   std::vector<int64_t> expected_y2_shape = {4};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");

//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_SUCCESS);
//   auto output0_desc1 = test_op.GetOutputDesc(0);
//   EXPECT_EQ(output0_desc1.GetShape().GetDims(), expected_y1_shape);
//   auto output0_desc2 = test_op.GetOutputDesc(1);
//   EXPECT_EQ(output0_desc2.GetShape().GetDims(), expected_y2_shape);
// }

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_int64_normal) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({8});
//   auto x1_dtype = DT_INT64;
//   auto x2_shape = std::vector<int64_t>({8});
//   auto x2_dtype = DT_INT64;
//   std::vector<int64_t> x1_value = {1, 2, 3, 4, 5, 6, 7, 8};
//   std::vector<int64_t> x2_value = {1, 2, 1, 4, 1, 6, 1, 8};
//   // expect result info
//   std::vector<int64_t> expected_y1_shape = {1};
//   std::vector<int64_t> expected_y2_shape = {4};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");

//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_SUCCESS);
//   auto output0_desc1 = test_op.GetOutputDesc(0);
//   EXPECT_EQ(output0_desc1.GetShape().GetDims(), expected_y1_shape);
//   auto output0_desc2 = test_op.GetOutputDesc(1);
//   EXPECT_EQ(output0_desc2.GetShape().GetDims(), expected_y2_shape);
// }

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_dtype_not_equal) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({8});
//   auto x1_dtype = DT_INT64;
//   auto x2_shape = std::vector<int64_t>({8});
//   auto x2_dtype = DT_INT32;
//   std::vector<int64_t> x1_value = {1, 2, 3, 4, 5, 6, 7, 8};
//   std::vector<int32_t> x2_value = {1, 2, 1, 4, 1, 6, 1, 8};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");

//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_FAILED);
// }

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_x1len_less_x2len) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({7});
//   auto x1_dtype = DT_INT32;
//   auto x2_shape = std::vector<int64_t>({8});
//   auto x2_dtype = DT_INT32;
//   std::vector<int32_t> x1_value = {2, 3, 4, 5, 6, 7, 8};
//   std::vector<int32_t> x2_value = {1, 2, 1, 4, 1, 6, 1, 8};
//   // expect result info
//   std::vector<int64_t> expected_y1_shape = {1};
//   std::vector<int64_t> expected_y2_shape = {4};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_SUCCESS);
//   auto output0_desc1 = test_op.GetOutputDesc(0);
//   EXPECT_EQ(output0_desc1.GetShape().GetDims(), expected_y1_shape);
//   auto output0_desc2 = test_op.GetOutputDesc(1);
//   EXPECT_EQ(output0_desc2.GetShape().GetDims(), expected_y2_shape);

// }

// TEST_F(BroadcastGradientArgsOpUT, BroadcastGradientArgs_all_equal) {
//   // input info
//   auto x1_shape = std::vector<int64_t>({8});
//   auto x1_dtype = DT_INT32;
//   auto x2_shape = std::vector<int64_t>({8});
//   auto x2_dtype = DT_INT32;
//   std::vector<int32_t> x1_value = {1, 2, 1, 4, 1, 6, 1, 8};
//   std::vector<int32_t> x2_value = {1, 2, 1, 4, 1, 6, 1, 8};
//   // expect result info
//   std::vector<int64_t> expected_y1_shape = {0};
//   std::vector<int64_t> expected_y2_shape = {0};

//   auto test_op = op::BroadcastGradientArgs("BroadcastGradientArgs");

//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x1, x1_shape, x1_dtype, FORMAT_ND, x1_value);
//   TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, x2, x2_shape, x2_dtype, FORMAT_ND, x2_value);

//   Runtime2TestParam mp_param;
//   mp_param.input_const = {true, true};
//   EXPECT_EQ(InferShapeTest(test_op, mp_param), ge::GRAPH_SUCCESS);
//   auto output0_desc1 = test_op.GetOutputDesc(0);
//   EXPECT_EQ(output0_desc1.GetShape().GetDims(), expected_y1_shape);
//   auto output0_desc2 = test_op.GetOutputDesc(1);
//   EXPECT_EQ(output0_desc2.GetShape().GetDims(), expected_y2_shape);
// }
