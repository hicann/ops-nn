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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

class SPARSE_TENSOR_DENSE_MAT_MUL_UT : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SPARSE_TENSOR_DENSE_MAT_MUL_UT SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SPARSE_TENSOR_DENSE_MAT_MUL_UT TearDown" << std::endl;
    }
};

template <typename T>
gert::Tensor* ConstructInputConstTensor(
    std::unique_ptr<uint8_t[]>& input_tensor_holder, const vector<T>& const_value, ge::DataType const_dtype,
    const gert::StorageShape tensorShape)
{
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor(
        tensorShape,                        // shape
        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
        gert::kFollowing,                   // placement
        const_dtype,                        // dt
        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    std::memcpy(tensor_data, &const_value[0], sizeof(T) * const_value.size());
    input_tensor->SetData(gert::TensorData(tensor_data, nullptr));
    return input_tensor;
}

TEST_F(SPARSE_TENSOR_DENSE_MAT_MUL_UT, SparseTensorDenseMatMul_InferShape_test_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SparseTensorDenseMatMul"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SparseTensorDenseMatMul")->infer_shape;
    vector<int32_t> indices = {0, 0, 1, 2, 2, 3};
    gert::StorageShape indicesShape = {{3, 2}, {3, 2}};
    auto indices_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * 6]);
    auto indices_tensor = ConstructInputConstTensor<int32_t>(indices_input_tensor, indices, ge::DT_INT32, indicesShape);

    vector<int32_t> values = {1, 2, 3};
    gert::StorageShape valuesShape = {{3}, {3}};
    auto values_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * 3]);
    auto values_tensor = ConstructInputConstTensor<int32_t>(values_input_tensor, values, ge::DT_INT32, valuesShape);

    vector<int64_t> shape = {3, 4};
    gert::StorageShape shapeShape = {{2}, {2}};
    auto shape_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int64_t) * 2]);
    auto shape_tensor = ConstructInputConstTensor<int64_t>(shape_input_tensor, shape, ge::DT_INT64, shapeShape);

    vector<int32_t> x2 = {1, 2, 3, 4, 5, 6, 7, 8};
    gert::StorageShape x2Shape = {{4, 2}, {4, 2}};
    auto x2_input_tensor = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(gert::Tensor) + sizeof(int32_t) * 8]);
    auto x2_tensor = ConstructInputConstTensor<int32_t>(x2_input_tensor, x2, ge::DT_INT32, x2Shape);

    gert::StorageShape output_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1})
                      .NodeIoNum(4, 1)
                      .InputShapes({indices_tensor, values_tensor, shape_tensor, x2_tensor})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"adjoint_a", ge::AnyValue::CreateFrom<bool>(false)},
                           {"adjoint_b", ge::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 2);
    EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 3);
    EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 2);
}

TEST_F(SPARSE_TENSOR_DENSE_MAT_MUL_UT, SparseTensorDenseMatMul_InferDtype_test_1)
{
    ge::op::SparseTensorDenseMatMul op;
    op.UpdateInputDesc("x1_indices", create_desc({2, 2}, ge::DT_INT32));
    op.UpdateInputDesc("x1_values", create_desc({2}, ge::DT_INT32));
    op.UpdateInputDesc("x1_shape", create_desc({2}, ge::DT_INT64));
    op.UpdateInputDesc("x2", create_desc({2, 2}, ge::DT_INT32));
    op.SetAttr("adjoint_a", false);
    op.SetAttr("adjoint_b", false);
    Runtime2TestParam param{{"adjoint_a", "adjoint_b"}};
    auto ret = InferDataTypeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto outputDtype = op.GetOutputDesc(0).GetDataType();
    EXPECT_EQ(outputDtype, ge::DT_INT32);
}

