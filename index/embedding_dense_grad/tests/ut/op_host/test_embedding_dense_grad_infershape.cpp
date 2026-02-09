/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file test_embedding_dense_grad_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"
#include "../../../op_graph/embedding_dense_grad_proto.h"

class EmbeddingDenseGradTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "embedding_dense_grad test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "embedding_dense_grad test TearDown" << std::endl;
    }
};

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_1) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({20, 30, 32});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({20, 30});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 2);
	embedding_dense_grad_op.SetAttr("padding_idx", -1);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 1);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2, 32};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_2) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({20, 300, 64});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({20, 300});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 2000);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 1);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2000, 64};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_3) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({7, 7, 7, 7, 1024});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({7, 7, 7, 7});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 99);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 0);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {99, 1024};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_4) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({3, 3, 32});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({3, 3});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 23);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 1);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {23, 32};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_5) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({-1, -1, -1});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({2, 2});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 2000);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", -1);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {2000, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_6) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({-1, -1, -1, -1, -1});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({-1, -1, -1, -1});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 20000);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 1);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {20000, -1};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_7) {
    ge::op::EmbeddingDenseGrad embedding_dense_grad_op;
    ge::TensorDesc grad;
    ge::Shape grad_shape({-1, -1, 256});
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
	ge::TensorDesc indices;
    ge::Shape indices_shape({-1, -1});
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    embedding_dense_grad_op.UpdateInputDesc("grad", grad);
	embedding_dense_grad_op.UpdateInputDesc("indices", indices);
	embedding_dense_grad_op.SetAttr("num_weights", 20010);
	embedding_dense_grad_op.SetAttr("padding_idx", 7);
	embedding_dense_grad_op.SetAttr("scale_grad_by_freq", 0);

    auto ret = embedding_dense_grad_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_desc = embedding_dense_grad_op.GetOutputDescByName("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
    std::vector<int64_t> expected_output_shape = {20010, 256};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    std::vector<bool> input_const = {false, false};
    std::vector<std::string> attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};

    Runtime2TestParam param;
    param.input_const = input_const;
    param.attrs = attrs;
    EXPECT_EQ(InferShapeTest(embedding_dense_grad_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = embedding_dense_grad_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}