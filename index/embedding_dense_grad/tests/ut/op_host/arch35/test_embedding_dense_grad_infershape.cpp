/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_dense_grad_infershape.cpp
 * \brief EmbeddingDenseGrad InferShape UT (V2 registry, op_host chain)
 */

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../op_graph/embedding_dense_grad_proto.h"

class EmbeddingDenseGradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "embedding_dense_grad test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "embedding_dense_grad test TearDown" << std::endl; }
};

static void RunEmbeddingDenseGradShapeTest(const std::vector<int64_t>& gradDims,
                                           const std::vector<int64_t>& indicesDims, int64_t numWeights,
                                           int64_t paddingIdx, int64_t scaleGradByFreq,
                                           const std::vector<int64_t>& expectedOut)
{
    ge::op::EmbeddingDenseGrad op;
    ge::TensorDesc grad;
    ge::Shape grad_shape(gradDims);
    grad.SetDataType(ge::DT_FLOAT);
    grad.SetShape(grad_shape);
    grad.SetOriginShape(grad_shape);
    ge::TensorDesc indices;
    ge::Shape indices_shape(indicesDims);
    indices.SetDataType(ge::DT_INT32);
    indices.SetShape(indices_shape);
    indices.SetOriginShape(indices_shape);

    op.UpdateInputDesc("grad", grad);
    op.UpdateInputDesc("indices", indices);
    op.SetAttr("num_weights", numWeights);
    op.SetAttr("padding_idx", paddingIdx);
    op.SetAttr("scale_grad_by_freq", scaleGradByFreq);

    Runtime2TestParam param;
    param.attrs = {"num_weights", "padding_idx", "scale_grad_by_freq"};
    ASSERT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expectedOut);
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_1)
{
    RunEmbeddingDenseGradShapeTest({20, 30, 32}, {20, 30}, 2, -1, 1, {2, 32});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_2)
{
    RunEmbeddingDenseGradShapeTest({20, 300, 64}, {20, 300}, 2000, 7, 1, {2000, 64});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_3)
{
    RunEmbeddingDenseGradShapeTest({7, 7, 7, 7, 1024}, {7, 7, 7, 7}, 99, 7, 0, {99, 1024});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_4)
{
    RunEmbeddingDenseGradShapeTest({3, 3, 32}, {3, 3}, 23, 7, 1, {23, 32});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_5)
{
    RunEmbeddingDenseGradShapeTest({-1, -1, -1}, {2, 2}, 2000, 7, -1, {2000, -1});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_6)
{
    RunEmbeddingDenseGradShapeTest({-1, -1, -1, -1, -1}, {-1, -1, -1, -1}, 20000, 7, 1, {20000, -1});
}

TEST_F(EmbeddingDenseGradTest, embedding_dense_grad_test_case_7)
{
    RunEmbeddingDenseGradShapeTest({-1, -1, 256}, {-1, -1}, 20010, 7, 0, {20010, 256});
}
