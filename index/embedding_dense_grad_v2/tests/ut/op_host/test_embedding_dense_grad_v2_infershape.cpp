/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_graph/embedding_dense_grad_v2_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
using namespace ge;
using namespace op;

class EmbeddingDenseGradV2ProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "EmbeddingDenseGradV2ProtoTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "EmbeddingDenseGradV2ProtoTest TearDown" << std::endl;
    }
};

TEST_F(EmbeddingDenseGradV2ProtoTest, infer_shape_test) {
    ge::op::EmbeddingDenseGradV2 op;
    op.UpdateInputDesc("grad", create_desc({1024, 4096}, ge::DT_FLOAT));
    op.UpdateInputDesc("sort_indices", create_desc({1024}, ge::DT_INT32));
    op.UpdateInputDesc("pos_idx", create_desc({1024}, ge::DT_INT32));
    op.SetAttr("num_weights", 100);
    op.SetAttr("padding_idx", -1);
    op.SetAttr("scale_grad_by_freq", false);

    std::vector<int64_t> expectedValueShape = {100, 4096};
    Runtime2TestParam param {{"num_weights", "padding_idx", "scale_grad_by_freq"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputValue = op.GetOutputDescByName("y");
    EXPECT_EQ(outputValue.GetShape().GetDims(), expectedValueShape);
}

TEST_F(EmbeddingDenseGradV2ProtoTest, infer_dtype_test) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingDenseGradV2"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EmbeddingDenseGradV2")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT32;
        ge::DataType inputRef3 = ge::DT_INT32;
        ge::DataType outputRef = ge::DT_FLOAT;
        auto contextHolder = gert::InferDataTypeContextFaker()
            .NodeIoNum(3, 1)
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&inputRef1, &inputRef2, &inputRef3})
            .OutputDataTypes({&outputRef})
            .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}
