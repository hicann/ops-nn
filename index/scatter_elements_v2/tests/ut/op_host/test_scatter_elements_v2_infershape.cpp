/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_elements_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "../../../op_graph/scatter_elements_v2_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"


class ScatterElementsV2Proto : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterElementsV2Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScatterElementsV2Proto Test TearDown" << std::endl;
    }
};

TEST_F(ScatterElementsV2Proto, scatter_elements_v2_fp32_shape)
{
    ge::op::ScatterElementsV2 op;
    op.UpdateInputDesc("var", create_desc({4096, 5933}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({4096, 5933}, ge::DT_FLOAT));
    op.UpdateInputDesc("updates", create_desc({4096, 5933}, ge::DT_FLOAT));

    std::vector<int64_t> expectedValueShape = {4096, 5933};
    Runtime2TestParam param{};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputValue = op.GetOutputDesc("var");
    EXPECT_EQ(outputValue.GetShape().GetDims(), expectedValueShape);
}

TEST_F(ScatterElementsV2Proto, scatter_elements_v2_fp32_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterElementsV2"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterElementsV2")->infer_datatype;

    if (dataTypeFunc != nullptr) {
        ge::DataType inputRef1 = ge::DT_FLOAT;
        ge::DataType inputRef2 = ge::DT_INT64;
        ge::DataType outputRef = ge::DT_FLOAT;
        auto contextHolder = gert::InferDataTypeContextFaker()
                                 .NodeIoNum(3, 1)
                                 .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                 .InputDataTypes({&inputRef1, &inputRef2, &inputRef1})
                                 .OutputDataTypes({&outputRef})
                                 .Build();
        auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), outputRef);
    }
}
