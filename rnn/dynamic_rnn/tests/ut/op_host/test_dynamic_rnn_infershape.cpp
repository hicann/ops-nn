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
 * \file test_dynamic_rnn_proto.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <vector>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry_base.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_graph/dynamic_rnn_proto.h"

using namespace ge;

class DynamicRnnTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_rnn test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_rnn test TearDown" << std::endl;
    }
};

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_1) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);
    WDesc.SetOriginShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
    BDesc.SetOriginShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    // auto ret = rnn_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // auto output_desc = rnn_op.GetOutputDesc("y");
    // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {t,batch, outputSize};
    // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

    Runtime2TestParam param{{"cell_type", "direction"}};
    EXPECT_EQ(InferShapeTest(rnn_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = rnn_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_2) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize,outputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    // auto ret = rnn_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_FAILED);

    Runtime2TestParam param{{"cell_type", "direction"}};
    EXPECT_EQ(InferShapeTest(rnn_op, param), ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_case_3) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, 4*outputSize});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);
    WDesc.SetOriginShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
    BDesc.SetOriginShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);
    rnn_op.SetAttr("direction", "BIDIRECTIONAL");

    // auto ret = rnn_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // auto output_desc = rnn_op.GetOutputDesc("y");
    std::vector<int64_t> expected_output1_shape = {t, batch, 2 * outputSize};
    // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output1_shape);

    Runtime2TestParam param{{"cell_type", "direction"}};
    EXPECT_EQ(InferShapeTest(rnn_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = rnn_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output1_shape);

    std::vector<int64_t> expected_output2_shape = {2 * t, batch, outputSize};
    auto output3_desc = rnn_op.GetOutputDesc(3);
    EXPECT_EQ(output3_desc.GetShape().GetDims(), expected_output2_shape);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_inferdtype_0) {
  ge::op::DynamicRNN op;

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicRNN"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicRNN")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);

  ge::DataType x_datatype1 = ge::DT_FLOAT16;
  ge::DataType w_datatype1 = ge::DT_FLOAT16;
  ge::DataType b_datatype1 = ge::DT_FLOAT16;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = gert::InferDataTypeContextFaker()
                            .IrInputNum(3)
                            .NodeIoNum(3, 8)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .InputDataTypes({&x_datatype1, &w_datatype1, &b_datatype1})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);
}

TEST_F(DynamicRnnTest, dynamic_rnn_test_with_unknowndim) {
    int t = 3;
    int batch = 16;
    int inputSize = 32;
    int outputSize = 48;
    ge::op::DynamicRNN rnn_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({t, batch, inputSize});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);

    ge::TensorDesc WDesc;
    ge::Shape wShape({inputSize+outputSize, -1});
    WDesc.SetDataType(ge::DT_FLOAT16);
    WDesc.SetShape(wShape);
    WDesc.SetOriginShape(wShape);

    ge::TensorDesc BDesc;
    ge::Shape bShape({4*outputSize});
    BDesc.SetDataType(ge::DT_FLOAT16);
    BDesc.SetShape(bShape);
    BDesc.SetOriginShape(bShape);

    rnn_op.UpdateInputDesc("x", XDesc);
    rnn_op.UpdateInputDesc("w", WDesc);
    rnn_op.UpdateInputDesc("b", BDesc);

    // auto ret = rnn_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_output_shape = {t,batch, -1};
    Runtime2TestParam param{{"cell_type", "direction"}};
    EXPECT_EQ(InferShapeTest(rnn_op, param), ge::GRAPH_SUCCESS);
    auto output0_desc = rnn_op.GetOutputDesc(0);
    EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);
}