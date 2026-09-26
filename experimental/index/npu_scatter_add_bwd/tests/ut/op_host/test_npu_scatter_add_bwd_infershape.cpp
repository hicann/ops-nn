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
 * \file test_npu_scatter_add_bwd_infershape.cpp
 * \brief NpuScatterAddBwd infershape/dtype UT
 */
#include <gtest/gtest.h>
#include <iostream>
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"

using namespace ge;

class NpuScatterAddBwdInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NpuScatterAddBwdInferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NpuScatterAddBwdInferShape Test TearDown" << std::endl; }
};

// 输出x_grad与输入x一致，s_grad与输入s一致
TEST_F(NpuScatterAddBwdInferShape, npu_scatter_add_bwd_infer_shape_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape yGrad = {{4, 16}, {4, 16}};
    gert::StorageShape x = {{8, 16}, {8, 16}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape xGrad = {{8, 16}, {8, 16}};
    gert::StorageShape sGrad = {{8}, {8}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1})
                      .InputShapes({&yGrad, &x, &s, &indices})
                      .OutputShapes({&xGrad, &sGrad})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* xGradOutShape = context->GetOutputShape(0);
    const gert::Shape* sGradOutShape = context->GetOutputShape(1);
    ASSERT_NE(xGradOutShape, nullptr);
    ASSERT_NE(sGradOutShape, nullptr);
    EXPECT_EQ(xGradOutShape->GetDimNum(), 2U);
    EXPECT_EQ(xGradOutShape->GetDim(0), 8);
    EXPECT_EQ(xGradOutShape->GetDim(1), 16);
    EXPECT_EQ(sGradOutShape->GetDimNum(), 1U);
    EXPECT_EQ(sGradOutShape->GetDim(0), 8);
}

// 输出dtype与对应输入一致
TEST_F(NpuScatterAddBwdInferShape, npu_scatter_add_bwd_infer_dtype_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAddBwd")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType yGradDtype = ge::DT_FLOAT16;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType sDtype = ge::DT_FLOAT16;
    ge::DataType idxDtype = ge::DT_INT32;
    ge::DataType xGradDtype = ge::DT_FLOAT16;
    ge::DataType sGradDtype = ge::DT_FLOAT16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(4, 2)
                             .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&yGradDtype, &xDtype, &sDtype, &idxDtype})
                             .OutputDataTypes({&xGradDtype, &sGradDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT16);
}
