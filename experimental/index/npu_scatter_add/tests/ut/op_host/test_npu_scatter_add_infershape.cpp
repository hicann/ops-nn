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
 * \file test_npu_scatter_add_infershape.cpp
 * \brief NpuScatterAdd infershape/dtype UT
 */
#include <gtest/gtest.h>
#include <iostream>
#include "exe_graph/runtime/storage_shape.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"

using namespace ge;

class NpuScatterAddInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NpuScatterAddInferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NpuScatterAddInferShape Test TearDown" << std::endl; }
};

// 输出shape与输入y一致（全输入场景）
TEST_F(NpuScatterAddInferShape, npu_scatter_add_infer_shape_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x = {{8, 16}, {8, 16}};
    gert::StorageShape y = {{4, 16}, {4, 16}};
    gert::StorageShape s = {{8}, {8}};
    gert::StorageShape indices = {{8}, {8}};
    gert::StorageShape sortIdx = {{8}, {8}};
    gert::StorageShape valid = {{1}, {1}};
    gert::StorageShape yOut = {{4, 16}, {4, 16}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(6, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1}, {1})
                      .InputShapes({&x, &y, &s, &indices, &sortIdx, &valid})
                      .OutputShapes({&yOut})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* outShape = context->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    EXPECT_EQ(outShape->GetDimNum(), 2U);
    EXPECT_EQ(outShape->GetDim(0), 4);
    EXPECT_EQ(outShape->GetDim(1), 16);
}

// 可选输入s、valid_token_num缺省场景
TEST_F(NpuScatterAddInferShape, npu_scatter_add_infer_shape_without_optional_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x = {{16, 32}, {16, 32}};
    gert::StorageShape y = {{8, 32}, {8, 32}};
    gert::StorageShape indices = {{16}, {16}};
    gert::StorageShape sortIdx = {{16}, {16}};
    gert::StorageShape yOut = {{8, 32}, {8, 32}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 0, 1, 1, 0}, {1})
                      .InputShapes({&x, &y, &indices, &sortIdx})
                      .OutputShapes({&yOut})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    const gert::Shape* outShape = context->GetOutputShape(0);
    ASSERT_NE(outShape, nullptr);
    EXPECT_EQ(outShape->GetDimNum(), 2U);
    EXPECT_EQ(outShape->GetDim(0), 8);
    EXPECT_EQ(outShape->GetDim(1), 32);
}

// 输出dtype与输入y一致
TEST_F(NpuScatterAddInferShape, npu_scatter_add_infer_dtype_success)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("NpuScatterAdd")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);

    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType yDtype = ge::DT_BF16;
    ge::DataType sDtype = ge::DT_BF16;
    ge::DataType idxDtype = ge::DT_INT32;
    ge::DataType outDtype = ge::DT_BF16;

    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(6, 1)
                             .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&xDtype, &yDtype, &sDtype, &idxDtype, &idxDtype, &idxDtype})
                             .OutputDataTypes({&outDtype})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
