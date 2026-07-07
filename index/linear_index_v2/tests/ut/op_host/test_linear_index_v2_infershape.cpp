/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "util/shape_util.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;

class LinearIndexV2InferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinearIndexV2InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "LinearIndexV2InferShape TearDown" << std::endl; }
};

TEST_F(LinearIndexV2InferShape, infershape_normal_int32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape strideShape = {{2}, {2}};
    gert::StorageShape valueSizeShape = {{1}, {1}};
    gert::StorageShape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&indicesShape, &strideShape, &valueSizeShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[3]");
}

TEST_F(LinearIndexV2InferShape, infershape_normal_int64)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape indicesShape = {{3}, {3}};
    gert::StorageShape strideShape = {{2}, {2}};
    gert::StorageShape valueSizeShape = {{1}, {1}};
    gert::StorageShape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&indicesShape, &strideShape, &valueSizeShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[3]");
}

TEST_F(LinearIndexV2InferShape, infershape_multi_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape indicesShape = {{2, 3}, {2, 3}};
    gert::StorageShape strideShape = {{2}, {2}};
    gert::StorageShape valueSizeShape = {{1}, {1}};
    gert::StorageShape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&indicesShape, &strideShape, &valueSizeShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), "[6]");
}

TEST_F(LinearIndexV2InferShape, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape indicesShape = {{-2}, {-2}};
    gert::StorageShape strideShape = {{2}, {2}};
    gert::StorageShape valueSizeShape = {{1}, {1}};
    gert::StorageShape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&indicesShape, &strideShape, &valueSizeShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(LinearIndexV2InferShape, infershape_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape indicesShape = {{-1, 3}, {-1, 3}};
    gert::StorageShape strideShape = {{2}, {2}};
    gert::StorageShape valueSizeShape = {{1}, {1}};
    gert::StorageShape outputShape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&indicesShape, &strideShape, &valueSizeShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(LinearIndexV2InferShape, inferdtype_int32)
{
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType inputRef = ge::DT_INT32;
    ge::DataType outputRef = ge::DT_INT32;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRef, &inputRef, &inputRef})
                              .OutputDataTypes({&outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST_F(LinearIndexV2InferShape, inferdtype_int64)
{
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LinearIndexV2")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType inputRef = ge::DT_INT64;
    ge::DataType outputRef = ge::DT_INT64;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(3)
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&inputRef, &inputRef, &inputRef})
                              .OutputDataTypes({&outputRef})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}
