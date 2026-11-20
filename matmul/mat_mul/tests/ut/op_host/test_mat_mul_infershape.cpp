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

namespace {
class MatMulInferShapeTest : public testing::Test {};

TEST_F(MatMulInferShapeTest, X1RankGreaterThan2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x1_shape = {2, 3, 4};
    gert::Shape x2_shape = {4, 5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1_shape, &x2_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MatMulInferShapeTest, X2RankGreaterThan2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape x1_shape = {2, 4};
    gert::Shape x2_shape = {2, 4, 5};
    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1_shape, &x2_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
} // namespace
