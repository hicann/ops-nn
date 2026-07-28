/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

TEST(HingeLossInferShape, same_shape_produces_predict_shape)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeLoss")->infer_shape;
    ASSERT_NE(func, nullptr);
    gert::StorageShape predict = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape target = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape loss;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&predict, &target})
                      .OutputShapes({&loss})
                      .Build();
    ASSERT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[1, -1, -1, 64]");
}

TEST(HingeLossInferShape, mismatched_shape_fails)
{
    auto func = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeLoss")->infer_shape;
    gert::StorageShape predict = {{1, 16, 64}, {1, 16, 64}};
    gert::StorageShape target = {{1, 8, 64}, {1, 8, 64}};
    gert::StorageShape loss;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&predict, &target})
                      .OutputShapes({&loss})
                      .Build();
    EXPECT_EQ(func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
