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

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
class TestFusedQuantMatmulInferShape : public testing::Test
{
};

TEST_F(TestFusedQuantMatmulInferShape, InferShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape = {{96, 1408}, {96, 1408}};
    gert::StorageShape x2Shape = {{2, 1664, 1408}, {2, 1664, 1408}};
    gert::StorageShape x2ScaleShape = {{1664}, {1664}};
    gert::StorageShape outputShape;

    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(11, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape, nullptr, nullptr, &x2ScaleShape,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
            .OutputShapes({&outputShape})
            .NodeAttrs(
                {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                 {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"fused_op_type", Ops::NN::AnyValue::CreateFrom<string>("swiglu")}})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[96, 1664]");
}

TEST_F(TestFusedQuantMatmulInferShape, dimNumNot3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape = {{96, 1408}, {96, 1408}};
    gert::StorageShape x2Shape = {{1664, 1408}, {1664, 1408}};
    gert::StorageShape x2ScaleShape = {{1664}, {1664}};
    gert::StorageShape outputShape;

    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(11, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape, nullptr, nullptr, &x2ScaleShape,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
            .OutputShapes({&outputShape})
            .NodeAttrs(
                {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                 {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"fused_op_type", Ops::NN::AnyValue::CreateFrom<string>("swiglu")}})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestFusedQuantMatmulInferShape, highestDimNot2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedQuantMatMul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape = {{96, 1408}, {96, 1408}};
    gert::StorageShape x2Shape = {{4, 1664, 1408}, {4, 1664, 1408}};
    gert::StorageShape x2ScaleShape = {{1664}, {1664}};
    gert::StorageShape outputShape;

    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(11, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape, nullptr, nullptr, &x2ScaleShape,
                          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr})
            .OutputShapes({&outputShape})
            .NodeAttrs(
                {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype)},
                 {"compute_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                 {"fused_op_type", Ops::NN::AnyValue::CreateFrom<string>("swiglu")}})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
}