/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
class TestWeightQuantBatchMatmulV2InferShape : public testing::Test
{
};

TEST_F(TestWeightQuantBatchMatmulV2InferShape, InferShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("WeightQuantBatchMatmulV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{32, 32, 96, 11264}, {32, 32, 96, 11264}};
    gert::StorageShape weightShape = {{1, 1664, 1408}, {1, 1664, 1408}};
    gert::StorageShape antiQuantScale = {{1664}, {1664}};
    gert::StorageShape antiQuantOffset = {{1664}, {1664}};
    gert::StorageShape yShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(7, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&xShape, &weightShape, &antiQuantScale, &antiQuantOffset, nullptr, nullptr, nullptr})
            .OutputShapes({&yShape})
            .NodeAttrs(
                {{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[32, 32, 96, 1664]");
}

TEST_F(TestWeightQuantBatchMatmulV2InferShape, UnknownDimNum)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("WeightQuantBatchMatmulV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = {-2};
    gert::Shape weightShape = {-2};
    gert::Shape antiQuantScale = {-2};
    gert::Shape antiQuantOffset = {-2};
    gert::Shape expectYShape = {-2};

    gert::Shape yShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xShape, &weightShape, &antiQuantScale, &antiQuantOffset})
                      .OutputShapes({&yShape})
                      .NodeAttrs(
                          {{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectYShape));
}
} // namespace
