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
class TestDualLevelQuantBatchMatmulInferShape : public testing::Test {};

TEST_F(TestDualLevelQuantBatchMatmulInferShape, InferShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DualLevelQuantBatchMatmul")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape = {{1, 7168}, {1536, 7168}};
    gert::StorageShape x2Shape = {{1536, 7168}, {112, 96, 16, 64}};
    gert::StorageShape x1Level0ScaleShape = {{1, 14}, {1, 14}};
    gert::StorageShape x1Level1ScaleShape = {{1, 112, 2}, {1, 112, 2}};
    gert::StorageShape x2Level0ScaleShape = {{14, 1536}, {14, 1536}};
    gert::StorageShape x2Level1ScaleShape = {{1536, 112, 2}, {1536, 112, 2}};
    gert::StorageShape yShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(7, 1)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&x1Shape, &x2Shape,
                &x1Level0ScaleShape, &x1Level1ScaleShape, &x2Level0ScaleShape, &x2Level1ScaleShape, nullptr})
            .OutputShapes({&yShape})
            .NodeAttrs(
                {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                 {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"level0_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(512)},
                 {"level1_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(32)}})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 1536]");
}
} // namespace