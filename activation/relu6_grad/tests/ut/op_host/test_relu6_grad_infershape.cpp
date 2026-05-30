/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class relu6grad : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "relu6grad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "relu6grad TearDown" << std::endl;
    }
};

TEST_F(relu6grad, relu6grad_infer_shape_same_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape Shape = {{16, 16}, {16, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {16, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(relu6grad, relu6grad_infer_shape_same_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape Shape = {{128, 256}, {128, 256}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {128, 256};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(relu6grad, relu6grad_infer_shape_same_bf16_3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape Shape = {{32, 64, 128}, {32, 64, 128}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {32, 64, 128};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

// gradients is scalar [1], features [16,16] -> output [16,16].
TEST_F(relu6grad, relu6grad_infer_shape_gradients_scalar_broadcast)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape gShape = {{1}, {1}};
    gert::StorageShape xShape = {{16, 16}, {16, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gShape, &xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {16, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

// Per-axis broadcast gradients=[16,1], features=[1,16] -> [16,16].
TEST_F(relu6grad, relu6grad_infer_shape_axis_broadcast_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape gShape = {{16, 1}, {16, 1}};
    gert::StorageShape xShape = {{1, 16}, {1, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gShape, &xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {16, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

// Cross-rank broadcast gradients=[4], features=[3,4] -> [3,4], bf16.
TEST_F(relu6grad, relu6grad_infer_shape_cross_rank_broadcast_bf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape gShape = {{4}, {4}};
    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gShape, &xShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {3, 4};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(relu6grad, relu6grad_infer_shape_dynamic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Relu6Grad")->infer_shape;
    gert::StorageShape Shape = {{-1, -1}, {-1, -1}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
