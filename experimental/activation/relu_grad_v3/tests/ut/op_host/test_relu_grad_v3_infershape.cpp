/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
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

class ReluGradV3Infershape : public testing::Tes {
protected:
    static void SetUpTestCase() { std::cout << "ReluGradV3Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReluGradV3Infershape TearDown" << std::endl; }
};

TEST_F(ReluGradV3Infershape, relu_grad_v3_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReluGradV3")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape yShape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape zShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &yShape})
                      .OutputShapes({&zShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, -1, 64]");
}

TEST_F(ReluGradV3Infershape, relu_grad_v3_infershape_y_scalar_broadcast_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReluGradV3")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{256, 32}, {256, 32}};
    gert::StorageShape yShape = {{1}, {1}};
    gert::StorageShape zShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &yShape})
                      .OutputShapes({&zShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[256, 32]");
}

TEST_F(ReluGradV3Infershape, relu_grad_v3_infershape_4d_broadcast_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReluGradV3")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{8, 32, 1, 64}, {8, 32, 1, 64}};
    gert::StorageShape yShape = {{1, 32, 128, 64}, {1, 32, 128, 64}};
    gert::StorageShape zShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &yShape})
                      .OutputShapes({&zShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[8, 32, 128, 64]");
}

TEST_F(ReluGradV3Infershape, relu_grad_v3_infershape_incompatible_broadcast_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ReluGradV3")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape = {{4, 3}, {4, 3}};
    gert::StorageShape zShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &yShape})
                      .OutputShapes({&zShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
