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

class HingeLossGradInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "HingeLossGradInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HingeLossGradInfershape TearDown" << std::endl;
    }
};

TEST_F(HingeLossGradInfershape, hinge_loss_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeLossGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape predictShape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape targetShape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape gradOutputShape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape outputShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&predictShape, &targetShape, &gradOutputShape})
            .OutputShapes({&outputShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, -1, 64]");
}

TEST_F(HingeLossGradInfershape, hinge_loss_grad_infershape_target_shape_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeLossGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape predictShape = {{1, 16, 64}, {1, 16, 64}};
    gert::StorageShape targetShape = {{1, 8, 64}, {1, 8, 64}};
    gert::StorageShape gradOutputShape = {{1, 16, 64}, {1, 16, 64}};
    gert::StorageShape outputShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&predictShape, &targetShape, &gradOutputShape})
            .OutputShapes({&outputShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(HingeLossGradInfershape, hinge_loss_grad_infershape_grad_output_shape_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("HingeLossGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape predictShape = {{1, 16, 64}, {1, 16, 64}};
    gert::StorageShape targetShape = {{1, 16, 64}, {1, 16, 64}};
    gert::StorageShape gradOutputShape = {{1, 16, 32}, {1, 16, 32}};
    gert::StorageShape outputShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&predictShape, &targetShape, &gradOutputShape})
            .OutputShapes({&outputShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
