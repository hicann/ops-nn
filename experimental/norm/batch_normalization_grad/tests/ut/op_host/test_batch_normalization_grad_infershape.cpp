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

class BatchNormalizationGradInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormalizationGradInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormalizationGradInfershape TearDown" << std::endl;
    }
};

TEST_F(BatchNormalizationGradInfershape, batch_normalization_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormalizationGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape gradOutputShape = {{16, 16, 64, 2}, {16, 16, 64, 2}};
    gert::StorageShape inputShape = {{16, 16, 64, 2}, {16, 16, 64, 2}};
    gert::StorageShape weightShape = {{16}, {16}};
    gert::StorageShape biasShape = {{16}, {16}};
    gert::StorageShape saveMeanShape = {{16}, {16}};
    gert::StorageShape saveInvstdShape = {{16}, {16}};
    gert::StorageShape gradInputShape;
    gert::StorageShape gradWeightShape;
    gert::StorageShape gradBiasShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(6, 3)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
            .InputShapes({&gradOutputShape, &inputShape, &weightShape,
                          &biasShape, &saveMeanShape, &saveInvstdShape})
            .OutputShapes({&gradInputShape, &gradWeightShape, &gradBiasShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto gradInput = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*gradInput), "[16, 16, 64, 2]");

    auto gradWeight = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*gradWeight), "[16]");

    auto gradBias = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*gradBias), "[16]");
}
