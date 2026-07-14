/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
#include "kernel_run_context_faker.h"
#include "register/op_impl_registry.h"

class GroupNormalizationGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormalizationGradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormalizationGradInfershape TearDown" << std::endl; }
};

TEST_F(GroupNormalizationGradInfershape, group_normalization_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormalizationGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 2, 64}, {1, 2, 64}};
    gert::StorageShape meanRstdShape = {{1, 2}, {1, 2}}; // [N, G]
    gert::StorageShape outputShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &xShape, &xShape, &meanRstdShape, &meanRstdShape})
                      .OutputShapes({&outputShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 2, 64]");
}

// negative: mean/rstd shape not [N, G] -> must fail
TEST_F(GroupNormalizationGradInfershape, group_normalization_grad_infershape_mean_mismatch_fail)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormalizationGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{1, 2, 64}, {1, 2, 64}};
    gert::StorageShape badMeanShape = {{1, 3}, {1, 3}}; // G=3 != x.G=2
    gert::StorageShape outputShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &xShape, &xShape, &badMeanShape, &badMeanShape})
                      .OutputShapes({&outputShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
