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

class CrossEntropyGradInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossEntropyGradInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossEntropyGradInfershape TearDown" << std::endl;
    }
};

TEST_F(CrossEntropyGradInfershape, cross_entropy_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape logitsShape = {{256, 32}, {256, 32}};
    gert::StorageShape targetShape = {{256}, {256}};
    gert::StorageShape outputShape;

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&logitsShape, &targetShape})
            .OutputShapes({&outputShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[256, 32]");
}
