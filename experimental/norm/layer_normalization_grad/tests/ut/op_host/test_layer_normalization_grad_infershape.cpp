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

class LayerNormalizationGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LayerNormalizationGradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LayerNormalizationGradInfershape TearDown" << std::endl; }
};

TEST_F(LayerNormalizationGradInfershape, layer_normalization_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormalizationGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // 输入: dy/x 2D [256,128], gamma 1D [128], mean/rstd 1D [256]
    gert::StorageShape dyShape = {{256, 128}, {256, 128}};
    gert::StorageShape xShape = {{256, 128}, {256, 128}};
    gert::StorageShape gammaShape = {{128}, {128}};
    gert::StorageShape meanShape = {{256}, {256}};
    gert::StorageShape rstdShape = {{256}, {256}};
    gert::StorageShape dxShape;
    gert::StorageShape dgammaShape;
    gert::StorageShape dbetaShape;

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dyShape, &xShape, &gammaShape, &meanShape, &rstdShape})
                      .OutputShapes({&dxShape, &dgammaShape, &dbetaShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto dx = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*dx), "[256, 128]");

    auto dgamma = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*dgamma), "[128]");

    auto dbeta = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*dbeta), "[128]");
}
