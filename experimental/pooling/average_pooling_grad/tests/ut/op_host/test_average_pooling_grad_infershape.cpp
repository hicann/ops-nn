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

#include <array>
#include <iostream>
#include <vector>

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

class AveragePoolingGradInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AveragePoolingGradInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AveragePoolingGradInfershape TearDown" << std::endl;
    }
};

TEST_F(AveragePoolingGradInfershape, average_pooling_grad_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AveragePoolingGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::array<int64_t, 4> origInputShapeValue = {2, 1, 4, 6};
    gert::StorageShape origInputShapeStorage = {{4}, {4}};
    gert::StorageShape gradOutputShapeStorage = {{2, 1, 4, 6}, {2, 1, 4, 6}};
    gert::Tensor origInputShapeTensor(
        origInputShapeStorage,
        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost,
        ge::DT_INT64,
        origInputShapeValue.data());
    gert::Tensor gradOutputTensor(
        gradOutputShapeStorage,
        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost,
        ge::DT_FLOAT,
        nullptr);

    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1}, {1})
            .InputTensors({&origInputShapeTensor, &gradOutputTensor})
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 1, 4, 6]");
}
