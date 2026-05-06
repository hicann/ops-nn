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

class AddExampleInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddExampleInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddExampleInfershape TearDown" << std::endl;
    }
};

TEST_F(AddExampleInfershape, add_example_infershape_test1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AddExample")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape x1Shape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape x2Shape = {{1, -1, -1, 64}, {1, -1, -1, 64}};
    gert::StorageShape outputShape;

    int64_t dtype = static_cast<int64_t> (ge::DT_FLOAT16);
    auto holder =
        gert::InferShapeContextFaker()
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&x1Shape, &x2Shape})
            .OutputShapes({&outputShape})
            .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, -1, -1, 64]");
}