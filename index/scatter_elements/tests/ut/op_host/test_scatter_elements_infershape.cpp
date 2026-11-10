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
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"

using namespace ge;

class ScatterElementsInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterElementsInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScatterElementsInfershape TearDown" << std::endl; }
};

static void RunScatterElementsInferShape(gert::Shape& varShape, const gert::Shape& expectedShape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterElements")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape indicesShape = {2, 1};
    gert::Shape updatesShape = {2, 8};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&varShape, &indicesShape, &updatesShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// 常规场景：输出shape与输入0的shape一致
TEST_F(ScatterElementsInfershape, scatter_elements_infershape_test_1)
{
    gert::Shape varShape = {4, 8};
    gert::Shape expectedShape = {4, 8};
    RunScatterElementsInferShape(varShape, expectedShape);
}

// 部分未知shape（-1）场景：输出shape透传
TEST_F(ScatterElementsInfershape, scatter_elements_infershape_partial_unknown)
{
    gert::Shape varShape = {2, -1};
    gert::Shape expectedShape = {2, -1};
    RunScatterElementsInferShape(varShape, expectedShape);
}

// 全未知shape（-1）场景：输出shape透传
TEST_F(ScatterElementsInfershape, scatter_elements_infershape_all_unknown_dim)
{
    gert::Shape varShape = {-1, -1};
    gert::Shape expectedShape = {-1, -1};
    RunScatterElementsInferShape(varShape, expectedShape);
}

// 未知rank（-2）场景：输出置为未知rank
TEST_F(ScatterElementsInfershape, scatter_elements_infershape_unknown_rank)
{
    gert::Shape varShape = {-2};
    gert::Shape expectedShape = {-2};
    RunScatterElementsInferShape(varShape, expectedShape);
}

// 与geir st用例scatter_elements_geir_dyn_dim0_float32对齐：第0维动态（-1），axis维保持静态
TEST_F(ScatterElementsInfershape, scatter_elements_infershape_dim0_unknown)
{
    gert::Shape varShape = {-1, 32, 48};
    gert::Shape expectedShape = {-1, 32, 48};
    RunScatterElementsInferShape(varShape, expectedShape);
}
