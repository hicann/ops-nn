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
#include "infershape_test_util.h"
#include "platform/platform_info.h"

using namespace ge;

class scatter_min : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "scatter_min SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "scatter_min TearDown" << std::endl; }
};

TEST_F(scatter_min, scatter_min_infershape_static)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMin")->infer_shape;
    gert::Shape varShape = {16, 8};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {16, 8};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// 未知rank（-2）场景：输出shape透传为(-2,)
TEST_F(scatter_min, scatter_min_infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMin")->infer_shape;
    gert::Shape varShape = {-2};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

TEST_F(scatter_min, scatter_min_infershape_dynamic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMin")->infer_shape;
    gert::Shape varShape = {2, 2, -1};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {2, 2, -1};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// 与geir st用例scatter_min_geir_dyn_dim0_float32对齐：var第0维动态（-1）
TEST_F(scatter_min, scatter_min_infershape_dim0_unknown)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMin")->infer_shape;
    gert::Shape varShape = {-1, 32};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {-1, 32};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}

// 与geir st用例scatter_min_geir_dyn_middim_int32对齐：var中间维动态（-1），updates该维不受影响
TEST_F(scatter_min, scatter_min_infershape_middim_unknown)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMin")->infer_shape;
    gert::Shape varShape = {16, -1, 8};
    gert::Shape outputShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&varShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expectedOutputShape = {16, -1, 8};
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedOutputShape));
}
