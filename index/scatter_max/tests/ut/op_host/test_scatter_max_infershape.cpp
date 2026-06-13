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

class scatter_max : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "scatter_max SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "scatter_max TearDown" << std::endl; }
};

TEST_F(scatter_max, scatter_max_infershape_static)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMax")->infer_shape;
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

TEST_F(scatter_max, scatter_max_infershape_dynamic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMax")->infer_shape;
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

TEST_F(scatter_max, scatter_max_infer_datatype)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterMax");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInstanceNum({1, 1, 1})
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
