/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

using namespace ge;

class SleepInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(SleepInferShapeTest, InferShape_success)
{
    std::string opType("Sleep");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    auto holder = gert::InferShapeContextFaker().NodeIoNum(0, 0).Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(SleepInferShapeTest, InferDataType_success)
{
    std::string opType("Sleep");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    auto holder = gert::InferDataTypeContextFaker().NodeIoNum(0, 0).Build();

    EXPECT_EQ(inferDataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
}
