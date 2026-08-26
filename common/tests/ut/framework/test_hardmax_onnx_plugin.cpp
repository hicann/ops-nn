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

#include "../../../src/framework/hardmax_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxHardMaxPluginTest, ParseAxisAttribute)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"axis","type":2,"i":2}]})");
    ge::Operator op_dest = CreateOperator("hardmax");
    int64_t axis = 0;

    EXPECT_EQ(domi::parse_params_hard_max(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("axis", axis), ge::GRAPH_SUCCESS);
    EXPECT_EQ(axis, 2);
}

TEST(OnnxHardMaxPluginTest, KeepsDefaultWithoutAttributes)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("hardmax");
    int64_t axis = 0;

    EXPECT_EQ(domi::parse_params_hard_max(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("axis", axis), ge::GRAPH_SUCCESS);
    EXPECT_EQ(axis, -1);
}
