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

#include "../../../framework/leaky_relu_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxLeakyReluPluginTest, ParseFloatAttributeFromString)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"alpha","type":1,"f":"0.125"}]})");
    ge::Operator op_dest = CreateOperator("leaky_relu");
    float negative_slope = 0.0f;

    EXPECT_EQ(domi::ParseParamsLeakyRelu(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("negative_slope", negative_slope), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(negative_slope, 0.125f);
}

TEST(OnnxLeakyReluPluginTest, KeepsDefaultWhenAttributeMissing)
{
    ge::Operator op_src = CreateSourceOperator(R"({})");
    ge::Operator op_dest = CreateOperator("leaky_relu");
    float negative_slope = 0.0f;

    EXPECT_EQ(domi::ParseParamsLeakyRelu(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("negative_slope", negative_slope), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(negative_slope, 0.01f);
}
