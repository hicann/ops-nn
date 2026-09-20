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

#include "../../../src/framework/group_normal_relu_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxGroupNormReluPluginTest, ParseEpsAndNumGroups)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"eps","type":1,"f":"0.001"},{"name":"num_groups","type":2,"i":8}]})");
    ge::Operator op_dest = CreateOperator("group_normal_relu");
    float eps = -1.0f;
    int num_groups = -1;

    EXPECT_EQ(domi::ParseOnnxParamsGroupNormRelu(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("eps", eps), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(eps, 0.001f);
    EXPECT_EQ(op_dest.GetAttr("num_groups", num_groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_groups, 8);
}

TEST(OnnxGroupNormReluPluginTest, KeepsDefaultWhenAttributeMissing)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("group_normal_relu");
    float eps = -1.0f;
    int num_groups = -1;

    EXPECT_EQ(domi::ParseOnnxParamsGroupNormRelu(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("eps", eps), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(eps, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("num_groups", num_groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_groups, 0);
}

TEST(OnnxGroupNormReluPluginTest, ParseZeroValuesWithoutValueFields)
{
    // GE 序列化 float/int 属性时会省略值为 0 的 "f"/"i" 字段，此时应解析为 0 而非回退默认值
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"eps","type":1},{"name":"num_groups","type":2}]})");
    ge::Operator op_dest = CreateOperator("group_normal_relu");
    float eps = -1.0f;
    int num_groups = -1;

    EXPECT_EQ(domi::ParseOnnxParamsGroupNormRelu(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("eps", eps), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(eps, 0.0f);
    EXPECT_EQ(op_dest.GetAttr("num_groups", num_groups), ge::GRAPH_SUCCESS);
    EXPECT_EQ(num_groups, 0);
}
