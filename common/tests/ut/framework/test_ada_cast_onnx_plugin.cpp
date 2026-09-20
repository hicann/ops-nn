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

#include "../../../src/framework/ada_cast_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxAdaCastPluginTest, ParsePixelAttribute)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"pixel","type":2,"i":1024}]})");
    ge::Operator op_dst = CreateOperator("AdaCast");
    int pixel = 0;

    EXPECT_EQ(domi::ParseParamsAdaCast(op_src, op_dst), domi::SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("pixel", pixel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pixel, 1024);
}

TEST(OnnxAdaCastPluginTest, KeepsDefaultWithoutAttributes)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dst = CreateOperator("AdaCast");
    int pixel = 0;

    EXPECT_EQ(domi::ParseParamsAdaCast(op_src, op_dst), domi::SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("pixel", pixel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pixel, 65535);
}

TEST(OnnxAdaCastPluginTest, ParseZeroPixelWithoutIField)
{
    // GE 序列化 int 属性时省略值为 0 的 "i" 字段，pixel 应为 0 而非默认 65535
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"pixel","type":2}]})");
    ge::Operator op_dst = CreateOperator("AdaCast");
    int pixel = -1;

    EXPECT_EQ(domi::ParseParamsAdaCast(op_src, op_dst), domi::SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("pixel", pixel), ge::GRAPH_SUCCESS);
    EXPECT_EQ(pixel, 0);
}
