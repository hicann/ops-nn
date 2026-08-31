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

#include "../../../framework/matmul_onnx_plugin.cpp"

namespace {
// ParseParamsMatMul 内部会调用 ChangeOutputFormatToNchw（对真实 onnx 图上的 op 设置 output format）。
// 单测构造的裸 ge::Operator 无 I/O，format 设置不可用，故这里只验证解析出的属性，
// 不依赖返回状态。
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "BatchMatMulV2"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = ge::Operator("src", "BatchMatMulV2");
    if (!attrs.empty()) {
        op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    }
    return op_src;
}
} // namespace

TEST(OnnxMatMulV3PluginTest, ParseAllIntAttrs)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"transA","type":7,"i":1},{"name":"transB","type":7,"i":1},)"
        R"({"name":"fixed_shift_value","type":7,"i":40},{"name":"enable_uncache","type":7,"i":1}]})");
    ge::Operator op_dst = CreateOperator("matmul");
    domi::ParseParamsMatMul(op_src, op_dst);

    bool trans_a = false;
    bool trans_b = false;
    int64_t fixed_shift_value = 0;
    int64_t enable_uncache = -1;
    EXPECT_EQ(op_dst.GetAttr("adj_x1", trans_a), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("adj_x2", trans_b), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("fixed_shift_value", fixed_shift_value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("enable_uncache", enable_uncache), ge::GRAPH_SUCCESS);
    EXPECT_TRUE(trans_a);
    EXPECT_TRUE(trans_b);
    EXPECT_EQ(fixed_shift_value, 40);
    EXPECT_EQ(enable_uncache, 1);
}

TEST(OnnxMatMulV3PluginTest, FallsBackOutOfRangeFixedShiftToDefault)
{
    // fixed_shift_value 超出 [34, 43] 时回退默认 42
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":[{"name":"fixed_shift_value","type":7,"i":100}]})");
    ge::Operator op_dst = CreateOperator("matmul");
    domi::ParseParamsMatMul(op_src, op_dst);

    int64_t fixed_shift_value = 0;
    EXPECT_EQ(op_dst.GetAttr("fixed_shift_value", fixed_shift_value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(fixed_shift_value, 42);
}

TEST(OnnxMatMulV3PluginTest, KeepsDefaultWithoutAttributeArray)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":{}})");
    ge::Operator op_dst = CreateOperator("matmul");
    domi::ParseParamsMatMul(op_src, op_dst);

    bool trans_a = true;
    bool trans_b = true;
    int64_t fixed_shift_value = 0;
    int64_t enable_uncache = -1;
    EXPECT_EQ(op_dst.GetAttr("adj_x1", trans_a), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("adj_x2", trans_b), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("fixed_shift_value", fixed_shift_value), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("enable_uncache", enable_uncache), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(trans_a);
    EXPECT_FALSE(trans_b);
    EXPECT_EQ(fixed_shift_value, 42);
    EXPECT_EQ(enable_uncache, 0);
}
