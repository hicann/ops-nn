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

#include "../../../framework/batch_matmul_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "BatchMatMul"); }
} // namespace

TEST(OnnxBatchMatMulPluginTest, SetDefaultTransposeAttrsWithoutAnyAttribute)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dst = CreateOperator("batchmatmul");

    EXPECT_EQ(domi::ParseParamsBatchMatMul(op_src, op_dst), domi::SUCCESS);

    bool adj_x1 = true;
    bool adj_x2 = true;
    EXPECT_EQ(op_dst.GetAttr("adj_x1", adj_x1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("adj_x2", adj_x2), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(adj_x1);
    EXPECT_FALSE(adj_x2);
}

TEST(OnnxBatchMatMulPluginTest, SetDefaultTransposeAttrsEvenWithAttributes)
{
    // onnx BatchMatMul 无 transpose 属性，送入属性后仍应保持 adj_x1/adj_x2 默认 false
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(R"({"attribute":[{"name":"transA","type":7,"i":1}]})"));
    ge::Operator op_dst = CreateOperator("batchmatmul");

    EXPECT_EQ(domi::ParseParamsBatchMatMul(op_src, op_dst), domi::SUCCESS);

    bool adj_x1 = true;
    bool adj_x2 = true;
    EXPECT_EQ(op_dst.GetAttr("adj_x1", adj_x1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("adj_x2", adj_x2), ge::GRAPH_SUCCESS);
    EXPECT_FALSE(adj_x1);
    EXPECT_FALSE(adj_x2);
}
