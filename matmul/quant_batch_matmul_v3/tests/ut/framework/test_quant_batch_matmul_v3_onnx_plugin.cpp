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

#include "../../../framework/quant_batch_matmulV3_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "QuantBatchMatmulV3"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    if (!attrs.empty()) {
        op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    }
    return op_src;
}
} // namespace

TEST(OnnxQuantBatchMatmulV3PluginTest, ParseAllIntAttrs)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"dtype","type":7,"i":3},{"name":"transpose_x1","type":7,"i":1},)"
        R"({"name":"transpose_x2","type":7,"i":1}]})");
    ge::Operator op_dst = CreateOperator("qbmmv3");
    EXPECT_EQ(domi::ParseParamsQuantBatchMatMulV3(op_src, op_dst), domi::SUCCESS);

    int dtype = 0;
    bool trans_x1 = false;
    bool trans_x2 = false;
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x1", trans_x1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x2", trans_x2), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dtype, 3);
    EXPECT_TRUE(trans_x1);
    EXPECT_TRUE(trans_x2);
}

TEST(OnnxQuantBatchMatmulV3PluginTest, KeepsDefaultWithoutAttributes)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":{}})");
    ge::Operator op_dst = CreateOperator("qbmmv3");
    EXPECT_EQ(domi::ParseParamsQuantBatchMatMulV3(op_src, op_dst), domi::SUCCESS);

    int dtype = 0;
    bool trans_x1 = true;
    bool trans_x2 = true;
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x1", trans_x1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x2", trans_x2), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dtype, 1);
    EXPECT_FALSE(trans_x1);
    EXPECT_FALSE(trans_x2);
}
