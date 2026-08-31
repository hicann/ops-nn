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

#include "../../../framework/npu_weight_quant_batchmatmul_v2_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "WeightQuantBatchMatmulV2"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    if (!attrs.empty()) {
        op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    }
    return op_src;
}
} // namespace

TEST(OnnxWeightQuantBatchMatmulV2PluginTest, ParseAllIntAttrs)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"antiquant_group_size","type":7,"i":128},{"name":"dtype","type":7,"i":4}]})");
    ge::Operator op_dst = CreateOperator("wqbmmv2");
    EXPECT_EQ(domi::ParseParamsWeightBatchQuantMatMulV2(op_src, op_dst), domi::SUCCESS);

    int group_size = 0;
    int dtype = 0;
    bool transpose_x = true;
    bool transpose_weight = true;
    EXPECT_EQ(op_dst.GetAttr("antiquant_group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_x", transpose_x), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("transpose_weight", transpose_weight), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 128);
    EXPECT_EQ(dtype, 4);
    EXPECT_FALSE(transpose_x);
    EXPECT_FALSE(transpose_weight);
}

TEST(OnnxWeightQuantBatchMatmulV2PluginTest, KeepsDefaultWithoutAttributes)
{
    ge::Operator op_src = CreateSourceOperator(R"({"attribute":{}})");
    ge::Operator op_dst = CreateOperator("wqbmmv2");
    EXPECT_EQ(domi::ParseParamsWeightBatchQuantMatMulV2(op_src, op_dst), domi::SUCCESS);

    int group_size = -1;
    int dtype = 0;
    EXPECT_EQ(op_dst.GetAttr("antiquant_group_size", group_size), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op_dst.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(group_size, 0);
    EXPECT_EQ(dtype, -1);
}
