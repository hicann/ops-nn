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

#include "../../../src/framework/bounding_box_decode_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }

ge::Operator CreateSourceOperator(const std::string& attrs)
{
    ge::Operator op_src = CreateOperator("src");
    op_src.SetAttr("attribute", ge::AscendString(attrs.c_str()));
    return op_src;
}
} // namespace

TEST(OnnxBoundingBoxDecodePluginTest, ParseFullAttributes)
{
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"max_shape","type":8,"ints":[640,640]},{"name":"means","type":7,"floats":[123.675,116.28,103.53]},{"name":"stds","type":7,"floats":[58.395,57.12,57.375]},{"name":"wh_ratio_clip","type":1,"f":"0.016"}]})");
    ge::Operator op_dest = CreateOperator("bounding_box_decode");
    std::vector<int64_t> max_shape;
    std::vector<float> means;
    std::vector<float> stds;
    float wh_ratio_clip = -1.0f;

    EXPECT_EQ(domi::ParseParamsBoundingBoxDecode(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("max_shape", max_shape), ge::GRAPH_SUCCESS);
    ASSERT_EQ(max_shape.size(), 2U);
    EXPECT_EQ(max_shape[0], 640);
    EXPECT_EQ(op_dest.GetAttr("means", means), ge::GRAPH_SUCCESS);
    ASSERT_EQ(means.size(), 3U);
    EXPECT_FLOAT_EQ(means[0], 123.675f);
    EXPECT_EQ(op_dest.GetAttr("stds", stds), ge::GRAPH_SUCCESS);
    ASSERT_EQ(stds.size(), 3U);
    EXPECT_FLOAT_EQ(stds[0], 58.395f);
    EXPECT_EQ(op_dest.GetAttr("wh_ratio_clip", wh_ratio_clip), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(wh_ratio_clip, 0.016f);
}

TEST(OnnxBoundingBoxDecodePluginTest, ParseZeroWhRatioClipWithoutFField)
{
    // GE 序列化 float 属性时会省略值为 0 的 "f" 字段，wh_ratio_clip 应解析为 0 而非默认 0.016
    ge::Operator op_src = CreateSourceOperator(
        R"({"attribute":[{"name":"max_shape","type":8,"ints":[640,640]},{"name":"means","type":7,"floats":[0.0,0.0]},{"name":"stds","type":7,"floats":[1.0,1.0]},{"name":"wh_ratio_clip","type":1}]})");
    ge::Operator op_dest = CreateOperator("bounding_box_decode");
    float wh_ratio_clip = -1.0f;

    EXPECT_EQ(domi::ParseParamsBoundingBoxDecode(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("wh_ratio_clip", wh_ratio_clip), ge::GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(wh_ratio_clip, 0.0f);
}
