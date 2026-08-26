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

#include "../../../src/framework/size_onnx_plugin.cpp"

namespace {
ge::Operator CreateOperator(const std::string& name) { return ge::Operator(name, "TestOp"); }
} // namespace

TEST(OnnxSizePluginTest, SetsInt64Dtype)
{
    ge::Operator op_src = CreateOperator("src");
    ge::Operator op_dest = CreateOperator("size");
    // SetAttr 走 ge::DataType 精确重载存储，读取时用同类型
    ge::DataType dtype = ge::DT_UNDEFINED;

    EXPECT_EQ(domi::ParseParamsSize(op_src, op_dest), domi::SUCCESS);
    EXPECT_EQ(op_dest.GetAttr("dtype", dtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(dtype, ge::DT_INT64);
}
