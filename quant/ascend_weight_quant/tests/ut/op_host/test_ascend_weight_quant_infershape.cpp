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

#include "../../../op_graph/ascend_weight_quant_proto.h"
#include "../../../op_host/ascend_weight_quant_def.cpp"

namespace {
// 断言算子原型（ge::op::AscendWeightQuant）注册内容：2 输入 1 输出，
// dst_type 默认 DT_INT8、支持 DT_INT4
TEST(AscendWeightQuantProtoTest, RegistersInputsOutputsAndDefaultAttr)
{
    ge::op::AscendWeightQuant op;

    EXPECT_EQ(op.GetInputsSize(), 2U);
    EXPECT_EQ(op.GetOutputsSize(), 1U);
    EXPECT_EQ(op.get_attr_dst_type(), ge::DT_INT8);

    op.set_attr_dst_type(ge::DT_INT4);
    EXPECT_EQ(op.get_attr_dst_type(), ge::DT_INT4);

    ge::TensorDesc inputDesc(ge::Shape({4, 1, 1, 1}), ge::FORMAT_FRACTAL_NZ, ge::DT_INT8);
    EXPECT_EQ(op.update_input_desc_x(inputDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_x().GetDataType(), ge::DT_INT8);
}

// 断言 op_def 注册内容：x/offset 为 int8 HWCN 必选输入，y 为 int8 HWCN 输出，
// dst_type 可选默认 2/DT_INT8
class AscendWeightQuantDefTest : public testing::Test {
protected:
    ops::AscendWeightQuant def{"AscendWeightQuant"};
    ops::OpAICoreConfig& GetCfg(const char* soc)
    {
        auto& configs = def.AICore().GetAICoreConfigs();
        auto iter = configs.find(ge::AscendString(soc));
        EXPECT_NE(iter, configs.end());
        return iter->second;
    }
};

TEST_F(AscendWeightQuantDefTest, RegistersSocConfigs)
{
    auto& configs = def.AICore().GetAICoreConfigs();
    EXPECT_EQ(configs.size(), 3U);
    EXPECT_NE(configs.find(ge::AscendString("ascend950")), configs.end());
    EXPECT_NE(configs.find(ge::AscendString("ascend960dt")), configs.end());
    EXPECT_NE(configs.find(ge::AscendString("ascend960pr")), configs.end());
}

TEST_F(AscendWeightQuantDefTest, RegistersRequiredInputs)
{
    auto& cfg = GetCfg("ascend950");
    auto& inputs = cfg.GetInputs();
    ASSERT_EQ(inputs.size(), 2U);

    EXPECT_EQ(std::string(inputs[0].GetParamName().GetString()), "x");
    EXPECT_EQ(std::string(inputs[1].GetParamName().GetString()), "offset");
    for (auto& input : inputs) {
        EXPECT_EQ(input.GetParamType(), ops::REQUIRED);
        auto& dtypes = input.GetDataTypes();
        ASSERT_EQ(dtypes.size(), 1U);
        EXPECT_EQ(dtypes[0], ge::DT_INT8);
        auto& formats = input.GetFormats();
        ASSERT_EQ(formats.size(), 1U);
        EXPECT_EQ(formats[0], ge::FORMAT_HWCN);
    }
}

TEST_F(AscendWeightQuantDefTest, RegistersOutputWithDtypeCombinations)
{
    auto& cfg = GetCfg("ascend950");
    auto& outputs = cfg.GetOutputs();
    ASSERT_EQ(outputs.size(), 1U);

    EXPECT_EQ(std::string(outputs[0].GetParamName().GetString()), "y");
    EXPECT_EQ(outputs[0].GetParamType(), ops::REQUIRED);
    auto& dtypes = outputs[0].GetDataTypes();
    ASSERT_EQ(dtypes.size(), 1U);
    EXPECT_EQ(dtypes[0], ge::DT_INT8);
    auto& formats = outputs[0].GetFormats();
    ASSERT_EQ(formats.size(), 1U);
    EXPECT_EQ(formats[0], ge::FORMAT_HWCN);
}

TEST_F(AscendWeightQuantDefTest, RegistersDstTypeAttrWithDefault)
{
    auto& attrs = def.GetAttrs();
    ASSERT_EQ(attrs.size(), 1U);

    EXPECT_EQ(std::string(attrs[0].GetName().GetString()), "dst_type");
    EXPECT_EQ(std::string(attrs[0].GetAttrDefaultVal("").GetString()), "2");
}
} // namespace
