/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv2d_v2_op_select_format_pass.cpp
 * \brief Conv2DV2 OpSelectFormat UT：校验两个分支（FRACTAL_Z / FRACTAL_Z_C04）下发的算子信息库JSON。
 *        真实的gert::OpCheckContext无法在GE之外构造，因此两个分支都用Conv2dV2CheckHelper(bool)
 *        直接指定，覆盖除“从context判定c04条件”以外的全部逻辑；context为空的入口路径单独用nullptr覆盖。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>
#include <utility>
#include <vector>

// OpSelectFormat的入口与helper均为内部链接，UT直接include源文件以驱动被测代码（同
// index/masked_scatter_with_position/tests/ut/op_host/test_masked_scatter_with_position_infershape.cpp 的做法）
#include "../../../op_graph/conv2d_v2_op_graph.cpp"

using namespace ge;
using Json = nlohmann::json;

namespace {
// 算子信息库JSON的5个tensor：classify -> name
const std::vector<std::pair<std::string, std::string>> TENSOR_NAME_MAP = {
    {"input0", "x"}, {"input1", "filter"}, {"input2", "bias"}, {"input3", "offset_w"}, {"output0", "y"}};

// 每张表的组合数
constexpr size_t COMBINATION_NUM = 4;

// 两个分支的完整期望JSON
const char* const EXPECTED_JSON_FRACTAL_Z =
    "{\"input0\":{\"dtype\":\"float16,float16,float16,float16\",\"format\":\"NCHW,NHWC,NCHW,NHWC\",\"name\":\"x\""
    ",\"unknownshape_format\":\"NCHW,NHWC,NCHW,NHWC\"},\"input1\":{\"dtype\":\"float16,float16,float16,float16\""
    ",\"format\":\"FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z\",\"name\":\"filter\",\"sub_format\":\"0\""
    ",\"unknownshape_format\":\"FRACTAL_Z,FRACTAL_Z,FRACTAL_Z,FRACTAL_Z\"},\"input2\":{\"dtype\":\""
    "float16,float16,float16,float16\",\"format\":\"ND,ND,ND,ND\",\"name\":\"bias\",\"unknownshape_format\":\""
    "ND,ND,ND,ND\"},\"input3\":{\"dtype\":\"int8,int8,int8,int8\",\"format\":\"ND,ND,ND,ND\",\"name\":\"offset_w\""
    ",\"unknownshape_format\":\"ND,ND,ND,ND\"},\"output0\":{\"dtype\":\"float16,float16,float16,float16\""
    ",\"format\":\"NCHW,NHWC,NHWC,NCHW\",\"name\":\"y\",\"unknownshape_format\":\"NCHW,NHWC,NHWC,NCHW\""
    "}}";

const char* const EXPECTED_JSON_FRACTAL_Z_C04 =
    "{\"input0\":{\"dtype\":\"float16,float16,float16,float16\",\"format\":\"NCHW,NHWC,NCHW,NHWC\",\"name\":\"x\""
    ",\"unknownshape_format\":\"NCHW,NHWC,NCHW,NHWC\"},\"input1\":{\"dtype\":\"float16,float16,float16,float16\""
    ",\"format\":\"FRACTAL_Z_C04,FRACTAL_Z_C04,FRACTAL_Z_C04,FRACTAL_Z_C04\",\"name\":\"filter\""
    ",\"sub_format\":\"0\",\"unknownshape_format\":\"FRACTAL_Z_C04,FRACTAL_Z_C04,FRACTAL_Z_C04,FRACTAL_Z_C04\""
    "},\"input2\":{\"dtype\":\"float16,float16,float16,float16\",\"format\":\"ND,ND,ND,ND\",\"name\":\"bias\""
    ",\"unknownshape_format\":\"ND,ND,ND,ND\"},\"input3\":{\"dtype\":\"int8,int8,int8,int8\",\"format\":\""
    "ND,ND,ND,ND\",\"name\":\"offset_w\",\"unknownshape_format\":\"ND,ND,ND,ND\"},\"output0\":{\"dtype\":\""
    "float16,float16,float16,float16\",\"format\":\"NCHW,NHWC,NHWC,NCHW\",\"name\":\"y\",\"unknownshape_format\""
    ":\"NCHW,NHWC,NHWC,NCHW\""
    "}}";

std::vector<std::string> SplitComma(const std::string& str)
{
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= str.size()) {
        size_t pos = str.find(',', begin);
        if (pos == std::string::npos) {
            result.push_back(str.substr(begin));
            break;
        }
        result.push_back(str.substr(begin, pos - begin));
        begin = pos + 1;
    }
    return result;
}

// 取JSON中某tensor的某个字段
std::vector<std::string> GetField(const Json& json, const std::string& classify, const std::string& field)
{
    EXPECT_TRUE(json.contains(classify)) << classify;
    if (!json.contains(classify) || !json[classify].contains(field)) {
        return {};
    }
    return SplitComma(json[classify][field].get<std::string>());
}
} // namespace

class Conv2DV2OpSelectFormatTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    // 用真实helper拼出JSON：c04EnableFlag为true取FRACTAL_Z_C04组合，false取FRACTAL_Z组合。
    // 真实的gert::OpCheckContext无法在GE之外构造，故直接指定分支；
    // 入口的nullptr路径另有用例覆盖（GetC04EnableFlag(nullptr)返回false）。
    static std::string BuildJsonString(bool c04EnableFlag)
    {
        ops::Conv2dV2CheckHelper helper(c04EnableFlag);
        ge::AscendString result;
        EXPECT_EQ(helper.OpSelectFormat(result), ge::GRAPH_SUCCESS);
        const char* jsonStr = result.GetString();
        EXPECT_NE(jsonStr, nullptr);
        return jsonStr == nullptr ? std::string() : std::string(jsonStr);
    }

    static Json BuildJson(bool c04EnableFlag)
    {
        const std::string jsonStr = BuildJsonString(c04EnableFlag);
        return jsonStr.empty() ? Json() : Json::parse(jsonStr);
    }
};

// JSON骨架：5个tensor齐备、name正确、三个数组等长且组合数为4、unknownshape_format与format一致
TEST_F(Conv2DV2OpSelectFormatTest, json_skeleton)
{
    for (bool c04EnableFlag : {false, true}) {
        SCOPED_TRACE(c04EnableFlag ? "c04=on" : "c04=off");
        Json json = BuildJson(c04EnableFlag);
        ASSERT_EQ(json.size(), TENSOR_NAME_MAP.size());
        for (const auto& tensor : TENSOR_NAME_MAP) {
            EXPECT_TRUE(json.contains(tensor.first)) << tensor.first;
            if (!json.contains(tensor.first)) {
                continue;
            }
            EXPECT_EQ(json[tensor.first]["name"].get<std::string>(), tensor.second);
            EXPECT_EQ(GetField(json, tensor.first, "dtype").size(), COMBINATION_NUM);
            EXPECT_EQ(GetField(json, tensor.first, "format").size(), COMBINATION_NUM);
            EXPECT_EQ(GetField(json, tensor.first, "unknownshape_format").size(), COMBINATION_NUM);
            EXPECT_EQ(GetField(json, tensor.first, "format"), GetField(json, tensor.first, "unknownshape_format"));
        }
    }
}

// 两个分支只差filter格式：filter格式全为FRACTAL_Z / FRACTAL_Z_C04，其余tensor逐项一致
TEST_F(Conv2DV2OpSelectFormatTest, branch_differ_only_in_filter_format)
{
    Json fractalZ = BuildJson(false);
    Json fractalZC04 = BuildJson(true);

    EXPECT_EQ(GetField(fractalZ, "input1", "format"), std::vector<std::string>(COMBINATION_NUM, "FRACTAL_Z"));
    EXPECT_EQ(GetField(fractalZC04, "input1", "format"), std::vector<std::string>(COMBINATION_NUM, "FRACTAL_Z_C04"));

    for (const auto& tensor : TENSOR_NAME_MAP) {
        SCOPED_TRACE(tensor.first);
        EXPECT_EQ(GetField(fractalZ, tensor.first, "dtype"), GetField(fractalZC04, tensor.first, "dtype"));
        if (tensor.first == "input1") {
            continue; // filter格式按分支不同
        }
        EXPECT_EQ(GetField(fractalZ, tensor.first, "format"), GetField(fractalZC04, tensor.first, "format"));
    }
}

// dtype/format的对应关系：输入固定fp16、filter固定fp16、offset_w固定int8、bias为ND
TEST_F(Conv2DV2OpSelectFormatTest, tensor_dtype_and_format)
{
    Json json = BuildJson(false);
    EXPECT_EQ(GetField(json, "input0", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float16"));
    EXPECT_EQ(GetField(json, "input1", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float16"));
    EXPECT_EQ(GetField(json, "input2", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float16"));
    EXPECT_EQ(GetField(json, "input3", "dtype"), std::vector<std::string>(COMBINATION_NUM, "int8"));
    EXPECT_EQ(GetField(json, "output0", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float16"));
    EXPECT_EQ(GetField(json, "input2", "format"), std::vector<std::string>(COMBINATION_NUM, "ND"));
    EXPECT_EQ(GetField(json, "input3", "format"), std::vector<std::string>(COMBINATION_NUM, "ND"));
}

// 支持format混合进出：x与y的格式组合里同时存在 NCHW->NHWC 与 NHWC->NCHW 的交叉项
TEST_F(Conv2DV2OpSelectFormatTest, support_mixed_in_out_format)
{
    Json json = BuildJson(false);
    auto xFormat = GetField(json, "input0", "format");
    auto yFormat = GetField(json, "output0", "format");
    ASSERT_EQ(xFormat.size(), COMBINATION_NUM);
    ASSERT_EQ(yFormat.size(), COMBINATION_NUM);
    EXPECT_EQ(xFormat, (std::vector<std::string>{"NCHW", "NHWC", "NCHW", "NHWC"}));
    EXPECT_EQ(yFormat, (std::vector<std::string>{"NCHW", "NHWC", "NHWC", "NCHW"}));
    EXPECT_NE(xFormat[2], yFormat[2]); // NCHW -> NHWC
    EXPECT_NE(xFormat[3], yFormat[3]); // NHWC -> NCHW
}

// weight的sub_format固定为0：两个分支均只下发单个0（对全部format组合生效），其余tensor不下发sub_format
TEST_F(Conv2DV2OpSelectFormatTest, weight_sub_format_is_zero)
{
    for (bool c04EnableFlag : {false, true}) {
        SCOPED_TRACE(c04EnableFlag ? "c04=on" : "c04=off");
        Json json = BuildJson(c04EnableFlag);
        ASSERT_TRUE(json.contains("input1"));
        EXPECT_EQ(GetField(json, "input1", "sub_format"), std::vector<std::string>{"0"});
        for (const auto& tensor : TENSOR_NAME_MAP) {
            if (tensor.first != "input1") {
                EXPECT_FALSE(json[tensor.first].contains("sub_format")) << tensor.first;
            }
        }
    }
}

// 完整JSON逐字节一致（两个分支），同时锁定tensor顺序
TEST_F(Conv2DV2OpSelectFormatTest, exact_json_fractal_z)
{
    EXPECT_EQ(BuildJsonString(false), std::string(EXPECTED_JSON_FRACTAL_Z));
}

TEST_F(Conv2DV2OpSelectFormatTest, exact_json_fractal_z_c04)
{
    EXPECT_EQ(BuildJsonString(true), std::string(EXPECTED_JSON_FRACTAL_Z_C04));
}

// 入口在context为nullptr（拿不到c04条件）时按FRACTAL_Z下发
TEST_F(Conv2DV2OpSelectFormatTest, null_context_uses_fractal_z)
{
    ge::AscendString result;
    ASSERT_EQ(ops::Conv2dV2OpSelectFormat(nullptr, result), ge::GRAPH_SUCCESS);
    ASSERT_NE(result.GetString(), nullptr);
    EXPECT_EQ(std::string(result.GetString()), std::string(EXPECTED_JSON_FRACTAL_Z));
}
