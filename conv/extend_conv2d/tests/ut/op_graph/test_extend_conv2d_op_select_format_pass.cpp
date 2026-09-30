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
 * \file test_extend_conv2d_op_select_format_pass.cpp
 * \brief ExtendConv2D OpSelectFormat UT：校验按filter格式过滤出的两个分支
 *        （FRACTAL_Z 96种 / FRACTAL_Z_C04 96种）下发的算子信息库JSON。
 *        真实的gert::OpCheckContext无法在GE之外构造，因此两个分支都直接调
 *        ExtendConv2dOpSelectFormatByFlag(bool)（入口按context判定c04后调用的就是它），
 *        覆盖除“从context判定c04条件”以外的全部逻辑；context为空的入口路径单独用nullptr覆盖。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <set>
#include <string>
#include <utility>
#include <vector>

// OpSelectFormat的入口与helper均为内部链接，UT直接include源文件以驱动被测代码（同
// index/masked_scatter_with_position/tests/ut/op_host/test_masked_scatter_with_position_infershape.cpp 的做法）
#include "../../../op_graph/extend_conv2d_op_graph.cpp"

using namespace ge;
using Json = nlohmann::json;

namespace {
// 算子信息库JSON的12个tensor：classify -> name（顺序即IR输入输出顺序）
const std::vector<std::pair<std::string, std::string>> TENSOR_NAME_MAP = {
    {"input0", "x"},           {"input1", "filter"}, {"input2", "bias"},
    {"input3", "offset_w"},    {"input4", "scale0"}, {"input5", "relu_weight0"},
    {"input6", "clip_value0"}, {"input7", "scale1"}, {"input8", "relu_weight1"},
    {"input9", "clip_value1"}, {"output0", "y0"},    {"output1", "y1"}};

// 每个分支的组合数（全量192按filter格式二分）
constexpr size_t COMBINATION_NUM = 96;

// 两个分支共同覆盖的dtype组合数：3种输入对 × 8种(y0,y1,scale)组合
constexpr size_t DISTINCT_DTYPE_COMBINATION_NUM = 24;

using DtypeTuple = std::array<std::string, 6>;

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

std::vector<std::string> GetField(const Json& json, const std::string& classify, const std::string& field)
{
    EXPECT_TRUE(json.contains(classify)) << classify;
    if (!json.contains(classify) || !json[classify].contains(field)) {
        return {};
    }
    return SplitComma(json[classify][field].get<std::string>());
}

// 收集JSON里所有组合的(x, filter, bias, scale, y0, y1) dtype元组
std::set<DtypeTuple> CollectDtypeTuples(const Json& json)
{
    auto x = GetField(json, "input0", "dtype");
    auto filter = GetField(json, "input1", "dtype");
    auto bias = GetField(json, "input2", "dtype");
    auto scale = GetField(json, "input4", "dtype");
    auto y0 = GetField(json, "output0", "dtype");
    auto y1 = GetField(json, "output1", "dtype");
    std::set<DtypeTuple> tuples;
    if (x.size() != COMBINATION_NUM || filter.size() != x.size() || bias.size() != x.size() ||
        scale.size() != x.size() || y0.size() != x.size() || y1.size() != x.size()) {
        return tuples;
    }
    for (size_t i = 0; i < x.size(); ++i) {
        tuples.insert({x[i], filter[i], bias[i], scale[i], y0[i], y1[i]});
    }
    return tuples;
}
} // namespace

class ExtendConv2DOpSelectFormatTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    // FRACTAL_Z分支：直接调入口（context=nullptr时c04判定为false），走真实生产路径
    static std::string BuildJsonStringFractalZ()
    {
        ge::AscendString result;
        EXPECT_EQ(ops::ExtendConv2dOpSelectFormat(nullptr, result), ge::GRAPH_SUCCESS);
        const char* jsonStr = result.GetString();
        EXPECT_NE(jsonStr, nullptr);
        return jsonStr == nullptr ? std::string() : std::string(jsonStr);
    }

    // FRACTAL_Z_C04分支：直接调按flag入参的实现（与入口共用同一段拼装逻辑）
    static std::string BuildJsonStringFractalZC04()
    {
        ge::AscendString result;
        EXPECT_EQ(ops::ExtendConv2dOpSelectFormatByFlag(true, result), ge::GRAPH_SUCCESS);
        const char* jsonStr = result.GetString();
        EXPECT_NE(jsonStr, nullptr);
        return jsonStr == nullptr ? std::string() : std::string(jsonStr);
    }

    static std::string BuildJsonString(bool c04EnableFlag)
    {
        return c04EnableFlag ? BuildJsonStringFractalZC04() : BuildJsonStringFractalZ();
    }

    static Json BuildJson(bool c04EnableFlag)
    {
        const std::string jsonStr = BuildJsonString(c04EnableFlag);
        return jsonStr.empty() ? Json() : Json::parse(jsonStr);
    }
};

// JSON骨架：12个tensor齐备且顺序与IR一致、name正确、三个数组等长、组合数96、unknownshape_format与format一致
TEST_F(ExtendConv2DOpSelectFormatTest, json_skeleton)
{
    for (bool c04EnableFlag : {false, true}) {
        SCOPED_TRACE(c04EnableFlag ? "c04=on" : "c04=off");
        Json json = BuildJson(c04EnableFlag);
        ASSERT_EQ(json.size(), TENSOR_NAME_MAP.size());

        const std::string jsonStr = BuildJsonString(c04EnableFlag);
        size_t lastPos = 0;
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

            // 顺序与IR一致
            size_t pos = jsonStr.find("\"" + tensor.first + "\"");
            EXPECT_NE(pos, std::string::npos) << tensor.first;
            EXPECT_GE(pos, lastPos) << tensor.first;
            lastPos = pos;
        }
    }
}

// filter格式按分支过滤：off全为FRACTAL_Z，on全为FRACTAL_Z_C04；
// x/y的格式序列两分支不同（全量列表里两个分支交错排列，过滤后组内顺序不同），
// 因此这里只校验ND类tensor格式恒定，以及(x,y)的(dtype,format)三元组集合两分支一致。
TEST_F(ExtendConv2DOpSelectFormatTest, filter_format_follows_c04_flag)
{
    Json fractalZ = BuildJson(false);
    Json fractalZC04 = BuildJson(true);

    EXPECT_EQ(GetField(fractalZ, "input1", "format"), std::vector<std::string>(COMBINATION_NUM, "FRACTAL_Z"));
    EXPECT_EQ(GetField(fractalZC04, "input1", "format"), std::vector<std::string>(COMBINATION_NUM, "FRACTAL_Z_C04"));

    // y0/y1为NCHW/NHWC（支持format混合进出），bias/offset_w/scale/relu_weight/clip_value为ND
    for (const char* ndTensor : {"input2", "input3", "input4", "input5", "input6", "input7", "input8", "input9"}) {
        EXPECT_EQ(GetField(fractalZ, ndTensor, "format"), std::vector<std::string>(COMBINATION_NUM, "ND")) << ndTensor;
        EXPECT_EQ(GetField(fractalZC04, ndTensor, "format"), std::vector<std::string>(COMBINATION_NUM, "ND"))
            << ndTensor;
    }

    // (x_dtype, x_format, y0_dtype, y0_format) 三元组集合两分支一致，仅组内顺序不同
    auto collect = [](const Json& json) {
        auto xDtype = GetField(json, "input0", "dtype");
        auto xFormat = GetField(json, "input0", "format");
        auto yDtype = GetField(json, "output0", "dtype");
        auto yFormat = GetField(json, "output0", "format");
        std::set<std::array<std::string, 4>> combos;
        for (size_t i = 0; i < xDtype.size(); ++i) {
            combos.insert({xDtype[i], xFormat[i], yDtype[i], yFormat[i]});
        }
        return combos;
    };
    auto fractalZCombos = collect(fractalZ);
    ASSERT_EQ(fractalZCombos.size(), 16U); // (x_dt,y_dt) 4种 × (x_fmt->y_fmt) 4种
    EXPECT_EQ(fractalZCombos, collect(fractalZC04));
}

// weight的sub_format固定为0：两个分支均只下发单个0（对全部format组合生效），其余tensor不下发sub_format
TEST_F(ExtendConv2DOpSelectFormatTest, weight_sub_format_is_zero)
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

// dtype约束：offset_w固定int8、relu_weight固定float32、bias为float16/int32、scale为int64/uint64
TEST_F(ExtendConv2DOpSelectFormatTest, tensor_dtype_invariants)
{
    Json json = BuildJson(false);
    EXPECT_EQ(GetField(json, "input3", "dtype"), std::vector<std::string>(COMBINATION_NUM, "int8"));
    EXPECT_EQ(GetField(json, "input5", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float32"));
    EXPECT_EQ(GetField(json, "input8", "dtype"), std::vector<std::string>(COMBINATION_NUM, "float32"));

    // 取值域：输入/输出为float16或int8，bias为float16或int32，scale为int64或uint64
    const std::vector<std::pair<std::string, std::set<std::string>>> expectDomain = {
        {"input0", {"float16", "int8"}},  {"input1", {"float16", "int8"}},  {"output0", {"float16", "int8"}},
        {"output1", {"float16", "int8"}}, {"input2", {"float16", "int32"}}, {"input4", {"int64", "uint64"}}};
    for (const auto& item : expectDomain) {
        std::set<std::string> dtypes;
        for (const auto& value : GetField(json, item.first, "dtype")) {
            dtypes.insert(value);
        }
        EXPECT_EQ(dtypes, item.second) << item.first;
    }
}

// 按filter格式过滤不丢dtype组合：两个分支覆盖的(x, filter, bias, scale, y0, y1)元组集合相同且为24种
TEST_F(ExtendConv2DOpSelectFormatTest, branches_cover_same_dtype_combinations)
{
    auto fractalZTuples = CollectDtypeTuples(BuildJson(false));
    auto fractalZC04Tuples = CollectDtypeTuples(BuildJson(true));
    ASSERT_EQ(fractalZTuples.size(), DISTINCT_DTYPE_COMBINATION_NUM);
    EXPECT_EQ(fractalZTuples, fractalZC04Tuples);
}

// 组合顺序：锁定两个分支的首尾组合取值
TEST_F(ExtendConv2DOpSelectFormatTest, combination_order)
{
    Json fractalZ = BuildJson(false);
    Json fractalZC04 = BuildJson(true);

    // idx0：fp16*fp16 输入，y0/y1=fp16，scale=int64，x/y为NCHW
    EXPECT_EQ(GetField(fractalZ, "input0", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZ, "input1", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZ, "input2", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZ, "input4", "dtype")[0], "int64");
    EXPECT_EQ(GetField(fractalZ, "output0", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZ, "output1", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZ, "input0", "format")[0], "NCHW");
    EXPECT_EQ(GetField(fractalZ, "output0", "format")[0], "NCHW");
    EXPECT_EQ(GetField(fractalZC04, "input0", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZC04, "input1", "dtype")[0], "float16");
    EXPECT_EQ(GetField(fractalZC04, "input4", "dtype")[0], "int64");
    EXPECT_EQ(GetField(fractalZC04, "input0", "format")[0], "NCHW");

    // idx95（末位）：fp16*int8 输入，scale=uint64，x/y格式不同（NHWC -> NCHW）
    const size_t last = COMBINATION_NUM - 1;
    EXPECT_EQ(GetField(fractalZ, "input0", "dtype")[last], "float16");
    EXPECT_EQ(GetField(fractalZ, "input1", "dtype")[last], "int8");
    EXPECT_EQ(GetField(fractalZ, "input2", "dtype")[last], "int32");
    EXPECT_EQ(GetField(fractalZ, "input4", "dtype")[last], "uint64");
    EXPECT_EQ(GetField(fractalZ, "input0", "format")[last], "NHWC");
    EXPECT_EQ(GetField(fractalZ, "output0", "format")[last], "NCHW");
    EXPECT_EQ(GetField(fractalZC04, "input0", "format")[last], "NHWC");
    EXPECT_EQ(GetField(fractalZC04, "output0", "format")[last], "NHWC");
}

// 入口在context为nullptr（拿不到c04条件）时下发FRACTAL_Z分支
TEST_F(ExtendConv2DOpSelectFormatTest, null_context_uses_fractal_z)
{
    ge::AscendString result;
    ASSERT_EQ(ops::ExtendConv2dOpSelectFormat(nullptr, result), ge::GRAPH_SUCCESS);
    ASSERT_NE(result.GetString(), nullptr);
    EXPECT_EQ(std::string(result.GetString()), BuildJsonString(false));
}
