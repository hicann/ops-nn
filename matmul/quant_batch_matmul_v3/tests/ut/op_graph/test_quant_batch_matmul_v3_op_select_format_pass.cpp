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
 * \file test_quant_batch_matmul_v3_op_select_format_pass.cpp
 * \brief OpSelectFormat dtype and format contract tests.
 */

#include <array>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "register/op_impl_registry.h"

namespace optiling {
ge::graphStatus QbmmV3OpSelectFormat(const gert::OpCheckContext* context, ge::AscendString& result);
}

namespace {
std::vector<std::string> SplitFormats(const std::string& values)
{
    std::vector<std::string> result;
    std::istringstream input(values);
    for (std::string value; std::getline(input, value, ',');) {
        result.push_back(value);
    }
    return result;
}

TEST(QuantBatchMatmulV3OpSelectFormat, EmitsCompleteAlignedTensorColumns)
{
    ge::AscendString result;
    ASSERT_EQ(optiling::QbmmV3OpSelectFormat(nullptr, result), ge::GRAPH_SUCCESS);
    const auto json = nlohmann::json::parse(result.GetString());
    const std::array<std::string, 7> keys = {"input0", "input1", "input2", "input3", "input4", "input5", "output0"};
    const std::array<std::string, 7> names = {"x1", "x2", "scale", "offset", "bias", "pertoken_scale", "y"};
    ASSERT_EQ(json.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        SCOPED_TRACE(keys[i]);
        ASSERT_TRUE(json.contains(keys[i]));
        const auto& tensor = json.at(keys[i]);
        EXPECT_EQ(tensor.at("name"), names[i]);
        EXPECT_EQ(SplitFormats(tensor.at("dtype").get<std::string>()).size(), 16);
        EXPECT_EQ(SplitFormats(tensor.at("format").get<std::string>()).size(), 16);
        EXPECT_EQ(tensor.at("format"), tensor.at("unknownshape_format"));
        if (i != 1) {
            for (const auto& format : SplitFormats(tensor.at("format").get<std::string>())) {
                EXPECT_EQ(format, "ND");
            }
        }
    }
}

TEST(QuantBatchMatmulV3OpSelectFormat, PreservesAllWeightScaleAndOutputCombinations)
{
    ge::AscendString result;
    ASSERT_EQ(optiling::QbmmV3OpSelectFormat(nullptr, result), ge::GRAPH_SUCCESS);
    const auto json = nlohmann::json::parse(result.GetString());
    const auto x1 = SplitFormats(json.at("input0").at("dtype"));
    const auto x2 = SplitFormats(json.at("input1").at("dtype"));
    const auto weightFormat = SplitFormats(json.at("input1").at("format"));
    const auto scale = SplitFormats(json.at("input2").at("dtype"));
    const auto offset = SplitFormats(json.at("input3").at("dtype"));
    const auto bias = SplitFormats(json.at("input4").at("dtype"));
    const auto pertoken = SplitFormats(json.at("input5").at("dtype"));
    const auto y = SplitFormats(json.at("output0").at("dtype"));
    ASSERT_EQ(x2.size(), 16);
    for (const auto* column : {&x1, &weightFormat, &scale, &offset, &bias, &pertoken, &y}) {
        ASSERT_EQ(column->size(), x2.size());
    }
    std::set<std::array<std::string, 4>> combinations;
    for (size_t i = 0; i < x2.size(); ++i) {
        EXPECT_EQ(x1[i], "int8");
        EXPECT_EQ(offset[i], "float32");
        EXPECT_EQ(bias[i], "int32");
        EXPECT_EQ(pertoken[i], "float32");
        combinations.insert({x2[i], weightFormat[i], scale[i], y[i]});
    }
    std::set<std::array<std::string, 4>> expected;
    for (const auto* weight : {"int4", "int8"}) {
        for (const auto* format : {"ND", "FRACTAL_NZ"}) {
            for (const auto* scaleType : {"int64", "uint64"}) {
                for (const auto* output : {"float16", "int8"}) {
                    expected.insert({weight, format, scaleType, output});
                }
            }
        }
    }
    EXPECT_EQ(combinations, expected);
}
} // namespace
