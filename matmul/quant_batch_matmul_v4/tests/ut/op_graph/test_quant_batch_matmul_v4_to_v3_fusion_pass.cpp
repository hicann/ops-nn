/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"

namespace {
constexpr char QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME[] = "QuantBatchMatmulV4ToV3FusionPass";
constexpr char TEST_CSV_FILE[] = "test_quant_batch_matmul_v4_to_v3_fusion_pass.csv";
constexpr size_t CASE_FIELD_NUM = 10;
constexpr size_t V4_INPUT_NUM = 10;
constexpr size_t V4_X1_INDEX = 0;
constexpr size_t V4_X2_INDEX = 1;
constexpr size_t V4_BIAS_INDEX = 2;
constexpr size_t V4_X1SCALE_INDEX = 3;
constexpr size_t V4_X2SCALE_INDEX = 4;
constexpr size_t V4_YSCALE_INDEX = 5;
constexpr size_t V4_X1OFFSET_INDEX = 6;
constexpr size_t V4_X2OFFSET_INDEX = 7;
constexpr size_t V4_YOFFSET_INDEX = 8;
constexpr size_t V4_X2TABLE_INDEX = 9;
constexpr int64_t CAPTURE_TENSOR_INDEX = 0;

struct QuantBatchMatmulV4ToV3FusionPassParam {
    std::string caseName;
    int dtype;
    bool hasBias;
    bool hasX1Scale;
    bool hasX2Scale;
    bool hasYScale;
    bool hasX1Offset;
    bool hasX2Offset;
    bool hasYOffset;
    bool hasX2Table;
};

static std::string GetDirName(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

static std::vector<std::string> SplitCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream lineStream(line);
    std::string field;
    while (std::getline(lineStream, field, ',')) {
        fields.emplace_back(field);
    }
    return fields;
}

static bool ToBool(const std::string &value)
{
    return std::stoi(value) != 0;
}

static QuantBatchMatmulV4ToV3FusionPassParam ParseParam(const std::vector<std::string> &fields)
{
    if (fields.size() != CASE_FIELD_NUM) {
        throw std::runtime_error("Invalid csv column size.");
    }
    QuantBatchMatmulV4ToV3FusionPassParam param;
    param.caseName = fields[0];
    param.dtype = std::stoi(fields[1]);
    param.hasBias = ToBool(fields[2]);
    param.hasX1Scale = ToBool(fields[3]);
    param.hasX2Scale = ToBool(fields[4]);
    param.hasYScale = ToBool(fields[5]);
    param.hasX1Offset = ToBool(fields[6]);
    param.hasX2Offset = ToBool(fields[7]);
    param.hasYOffset = ToBool(fields[8]);
    param.hasX2Table = ToBool(fields[9]);
    return param;
}

static std::vector<QuantBatchMatmulV4ToV3FusionPassParam> GetParams()
{
    std::ifstream csvData(GetDirName(__FILE__) + "/" + TEST_CSV_FILE, std::ios::in);
    if (!csvData.is_open()) {
        throw std::runtime_error("Open csv file failed.");
    }

    std::vector<QuantBatchMatmulV4ToV3FusionPassParam> params;
    std::string line;
    while (std::getline(csvData, line)) {
        if (line.empty()) {
            continue;
        }
        auto fields = SplitCsvLine(line);
        if (!fields.empty() && fields[0] == "case_name") {
            continue;
        }
        params.emplace_back(ParseParam(fields));
    }
    return params;
}
}  // namespace

class QuantBatchMatmulV4ToV3FusionPassTest
    : public testing::TestWithParam<QuantBatchMatmulV4ToV3FusionPassParam> {};

TEST_P(QuantBatchMatmulV4ToV3FusionPassTest, General)
{
    const auto &param = GetParam();
    std::vector<ge::fusion::PatternUniqPtr> patternGraphs;
    auto graphBuilder = ge::es::EsGraphBuilder(QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME);
    auto inputs = graphBuilder.CreateInputs<V4_INPUT_NUM>();

    if (!param.hasBias) {
        inputs[V4_BIAS_INDEX] = nullptr;
    }
    if (!param.hasX1Scale) {
        inputs[V4_X1SCALE_INDEX] = nullptr;
    }
    if (!param.hasX2Scale) {
        inputs[V4_X2SCALE_INDEX] = nullptr;
    }
    if (!param.hasYScale) {
        inputs[V4_YSCALE_INDEX] = nullptr;
    }
    if (!param.hasX1Offset) {
        inputs[V4_X1OFFSET_INDEX] = nullptr;
    }
    if (!param.hasX2Offset) {
        inputs[V4_X2OFFSET_INDEX] = nullptr;
    }
    if (!param.hasYOffset) {
        inputs[V4_YOFFSET_INDEX] = nullptr;
    }
    if (!param.hasX2Table) {
        inputs[V4_X2TABLE_INDEX] = nullptr;
    }

    auto y = ge::es::QuantBatchMatmulV4(inputs[V4_X1_INDEX], inputs[V4_X2_INDEX], inputs[V4_BIAS_INDEX],
        inputs[V4_X1SCALE_INDEX], inputs[V4_X2SCALE_INDEX], inputs[V4_YSCALE_INDEX], inputs[V4_X1OFFSET_INDEX],
        inputs[V4_X2OFFSET_INDEX], inputs[V4_YOFFSET_INDEX], inputs[V4_X2TABLE_INDEX], param.dtype);

    auto graph = graphBuilder.BuildAndReset({y});
    graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, QUANT_BATCH_MATMUL_V4_TO_V3_FUSION_PASS_NAME);
    auto pattern = std::make_unique<ge::fusion::Pattern>(std::move(*graph));
    pattern->CaptureTensor({*y.GetProducer(), CAPTURE_TENSOR_INDEX});

    patternGraphs.emplace_back(std::move(pattern));
}

INSTANTIATE_TEST_CASE_P(QuantBatchMatmulV4ToV3FusionPass, QuantBatchMatmulV4ToV3FusionPassTest,
    testing::ValuesIn(GetParams()));
