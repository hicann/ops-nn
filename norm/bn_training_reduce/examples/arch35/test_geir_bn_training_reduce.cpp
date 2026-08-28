/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ge_api.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "ops_proto_nn.h"
#include "tensor.h"
#include "types.h"

// The selected standard domain header owns BNTrainingReduce. Data remains in
// ops_proto_legacy.h in this CANN release, so keep its tiny construction class
// test-local instead of including the duplicate-heavy compatibility header.
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

namespace {

constexpr float kRtol = 1.0e-4F;
constexpr float kAtol = 1.0e-4F;

struct CaseDef {
    std::string id;
    std::string category;
    std::string mode;
    std::string capabilities;
    std::string dtype;
    std::string inputFormat;
    std::string graphShape;
    std::string feedShapes;
    std::string pattern;
    std::string expectation;
    std::string acceptance;
};

struct InputData {
    ge::Tensor tensor;
    std::vector<float> quantized;
};

struct Expected {
    std::vector<float> sum;
    std::vector<float> squareSum;
};

struct Observed {
    bool readable = false;
    bool valuePass = false;
    std::vector<float> sum;
    std::vector<float> squareSum;
};

struct GraphBundle {
    ge::Graph graph;
    ge::Operator reduce;
    std::string node;
};

std::vector<std::string> Split(const std::string& text, char delimiter)
{
    std::vector<std::string> fields;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, delimiter)) {
        fields.push_back(field);
    }
    if (!text.empty() && text.back() == delimiter) {
        fields.emplace_back();
    }
    return fields;
}

bool LoadCases(const std::string& path, std::vector<CaseDef>& cases)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        std::fprintf(stderr, "GEIR-INFRA manifest=%s reason=open_failed value=FAIL\n", path.c_str());
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        return false;
    }
    const std::string expectedHeader = "id\tcategory\tmode\tcapabilities\tdtype\tinput_format\tgraph_shape\tfeed_"
                                       "shapes\tpattern\texpectation\tacceptance";
    if (line != expectedHeader) {
        std::fprintf(stderr, "GEIR-INFRA manifest=%s reason=bad_header value=FAIL\n", path.c_str());
        return false;
    }
    std::map<std::string, bool> ids;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = Split(line, '\t');
        if (fields.size() != 11U || fields[0].empty() || ids[fields[0]]) {
            std::fprintf(stderr, "GEIR-INFRA manifest=%s reason=bad_or_duplicate_row row=%s value=FAIL\n", path.c_str(),
                         line.c_str());
            return false;
        }
        ids[fields[0]] = true;
        cases.push_back({fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7],
                         fields[8], fields[9], fields[10]});
    }
    return !cases.empty();
}

std::vector<int64_t> ParseShape(const std::string& text)
{
    if (text == "UNKNOWN_RANK") {
        return ge::UNKNOWN_RANK;
    }
    if (text.size() < 2U || text.front() != '[' || text.back() != ']') {
        return {};
    }
    const std::string inner = text.substr(1, text.size() - 2U);
    if (inner.empty()) {
        return {};
    }
    std::vector<int64_t> dims;
    for (const auto& field : Split(inner, ',')) {
        dims.push_back(std::stoll(field));
    }
    return dims;
}

std::string ShapeString(const std::vector<int64_t>& dims)
{
    if (dims == ge::UNKNOWN_RANK) {
        return "UNKNOWN_RANK";
    }
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << dims[i];
    }
    out << ']';
    return out.str();
}

std::string ShapeString(const ge::Shape& shape) { return ShapeString(shape.GetDims()); }

size_t Numel(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return 1U;
    }
    size_t count = 1U;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            return 0U;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

ge::DataType ParseDtype(const std::string& dtype)
{
    if (dtype == "F16") {
        return ge::DT_FLOAT16;
    }
    if (dtype == "BF16") {
        return ge::DT_BF16;
    }
    if (dtype == "I32") {
        return ge::DT_INT32;
    }
    return ge::DT_FLOAT;
}

const char* DtypeName(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BFLOAT16";
        case ge::DT_FLOAT:
            return "FLOAT32";
        case ge::DT_INT32:
            return "INT32";
        case ge::DT_UNDEFINED:
            return "UNDEFINED";
        default:
            return "OTHER";
    }
}

ge::Format ParseFormat(const std::string& format)
{
    if (format == "NHWC") {
        return ge::FORMAT_NHWC;
    }
    if (format == "NCDHW") {
        return ge::FORMAT_NCDHW;
    }
    return ge::FORMAT_NCHW;
}

const char* FormatName(ge::Format format)
{
    if (format == ge::FORMAT_ND) {
        return "ND";
    }
    if (format == ge::FORMAT_NCHW) {
        return "NCHW";
    }
    if (format == ge::FORMAT_NHWC) {
        return "NHWC";
    }
    if (format == ge::FORMAT_NCDHW) {
        return "NCDHW";
    }
    return "OTHER";
}

uint32_t RoundRightShiftToEven(uint32_t value, uint32_t shift)
{
    const uint32_t truncated = value >> shift;
    const uint32_t remainder = value & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1U);
    return truncated + static_cast<uint32_t>(remainder > halfway || (remainder == halfway && (truncated & 1U) != 0U));
}

aclFloat16 FloatToFloat16(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0xFFU) {
        return static_cast<aclFloat16>(sign | 0x7C00U | (mantissa == 0U ? 0U : 0x0200U));
    }
    if (exponent == 0U) {
        return static_cast<aclFloat16>(sign);
    }

    int32_t halfExponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<aclFloat16>(sign);
        }
        const uint32_t rounded = RoundRightShiftToEven(mantissa | 0x800000U, 14U - halfExponent);
        return static_cast<aclFloat16>(sign | rounded);
    }
    if (halfExponent >= 31) {
        return static_cast<aclFloat16>(sign | 0x7C00U);
    }

    uint32_t roundedMantissa = RoundRightShiftToEven(mantissa, 13U);
    if (roundedMantissa == 0x400U) {
        roundedMantissa = 0U;
        ++halfExponent;
        if (halfExponent >= 31) {
            return static_cast<aclFloat16>(sign | 0x7C00U);
        }
    }
    return static_cast<aclFloat16>(sign | (static_cast<uint32_t>(halfExponent) << 10U) | roundedMantissa);
}

float Float16ToFloat(aclFloat16 value)
{
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    uint32_t exponent = (static_cast<uint32_t>(value) >> 10U) & 0x1FU;
    uint32_t mantissa = static_cast<uint32_t>(value) & 0x03FFU;
    uint32_t bits = sign;
    if (exponent == 0U && mantissa != 0U) {
        int32_t normalizedExponent = -14;
        while ((mantissa & 0x0400U) == 0U) {
            mantissa <<= 1U;
            --normalizedExponent;
        }
        mantissa &= 0x03FFU;
        bits |= static_cast<uint32_t>(normalizedExponent + 127) << 23U;
        bits |= mantissa << 13U;
    } else if (exponent == 0x1FU) {
        bits |= 0x7F800000U | (mantissa << 13U);
    } else if (exponent != 0U) {
        bits |= (exponent - 15U + 127U) << 23U;
        bits |= mantissa << 13U;
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> GenerateValues(const std::vector<int64_t>& shape, const std::string& pattern)
{
    const size_t count = Numel(shape);
    std::vector<float> values(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        const int64_t signedIndex = static_cast<int64_t>(i % 17U) - 8;
        values[i] = static_cast<float>(signedIndex) * 0.125F;
    }
    if (pattern == "zero") {
        std::fill(values.begin(), values.end(), 0.0F);
    } else if (pattern == "alternate") {
        for (size_t i = 0; i < count; ++i) {
            values[i] = (i % 2U == 0U) ? 16.0F : -16.0F;
        }
    } else if (pattern == "60000") {
        std::fill(values.begin(), values.end(), 60000.0F);
    } else if (pattern == "1e20") {
        std::fill(values.begin(), values.end(), 1.0e20F);
    } else if (pattern == "partition" && shape.size() == 4U) {
        const int64_t n = shape[0];
        const int64_t c = shape[1];
        const int64_t h = shape[2];
        const int64_t w = shape[3];
        for (int64_t ni = 0; ni < n; ++ni) {
            for (int64_t ci = 0; ci < c; ++ci) {
                for (int64_t hi = 0; hi < h; ++hi) {
                    for (int64_t wi = 0; wi < w; ++wi) {
                        const size_t offset = static_cast<size_t>(((ni * c + ci) * h + hi) * w + wi);
                        const int64_t code = ((ni % 2) * 13 + ci * 7 + hi * 3 + wi) % 19 - 9;
                        values[offset] = static_cast<float>(code) * 0.0625F;
                    }
                }
            }
        }
    }

    if (shape.size() == 4U && count != 0U && shape[1] > 0) {
        const int64_t channel = shape[1] > 1 ? 1 : 0;
        const size_t first = static_cast<size_t>(channel * shape[2] * shape[3]);
        if (pattern == "nan") {
            values[first] = std::numeric_limits<float>::quiet_NaN();
        } else if (pattern == "posinf") {
            values[first] = std::numeric_limits<float>::infinity();
        } else if (pattern == "posneginf") {
            values[first] = std::numeric_limits<float>::infinity();
            if (first + 1U < values.size()) {
                values[first + 1U] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    return values;
}

InputData MakeInput(const CaseDef& test, const std::vector<int64_t>& shape)
{
    const ge::DataType dtype = ParseDtype(test.dtype);
    std::vector<float> source = GenerateValues(shape, test.pattern);
    std::vector<uint8_t> bytes;
    std::vector<float> quantized(source.size());
    if (dtype == ge::DT_FLOAT16) {
        std::vector<aclFloat16> raw(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            raw[i] = FloatToFloat16(source[i]);
            quantized[i] = Float16ToFloat(raw[i]);
        }
        bytes.resize(raw.size() * sizeof(aclFloat16));
        if (!bytes.empty()) {
            std::memcpy(bytes.data(), raw.data(), bytes.size());
        }
    } else if (dtype == ge::DT_BF16) {
        std::vector<uint16_t> raw(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            raw[i] = FloatToBfloat16(source[i]);
            quantized[i] = Bfloat16ToFloat(raw[i]);
        }
        bytes.resize(raw.size() * sizeof(uint16_t));
        if (!bytes.empty()) {
            std::memcpy(bytes.data(), raw.data(), bytes.size());
        }
    } else if (dtype == ge::DT_INT32) {
        std::vector<int32_t> raw(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            raw[i] = static_cast<int32_t>(source[i]);
            quantized[i] = static_cast<float>(raw[i]);
        }
        bytes.resize(raw.size() * sizeof(int32_t));
        if (!bytes.empty()) {
            std::memcpy(bytes.data(), raw.data(), bytes.size());
        }
    } else {
        quantized = source;
        bytes.resize(source.size() * sizeof(float));
        if (!bytes.empty()) {
            std::memcpy(bytes.data(), source.data(), bytes.size());
        }
    }

    ge::TensorDesc desc(ge::Shape(shape), ParseFormat(test.inputFormat), dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(static_cast<int64_t>(shape.size()));
    return {ge::Tensor(desc, bytes), std::move(quantized)};
}

Expected ComputeExpected(const std::vector<float>& input, const std::vector<int64_t>& shape)
{
    Expected expected;
    if (shape.size() != 4U || shape[1] < 0) {
        return expected;
    }
    const int64_t n = shape[0];
    const int64_t c = shape[1];
    const int64_t h = shape[2];
    const int64_t w = shape[3];
    expected.sum.assign(static_cast<size_t>(c), 0.0F);
    expected.squareSum.assign(static_cast<size_t>(c), 0.0F);
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            for (int64_t hi = 0; hi < h; ++hi) {
                for (int64_t wi = 0; wi < w; ++wi) {
                    const size_t offset = static_cast<size_t>(((ni * c + ci) * h + hi) * w + wi);
                    const float value = input[offset];
                    expected.sum[static_cast<size_t>(ci)] += value;
                    expected.squareSum[static_cast<size_t>(ci)] += value * value;
                }
            }
        }
    }
    return expected;
}

bool Close(float actual, float expected, float& absoluteError)
{
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (std::isinf(expected)) {
        return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    }
    if (!std::isfinite(actual)) {
        return false;
    }
    absoluteError = std::fabs(actual - expected);
    return absoluteError <= kAtol + kRtol * std::fabs(expected);
}

std::string OutputShapes(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 2U) {
        return "COUNT_" + std::to_string(outputs.size());
    }
    return "{" + ShapeString(outputs[0].GetTensorDesc().GetShape()) + "," +
           ShapeString(outputs[1].GetTensorDesc().GetShape()) + "}";
}

std::string OutputDtypes(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 2U) {
        return "COUNT_" + std::to_string(outputs.size());
    }
    return "{" + std::string(DtypeName(outputs[0].GetTensorDesc().GetDataType())) + "," +
           DtypeName(outputs[1].GetTensorDesc().GetDataType()) + "}";
}

std::string OutputFormats(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 2U) {
        return "COUNT_" + std::to_string(outputs.size());
    }
    return "{" + std::string(FormatName(outputs[0].GetTensorDesc().GetFormat())) + "," +
           FormatName(outputs[1].GetTensorDesc().GetFormat()) + "}";
}

Observed ValidateOutputs(const CaseDef& test, const std::vector<int64_t>& feedShape, const Expected& expected,
                         const std::vector<ge::Tensor>& outputs, std::string& maxErrors, std::string& actualNumel)
{
    Observed observed;
    if (outputs.size() != 2U || feedShape.size() != 4U) {
        maxErrors = "{NA,NA}";
        actualNumel = "{NA,NA}";
        return observed;
    }
    const int64_t channel = feedShape[1];
    const std::vector<int64_t> expectedShape = {channel};
    bool metadataPass = true;
    for (const auto& output : outputs) {
        const ge::TensorDesc desc = output.GetTensorDesc();
        metadataPass = metadataPass && desc.GetShape().GetDims() == expectedShape;
        metadataPass = metadataPass && desc.GetDataType() == ge::DT_FLOAT;
        metadataPass = metadataPass && desc.GetFormat() == ge::FORMAT_ND;
        metadataPass = metadataPass && output.GetSize() == static_cast<size_t>(channel) * sizeof(float);
    }

    observed.sum.resize(static_cast<size_t>(channel));
    observed.squareSum.resize(static_cast<size_t>(channel));
    if (channel > 0) {
        if (outputs[0].GetData() == nullptr || outputs[1].GetData() == nullptr) {
            maxErrors = "{NA,NA}";
            actualNumel = "{0,0}";
            return observed;
        }
        std::memcpy(observed.sum.data(), outputs[0].GetData(), observed.sum.size() * sizeof(float));
        std::memcpy(observed.squareSum.data(), outputs[1].GetData(), observed.squareSum.size() * sizeof(float));
    }
    observed.readable = true;

    float maxSumError = 0.0F;
    float maxSquareError = 0.0F;
    bool valuesPass = expected.sum.size() == observed.sum.size() &&
                      expected.squareSum.size() == observed.squareSum.size();
    for (size_t i = 0; valuesPass && i < observed.sum.size(); ++i) {
        float error = 0.0F;
        valuesPass = Close(observed.sum[i], expected.sum[i], error);
        maxSumError = std::max(maxSumError, error);
        error = 0.0F;
        valuesPass = valuesPass && Close(observed.squareSum[i], expected.squareSum[i], error);
        maxSquareError = std::max(maxSquareError, error);
        if (test.pattern == "zero") {
            valuesPass = valuesPass && observed.sum[i] == 0.0F && observed.squareSum[i] == 0.0F;
        }
        if (test.pattern == "alternate" && std::isfinite(observed.squareSum[i])) {
            valuesPass = valuesPass && observed.squareSum[i] >= 0.0F;
        }
    }
    observed.valuePass = metadataPass && valuesPass;
    std::ostringstream errors;
    errors << '{' << maxSumError << ',' << maxSquareError << '}';
    maxErrors = errors.str();
    actualNumel = "{" + std::to_string(outputs[0].GetSize() / sizeof(float)) + "," +
                  std::to_string(outputs[1].GetSize() / sizeof(float)) + "}";
    return observed;
}

bool IsUnknownRank(const ge::TensorDesc& desc) { return desc.GetShape().GetDims() == ge::UNKNOWN_RANK; }

bool PrintAndCheckShapeInit(const std::string& caseId, const GraphBundle& bundle)
{
    const ge::TensorDesc sumDesc = bundle.reduce.GetOutputDescByName("sum");
    const ge::TensorDesc squareDesc = bundle.reduce.GetOutputDescByName("square_sum");
    const bool valid = IsUnknownRank(sumDesc) && IsUnknownRank(squareDesc) && sumDesc.GetDataType() == ge::DT_FLOAT &&
                       squareDesc.GetDataType() == ge::DT_FLOAT;
    std::printf("SHAPE-INIT case=%s outputs={sum:{shape:%s,dtype:%s},square_sum:{shape:%s,dtype:%s}} value=%s\n",
                caseId.c_str(), ShapeString(sumDesc.GetShape()).c_str(), DtypeName(sumDesc.GetDataType()),
                ShapeString(squareDesc.GetShape()).c_str(), DtypeName(squareDesc.GetDataType()),
                valid ? "PASS" : "FAIL");
    return valid;
}

GraphBundle BuildGraph(const CaseDef& test)
{
    GraphBundle bundle;
    bundle.node = test.id == "route-target" ? "bn_training_reduce_route" : "bn_training_reduce_" + test.id;
    bundle.graph = ge::Graph((bundle.node + "_graph").c_str());
    auto reduce = ge::op::BNTrainingReduce(bundle.node.c_str());
    const ge::TensorDesc outDesc(ge::Shape(ge::UNKNOWN_RANK), ParseFormat(test.inputFormat), ge::DT_FLOAT);
    reduce.update_output_desc_sum(outDesc);
    reduce.update_output_desc_square_sum(outDesc);

    std::vector<ge::Operator> inputs;
    const std::vector<int64_t> graphShape = test.mode == "missing-input" ? std::vector<int64_t>{1, 1, 1, 1} :
                                                                           ParseShape(test.graphShape);
    ge::TensorDesc xDesc(ge::Shape(graphShape), ParseFormat(test.inputFormat), ParseDtype(test.dtype));
    xDesc.SetRealDimCnt(graphShape == ge::UNKNOWN_RANK ? 0 : static_cast<int64_t>(graphShape.size()));
    auto data = ge::op::Data((bundle.node + "_x").c_str()).set_attr_index(0);
    data.update_input_desc_x(xDesc);
    data.update_output_desc_y(xDesc);
    reduce.set_input_x(data);
    reduce.update_input_desc_x(xDesc);
    bundle.graph.AddOp(data);
    inputs.push_back(data);
    bundle.graph.AddOp(reduce);
    const std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs = {{reduce, {0U, 1U}}};
    if (!inputs.empty()) {
        bundle.graph.SetInputs(inputs);
    }
    bundle.graph.SetOutputs(outputs);
    bundle.reduce = reduce;
    return bundle;
}

const char* GraphMode(const CaseDef& test)
{
    return test.mode == "dynamic" || test.mode == "unknown-rank" ? "dynamic" : "static";
}

std::string Sanitize(std::string text)
{
    if (text.size() > 160U) {
        text.resize(160U);
    }
    for (char& ch : text) {
        const bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                          ch == '_' || ch == '-' || ch == '.';
        if (!safe) {
            ch = '_';
        }
    }
    return text.empty() ? "NONE" : text;
}

std::string CurrentError()
{
    const ge::AscendString message = ge::GEGetErrorMsgV2();
    return message.GetString() == nullptr ? "NONE" : Sanitize(message.GetString());
}

std::string ClassifyError(const std::string& raw)
{
    std::string lower = raw;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lower.find("dtype") != std::string::npos || lower.find("data_type") != std::string::npos ||
        lower.find("datatype") != std::string::npos) {
        return "dtype_not_supported";
    }
    if (lower.find("format") != std::string::npos || lower.find("shape") != std::string::npos ||
        lower.find("rank") != std::string::npos || lower.find("dim") != std::string::npos) {
        return "shape_mismatch";
    }
    if (lower.find("null") != std::string::npos || lower.find("missing") != std::string::npos ||
        lower.find("input") != std::string::npos || lower.find("anchor") != std::string::npos ||
        lower.find("edge") != std::string::npos) {
        return "null_input";
    }
    return "unclassified";
}

Observed RunPositiveFeed(ge::Session& session, uint32_t gid, const CaseDef& test, const GraphBundle& bundle,
                         const std::vector<int64_t>& feedShape, const std::string& feedLabel,
                         const std::string& runMode, bool& runCompleted)
{
    const InputData input = MakeInput(test, feedShape);
    const Expected expected = ComputeExpected(input.quantized, feedShape);
    std::vector<ge::Tensor> outputs;
    const ge::Status status = session.RunGraph(gid, {input.tensor}, outputs);
    runCompleted = status == ge::SUCCESS;
    std::string maxErrors = "{NA,NA}";
    std::string actualNumel = "{NA,NA}";
    Observed observed;
    if (runCompleted) {
        observed = ValidateOutputs(test, feedShape, expected, outputs, maxErrors, actualNumel);
    }
    const std::string runError = runCompleted ? "NONE" : CurrentError();
    const int64_t channel = feedShape.size() == 4U ? feedShape[1] : -1;
    std::printf("GRAPH-RUN mode=%s case=%s node=%s gid=%u feed=%s expected_shape={[%ld],[%ld]} actual_shape=%s "
                "expected_dtype={FLOAT32,FLOAT32} actual_dtype=%s expected_format={ND,ND} actual_format=%s "
                "expected_numel={%ld,%ld} actual_numel=%s max_abs=%s rtol=%.1e atol=%.1e status=%u error=%s value=%s\n",
                runMode.c_str(), feedLabel.c_str(), bundle.node.c_str(), gid, ShapeString(feedShape).c_str(), channel,
                channel, OutputShapes(outputs).c_str(), OutputDtypes(outputs).c_str(), OutputFormats(outputs).c_str(),
                channel, channel, actualNumel.c_str(), maxErrors.c_str(), kRtol, kAtol, status, runError.c_str(),
                observed.valuePass ? "PASS" : "FAIL");
    return observed;
}

bool CheckPartitionInvariant(const Observed& whole, const Observed& half, float& maxError)
{
    if (!whole.readable || !half.readable || whole.sum.size() != half.sum.size()) {
        return false;
    }
    bool pass = true;
    maxError = 0.0F;
    for (size_t i = 0; i < whole.sum.size(); ++i) {
        float error = 0.0F;
        pass = pass && Close(whole.sum[i], half.sum[i] * 2.0F, error);
        maxError = std::max(maxError, error);
        error = 0.0F;
        pass = pass && Close(whole.squareSum[i], half.squareSum[i] * 2.0F, error);
        maxError = std::max(maxError, error);
    }
    return pass;
}

bool RunAcceptCase(ge::Session& session, uint32_t gid, const CaseDef& test, bool& infraFailure)
{
    GraphBundle bundle = BuildGraph(test);
    const char* graphMode = GraphMode(test);
    const ge::Status addStatus = session.AddGraph(gid, bundle.graph, std::map<ge::AscendString, ge::AscendString>{});
    std::printf("GRAPH-ADD mode=%s case=%s node=%s gid=%u declared_input_shape=%s status=%u\n", graphMode,
                test.id.c_str(), bundle.node.c_str(), gid, test.graphShape.c_str(), addStatus);
    if (addStatus != ge::SUCCESS) {
        std::printf("GRAPH-RUN mode=%s case=%s node=%s gid=%u actual_stage=ADD_GRAPH status=%u error=%s value=FAIL\n",
                    graphMode, test.id.c_str(), bundle.node.c_str(), gid, addStatus, CurrentError().c_str());
        return false;
    }
    if (!PrintAndCheckShapeInit(test.id, bundle)) {
        infraFailure = true;
    }

    const auto feedFields = Split(test.feedShapes, ';');
    bool pass = true;
    std::vector<Observed> observations;
    for (size_t i = 0; i < feedFields.size(); ++i) {
        bool completed = false;
        const std::string label = feedFields.size() == 1U ? test.id : test.id + "#feed" + std::to_string(i);
        observations.push_back(
            RunPositiveFeed(session, gid, test, bundle, ParseShape(feedFields[i]), label, graphMode, completed));
        pass = pass && completed && observations.back().valuePass;
    }
    if (test.mode == "partition") {
        float maxError = 0.0F;
        const bool invariantPass = observations.size() == 2U &&
                                   CheckPartitionInvariant(observations[0], observations[1], maxError);
        std::printf("INVARIANT case=%s name=batch_partition_additivity actual=whole_vs_two_halves max_abs=%g "
                    "rtol=%.1e atol=%.1e value=%s\n",
                    test.id.c_str(), maxError, kRtol, kAtol, invariantPass ? "PASS" : "FAIL");
        pass = pass && invariantPass;
    }
    session.RemoveGraph(gid);
    return pass;
}

bool RunRejectCase(ge::Session& session, uint32_t gid, const CaseDef& test, bool& infraFailure)
{
    GraphBundle bundle = BuildGraph(test);
    const char* graphMode = GraphMode(test);
    const ge::Status addStatus = session.AddGraph(gid, bundle.graph, std::map<ge::AscendString, ge::AscendString>{});
    std::printf("GRAPH-ADD mode=%s case=%s node=%s gid=%u declared_input_shape=%s status=%u\n", graphMode,
                test.id.c_str(), bundle.node.c_str(), gid, test.graphShape.c_str(), addStatus);
    std::string stage = "ADD_GRAPH";
    ge::Status status = addStatus;
    std::string rawError = CurrentError();
    if (addStatus == ge::SUCCESS) {
        if (!PrintAndCheckShapeInit(test.id, bundle)) {
            infraFailure = true;
        }
        stage = "RUN_GRAPH";
        std::vector<ge::Tensor> feeds;
        if (test.mode != "missing-input") {
            feeds.push_back(MakeInput(test, ParseShape(test.feedShapes)).tensor);
        }
        std::vector<ge::Tensor> outputs;
        status = session.RunGraph(gid, feeds, outputs);
        rawError = CurrentError();
        session.RemoveGraph(gid);
    }
    const std::string actualError = status == ge::SUCCESS ? "accepted" : ClassifyError(rawError);
    const bool pass = status != ge::SUCCESS && actualError == test.expectation;
    std::printf(
        "GRAPH-REJECT case=%s node=%s inputs={shape:%s,format:%s,dtype:%s} actual_stage=%s status=%u value=%s\n",
        test.id.c_str(), bundle.node.c_str(), test.graphShape.c_str(), test.inputFormat.c_str(), test.dtype.c_str(),
        stage.c_str(), status, pass ? "PASS" : "FAIL");
    std::printf("REJECT-DETAIL case=%s expected_error=%s actual_error=%s raw_error=%s\n", test.id.c_str(),
                test.expectation.c_str(), actualError.c_str(), rawError.c_str());
    return pass;
}

bool RunRuntimeInputContractCase(ge::Session& session, uint32_t gid, const CaseDef& test, bool& infraFailure)
{
    GraphBundle bundle = BuildGraph(test);
    const char* graphMode = GraphMode(test);
    const ge::Status addStatus = session.AddGraph(gid, bundle.graph, std::map<ge::AscendString, ge::AscendString>{});
    std::printf("GRAPH-ADD mode=%s case=%s node=%s gid=%u declared_input_shape=%s status=%u\n", graphMode,
                test.id.c_str(), bundle.node.c_str(), gid, test.graphShape.c_str(), addStatus);
    std::string stage = "ADD_GRAPH";
    ge::Status status = addStatus;
    std::string rawError = CurrentError();
    if (addStatus == ge::SUCCESS) {
        if (!PrintAndCheckShapeInit(test.id, bundle)) {
            infraFailure = true;
        }
        stage = "RUN_GRAPH";
        std::vector<ge::Tensor> outputs;
        status = session.RunGraph(gid, {}, outputs);
        rawError = CurrentError();
        session.RemoveGraph(gid);
    }
    const std::string actualError = status == ge::SUCCESS ? "accepted" : ClassifyError(rawError);
    const bool pass = status != ge::SUCCESS && actualError == test.expectation;
    std::printf("RUNTIME-INPUT-CONTRACT case=%s node=%s graph=LEGAL_DATA_TO_OP runtime_feeds=EMPTY actual_stage=%s "
                "status=%u expected_error=%s actual_error=%s kernel_start_policy=ALLOWED value=%s\n",
                test.id.c_str(), bundle.node.c_str(), stage.c_str(), status, test.expectation.c_str(),
                actualError.c_str(), pass ? "PASS" : "FAIL");
    std::printf("RUNTIME-CONTRACT-DETAIL case=%s raw_error=%s\n", test.id.c_str(), rawError.c_str());
    return pass;
}

bool InitializeGe()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    const ge::Status status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        std::fprintf(stderr, "GEIR-INFRA stage=GEInitialize status=%u error=%s value=FAIL\n", status,
                     CurrentError().c_str());
        return false;
    }
    return true;
}

int RunRoute()
{
    if (!InitializeGe()) {
        return 2;
    }
    ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
    CaseDef route = {"route-target", "positive",  "static", "route",  "F32",   "NCHW",
                     "[1,1,1,1]",    "[1,1,1,1]", "finite", "oracle", "accept"};
    GraphBundle bundle = BuildGraph(route);
    constexpr uint32_t gid = 7000U;
    const ge::Status addStatus = session.AddGraph(gid, bundle.graph, std::map<ge::AscendString, ge::AscendString>{});
    std::printf("GRAPH-ADD mode=route signature=F32_NCHW_1x1x1x1 node=%s gid=%u declared_input_shape=%s status=%u\n",
                bundle.node.c_str(), gid, route.graphShape.c_str(), addStatus);
    int probeExit = 0;
    bool businessPass = false;
    if (addStatus != ge::SUCCESS) {
        probeExit = 3;
        std::printf("GRAPH-RUN mode=route case=route-target node=%s gid=%u actual_stage=ADD_GRAPH status=%u "
                    "error=%s value=FAIL\n",
                    bundle.node.c_str(), gid, addStatus, CurrentError().c_str());
    } else if (!PrintAndCheckShapeInit("route-target", bundle)) {
        probeExit = 4;
    } else {
        bool completed = false;
        const Observed observed = RunPositiveFeed(session, gid, route, bundle, {1, 1, 1, 1}, "route-target", "route",
                                                  completed);
        if (!completed || !observed.readable) {
            probeExit = 5;
        }
        businessPass = observed.valuePass;
        session.RemoveGraph(gid);
    }
    const ge::Status finalizeStatus = ge::GEFinalize();
    if (finalizeStatus != ge::SUCCESS && probeExit == 0) {
        probeExit = 6;
    }
    std::printf("ROUTE-PROBE op=BNTrainingReduce node=%s status=DONE value=%s probe_exit=%d\n", bundle.node.c_str(),
                businessPass ? "PASS" : "FAIL", probeExit);
    return probeExit;
}

int RunSelected(const std::vector<CaseDef>& cases, const std::string& selectedId, bool printSummary)
{
    if (!InitializeGe()) {
        return 2;
    }
    ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
    int total = 0;
    int passed = 0;
    bool infraFailure = false;
    uint32_t gid = 8000U;
    for (const auto& test : cases) {
        if (!selectedId.empty() && test.id != selectedId) {
            continue;
        }
        ++total;
        bool pass = false;
        if (test.acceptance == "reject") {
            pass = RunRejectCase(session, gid, test, infraFailure);
        } else if (test.acceptance == "runtime-contract") {
            pass = RunRuntimeInputContractCase(session, gid, test, infraFailure);
        } else {
            pass = RunAcceptCase(session, gid, test, infraFailure);
        }
        std::printf("CASE %s %s\n", test.id.c_str(), pass ? "PASS" : "FAIL");
        if (pass) {
            ++passed;
        }
        ++gid;
    }
    const ge::Status finalizeStatus = ge::GEFinalize();
    if (finalizeStatus != ge::SUCCESS) {
        infraFailure = true;
    }
    if (printSummary) {
        std::printf("GEIR-SUMMARY total=%d pass=%d fail=%d skip=0\n", total, passed, total - passed);
    }
    if (total == 0) {
        std::fprintf(stderr, "GEIR-INFRA selected=%s reason=no_matching_case value=FAIL\n", selectedId.c_str());
        return 2;
    }
    return infraFailure ? 2 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        return RunRoute();
    }
    if (std::string(argv[1]) == "--route") {
        return RunRoute();
    }
    std::string manifest;
    std::string selected;
    bool printSummary = true;
    if (std::string(argv[1]) == "--test" && argc == 3) {
        manifest = argv[2];
    } else if (std::string(argv[1]) == "--case" && argc == 4) {
        selected = argv[2];
        manifest = argv[3];
        printSummary = false;
    } else {
        std::fprintf(stderr, "usage: %s {--route|--test|--case ID} cases.tsv\n", argv[0]);
        return 2;
    }
    std::vector<CaseDef> cases;
    if (!LoadCases(manifest, cases)) {
        return 2;
    }
    return RunSelected(cases, selected, printSummary);
}
