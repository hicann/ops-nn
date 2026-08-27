/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_selu_dynamic.cpp
 * @brief Selu dynamic-shape and dynamic-rank GE IR verification.
 *
 * Both scenarios reuse one GE Session. Each scenario adds one dynamic graph and
 * runs that graph three times with distinct concrete shapes. Every run checks
 * the only output's shape, dtype, and all values against the CPU golden
 * y = scale * (max(0, x) + min(0, alpha * (exp(x) - 1))) computed in double.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/selu_proto.h"

namespace {

constexpr int kSuccess = 0;
constexpr int kFailed = -1;
constexpr double kAlpha = 1.6732632423543772848170429916717;
constexpr double kScale = 1.0507009873554804934193349852946;
constexpr float kAtol = 1.0e-4F;
constexpr float kRtol = 1.0e-4F;

using ShapeVector = std::vector<int64_t>;
using ShapeCases = std::vector<ShapeVector>;

std::string ShapeToString(const ShapeVector& shape)
{
    std::string result = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0U) {
            result += ",";
        }
        result += std::to_string(shape[index]);
    }
    result += "]";
    return result;
}

int64_t ElementCount(const ShapeVector& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

std::vector<float> MakeInput(int64_t count)
{
    const std::vector<float> pattern = {-8.0F, -1.0F, -0.25F, 0.0F, 0.5F, 3.0F, 7.0F, 1000.0F};
    std::vector<float> input(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
        input[static_cast<size_t>(index)] = pattern[static_cast<size_t>(index) % pattern.size()];
    }
    return input;
}

std::vector<float> ComputeGolden(const std::vector<float>& input)
{
    std::vector<float> golden(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const double value = static_cast<double>(input[index]);
        const double linear = std::max(value, 0.0);
        const double saturate = std::min(0.0, kAlpha * std::expm1(value));
        golden[index] = static_cast<float>(kScale * (linear + saturate));
    }
    return golden;
}

ge::TensorDesc MakeTensorDesc(const ShapeVector& shape, bool hostPlacement)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    if (hostPlacement) {
        desc.SetPlacement(ge::kPlacementHost);
    }
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

struct SeluFeeds {
    std::vector<float> input;
    std::vector<ge::Tensor> tensors;
};

SeluFeeds MakeFeeds(const ShapeVector& shape)
{
    SeluFeeds feeds;
    feeds.input = MakeInput(ElementCount(shape));
    const ge::TensorDesc desc = MakeTensorDesc(shape, true);
    const uint32_t byteSize = static_cast<uint32_t>(feeds.input.size() * sizeof(float));
    feeds.tensors.emplace_back(desc, reinterpret_cast<uint8_t*>(feeds.input.data()), byteSize);
    return feeds;
}

int BuildGraph(const std::string& scenarioName, const ShapeVector& declaredShape, ge::Graph& graph)
{
    const ge::TensorDesc inputDesc = MakeTensorDesc(declaredShape, false);

    const std::string dataName = scenarioName + "_x";
    auto data = ge::op::Data(dataName.c_str()).set_attr_index(0);
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    graph.AddOp(data);

    const std::string opName = "selu_" + scenarioName;
    auto selu = ge::op::Selu(opName.c_str());
    selu.set_input_x(data);
    selu.update_input_desc_x(inputDesc);

    const ge::TensorDesc outputDesc = MakeTensorDesc(declaredShape, false);
    selu.update_output_desc_y(outputDesc);
    graph.AddOp(selu);

    std::vector<ge::Operator> inputs = {data};
    std::vector<ge::Operator> outputs = {selu};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return kSuccess;
}

bool HasExpectedShape(const ge::Shape& actual, const ShapeVector& expected)
{
    if (actual.GetDimNum() != expected.size()) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (actual.GetDim(index) != expected[index]) {
            return false;
        }
    }
    return true;
}

bool VerifyOutput(const std::vector<ge::Tensor>& outputs, const ShapeVector& shape, const std::vector<float>& input)
{
    if (outputs.size() != 1U) {
        std::cerr << "Expected exactly one Selu output, got " << outputs.size() << std::endl;
        return false;
    }

    const ge::Tensor& output = outputs[0];
    const ge::TensorDesc outputDesc = output.GetTensorDesc();
    if (!HasExpectedShape(outputDesc.GetShape(), shape)) {
        std::cerr << "Selu output shape mismatch for " << ShapeToString(shape) << std::endl;
        return false;
    }
    if (outputDesc.GetDataType() != ge::DT_FLOAT) {
        std::cerr << "Selu output dtype mismatch: expected DT_FLOAT, got " << outputDesc.GetDataType() << std::endl;
        return false;
    }

    const std::vector<float> golden = ComputeGolden(input);
    const size_t expectedBytes = golden.size() * sizeof(float);
    const float* actual = reinterpret_cast<const float*>(output.GetData());
    if (actual == nullptr || output.GetSize() != expectedBytes) {
        std::cerr << "Selu output buffer mismatch: actual bytes=" << output.GetSize()
                  << ", expected bytes=" << expectedBytes << std::endl;
        return false;
    }

    for (size_t index = 0; index < golden.size(); ++index) {
        const float error = std::fabs(actual[index] - golden[index]);
        const float tolerance = kAtol + kRtol * std::fabs(golden[index]);
        if (!std::isfinite(actual[index]) || error > tolerance) {
            std::cerr << "Selu value mismatch at index " << index << ": actual=" << actual[index]
                      << ", expected=" << golden[index] << ", tolerance=" << tolerance << std::endl;
            return false;
        }
    }
    return true;
}

int RunScenario(ge::Session& session, uint32_t graphId, const std::string& scenarioName,
                const ShapeVector& declaredShape, const ShapeCases& concreteShapes)
{
    std::cout << "Scenario " << scenarioName << ", declared shape " << ShapeToString(declaredShape) << std::endl;

    const std::string graphName = "selu_" + scenarioName + "_graph";
    ge::Graph graph(graphName.c_str());
    if (BuildGraph(scenarioName, declaredShape, graph) != kSuccess) {
        std::cerr << "ERROR: unable to build graph for scenario " << scenarioName << std::endl;
        return kFailed;
    }

    std::map<ge::AscendString, ge::AscendString> graphOptions;
    ge::Status status = session.AddGraph(graphId, graph, graphOptions);
    if (status != kSuccess) {
        std::cerr << "ERROR: AddGraph failed for scenario " << scenarioName << ": " << ge::GEGetErrorMsgV2().GetString()
                  << std::endl;
        return kFailed;
    }

    size_t passed = 0U;
    for (const ShapeVector& shape : concreteShapes) {
        std::cout << "Run concrete shape " << ShapeToString(shape) << std::endl;
        SeluFeeds feeds = MakeFeeds(shape);
        std::vector<ge::Tensor> outputs;
        status = session.RunGraph(graphId, feeds.tensors, outputs);
        if (status != kSuccess) {
            std::cerr << "ERROR: RunGraph failed for scenario " << scenarioName << ", shape " << ShapeToString(shape)
                      << ": " << ge::GEGetErrorMsgV2().GetString() << std::endl;
            continue;
        }
        if (!VerifyOutput(outputs, shape, feeds.input)) {
            std::cerr << "ERROR: output verification failed for scenario " << scenarioName << ", shape "
                      << ShapeToString(shape) << std::endl;
            continue;
        }

        ++passed;
        std::cout << "Shape, dtype and values PASSED for " << ShapeToString(shape) << std::endl;
    }

    std::cout << "Scenario " << scenarioName << " summary: " << passed << "/" << concreteShapes.size() << " passed"
              << std::endl;
    return passed == concreteShapes.size() ? kSuccess : kFailed;
}

} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    ge::Status status = ge::GEInitialize(globalOptions);
    if (status != kSuccess) {
        std::cerr << "ERROR: GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }

    std::map<ge::AscendString, ge::AscendString> buildOptions;
    ge::Session* session = new (std::nothrow) ge::Session(buildOptions);
    if (session == nullptr) {
        std::cerr << "ERROR: unable to create GE Session" << std::endl;
        (void)ge::GEFinalize();
        return kFailed;
    }

    const int dimResult = RunScenario(*session, 0U, "unknown_dim_minus_1", {-1, -1}, {{4, 2}, {1, 8}, {3, 5}});
    const int rankResult = RunScenario(*session, 1U, "unknown_rank_minus_2", {-2}, {{8}, {4, 2}, {2, 3, 4}});

    delete session;
    session = nullptr;
    status = ge::GEFinalize();
    if (status != kSuccess) {
        std::cerr << "ERROR: GEFinalize failed" << std::endl;
        return kFailed;
    }

    if (dimResult != kSuccess || rankResult != kSuccess) {
        std::cerr << "ERROR: Selu dynamic GEIR verification did not pass all scenarios" << std::endl;
        return kFailed;
    }

    std::cout << "Selu dynamic GEIR verification PASSED (-1 and -2)" << std::endl;
    return kSuccess;
}
