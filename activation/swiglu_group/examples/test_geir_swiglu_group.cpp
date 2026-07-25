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
 * \file test_geir_swiglu_group.cpp
 * \brief GE IR invocation and result verification for SwigluGroup.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
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
#include "../op_graph/swiglu_group_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;
constexpr float RTOL = 1.0e-4f;
constexpr float ATOL = 1.0e-4f;
const vector<int64_t> X_SHAPE = {4, 512};
const vector<int64_t> Y_SHAPE = {4, 256};

struct SwigluGroupCase {
    const char* name;
    bool withWeight;
    bool withGroupIndex;
    float clampLimit;
};

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

TensorDesc CreateHostTensorDesc(const vector<int64_t>& shape, DataType dataType)
{
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dataType);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

template <typename T>
Tensor CreateTensor(const TensorDesc& desc, const vector<T>& data)
{
    return Tensor(desc, reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}

vector<float> CreateXData()
{
    vector<float> data(static_cast<size_t>(GetShapeSize(X_SHAPE)));
    const int64_t rowCount = X_SHAPE[0];
    const int64_t hiddenSize = X_SHAPE[1] / 2;
    for (int64_t row = 0; row < rowCount; ++row) {
        for (int64_t col = 0; col < hiddenSize; ++col) {
            const size_t rowOffset = static_cast<size_t>(row * X_SHAPE[1]);
            data[rowOffset + static_cast<size_t>(col)] = static_cast<float>((col + row) % 9 - 4) * 0.5f;
            data[rowOffset + static_cast<size_t>(hiddenSize + col)] = static_cast<float>((col * 3 + row) % 11 - 5) *
                                                                      0.4f;
        }
    }
    return data;
}

vector<float> ComputeGolden(const vector<float>& x, const vector<float>* weight, float clampLimit)
{
    vector<float> golden(static_cast<size_t>(GetShapeSize(Y_SHAPE)));
    const int64_t rowCount = X_SHAPE[0];
    const int64_t hiddenSize = X_SHAPE[1] / 2;
    for (int64_t row = 0; row < rowCount; ++row) {
        for (int64_t col = 0; col < hiddenSize; ++col) {
            const size_t xOffset = static_cast<size_t>(row * X_SHAPE[1]);
            float x0 = x[xOffset + static_cast<size_t>(col)];
            float x1 = x[xOffset + static_cast<size_t>(hiddenSize + col)];
            if (clampLimit != DEFAULT_CLAMP_LIMIT) {
                x0 = std::min(x0, clampLimit);
                x1 = std::min(std::max(x1, -clampLimit), clampLimit);
            }
            float value = x0 / (1.0f + std::exp(-x0)) * x1;
            if (weight != nullptr) {
                value *= (*weight)[static_cast<size_t>(row)];
            }
            golden[static_cast<size_t>(row * hiddenSize + col)] = value;
        }
    }
    return golden;
}

int CreateSwigluGroupGraph(const SwigluGroupCase& testCase, Graph& graph, vector<Tensor>& input,
                           vector<Operator>& inputs, vector<Operator>& outputs, vector<float>& xHostData,
                           vector<float>& weightHostData)
{
    auto swigluGroup = op::SwigluGroup("swiglu_group");

    xHostData = CreateXData();
    TensorDesc xDesc = CreateHostTensorDesc(X_SHAPE, ge::DT_FLOAT);
    auto xData = op::Data("x").set_attr_index(static_cast<int64_t>(inputs.size()));
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);
    swigluGroup.set_input_x(xData);
    swigluGroup.update_input_desc_x(xDesc);
    input.push_back(CreateTensor(xDesc, xHostData));
    inputs.push_back(xData);

    if (testCase.withWeight) {
        const vector<int64_t> weightShape = {4, 1};
        weightHostData = {0.5f, -1.0f, 1.5f, 2.0f};
        TensorDesc weightDesc = CreateHostTensorDesc(weightShape, ge::DT_FLOAT);
        auto weightData = op::Data("weight").set_attr_index(static_cast<int64_t>(inputs.size()));
        weightData.update_input_desc_x(weightDesc);
        weightData.update_output_desc_y(weightDesc);
        graph.AddOp(weightData);
        swigluGroup.set_input_weight(weightData);
        swigluGroup.update_input_desc_weight(weightDesc);
        input.push_back(CreateTensor(weightDesc, weightHostData));
        inputs.push_back(weightData);
    }

    if (testCase.withGroupIndex) {
        const vector<int64_t> groupIndexShape = {2};
        const vector<int64_t> groupIndexHostData = {1, 3};
        TensorDesc groupIndexDesc = CreateHostTensorDesc(groupIndexShape, ge::DT_INT64);
        auto groupIndexData = op::Data("group_index").set_attr_index(static_cast<int64_t>(inputs.size()));
        groupIndexData.update_input_desc_x(groupIndexDesc);
        groupIndexData.update_output_desc_y(groupIndexDesc);
        graph.AddOp(groupIndexData);
        swigluGroup.set_input_group_index(groupIndexData);
        swigluGroup.update_input_desc_group_index(groupIndexDesc);
        input.push_back(CreateTensor(groupIndexDesc, groupIndexHostData));
        inputs.push_back(groupIndexData);
    }

    TensorDesc yDesc(ge::Shape(Y_SHAPE), FORMAT_ND, ge::DT_FLOAT);
    swigluGroup.update_output_desc_y(yDesc);
    swigluGroup.set_attr_clamp_limit(testCase.clampLimit);
    graph.AddOp(swigluGroup);
    outputs.push_back(swigluGroup);
    return SUCCESS;
}

bool VerifyOutput(const SwigluGroupCase& testCase, const vector<Tensor>& output, const vector<float>& golden)
{
    if (output.size() != 1) {
        std::cout << testCase.name << ": expected one output, got " << output.size() << std::endl;
        return false;
    }

    const Tensor& y = output[0];
    const TensorDesc yDesc = y.GetTensorDesc();
    if (yDesc.GetDataType() != ge::DT_FLOAT || yDesc.GetShape().GetDims() != Y_SHAPE ||
        y.GetSize() != golden.size() * sizeof(float) || y.GetData() == nullptr) {
        std::cout << testCase.name << ": invalid output metadata, dtype=" << yDesc.GetDataType()
                  << ", elements=" << yDesc.GetShape().GetShapeSize() << ", bytes=" << y.GetSize() << std::endl;
        return false;
    }

    const float* actual = reinterpret_cast<const float*>(y.GetData());
    float maxError = 0.0f;
    size_t maxErrorIndex = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        const float error = std::abs(actual[i] - golden[i]);
        const float tolerance = ATOL + RTOL * std::abs(golden[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIndex = i;
        }
        if (error > tolerance) {
            std::cout << testCase.name << ": mismatch at " << i << ", actual=" << actual[i]
                      << ", expected=" << golden[i] << ", error=" << error << ", tolerance=" << tolerance << std::endl;
            return false;
        }
    }

    std::cout << "[PASS] " << testCase.name << ": shape=[4, 256], dtype=float32, max_error=" << maxError << " at index "
              << maxErrorIndex << std::endl;
    return true;
}

int RunSwigluGroupCase(const SwigluGroupCase& testCase)
{
    printf("%s - INFO - [XIR]: Run %s\n", GetTime().c_str(), testCase.name);
    Graph graph(testCase.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    vector<float> xHostData;
    vector<float> weightHostData;
    Status ret = CreateSwigluGroupGraph(testCase, graph, input, inputs, outputs, xHostData, weightHostData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create %s graph failed\n", GetTime().c_str(), testCase.name);
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add %s graph failed\n", GetTime().c_str(), testCase.name);
        std::cout << "Error message: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        delete session;
        return FAILED;
    }

    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run %s graph failed\n", GetTime().c_str(), testCase.name);
        std::cout << "Error message: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        delete session;
        return FAILED;
    }

    const vector<float>* weight = testCase.withWeight ? &weightHostData : nullptr;
    const vector<float> golden = ComputeGolden(xHostData, weight, testCase.clampLimit);
    const bool passed = VerifyOutput(testCase, output, golden);
    delete session;
    return passed ? SUCCESS : FAILED;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        std::cout << "Return code: " << ret << std::endl;
        std::cout << "Error message: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    const vector<SwigluGroupCase> testCases = {
        {"basic", false, false, DEFAULT_CLAMP_LIMIT},
        {"weight", true, false, DEFAULT_CLAMP_LIMIT},
        {"group_index", false, true, DEFAULT_CLAMP_LIMIT},
        {"weight_group_clamp", true, true, 1.0f},
    };

    bool passed = true;
    for (const auto& testCase : testCases) {
        if (RunSwigluGroupCase(testCase) != SUCCESS) {
            passed = false;
            break;
        }
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: SwigluGroup GE IR verification %s\n", GetTime().c_str(), passed ? "PASSED" : "FAILED");
    return passed ? SUCCESS : FAILED;
}
