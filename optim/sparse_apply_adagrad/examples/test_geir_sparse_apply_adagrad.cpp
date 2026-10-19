/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "graph/operator_factory.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/sparse_apply_adagrad_proto.h"

namespace {
constexpr int64_t ROWS = 4;
constexpr int64_t COLS = 8;
constexpr float LEARNING_RATE = 0.1f;
constexpr float EPSILON = 1e-8f;

template <typename T>
bool AddInput(const char* name, const std::vector<int64_t>& shape, ge::DataType dtype, const std::vector<T>& values,
              ge::Graph& graph, std::vector<ge::Operator>& inputs, std::vector<ge::Tensor>& tensors)
{
    auto data = ge::OperatorFactory::CreateOperator(name, "Data");
    data.SetAttr("index", static_cast<int64_t>(inputs.size()));
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(static_cast<int32_t>(shape.size()));
    ge::Tensor tensor(desc);
    if (tensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T)) !=
            ge::GRAPH_SUCCESS ||
        data.UpdateInputDesc("x", desc) != ge::GRAPH_SUCCESS || data.UpdateOutputDesc("y", desc) != ge::GRAPH_SUCCESS ||
        graph.AddOp(data) != ge::GRAPH_SUCCESS) {
        std::cerr << "Failed to create input: " << name << std::endl;
        return false;
    }
    inputs.push_back(data);
    tensors.push_back(tensor);
    return true;
}

bool CheckOutput(const ge::Tensor& tensor, const std::vector<float>& expected, const char* name)
{
    if (tensor.GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        tensor.GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>({ROWS, COLS}) ||
        tensor.GetSize() != expected.size() * sizeof(float) || tensor.GetData() == nullptr) {
        std::cerr << "Unexpected output descriptor or size: " << name << std::endl;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        float actual = 0.0f;
        std::memcpy(&actual, tensor.GetData() + i * sizeof(float), sizeof(float));
        const float tolerance = 1e-5f + 1e-5f * std::fabs(expected[i]);
        if (!std::isfinite(actual) || std::fabs(actual - expected[i]) > tolerance) {
            std::cerr << name << "[" << i << "]: expected " << expected[i] << ", got " << actual << std::endl;
            return false;
        }
    }
    return true;
}

bool RunExample(bool updateSlots)
{
    std::vector<float> var(ROWS * COLS, 1.0f);
    std::vector<float> accum(ROWS * COLS, 0.5f);
    const std::vector<float> lr = {LEARNING_RATE};
    const std::vector<int32_t> indices = {1, 3};
    std::vector<float> grad(indices.size() * COLS);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = (i % 2 == 0 ? 1.0f : -1.0f) * 0.1f * static_cast<float>(i + 1);
    }

    // Compute expectations before RunGraph, which may update var and accum in place.
    auto expectedVar = var;
    auto expectedAccum = accum;
    for (size_t row = 0; row < indices.size(); ++row) {
        for (int64_t col = 0; col < COLS; ++col) {
            const size_t offset = indices[row] * COLS + col;
            const float gradient = grad[row * COLS + col];
            if (updateSlots) {
                expectedAccum[offset] += gradient * gradient;
            }
            expectedVar[offset] -= LEARNING_RATE * gradient / (std::sqrt(expectedAccum[offset]) + EPSILON);
        }
    }

    ge::Graph graph("sparse_apply_adagrad_example");
    std::vector<ge::Operator> inputs;
    std::vector<ge::Tensor> inputTensors;
    if (!AddInput("var", {ROWS, COLS}, ge::DT_FLOAT, var, graph, inputs, inputTensors) ||
        !AddInput("accum", {ROWS, COLS}, ge::DT_FLOAT, accum, graph, inputs, inputTensors) ||
        !AddInput("lr", {1}, ge::DT_FLOAT, lr, graph, inputs, inputTensors) ||
        !AddInput("grad", {static_cast<int64_t>(indices.size()), COLS}, ge::DT_FLOAT, grad, graph, inputs,
                  inputTensors) ||
        !AddInput("indices", {static_cast<int64_t>(indices.size())}, ge::DT_INT32, indices, graph, inputs,
                  inputTensors)) {
        return false;
    }

    auto adagrad = ge::op::SparseApplyAdagrad("SparseApplyAdagrad")
                       .set_input_var(inputs[0])
                       .set_input_accum(inputs[1])
                       .set_input_lr(inputs[2])
                       .set_input_grad(inputs[3])
                       .set_input_indices(inputs[4])
                       .set_attr_use_locking(false)
                       .set_attr_update_slots(updateSlots);
    ge::TensorDesc outputDesc(ge::Shape({ROWS, COLS}), ge::FORMAT_ND, ge::DT_FLOAT);
    outputDesc.SetRealDimCnt(2);
    if (adagrad.update_output_desc_var(outputDesc) != ge::GRAPH_SUCCESS ||
        adagrad.update_output_desc_accum(outputDesc) != ge::GRAPH_SUCCESS ||
        graph.AddOp(adagrad) != ge::GRAPH_SUCCESS) {
        std::cerr << "Failed to create SparseApplyAdagrad" << std::endl;
        return false;
    }
    const std::vector<ge::Operator> outputs = {adagrad};
    graph.SetInputs(inputs).SetOutputs(outputs);

    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    constexpr uint32_t graphId = 0;
    auto status = session.AddGraph(graphId, graph, options);
    if (status != ge::SUCCESS) {
        std::cerr << "AddGraph failed: " << status << std::endl;
        return false;
    }
    std::vector<ge::Tensor> outputTensors;
    status = session.RunGraph(graphId, inputTensors, outputTensors);
    if (status != ge::SUCCESS) {
        std::cerr << "RunGraph failed: " << status << std::endl;
        return false;
    }
    if (outputTensors.size() != 2) {
        std::cerr << "Expected var and accum outputs, got " << outputTensors.size() << std::endl;
        return false;
    }
    if (!CheckOutput(outputTensors[0], expectedVar, "var") || !CheckOutput(outputTensors[1], expectedAccum, "accum")) {
        return false;
    }
    std::cout << "SparseApplyAdagrad update_slots=" << std::boolalpha << updateSlots
              << ": var and accum match, including unchanged rows." << std::endl;
    return true;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    const auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        std::cerr << "GEInitialize failed: " << status << std::endl;
        return 1;
    }
    // RunExample destroys each Session before GEFinalize.
    const bool success = RunExample(true) && RunExample(false);
    const auto finalizeStatus = ge::GEFinalize();
    if (finalizeStatus != ge::SUCCESS) {
        std::cerr << "GEFinalize failed: " << finalizeStatus << std::endl;
        return 1;
    }
    return success ? 0 : 1;
}
