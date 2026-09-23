/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GRU_BLOCK_CELL_GRAD_GEIR_EXAMPLE_COMMON_H_
#define GRU_BLOCK_CELL_GRAD_GEIR_EXAMPLE_COMMON_H_
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "ge/ge_api.h"
#include "graph/operator_factory.h"
#include "../../op_graph/gru_block_cell_grad_proto.h"

namespace gru_example {
using Shapes = std::vector<std::vector<int64_t>>;
constexpr const char* INPUT_NAMES[] = {"x", "h_prev", "w_ru", "w_c", "b_ru", "b_c", "r", "u", "c", "d_h"};
constexpr const char* OUTPUT_NAMES[] = {"d_x", "d_h_prev", "d_c_bar", "d_r_bar_u_bar"};
constexpr std::array<std::array<int64_t, 3>, 3> CASES = {{{2, 3, 4}, {1, 7, 5}, {3, 5, 7}}};

inline Shapes InputShapes(const std::array<int64_t, 3>& dims)
{
    const auto b = dims[0], i = dims[1], c = dims[2];
    return {{b, i}, {b, c}, {i + c, 2 * c}, {i + c, c}, {2 * c}, {c}, {b, c}, {b, c}, {b, c}, {b, c}};
}
inline Shapes OutputShapes(const std::array<int64_t, 3>& dims)
{
    return {{dims[0], dims[1]}, {dims[0], dims[2]}, {dims[0], dims[2]}, {dims[0], 2 * dims[2]}};
}
inline size_t Count(const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto dim : shape)
        n *= static_cast<size_t>(dim);
    return n;
}
inline std::string ShapeString(const std::vector<int64_t>& shape)
{
    std::string out = "[";
    for (size_t j = 0; j < shape.size(); ++j)
        out += (j == 0 ? "" : ",") + std::to_string(shape[j]);
    return out + "]";
}
template <class T>
bool Read(const std::string& file, size_t count, std::vector<T>& values)
{
    values.resize(count);
    std::ifstream stream(file, std::ios::binary);
    if (!stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T))) ||
        stream.peek() != std::ifstream::traits_type::eof()) {
        std::cerr << "ERROR: invalid data file " << file << '\n';
        return false;
    }
    return true;
}
inline bool Verify(const std::vector<ge::Tensor>& outputs, size_t caseId, const std::string& dataDir)
{
    auto expectedShapes = OutputShapes(CASES[caseId]);
    if (outputs.size() != expectedShapes.size())
        return false;
    // Frozen user spec.yaml numerical_tolerance.float32 (max_relative).
    constexpr double rtol = 1.0e-5;
    constexpr double atol = 1.0e-6;
    for (size_t j = 0; j < outputs.size(); ++j) {
        const auto& output = outputs[j];
        if (output.GetTensorDesc().GetShape().GetDims() != expectedShapes[j] ||
            output.GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
            output.GetSize() != Count(expectedShapes[j]) * sizeof(float)) {
            std::cerr << "ERROR: output " << j << " shape/dtype/bytes mismatch\n";
            return false;
        }
        std::vector<double> golden;
        if (!Read(dataDir + "/case" + std::to_string(caseId) + "_golden" + std::to_string(j) + ".bin",
                  Count(expectedShapes[j]), golden))
            return false;
        for (size_t k = 0; k < golden.size(); ++k) {
            float value;
            std::memcpy(&value, output.GetData() + k * sizeof(float), sizeof(float));
            if (!std::isfinite(value) || !std::isfinite(golden[k]) ||
                std::abs(static_cast<double>(value) - golden[k]) > atol + rtol * std::abs(golden[k])) {
                std::cerr << "ERROR: output " << j << " index " << k << " actual=" << value << " golden=" << golden[k]
                          << '\n';
                return false;
            }
        }
    }
    return true;
}
inline bool RunScenario(ge::Session& session, uint32_t graphId, const char* scenario, int unknown,
                        const std::string& dataDir)
{
    ge::Graph graph(scenario);
    ge::op::GRUBlockCellGrad grad(scenario);
    auto declared = InputShapes(CASES[0]);
    auto declaredOutputs = OutputShapes(CASES[0]);
    for (auto& shape : declared) {
        if (unknown == -1)
            std::fill(shape.begin(), shape.end(), -1);
        if (unknown == -2)
            shape = {-2};
    }
    for (auto& shape : declaredOutputs) {
        if (unknown == -1)
            std::fill(shape.begin(), shape.end(), -1);
        if (unknown == -2)
            shape = {-2};
    }
    std::cout << "Scenario " << scenario << ", declared shape " << ShapeString(declared[0]) << '\n';
    std::vector<ge::Operator> inputs;
    for (size_t j = 0; j < declared.size(); ++j) {
        const std::string name = std::string(scenario) + "_" + INPUT_NAMES[j];
        auto data = ge::OperatorFactory::CreateOperator(name.c_str(), "Data");
        data.SetAttr("index", static_cast<int64_t>(j));
        ge::TensorDesc desc(ge::Shape(declared[j]), ge::FORMAT_ND, ge::DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        if (data.UpdateInputDesc("x", desc) != ge::GRAPH_SUCCESS ||
            data.UpdateOutputDesc("y", desc) != ge::GRAPH_SUCCESS ||
            grad.UpdateInputDesc(INPUT_NAMES[j], desc) != ge::GRAPH_SUCCESS)
            return false;
        grad.SetInput(INPUT_NAMES[j], data, "y");
        inputs.push_back(data);
    }
    for (size_t j = 0; j < declaredOutputs.size(); ++j) {
        ge::TensorDesc desc(ge::Shape(declaredOutputs[j]), ge::FORMAT_ND, ge::DT_FLOAT);
        if (grad.UpdateOutputDesc(OUTPUT_NAMES[j], desc) != ge::GRAPH_SUCCESS)
            return false;
    }
    graph.SetInputs(inputs).SetOutputs(std::vector<ge::Operator>{grad});
    std::map<ge::AscendString, ge::AscendString> graphOptions;
    if (session.AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        std::cerr << "ERROR: AddGraph " << scenario << " " << ge::GEGetErrorMsgV2().GetString() << '\n';
        return false;
    }
    std::cout << "AddGraph once graph_id=" << graphId << " session=" << &session << '\n';
    const size_t total = unknown == 0 ? 1 : CASES.size();
    for (size_t caseId = 0; caseId < total; ++caseId) {
        auto shapes = InputShapes(CASES[caseId]);
        std::vector<ge::Tensor> feeds;
        std::vector<std::vector<float>> storage(shapes.size());
        for (size_t j = 0; j < shapes.size(); ++j) {
            if (!Read(dataDir + "/case" + std::to_string(caseId) + "_input" + std::to_string(j) + ".bin",
                      Count(shapes[j]), storage[j]))
                return false;
            ge::TensorDesc desc(ge::Shape(shapes[j]), ge::FORMAT_ND, ge::DT_FLOAT);
            desc.SetPlacement(ge::kPlacementHost);
            feeds.emplace_back(desc, reinterpret_cast<const uint8_t*>(storage[j].data()),
                               storage[j].size() * sizeof(float));
        }
        std::cout << "Run concrete shape " << ShapeString(shapes[0]) << '\n';
        std::cout << "Execution B/I/C=" << CASES[caseId][0] << "/" << CASES[caseId][1] << "/" << CASES[caseId][2]
                  << " graph_id=" << graphId << " session=" << &session << '\n';
        std::vector<ge::Tensor> outputs;
        if (session.RunGraph(graphId, feeds, outputs) != ge::SUCCESS) {
            std::cerr << "ERROR: RunGraph " << scenario << " " << ge::GEGetErrorMsgV2().GetString() << '\n';
            return false;
        }
        if (!Verify(outputs, caseId, dataDir))
            return false;
        std::cout << "Shape, dtype and values PASSED for " << ShapeString(shapes[0]) << '\n';
    }
    std::cout << "Scenario " << scenario << " summary: " << total << "/" << total << " passed\n";
    return true;
}
inline int Main(int argc, char** argv, bool dynamic)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " DEVICE_ID DATA_DIR (generated with prepare_geir_data.py)\n";
        return 1;
    }
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", argv[1]}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS)
        return 1;
    bool success;
    {
        ge::Session session(options);
        if (dynamic) {
            success = RunScenario(session, 0, "unknown_dim_minus_1", -1, argv[2]) &&
                      RunScenario(session, 1, "unknown_rank_minus_2", -2, argv[2]);
            std::cout << "Rank coverage: fixed interface (8 rank-2 inputs, 2 rank-1 biases); three different legal "
                         "ranks are not applicable\n";
        } else {
            success = RunScenario(session, 0, "static", 0, argv[2]);
        }
    }
    success = ge::GEFinalize() == ge::SUCCESS && success;
    if (success)
        std::cout << "GRUBlockCellGrad " << (dynamic ? "dynamic" : "static") << " GEIR verification PASSED\n";
    return success ? 0 : 1;
}
} // namespace gru_example
#endif
