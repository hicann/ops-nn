/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"
#include "../op_graph/foreach_mul_list_proto.h"

namespace {
constexpr int kSuccess = 0;
constexpr int kFailed = 1;

bool CheckOutputs(const std::vector<ge::Tensor>& outputs, const std::vector<std::vector<int64_t>>& shapes,
                  const std::vector<std::vector<float>>& x1, const std::vector<std::vector<float>>& x2)
{
    if (outputs.size() != shapes.size()) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        if (output.GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
            output.GetTensorDesc().GetShape().GetDims() != shapes[i] ||
            output.GetSize() != x1[i].size() * sizeof(float) || output.GetData() == nullptr) {
            std::cerr << "Unexpected dtype, shape or byte size for output " << i << std::endl;
            return false;
        }
        std::cout << "y[" << i << "]:";
        for (size_t j = 0; j < x1[i].size(); ++j) {
            float actual;
            std::memcpy(&actual, output.GetData() + j * sizeof(float), sizeof(float));
            // These inputs and products are exactly representable in float32.
            const float expected = x1[i][j] * x2[i][j];
            if (actual != expected) {
                std::cerr << "Output mismatch at [" << i << "][" << j << "]: expected " << expected << ", got "
                          << actual << std::endl;
                return false;
            }
            std::cout << " " << actual;
        }
        std::cout << std::endl;
    }
    return true;
}

int RunExample()
{
    // Corresponding tensors have the same shape; different list entries may have different shapes.
    const std::vector<std::vector<int64_t>> shapes = {{2, 3}, {4}, {1, 2, 2}};
    const std::vector<std::vector<float>> x1 = {{1, -2, 0, 0.5F, -1.5F, 4}, {2, -3, 0.25F, 8}, {1, 2, -4, 0}};
    const std::vector<std::vector<float>> x2 = {{6, 5, -4, 3, -2, 0.5F}, {-2, -0.5F, 4, 0}, {0.5F, -2, -1, 8}};
    const uint32_t count = static_cast<uint32_t>(shapes.size());

    ge::Graph graph("foreach_mul_list_example");
    auto mul = ge::op::ForeachMulList("foreach_mul_list");
    mul.create_dynamic_input_x1(count);
    mul.create_dynamic_input_x2(count);
    mul.create_dynamic_output_y(count);

    std::vector<ge::Operator> inputNodes;
    std::vector<ge::Tensor> inputs;
    for (uint32_t list = 0; list < 2; ++list) {
        const auto& values = list == 0 ? x1 : x2;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t index = list * count + i;
            const std::string name = "input_" + std::to_string(index);
            auto data = ge::op::Data(name.c_str()).set_attr_index(index);
            ge::TensorDesc desc(ge::Shape(shapes[i]), ge::FORMAT_ND, ge::DT_FLOAT);
            desc.SetPlacement(ge::kPlacementHost);
            if (data.update_input_desc_x(desc) != ge::GRAPH_SUCCESS ||
                data.update_output_desc_y(desc) != ge::GRAPH_SUCCESS || graph.AddOp(data) != ge::GRAPH_SUCCESS) {
                std::cerr << "Cannot create input " << index << std::endl;
                return kFailed;
            }
            ge::graphStatus status;
            if (list == 0) {
                mul.set_dynamic_input_x1(i, data);
                status = mul.update_dynamic_input_desc_x1(i, desc);
            } else {
                mul.set_dynamic_input_x2(i, data);
                status = mul.update_dynamic_input_desc_x2(i, desc);
            }
            if (status != ge::GRAPH_SUCCESS) {
                std::cerr << "Cannot set input descriptor " << index << std::endl;
                return kFailed;
            }
            inputNodes.push_back(data);
            inputs.emplace_back(desc, reinterpret_cast<const uint8_t*>(values[i].data()),
                                values[i].size() * sizeof(float));
        }
    }

    std::vector<size_t> outputIndices;
    for (uint32_t i = 0; i < count; ++i) {
        const ge::TensorDesc desc(ge::Shape(shapes[i]), ge::FORMAT_ND, ge::DT_FLOAT);
        if (mul.update_dynamic_output_desc_y(i, desc) != ge::GRAPH_SUCCESS) {
            std::cerr << "Cannot set output descriptor " << i << std::endl;
            return kFailed;
        }
        outputIndices.push_back(i);
    }
    if (graph.AddOp(mul) != ge::GRAPH_SUCCESS) {
        std::cerr << "Cannot add ForeachMulList to the graph" << std::endl;
        return kFailed;
    }
    graph.SetInputs(inputNodes);
    graph.SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{{mul, outputIndices}});

    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    if (session.AddGraph(0, graph) != ge::GRAPH_SUCCESS) {
        std::cerr << "AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::GRAPH_SUCCESS) {
        std::cerr << "RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    return CheckOutputs(outputs, shapes, x1, x2) ? kSuccess : kFailed;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options{{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::GRAPH_SUCCESS) {
        std::cerr << "GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    int result = kFailed;
    try {
        result = RunExample();
    } catch (const std::exception& error) {
        std::cerr << "Example failed: " << error.what() << std::endl;
    }
    // Destroy the Session in RunExample before finalizing GE, including on failures.
    if (ge::GEFinalize() != ge::GRAPH_SUCCESS) {
        std::cerr << "GEFinalize failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    if (result == kSuccess) {
        std::cout << "ForeachMulList graph example passed: all 3 output tensors (14 values) match" << std::endl;
    }
    return result;
}
