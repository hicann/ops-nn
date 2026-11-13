/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 */
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "ops_proto_math.h"
#include "../op_graph/act_ulq_clamp_min_grad_proto.h"

namespace {
struct GeFinalizer {
    ~GeFinalizer() { ge::GEFinalize(); }
};
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed\n";
        return 1;
    }
    GeFinalizer finalizer;
    // Session is destroyed before GEFinalize on every return path.
    ge::Session session(
        std::map<ge::AscendString, ge::AscendString>{{"ge.exec.placement", "DEVICE"}, {"ge.jit_compile", "0"}});
    ge::Graph graph("act_ulq_clamp_min_grad_example");
    ge::TensorDesc inputDesc(ge::Shape({4, 2}), ge::FORMAT_ND, ge::DT_FLOAT);
    inputDesc.SetPlacement(ge::kPlacementHost);
    const std::array<std::vector<float>, 3> values = {std::vector<float>(8, 2.0f), std::vector<float>(8, -2.0f),
                                                      std::vector<float>(8, 3.0f)};
    std::vector<ge::Operator> data;
    std::vector<ge::Tensor> inputs;
    for (int index = 0; index < 3; ++index) {
        const std::string name = "input_" + std::to_string(index);
        auto node = ge::op::Data(name.c_str()).set_attr_index(index);
        node.update_input_desc_x(inputDesc);
        node.update_output_desc_y(inputDesc);
        graph.AddOp(node);
        data.push_back(node);
        inputs.emplace_back(inputDesc, reinterpret_cast<const uint8_t*>(values[index].data()),
                            values[index].size() * sizeof(float));
    }
    auto op = ge::op::ActULQClampMinGrad("act_ulq_clamp_min_grad");
    op.update_input_desc_y_grad(inputDesc);
    op.update_input_desc_clamp_min_mask(inputDesc);
    op.update_input_desc_x_clamped_loss(inputDesc);
    op.set_input_y_grad(data[0]);
    op.set_input_clamp_min_mask(data[1]);
    op.set_input_x_clamped_loss(data[2]);
    op.update_output_desc_clamp_min_grad(
        ge::TensorDesc(ge::Shape(std::vector<int64_t>{}), ge::FORMAT_ND, ge::DT_FLOAT));
    graph.AddOp(op);
    graph.SetInputs(data).SetOutputs({op});
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed\n";
        return 1;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << "\n";
        return 1;
    }
    if (outputs.size() != 1 || outputs[0].GetSize() != sizeof(float) ||
        outputs[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT) {
        std::cerr << "Unexpected output descriptor\n";
        return 1;
    }
    double expected = 0;
    for (size_t index = 0; index < values[0].size(); ++index) {
        expected += static_cast<double>(values[0][index]) *
                    (values[2][index] + std::abs(static_cast<double>(values[1][index])));
    }
    float actual;
    std::memcpy(&actual, outputs[0].GetData(), sizeof(actual));
    std::cout << "expected=" << expected << " actual=" << actual << '\n';
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5 * std::abs(expected)) {
        std::cerr << "Precision check failed\n";
        return 1;
    }
    std::cout << "Precision is ok\n";
    return 0;
}
