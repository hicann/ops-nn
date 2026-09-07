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
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include "array_ops.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/centralization_proto.h"

namespace {
constexpr uint32_t kDeviceId = 0;

ge::Tensor BuildInput(const std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape({2, 3, 4}), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}

bool CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        return false;
    }
    const auto& output = outputs[0];
    const auto* values = reinterpret_cast<const float*>(output.GetData());
    const std::vector<float> expected = {
        -4.0f, -4.0f, -4.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f, 4.0f,
        -4.0f, -4.0f, -4.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 4.0f, 4.0f, 4.0f,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 1e-5f) {
            std::cerr << "Unexpected output at " << i << ": " << values[i] << std::endl;
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    const std::vector<float> values = {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    };
    ge::Tensor input = BuildInput(values);
    ge::TensorDesc desc = input.GetTensorDesc();
    auto data = ge::op::Data("x_data").set_attr_index(0);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    auto op = ge::op::Centralization("centralization");
    op.set_input_x(data).set_attr_axes({1});
    op.update_output_desc_y(desc);
    ge::Graph graph("centralization_graph");
    graph.AddOp(data);
    graph.AddOp(op);
    graph.SetInputs({data}).SetOutputs({op});

    std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        return 1;
    }
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        ge::GEFinalize();
        return 1;
    }
    std::vector<ge::Tensor> outputs;
    const auto status = session.RunGraph(0, {input}, outputs);
    const bool passed = status == ge::SUCCESS && CheckOutput(outputs);
    ge::GEFinalize();
    return passed ? 0 : 1;
}
