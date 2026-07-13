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
#include <ctime>
#include <iostream>
#include <map>
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

#include "../op_graph/avg_pool1d_avg_matrix_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

ge::Tensor BuildFloatTensor(const std::vector<int64_t>& shape, const std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    auto data = ge::op::Data("x").set_attr_index(0);
    const std::vector<int64_t> inputShape = {1, 1, 1, 4};
    const std::vector<float> inputValues = {1.f, 2.f, 3.f, 4.f};
    ge::Tensor inputTensor = BuildFloatTensor(inputShape, inputValues);
    ge::TensorDesc inputDesc = inputTensor.GetTensorDesc();
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    graph.AddOp(data);

    auto avgPool = ge::op::AvgPool1DAvgMatrix("avg_pool1d_avg_matrix");
    avgPool.set_input_x(data);
    avgPool.set_attr_ksize(2);
    avgPool.set_attr_strides(2);
    avgPool.set_attr_pads({0, 0});
    avgPool.set_attr_ceil_mode(false);
    avgPool.set_attr_count_include_pad(false);

    inputs.push_back(inputTensor);
    graphInputs.push_back(data);
    graphOutputs.push_back(avgPool);
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto& tensor = outputs[0];
    const int64_t elemCount = tensor.GetTensorDesc().GetShape().GetShapeSize();
    if (elemCount != 32) {
        std::cerr << "Unexpected output element count: " << elemCount << std::endl;
        return kFailed;
    }

    const auto* data = reinterpret_cast<const float*>(tensor.GetData());
    for (int64_t i = 0; i < elemCount; ++i) {
        std::cout << "avg_pool1d_avg_matrix output[" << i << "] = " << data[i] << std::endl;
        if (data[i] != 0.5f) {
            std::cerr << "Unexpected avg_pool1d_avg_matrix result at index " << i << ": " << data[i] << ", expected 0.5"
                      << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start AvgPool1DAvgMatrix GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("avg_pool1d_avg_matrix_graph");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    if (CreateGraph(inputs, graphInputs, graphOutputs, graph) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    const int ret = CheckOutput(outputs);
    ge::GEFinalize();
    if (ret != kSuccess) {
        return kFailed;
    }
    std::cout << GetTime() << " - INFO - AvgPool1DAvgMatrix GEIR example success" << std::endl;
    return kSuccess;
}
