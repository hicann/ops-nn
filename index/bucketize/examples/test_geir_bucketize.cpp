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

#include "../op_graph/bucketize_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr int32_t kExpectedBucket = 1;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

ge::Tensor BuildInt32Tensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT32);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int32_t));
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    auto data = ge::op::Data("x").set_attr_index(0);
    const std::vector<int64_t> inputShape = {6};
    const std::vector<int32_t> inputValues = {0, 2, 4, 6, 8, 10};
    ge::Tensor inputTensor = BuildInt32Tensor(inputShape, inputValues);
    ge::TensorDesc inputDesc = inputTensor.GetTensorDesc();
    data.update_input_desc_x(inputDesc);
    data.update_output_desc_y(inputDesc);
    graph.AddOp(data);

    auto bucketize = ge::op::Bucketize("bucketize");
    bucketize.set_input_x(data);
    bucketize.set_attr_boundaries({1.0f, 3.0f, 5.0f, 7.0f});
    bucketize.set_attr_dtype(ge::DT_INT32);
    bucketize.set_attr_right(false);

    inputs.push_back(inputTensor);
    graphInputs.push_back(data);
    graphOutputs.push_back(bucketize);
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
    if (elemCount != 6) {
        std::cerr << "Unexpected output element count: " << elemCount << std::endl;
        return kFailed;
    }

    const auto* data = reinterpret_cast<const int32_t*>(tensor.GetData());
    const int32_t expected[] = {0, 1, 2, 3, 4, 4};
    for (int64_t i = 0; i < elemCount; ++i) {
        std::cout << "bucketize output[" << i << "] = " << data[i] << std::endl;
        if (data[i] != expected[i]) {
            std::cerr << "Unexpected bucketize result at index " << i << ": " << data[i] << ", expected " << expected[i]
                      << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start Bucketize GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("bucketize_graph");
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
    std::cout << GetTime() << " - INFO - Bucketize GEIR example success" << std::endl;
    return kSuccess;
}
