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
 * @file test_geir_in_training_update_v2.cpp
 * @brief Minimal INTrainingUpdateV2 GE graph example for Ascend 950.
 */

#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/in_training_update_v2_proto.h"

namespace {
constexpr int32_t SUCCESS_CODE = 0;
constexpr int32_t FAILED_CODE = -1;

using ge::AscendString;
using ge::Graph;
using ge::Operator;
using ge::Session;
using ge::Status;
using ge::Tensor;
using ge::TensorDesc;

struct HostTensor {
    Tensor tensor;
    std::unique_ptr<float[]> data;
};

HostTensor MakeTensor(const std::vector<int64_t>& shape, const std::vector<float>& values)
{
    size_t count = 1;
    for (int64_t dim : shape) {
        count *= static_cast<size_t>(dim);
    }
    HostTensor result;
    result.data = std::make_unique<float[]>(count);
    for (size_t index = 0; index < count; ++index) {
        result.data[index] = values[index % values.size()];
    }
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    desc.SetRealDimCnt(shape.size());
    desc.SetPlacement(ge::kPlacementHost);
    result.tensor = Tensor(desc, reinterpret_cast<uint8_t*>(result.data.get()), count * sizeof(float));
    return result;
}

bool CheckAll(const Tensor& tensor, const std::vector<float>& expected, float tolerance)
{
    const int64_t count = tensor.GetTensorDesc().GetShape().GetShapeSize();
    const auto* data = reinterpret_cast<const float*>(tensor.GetData());
    if (data == nullptr || count <= 0) {
        return false;
    }
    for (int64_t index = 0; index < count; ++index) {
        const float target = expected[static_cast<size_t>(index) % expected.size()];
        if (!std::isfinite(data[index]) || !std::isfinite(target) || std::fabs(data[index] - target) > tolerance) {
            std::printf("mismatch at %" PRId64 ": actual=%f expected=%f\n", index, data[index], target);
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("GEInitialize failed\n");
        return FAILED_CODE;
    }

    Graph graph("in_training_update_v2_geir");
    auto update = ge::op::INTrainingUpdateV2("in_training_update_v2");
    update.set_attr_momentum(0.1f).set_attr_epsilon(1.0e-5f);

    const std::vector<int64_t> xShape = {1, 2, 2, 2};
    const std::vector<int64_t> statShape = {1, 2, 1, 1};
    std::vector<HostTensor> hostTensors;
    hostTensors.emplace_back(MakeTensor(xShape, {-1.0f, 1.0f, -1.0f, 1.0f}));
    hostTensors.emplace_back(MakeTensor(statShape, {0.0f}));
    hostTensors.emplace_back(MakeTensor(statShape, {4.0f}));
    hostTensors.emplace_back(MakeTensor(statShape, {2.0f}));
    hostTensors.emplace_back(MakeTensor(statShape, {0.5f}));
    hostTensors.emplace_back(MakeTensor(statShape, {0.25f}));
    hostTensors.emplace_back(MakeTensor(statShape, {1.0f}));

    std::vector<Operator> graphInputs;
    std::vector<Tensor> inputs;
    for (size_t index = 0; index < hostTensors.size(); ++index) {
        auto data = ge::op::Data("input_" + std::to_string(index)).set_attr_index(static_cast<int64_t>(index));
        const TensorDesc& desc = hostTensors[index].tensor.GetTensorDesc();
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        if (graph.AddOp(data) != ge::GRAPH_SUCCESS) {
            std::printf("AddOp failed for input %zu\n", index);
            if (ge::GEFinalize() != ge::SUCCESS) {
                std::printf("GEFinalize failed after AddOp failure\n");
            }
            return FAILED_CODE;
        }
        graphInputs.push_back(data);
        inputs.push_back(hostTensors[index].tensor);
        switch (index) {
            case 0:
                update.set_input_x(data);
                break;
            case 1:
                update.set_input_sum(data);
                break;
            case 2:
                update.set_input_square_sum(data);
                break;
            case 3:
                update.set_input_gamma(data);
                break;
            case 4:
                update.set_input_beta(data);
                break;
            case 5:
                update.set_input_mean(data);
                break;
            default:
                update.set_input_variance(data);
                break;
        }
    }

    update.update_output_desc_y(TensorDesc(ge::Shape(xShape), ge::FORMAT_NCHW, ge::DT_FLOAT));
    update.update_output_desc_batch_mean(TensorDesc(ge::Shape(statShape), ge::FORMAT_NCHW, ge::DT_FLOAT));
    update.update_output_desc_batch_variance(TensorDesc(ge::Shape(statShape), ge::FORMAT_NCHW, ge::DT_FLOAT));
    graph.SetInputs(graphInputs).SetOutputs({update});

    Session session(std::map<AscendString, AscendString>{});
    Status status = session.AddGraph(0, graph, std::map<AscendString, AscendString>{});
    std::vector<Tensor> outputs;
    if (status == ge::SUCCESS) {
        status = session.RunGraph(0, inputs, outputs);
    }

    bool passed = status == ge::SUCCESS && outputs.size() == 3;
    if (passed) {
        const float scale = 2.0f / std::sqrt(1.0f + 1.0e-5f);
        passed = CheckAll(outputs[0], {0.5f - scale, 0.5f + scale, 0.5f - scale, 0.5f + scale}, 1.0e-4f);
        passed = passed && CheckAll(outputs[1], {0.225f}, 1.0e-6f);
        passed = passed && CheckAll(outputs[2], {1.0333333f}, 1.0e-5f);
    }

    const Status finalizeStatus = ge::GEFinalize();
    passed = passed && finalizeStatus == ge::SUCCESS;
    std::printf("INTrainingUpdateV2 GEIR example: %s\n", passed ? "PASS" : "FAIL");
    return passed ? SUCCESS_CODE : FAILED_CODE;
}
