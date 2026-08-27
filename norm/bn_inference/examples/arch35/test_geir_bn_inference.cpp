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
#include <iostream>
#include <map>
#include <memory>
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
#include "../../op_graph/bn_inference_proto.h"

namespace {
constexpr int32_t kSuccess = 0;
constexpr int32_t kFailed = -1;

struct FormatConfig {
    ge::Format format = ge::FORMAT_RESERVED;
    std::vector<int64_t> shape;
    int64_t channels = 0;
};

bool ResolveFormat(const std::string& name, FormatConfig& config)
{
    if (name == "NCHW") {
        config = {ge::FORMAT_NCHW, {2, 3, 4, 5}, 3};
        return true;
    }
    if (name == "NHWC") {
        config = {ge::FORMAT_NHWC, {2, 4, 5, 3}, 3};
        return true;
    }
    if (name == "NCDHW") {
        config = {ge::FORMAT_NCDHW, {2, 3, 2, 4, 5}, 3};
        return true;
    }
    if (name == "NDHWC") {
        config = {ge::FORMAT_NDHWC, {2, 2, 4, 5, 3}, 3};
        return true;
    }
    if (name == "ND4") {
        config = {ge::FORMAT_ND, {2, 3, 4, 5}, 3};
        return true;
    }
    if (name == "ND5") {
        config = {ge::FORMAT_ND, {2, 3, 2, 4, 5}, 3};
        return true;
    }
    return false;
}

int64_t ElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (const int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

bool MakeHostTensor(const std::vector<int64_t>& shape, ge::Format format, float base, ge::TensorDesc& desc,
                    ge::Tensor& tensor)
{
    desc = ge::TensorDesc(ge::Shape(shape), format, ge::DT_FLOAT);
    desc.SetRealDimCnt(shape.size());
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(format);
    desc.SetOriginFormat(format);

    const int64_t count = ElementCount(shape);
    std::vector<float> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = base + static_cast<float>(i % 17) * 0.01F;
    }
    tensor = ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
    return true;
}

ge::op::Data AddInput(ge::Graph& graph, std::vector<ge::Operator>& graphInputs, std::vector<ge::Tensor>& tensors,
                      const std::string& name, int64_t index, const std::vector<int64_t>& shape, ge::Format format,
                      float base)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    ge::TensorDesc desc;
    ge::Tensor tensor;
    (void)MakeHostTensor(shape, format, base, desc, tensor);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    graph.AddOp(data);
    graphInputs.emplace_back(data);
    tensors.emplace_back(tensor);
    return data;
}

bool ParseOptionalMask(const std::string& value, bool& hasScale, bool& hasOffset)
{
    hasScale = value == "scale" || value == "both";
    hasOffset = value == "offset" || value == "both";
    return value == "none" || value == "scale" || value == "offset" || value == "both";
}
} // namespace

int main(int argc, char* argv[])
{
    const std::string formatName = argc > 1 ? argv[1] : "NCHW";
    const std::string optionalMask = argc > 2 ? argv[2] : "both";
    FormatConfig config;
    bool hasScale = false;
    bool hasOffset = false;
    if (!ResolveFormat(formatName, config) || !ParseOptionalMask(optionalMask, hasScale, hasOffset)) {
        std::cerr << "Usage: " << argv[0] << " [NCHW|NHWC|NCDHW|NDHWC|ND4|ND5] [none|scale|offset|both]" << std::endl;
        return kFailed;
    }

    const std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                        {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("test_geir_bn_inference");
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Tensor> inputTensors;

    auto x = AddInput(graph, graphInputs, inputTensors, "x", 0, config.shape, config.format, -0.5F);
    auto mean = AddInput(graph, graphInputs, inputTensors, "mean", 1, {config.channels}, ge::FORMAT_ND, 0.1F);
    auto variance = AddInput(graph, graphInputs, inputTensors, "variance", 2, {config.channels}, ge::FORMAT_ND, 1.0F);
    // Empty shape means a rank-0 scalar, not shape [1].
    auto momentum = AddInput(graph, graphInputs, inputTensors, "momentum", 3, {}, ge::FORMAT_ND, 0.9F);

    auto bn = ge::op::BNInference("bn_inference");
    bn.set_input_x(x);
    bn.set_input_mean(mean);
    bn.set_input_variance(variance);
    bn.set_input_momentum(momentum);

    if (hasScale) {
        auto scale = AddInput(graph, graphInputs, inputTensors, "scale", 4, {config.channels}, ge::FORMAT_ND, 1.0F);
        bn.set_input_scale(scale);
    }
    if (hasOffset) {
        const int64_t offsetIndex = hasScale ? 5 : 4;
        auto offset = AddInput(graph, graphInputs, inputTensors, "offset", offsetIndex, {config.channels},
                               ge::FORMAT_ND, 0.2F);
        bn.set_input_offset(offset);
    }
    bn.set_attr_epsilon(1e-5F);
    bn.set_attr_use_global_stats(true);
    bn.set_attr_mode(1);

    ge::TensorDesc xDesc(ge::Shape(config.shape), config.format, ge::DT_FLOAT);
    xDesc.SetOriginFormat(config.format);
    bn.update_output_desc_y(xDesc);
    graph.SetInputs(graphInputs).SetOutputs({bn});

    const std::map<ge::AscendString, ge::AscendString> sessionOptions;
    const std::map<ge::AscendString, ge::AscendString> graphOptions;
    std::unique_ptr<ge::Session> session(new (std::nothrow) ge::Session(sessionOptions));
    if (session == nullptr || session->AddGraph(0, graph, graphOptions) != ge::SUCCESS) {
        std::cerr << "AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        (void)ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    const ge::Status status = session->RunGraph(0, inputTensors, outputs);
    if (status != ge::SUCCESS || outputs.size() != 1U) {
        std::cerr << "RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        session.reset();
        (void)ge::GEFinalize();
        return kFailed;
    }

    std::cout << "BNInference GEIR succeeded: format=" << formatName << ", optional=" << optionalMask
              << ", outputElements=" << outputs[0].GetTensorDesc().GetShape().GetShapeSize() << std::endl;
    session.reset();
    return ge::GEFinalize() == ge::SUCCESS ? kSuccess : kFailed;
}
