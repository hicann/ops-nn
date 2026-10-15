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
 * @file test_geir_rms_norm_quant_v2.cpp
 * @brief RmsNormQuantV2 GE graph example with two INT8 outputs on Ascend 950.
 */

#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "ops_proto_legacy.h"
#include "tensor.h"

#include "../op_graph/rms_norm_quant_v2_proto.h"

namespace {
constexpr int32_t SUCCESS_CODE = 0;
constexpr int32_t FAILED_CODE = 1;

ge::Tensor MakeTensor(const std::vector<int64_t>& shape, const std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    const auto* bytes = static_cast<const uint8_t*>(static_cast<const void*>(values.data()));
    return ge::Tensor(desc, bytes, values.size() * sizeof(float));
}

bool CheckOutput(const ge::Tensor& tensor, const std::vector<int64_t>& shape, uint8_t expected)
{
    const auto& desc = tensor.GetTensorDesc();
    if (desc.GetShape().GetDims() != shape || desc.GetDataType() != ge::DT_INT8 || tensor.GetData() == nullptr ||
        tensor.GetSize() != static_cast<size_t>(desc.GetShape().GetShapeSize())) {
        std::printf("Unexpected output shape, type or data size\n");
        return false;
    }
    for (size_t index = 0; index < tensor.GetSize(); ++index) {
        if (tensor.GetData()[index] != expected) {
            std::printf("Output mismatch at %zu: got %u, expected %u\n", index,
                        static_cast<unsigned int>(tensor.GetData()[index]), static_cast<unsigned int>(expected));
            return false;
        }
    }
    return true;
}

int32_t RunExample()
{
    const std::vector<int64_t> xShape = {2, 16};
    // x and gamma are all ones. Division by scales1=0.5 and scales2=0.25 gives y1=2 and y2=4.
    const std::vector<ge::Tensor> inputs = {MakeTensor(xShape, std::vector<float>(32, 1.0f)),
                                            MakeTensor({16}, std::vector<float>(16, 1.0f)), MakeTensor({1}, {0.5f}),
                                            MakeTensor({1}, {0.25f})};
    std::vector<ge::Operator> dataNodes;
    const char* const inputNames[] = {"x", "gamma", "scales1", "scales2"};
    ge::Graph graph("rms_norm_quant_v2_example");
    for (size_t index = 0; index < inputs.size(); ++index) {
        auto data = ge::op::Data(inputNames[index]).set_attr_index(static_cast<int64_t>(index));
        const auto& desc = inputs[index].GetTensorDesc();
        if (data.update_input_desc_x(desc) != ge::GRAPH_SUCCESS ||
            data.update_output_desc_y(desc) != ge::GRAPH_SUCCESS || graph.AddOp(data) != ge::GRAPH_SUCCESS) {
            std::printf("Failed to create input %s\n", inputNames[index]);
            return FAILED_CODE;
        }
        dataNodes.push_back(data);
    }

    auto quant = ge::op::RmsNormQuantV2("rms_norm_quant_v2");
    quant.set_input_x(dataNodes[0])
        .set_input_gamma(dataNodes[1])
        .set_input_scales1(dataNodes[2])
        .set_input_scales2(dataNodes[3])
        .set_attr_epsilon(1.0e-6f)
        .set_attr_div_mode(true)
        .set_attr_dst_type(ge::DT_INT8);
    // zero_points1, zero_points2 and beta are omitted. Both outputs are valid because scales2 is present.
    const ge::TensorDesc outputDesc(ge::Shape(xShape), ge::FORMAT_ND, ge::DT_INT8);
    if (quant.update_output_desc_y1(outputDesc) != ge::GRAPH_SUCCESS ||
        quant.update_output_desc_y2(outputDesc) != ge::GRAPH_SUCCESS || graph.AddOp(quant) != ge::GRAPH_SUCCESS) {
        std::printf("Failed to create RmsNormQuantV2 node\n");
        return FAILED_CODE;
    }
    graph.SetInputs(dataNodes).SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{{quant, {0, 1}}});

    const std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::printf("AddGraph failed\n");
        return FAILED_CODE;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::printf("RunGraph failed\n");
        return FAILED_CODE;
    }
    if (outputs.size() != 2 || !CheckOutput(outputs[0], xShape, 2) || !CheckOutput(outputs[1], xShape, 4)) {
        std::printf("RmsNormQuantV2 output verification failed\n");
        return FAILED_CODE;
    }
    std::printf("RmsNormQuantV2 PASS: y1=[2,...], y2=[4,...], both shapes=[2,16]\n");
    return SUCCESS_CODE;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::printf("GEInitialize failed\n");
        return FAILED_CODE;
    }
    const int32_t result = RunExample();
    // RunExample destroys the session before the global GE resources are released.
    if (ge::GEFinalize() != ge::SUCCESS) {
        std::printf("GEFinalize failed\n");
        return FAILED_CODE;
    }
    return result;
}
