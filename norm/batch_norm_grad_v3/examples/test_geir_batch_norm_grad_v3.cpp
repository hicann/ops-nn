/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend 950/960, with the CANN environment sourced and BatchNormGradV3 installed.
// Build from this directory:
// g++ -std=c++17 test_geir_batch_norm_grad_v3.cpp -o test_geir_batch_norm_grad_v3 \
//     -I$ASCEND_HOME_PATH/include -L$ASCEND_HOME_PATH/lib64 \
//     -I$ASCEND_HOME_PATH/opp/built-in/op_graph/inc \
//     -lgraph -lgraph_base -lge_runner -lascendcl
// Run on device 0: ./test_geir_batch_norm_grad_v3 [--infer | --empty-infer]
// Training checks all three gradients. Nonempty inference checks dx.
// Empty inference checks that GE infers all three output shapes as empty.

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "ops_proto_legacy.h"
#include "../op_graph/batch_norm_grad_v3_proto.h"

namespace {
constexpr float EPSILON = 1e-5f;
constexpr float TOLERANCE = 1e-4f;
const std::vector<int64_t> EMPTY_INPUT_SHAPE = {0, 3, 2, 2};

bool CheckOutput(const ge::Tensor& output, const std::vector<float>& expected, const char* name)
{
    if (output.GetTensorDesc().GetDataType() != ge::DT_FLOAT || output.GetSize() != expected.size() * sizeof(float) ||
        output.GetData() == nullptr) {
        std::cerr << name << ": unexpected output dtype or size\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        float value = 0;
        std::memcpy(&value, output.GetData() + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value) || std::abs(value - expected[i]) > TOLERANCE * (1.0f + std::abs(expected[i]))) {
            std::cerr << name << '[' << i << "] expected " << expected[i] << ", got " << value << '\n';
            return false;
        }
    }
    std::cout << "PASS " << name << '\n';
    return true;
}

bool CheckEmptyOutput(const ge::Tensor& output, const std::vector<int64_t>& shape, const char* name)
{
    const auto desc = output.GetTensorDesc();
    if (desc.GetDataType() != ge::DT_FLOAT || desc.GetShape().GetDims() != shape || output.GetSize() != 0) {
        std::cerr << name << ": unexpected empty output shape, dtype, or size\n";
        return false;
    }
    std::cout << "PASS empty " << name << '\n';
    return true;
}

bool BuildGraph(bool training, bool emptyInference, ge::Graph& graph, std::vector<ge::Tensor>& inputs)
{
    // Use the supplied mean=0, inverse standard deviation=1 to make the training
    // reference hand-checkable. These are inputs, not recomputed from x.
    const auto xShape = emptyInference ? EMPTY_INPUT_SHAPE : std::vector<int64_t>{1, 2, 2, 2};
    const std::vector<int64_t> channelShape = {xShape[1]};
    const auto channels = static_cast<size_t>(xShape[1]);
    const auto dataValues = emptyInference ? std::vector<float>{} : std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<float> ones(channels, 1.0f);
    const std::vector<float> zeros(channels, 0.0f);
    const std::vector<std::vector<float>> values = {dataValues, dataValues, ones, zeros, ones, zeros, ones};
    const float emptyData = 0.0f;
    const std::vector<const char*> names = {"dy",          "x",         "weight",   "running_mean",
                                            "running_var", "save_mean", "save_rstd"};
    ge::op::BatchNormGradV3 op("batch_norm_grad_v3");
    std::vector<ge::Operator> dataOps;
    for (size_t i = 0; i < names.size(); ++i) {
        const auto& shape = i < 2 ? xShape : channelShape;
        ge::TensorDesc desc(ge::Shape(shape), i < 2 ? ge::FORMAT_NCHW : ge::FORMAT_ND, ge::DT_FLOAT);
        desc.SetRealDimCnt(shape.size());
        desc.SetPlacement(ge::kPlacementHost);
        const auto dataName = std::string("data_") + names[i];
        auto data = ge::op::Data(dataName.c_str()).set_attr_index(static_cast<int64_t>(i));
        if (data.UpdateInputDesc("x", desc) != ge::GRAPH_SUCCESS ||
            data.UpdateOutputDesc("y", desc) != ge::GRAPH_SUCCESS ||
            op.UpdateInputDesc(names[i], desc) != ge::GRAPH_SUCCESS || graph.AddOp(data) != ge::GRAPH_SUCCESS) {
            std::cerr << "Failed to create input " << names[i] << '\n';
            return false;
        }
        op.SetInput(names[i], data);
        const auto* inputData = values[i].empty() ? &emptyData : values[i].data();
        inputs.emplace_back(desc, reinterpret_cast<const uint8_t*>(inputData), values[i].size() * sizeof(float));
        dataOps.push_back(data);
    }
    const bool allOutputs = training || emptyInference;
    op.set_attr_is_training(training).set_attr_epsilon(EPSILON).set_attr_output_mask({true, allOutputs, allOutputs});
    // Let InferShape determine the output shapes for the empty graph.
    const auto dxShape = emptyInference ? std::vector<int64_t>(xShape.size(), -1) : xShape;
    const auto paramShape = emptyInference ? std::vector<int64_t>{-1} : channelShape;
    const ge::TensorDesc dxDesc(ge::Shape(dxShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    const ge::TensorDesc paramDesc(ge::Shape(paramShape), ge::FORMAT_ND, ge::DT_FLOAT);
    if (op.update_output_desc_dx(dxDesc) != ge::GRAPH_SUCCESS ||
        op.update_output_desc_dweight(paramDesc) != ge::GRAPH_SUCCESS ||
        op.update_output_desc_dbias(paramDesc) != ge::GRAPH_SUCCESS || graph.AddOp(op) != ge::GRAPH_SUCCESS) {
        std::cerr << "Failed to set output descriptors\n";
        return false;
    }
    graph.SetInputs(dataOps);
    graph.SetOutputs(std::vector<std::pair<ge::Operator, std::vector<size_t>>>{
        {op, allOutputs ? std::vector<size_t>{0, 1, 2} : std::vector<size_t>{0}}});
    return true;
}

bool RunExample(bool training, bool emptyInference)
{
    ge::Graph graph("batch_norm_grad_v3_example");
    std::vector<ge::Tensor> inputs;
    if (!BuildGraph(training, emptyInference, graph, inputs)) {
        return false;
    }
    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    auto status = session.AddGraph(0, graph);
    if (status != ge::SUCCESS) {
        std::cerr << "AddGraph failed: " << status << '\n';
        return false;
    }
    std::vector<ge::Tensor> outputs;
    status = session.RunGraph(0, inputs, outputs);
    const size_t expectedOutputCount = (training || emptyInference) ? 3U : 1U;
    if (status != ge::SUCCESS || outputs.size() != expectedOutputCount) {
        std::cerr << "RunGraph failed or returned an unexpected output count: " << status << '\n';
        return false;
    }
    if (emptyInference) {
        const bool dxOk = CheckEmptyOutput(outputs[0], EMPTY_INPUT_SHAPE, "dx");
        const bool weightOk = CheckEmptyOutput(outputs[1], {0}, "dweight");
        const bool biasOk = CheckEmptyOutput(outputs[2], {0}, "dbias");
        return dxOk && weightOk && biasOk;
    }
    // For each channel: dbias=sum(dy), dweight=sum(dy*x),
    // dx=dy-(dbias+x*dweight)/4 when weight=invstd=1 and save_mean=0.
    std::vector<float> expectedDx = {-1.5f, -4.0f, -6.5f, -9.0f, -127.5f, -158.0f, -188.5f, -219.0f};
    if (!training) {
        expectedDx = {0, 1, 2, 3, 4, 5, 6, 7};
        for (auto& value : expectedDx) {
            value /= std::sqrt(1.0f + EPSILON);
        }
    }
    const bool dxOk = CheckOutput(outputs[0], expectedDx, "dx");
    if (!training) {
        return dxOk;
    }
    const bool weightOk = CheckOutput(outputs[1], {14.0f, 126.0f}, "dweight");
    const bool biasOk = CheckOutput(outputs[2], {6.0f, 22.0f}, "dbias");
    return dxOk && weightOk && biasOk;
}
} // namespace

int main(int argc, char* argv[])
{
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc > 2 || (argc == 2 && mode != "--infer" && mode != "--empty-infer")) {
        std::cerr << "Usage: " << argv[0] << " [--infer | --empty-infer]\n";
        return 1;
    }
    const bool training = argc == 1;
    const bool emptyInference = mode == "--empty-infer";
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    const auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        std::cerr << "GEInitialize failed: " << status << '\n';
        return 1;
    }
    // RunExample destroys the Session before GEFinalize, including failure paths.
    const bool passed = RunExample(training, emptyInference);
    const auto finalizeStatus = ge::GEFinalize();
    if (!passed || finalizeStatus != ge::SUCCESS) {
        return 1;
    }
    const char* scenario = emptyInference ? "empty inference" : (training ? "training" : "inference");
    std::cout << "PASS BatchNormGradV3 GEIR " << scenario << '\n';
    return 0;
}
