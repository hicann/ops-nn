/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_bn_infer.cpp
 * \brief GE IR example for BNInfer.
 */

#include <ctime>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
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

#include "../op_graph/bn_infer_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int32_t RET_FAILED = -1;
constexpr int32_t RET_SUCCESS = 0;
std::vector<std::unique_ptr<uint8_t[]>> g_inputHolder;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

bool GetShapeSize(const vector<int64_t>& shape, size_t& size)
{
    size = 1;
    for (auto dim : shape) {
        if (dim < 0 || (size != 0 && static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / size)) {
            return false;
        }
        size *= static_cast<size_t>(dim);
    }
    return true;
}

int32_t GenFloatData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, float baseValue)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 0;
    if (!GetShapeSize(shape, size) || size > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return RET_FAILED;
    }
    size_t dataSize = size * sizeof(float);
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[dataSize]);
    if (data == nullptr) {
        return RET_FAILED;
    }
    auto typedData = reinterpret_cast<float*>(data.get());
    for (size_t i = 0; i < size; ++i) {
        typedData[i] = baseValue + static_cast<float>(i % 7) * 0.125f;
    }
    tensor = Tensor(desc, data.get(), dataSize);
    g_inputHolder.emplace_back(std::move(data));
    return RET_SUCCESS;
}

int32_t AddFloatInput(Graph& graph, vector<Tensor>& input, vector<Operator>& inputs, const string& name, uint32_t index,
                      const vector<int64_t>& shape, Tensor& tensor, op::Data& dataOp, float baseValue)
{
    dataOp = op::Data(name.c_str()).set_attr_index(index);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    if (GenFloatData(shape, tensor, desc, baseValue) != RET_SUCCESS) {
        return RET_FAILED;
    }

    dataOp.update_input_desc_x(desc);
    dataOp.update_output_desc_y(desc);
    input.push_back(tensor);
    graph.AddOp(dataOp);
    inputs.push_back(dataOp);
    return RET_SUCCESS;
}

int32_t CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto bnInfer = op::BNInfer("bnInfer1");
    vector<int64_t> xShape = {2, 3, 4};
    vector<int64_t> paramShape = {3};

    Tensor xTensor;
    Tensor scaleTensor;
    Tensor offsetTensor;
    Tensor meanTensor;
    Tensor varianceTensor;
    op::Data xData = op::Data("data_x");
    op::Data scaleData = op::Data("data_scale");
    op::Data offsetData = op::Data("data_offset");
    op::Data meanData = op::Data("data_mean");
    op::Data varianceData = op::Data("data_variance");

    if (AddFloatInput(graph, input, inputs, "data_x", 0, xShape, xTensor, xData, 0.5f) != RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddFloatInput(graph, input, inputs, "data_scale", 1, paramShape, scaleTensor, scaleData, 1.0f) != RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddFloatInput(graph, input, inputs, "data_offset", 2, paramShape, offsetTensor, offsetData, 0.0f) !=
        RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddFloatInput(graph, input, inputs, "data_mean", 3, paramShape, meanTensor, meanData, 0.25f) != RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddFloatInput(graph, input, inputs, "data_variance", 4, paramShape, varianceTensor, varianceData, 1.0f) !=
        RET_SUCCESS) {
        return RET_FAILED;
    }

    bnInfer.set_input_x(xData);
    bnInfer.set_input_scale(scaleData);
    bnInfer.set_input_offset(offsetData);
    bnInfer.set_input_mean(meanData);
    bnInfer.set_input_variance(varianceData);
    bnInfer.set_attr_epsilon(1e-5f);
    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    bnInfer.update_output_desc_y(yDesc);
    outputs.push_back(bnInfer);
    return RET_SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    Graph graph("tc_ge_irrun_bn_infer");
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return RET_FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    if (CreateOppInGraph(input, inputs, outputs, graph) != RET_SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return RET_FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> buildOptions;
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return RET_FAILED;
    }

    uint32_t graphId = 0;
    map<AscendString, AscendString> graphOptions;
    if (session->AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return RET_FAILED;
    }

    vector<Tensor> output;
    if (session->RunGraph(graphId, input, output) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return RET_FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    delete session;
    GEFinalize();
    return RET_SUCCESS;
}
