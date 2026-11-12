/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// test_geir for ForeachMinimumList (GEIR verify; 2 dynamic input lists + 1 dynamic output).
// For every tensor pair in the lists: y[i] = min(x1[i], x2[i]) element-wise.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "array_ops.h"
#include "gnode.h"
#include "named_io_node_builder.h"
#include "../op_graph/foreach_minimum_list_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int32_t RET_SUCCESS = 0;
constexpr int32_t RET_FAILED = -1;
constexpr size_t kInputListCount = 2; // x1 and x2
constexpr size_t kTimeBufferSize = 64;
const char* const kUnknownTime = "unknown-time";
const char* const kGraphName = "tc_ge_irrun_test";
const char* const kOpType = "ForeachMinimumList";
const char* const kOpName = "foreachMinimumList1";

// One tensor pair in the lists together with its expected minimum result.
struct TensorCase {
    vector<int64_t> shape;
    vector<float> x1;
    vector<float> x2;
    vector<float> y;
};

const vector<TensorCase> kCases = {
    {{2, 3},
     {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
     {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f},
     {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f}},
    {{3}, {7.0f, 8.0f, 9.0f}, {9.0f, 8.0f, 7.0f}, {7.0f, 8.0f, 7.0f}},
};

string GetTime()
{
    const time_t currentTime = time(nullptr);
    if (currentTime == static_cast<time_t>(-1)) {
        return kUnknownTime;
    }
    std::tm localTime = {};
    if (localtime_r(&currentTime, &localTime) == nullptr) {
        return kUnknownTime;
    }
    char buffer[kTimeBufferSize] = {};
    if (strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S,000", &localTime) == 0) {
        return kUnknownTime;
    }
    return buffer;
}

bool GetShapeSize(const vector<int64_t>& shape, size_t& count)
{
    count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            return false;
        }
        const size_t dimSize = static_cast<size_t>(dim);
        if (dimSize != 0 && count > std::numeric_limits<size_t>::max() / dimSize) {
            return false;
        }
        count *= dimSize;
    }
    return true;
}

TensorDesc MakeDesc(const vector<int64_t>& shape) { return TensorDesc(ge::Shape(shape), FORMAT_ND, DT_FLOAT); }

bool MakeData(const string& name, int32_t index, const vector<int64_t>& shape, const vector<float>& values,
              vector<Tensor>& input, Operator& data)
{
    size_t elementCount = 0;
    if (!GetShapeSize(shape, elementCount) || elementCount != values.size()) {
        printf("invalid host data for node %s\n", name.c_str());
        return false;
    }
    data = op::Data(name.c_str()).set_attr_index(index);
    TensorDesc desc = MakeDesc(shape);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    Tensor tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
    if (data.UpdateInputDesc("x", desc) != GRAPH_SUCCESS || data.UpdateOutputDesc("y", desc) != GRAPH_SUCCESS) {
        printf("failed to set desc for node %s\n", name.c_str());
        return false;
    }
    input.push_back(tensor);
    return true;
}

bool AddDataNode(Graph& graph, vector<Tensor>& input, vector<Operator>& inputs, vector<GNode>& dataNodes,
                 const string& name, int32_t index, const vector<int64_t>& shape, const vector<float>& values)
{
    Operator data;
    if (!MakeData(name, index, shape, values, input, data)) {
        return false;
    }
    GNode node = graph.AddNodeByOp(data);
    AscendString nodeType;
    if (node.GetType(nodeType) != GRAPH_SUCCESS) {
        printf("AddNodeByOp failed for node %s\n", name.c_str());
        return false;
    }
    dataNodes.push_back(node);
    inputs.push_back(data);
    return true;
}

bool ConnectPorts(Graph& graph, vector<GNode>& dataNodes, GNode& opNode)
{
    for (size_t port = 0; port < dataNodes.size(); ++port) {
        if (graph.AddDataEdge(dataNodes[port], 0, opNode, static_cast<int32_t>(port)) != GRAPH_SUCCESS) {
            printf("AddDataEdge failed at port %zu\n", port);
            return false;
        }
    }
    return true;
}

bool SetGraphOutputs(Graph& graph, GNode& opNode)
{
    vector<std::pair<GNode, int32_t>> outPorts;
    outPorts.reserve(kCases.size());
    for (size_t i = 0; i < kCases.size(); ++i) {
        outPorts.emplace_back(opNode, static_cast<int32_t>(i));
    }
    if (graph.SetOutputs(outPorts) != GRAPH_SUCCESS) {
        printf("SetOutputs failed\n");
        return false;
    }
    return true;
}

// Build ForeachMinimumList with NamedIoNodeBuilder so that the dynamic input/output ports follow the
// IR definition order (x1..., x2..., y...), avoiding the port-name ordering of legacy Operator serialization.
int32_t CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, Graph& graph)
{
    vector<GNode> dataNodes;
    dataNodes.reserve(kCases.size() * kInputListCount);

    NamedIoNodeBuilder builder(graph);
    builder.Type(kOpType).Name(kOpName);

    int32_t index = 0;
    for (size_t i = 0; i < kCases.size(); ++i) {
        const string suffix = std::to_string(i);
        if (!AddDataNode(graph, input, inputs, dataNodes, "x1_" + suffix, index++, kCases[i].shape, kCases[i].x1)) {
            return RET_FAILED;
        }
        // Dynamic input ports follow the IR naming rule: x10, x11, x20, x21, ...
        builder.AddInput(("x1" + suffix).c_str(), MakeDesc(kCases[i].shape));
    }
    for (size_t i = 0; i < kCases.size(); ++i) {
        const string suffix = std::to_string(i);
        if (!AddDataNode(graph, input, inputs, dataNodes, "x2_" + suffix, index++, kCases[i].shape, kCases[i].x2)) {
            return RET_FAILED;
        }
        builder.AddInput(("x2" + suffix).c_str(), MakeDesc(kCases[i].shape));
    }
    for (size_t i = 0; i < kCases.size(); ++i) {
        builder.AddOutput(("y" + std::to_string(i)).c_str(), MakeDesc(kCases[i].shape));
    }

    AscendString errorMsg;
    auto opNode = builder.Build(errorMsg);
    if (opNode == nullptr) {
        const char* message = errorMsg.GetString();
        printf("NamedIoNodeBuilder Build failed: %s\n", message != nullptr ? message : "");
        return RET_FAILED;
    }
    if (!ConnectPorts(graph, dataNodes, *opNode) || !SetGraphOutputs(graph, *opNode)) {
        return RET_FAILED;
    }
    return RET_SUCCESS;
}

bool CheckOutput(const vector<Tensor>& output)
{
    if (output.size() != kCases.size()) {
        printf("unexpected output count: %zu, expected %zu\n", output.size(), kCases.size());
        return false;
    }
    for (size_t i = 0; i < kCases.size(); ++i) {
        const Tensor& tensor = output[i];
        const size_t expectedBytes = kCases[i].y.size() * sizeof(float);
        if (tensor.GetTensorDesc().GetDataType() != DT_FLOAT ||
            tensor.GetTensorDesc().GetShape().GetDims() != kCases[i].shape || tensor.GetSize() != expectedBytes ||
            tensor.GetData() == nullptr) {
            printf("output[%zu] has unexpected metadata or buffer\n", i);
            return false;
        }
        for (size_t j = 0; j < kCases[i].y.size(); ++j) {
            float value = 0.0f;
            std::memcpy(&value, tensor.GetData() + j * sizeof(float), sizeof(float));
            if (value != kCases[i].y[j]) {
                printf("output[%zu][%zu] = %f, expected %f\n", i, j, value, kCases[i].y[j]);
                return false;
            }
        }
    }
    printf("ForeachMinimumList output metadata and all values matched\n");
    return true;
}

int32_t RunGraph(Graph& graph, const vector<Tensor>& input)
{
    map<AscendString, AscendString> sessionOptions;
    map<AscendString, AscendString> graphOptions;
    uint32_t graphId = 0;
    vector<Tensor> output;

    Session session(sessionOptions);
    if (session.AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        return RET_FAILED;
    }
    if (session.RunGraph(graphId, input, output) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        return RET_FAILED;
    }
    if (!CheckOutput(output)) {
        printf("%s - ERROR - [XIR]: CheckOutput failed\n", GetTime().c_str());
        return RET_FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    return RET_SUCCESS;
}

bool ParseDeviceId(int argc, char* argv[], int32_t& deviceId)
{
    deviceId = 0;
    if (argc <= 1) {
        return true;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
        printf("invalid device id: %s\n", argv[1]);
        return false;
    }
    deviceId = static_cast<int32_t>(parsed);
    return true;
}
} // namespace

int main(int argc, char* argv[])
{
    int32_t deviceId = 0;
    if (!ParseDeviceId(argc, argv, deviceId)) {
        return RET_FAILED;
    }
    const string deviceIdStr = std::to_string(deviceId);
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", deviceIdStr.c_str()},
                                                     {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return RET_FAILED;
    }
    printf("%s - INFO - [XIR]: Using device %d\n", GetTime().c_str(), deviceId);

    int32_t result = RET_FAILED;
    {
        Graph graph(kGraphName);
        vector<Tensor> input;
        vector<Operator> inputs;
        if (CreateOppInGraph(input, inputs, graph) != RET_SUCCESS) {
            printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        } else {
            graph.SetInputs(inputs);
            result = RunGraph(graph, input);
        }
    }

    if (ge::GEFinalize() != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        result = RET_FAILED;
    }
    return result;
}
