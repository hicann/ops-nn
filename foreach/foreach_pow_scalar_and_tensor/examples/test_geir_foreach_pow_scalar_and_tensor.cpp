/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// test_geir for ForeachPowScalarAndTensor (GEIR graph-mode verify, static scalar + dynamic tensor-list input).
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <new>
#include <ctime>
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
#include "../op_graph/foreach_pow_scalar_and_tensor_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t t;
    time(&t);
    char b[64];
    strftime(b, sizeof(b), "%Y-%m-%d %H:%M:%S,000", localtime(&t));
    return b;
}

static Operator MkF32(const string& name, int idx, vector<int64_t> shape, vector<Tensor>& input, float value)
{
    auto d = op::Data(name.c_str()).set_attr_index(idx);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT);
    size_t n = 1;
    for (auto s : shape)
        n *= s;
    float* p = new (std::nothrow) float[n];
    for (size_t i = 0; i < n; ++i)
        p[i] = value;
    Tensor t(desc, (uint8_t*)p, n * sizeof(float));
    input.push_back(t);
    delete[] p;
    d.update_input_desc_x(desc);
    d.update_output_desc_y(desc);
    return d;
}

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    const int N = 2; // 2 tensors per list
    vector<int64_t> tShape = {4};
    int idx = 0;
    // 用 NamedIoNodeBuilder 按 IR 定义顺序（scalar, x）构建节点，
    // 避免旧 Operator 序列化按端口名排序导致静态输入前置、动态实例表错位
    vector<GNode> dataNodes;
    auto addData = [&](const string& name, const vector<int64_t>& shape, float value) {
        auto d = MkF32(name, idx++, shape, input, value);
        dataNodes.push_back(graph.AddNodeByOp(d));
        inputs.push_back(d);
    };
    NamedIoNodeBuilder builder(graph);
    builder.Type("ForeachPowScalarAndTensor").Name("foreachPowScalarAndTensor1");
    TensorDesc xd(ge::Shape(tShape), FORMAT_ND, DT_FLOAT);
    TensorDesc sd(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
    // IR 顺序为静态输入 scalar 在前、动态输入 x 在后，构图顺序需与之一致
    addData("scalar", {1}, 2.0f);
    builder.AddInput("scalar", sd);
    int port = 1;
    for (int i = 0; i < N; ++i) {
        addData("x_" + std::to_string(i), tShape, 1.0f);
        builder.AddInput(("x" + std::to_string(i)).c_str(), xd);
        port++;
    }
    for (int i = 0; i < N; ++i) {
        builder.AddOutput(("y" + std::to_string(i)).c_str(), xd);
    }
    AscendString errMsg;
    auto opNode = builder.Build(errMsg);
    if (opNode == nullptr) {
        printf("NamedIoNodeBuilder Build failed: %s\n", errMsg.GetString());
        return FAILED;
    }
    for (int k = 0; k < port; ++k) {
        if (graph.AddDataEdge(dataNodes[k], 0, *opNode, k) != GRAPH_SUCCESS) {
            printf("AddDataEdge failed at port %d\n", k);
            return FAILED;
        }
    }
    vector<std::pair<GNode, int32_t>> outPorts;
    for (int i = 0; i < N; ++i) {
        outPorts.push_back({*opNode, i});
    }
    if (graph.SetOutputs(outPorts) != GRAPH_SUCCESS) {
        printf("SetOutputs failed\n");
        return FAILED;
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;
    map<AscendString, AscendString> go = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(go) != SUCCESS) {
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());
    vector<Operator> inputs{}, outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        return FAILED;
    }
    if (!inputs.empty())
        graph.SetInputs(inputs); // outputs 已在 CreateOppInGraph 内经 GNode 端口设置
    map<AscendString, AscendString> opts = {};
    Session* session = new Session(opts);
    if (session == nullptr) {
        return FAILED;
    }
    uint32_t gid = 0;
    session->AddGraph(gid, graph, opts);
    vector<ge::Tensor> output;
    if (session->RunGraph(gid, input, output) != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success, output num %zu\n", GetTime().c_str(),
           output.size());
    if (!output.empty() && output[0].GetData() != nullptr) {
        printf("%s - INFO - [XIR]: y[0][0] = %f (expect 2.0^1.0 = 2.0)\n", GetTime().c_str(),
               reinterpret_cast<const float*>(output[0].GetData())[0]);
    }
    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
