/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// test_geir for InplaceIndexFill (GEIR graph-mode verification). Style follows the scatter_min test_geir template.
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
#include "../op_graph/inplace_index_fill_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

static int32_t GenFloatOnes(vector<int64_t> shape, Tensor& t, TensorDesc& desc, float v)
{
    desc.SetRealDimCnt(shape.size());
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    float* p = new (std::nothrow) float[n];
    for (size_t i = 0; i < n; ++i)
        p[i] = v;
    t = Tensor(desc, (uint8_t*)p, n * sizeof(float));
    delete[] p;
    return SUCCESS;
}

static int32_t GenInt32Idx(vector<int64_t> shape, Tensor& t, TensorDesc& desc, int32_t mod)
{
    desc.SetRealDimCnt(shape.size());
    size_t n = 1;
    for (auto d : shape)
        n *= d;
    int32_t* p = new (std::nothrow) int32_t[n];
    for (size_t i = 0; i < n; ++i)
        p[i] = (int32_t)(i % mod); // valid indices in [0, mod)
    t = Tensor(desc, (uint8_t*)p, n * sizeof(int32_t));
    delete[] p;
    return SUCCESS;
}

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto op1 = op::InplaceIndexFill("InplaceIndexFill1");
    vector<int64_t> xShape = {4, 5};
    vector<int64_t> idxShape = {3};
    vector<int64_t> valShape = {3, 5}; // value.shape与x.shape在dim轴外一致，dim轴为indices长度

    // x (float32)
    auto dX = op::Data("data_x").set_attr_index(0);
    TensorDesc descX(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    descX.SetPlacement(ge::kPlacementHost);
    Tensor tX;
    GenFloatOnes(xShape, tX, descX, 1.0f);
    dX.update_input_desc_x(descX);
    dX.update_output_desc_x(descX);
    input.push_back(tX);
    graph.AddOp(dX);
    inputs.push_back(dX);
    op1.set_input_x(dX);

    // indices (int32)
    auto dIdx = op::Data("data_indices").set_attr_index(1);
    TensorDesc descIdx(ge::Shape(idxShape), FORMAT_ND, DT_INT32);
    descIdx.SetPlacement(ge::kPlacementHost);
    Tensor tIdx;
    GenInt32Idx(idxShape, tIdx, descIdx, 5);
    dIdx.update_input_desc_x(descIdx);
    dIdx.update_output_desc_y(descIdx);
    input.push_back(tIdx);
    graph.AddOp(dIdx);
    inputs.push_back(dIdx);
    op1.set_input_indices(dIdx);

    // value (float32)
    auto dVal = op::Data("data_value").set_attr_index(2);
    TensorDesc descVal(ge::Shape(valShape), FORMAT_ND, DT_FLOAT);
    descVal.SetPlacement(ge::kPlacementHost);
    Tensor tVal;
    GenFloatOnes(valShape, tVal, descVal, 2.0f);
    dVal.update_input_desc_x(descVal);
    dVal.update_output_desc_y(descVal);
    input.push_back(tVal);
    graph.AddOp(dVal);
    inputs.push_back(dVal);
    op1.set_input_value(dVal);

    TensorDesc outDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT);
    op1.update_output_desc_x(outDesc);
    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;
    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != SUCCESS) {
        printf("GEInit failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<Operator> inputs{}, outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("create graph failed\n");
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty())
        graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> build_options = {};
    Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("session null\n");
        return FAILED;
    }
    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    if (session->RunGraph(graph_id, input, output) != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
