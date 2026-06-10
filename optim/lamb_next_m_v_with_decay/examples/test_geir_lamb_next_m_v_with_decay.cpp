/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_lamb_next_m_v_with_decay.cpp
 * \brief
 */

#include <iostream>
#include <string.h>
#include <vector>
#include <string>
#include <map>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "../op_graph/lamb_next_m_v_with_decay_proto.h"

#define FAILED (-1)
#define SUCCESS 0
using namespace ge;
using std::vector;

#define ADD_INPUT(idx, name, dtype, shape)                                                  \
    vector<int64_t> ph##idx##_shape = shape;                                                 \
    auto ph##idx = op::Data(std::string("ph") + #idx).set_attr_index(0);                     \
    TensorDesc ph##idx##_desc = TensorDesc(ge::Shape(ph##idx##_shape), FORMAT_ND, dtype);    \
    float* d##idx = new (std::nothrow) float[1024];                                          \
    for (int i = 0; i < 1024; ++i) { d##idx[i] = 1.0f; }                                     \
    Tensor t##idx(ph##idx##_desc, (uint8_t*)d##idx, 1024 * sizeof(float));                   \
    ph##idx.update_input_desc_x(ph##idx##_desc);                                             \
    ph##idx.update_output_desc_y(ph##idx##_desc);                                            \
    input.push_back(t##idx);                                                                 \
    graph.AddOp(ph##idx);                                                                    \
    op0.set_input_##name(ph##idx);                                                           \
    inputs.push_back(ph##idx)

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    auto op0 = op::LambNextMVWithDecay("lamb_next_mv_with_decay_0");
    std::vector<int64_t> fullShape = {2, 16};
    std::vector<int64_t> scalarShape = {1};
    ADD_INPUT(1, input_mul3, inDtype, fullShape);
    ADD_INPUT(2, input_mul2, inDtype, fullShape);
    ADD_INPUT(3, input_realdiv1, inDtype, scalarShape);
    ADD_INPUT(4, input_mul1, inDtype, fullShape);
    ADD_INPUT(5, input_mul0, inDtype, fullShape);
    ADD_INPUT(6, input_realdiv0, inDtype, scalarShape);
    ADD_INPUT(7, input_mul4, inDtype, fullShape);
    ADD_INPUT(8, mul0_x, inDtype, scalarShape);
    ADD_INPUT(9, mul1_sub, inDtype, scalarShape);
    ADD_INPUT(10, mul2_x, inDtype, scalarShape);
    ADD_INPUT(11, mul3_sub1, inDtype, scalarShape);
    ADD_INPUT(12, mul4_x, inDtype, scalarShape);
    ADD_INPUT(13, add2_y, inDtype, scalarShape);
    outputs.push_back(op0);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("test_lamb_next_m_v_with_decay");
    std::vector<ge::Tensor> input;
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(global_options) != SUCCESS) { return FAILED; }
    std::vector<Operator> inputs{}, outputs{};
    DataType inDtype = DT_FLOAT;
    if (CreateOppInGraph(inDtype, input, inputs, outputs, graph) != SUCCESS) { return FAILED; }
    if (!inputs.empty() && !outputs.empty()) { graph.SetInputs(inputs).SetOutputs(outputs); }
    ge::Session* session = new Session({});
    if (session == nullptr) { return FAILED; }
    uint32_t graph_id = 0;
    session->AddGraph(graph_id, graph, {});
    std::vector<ge::Tensor> output;
    if (session->RunGraph(graph_id, input, output) != SUCCESS) { delete session; GEFinalize(); return FAILED; }
    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
