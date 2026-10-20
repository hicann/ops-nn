/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_multi_add_rms_norm_dynamic_quant.cpp
 * \brief geir(graph mode) end-to-end test for MultiAddRmsNormDynamicQuant.
 *   GE-only fused op (no aclnn entry): x1(TensorList) + x2 -> RmsNorm(gamma) -> dual-smooth DynamicQuant.
 *   Inputs fp16/bf16 (here fp16=1.0); dual smooth_scale -> all 6 outputs (y1/y2/x/y/scale1/scale2) exercised.
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <new>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../../op_graph/multi_add_rms_norm_dynamic_quant_proto.h"

#define FAILED -1
#define SUCCESS 0

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

// fp16 Data placeholder filled with 1.0 (0x3C00). Structural GE-path test: values need only be valid.
static Operator MkHalf(const string& name, int idx, vector<int64_t> shape, vector<Tensor>& input)
{
    auto d = op::Data(name.c_str()).set_attr_index(idx);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, DT_FLOAT16);
    desc.SetFormat(FORMAT_ND);
    size_t n = 1;
    for (auto s : shape) {
        n *= s;
    }
    uint16_t* p = new (std::nothrow) uint16_t[n];
    for (size_t i = 0; i < n; ++i) {
        p[i] = 0x3C00; // 1.0 in IEEE half
    }
    Tensor t(desc, reinterpret_cast<uint8_t*>(p), n * sizeof(uint16_t));
    input.push_back(t);
    delete[] p;
    d.update_input_desc_x(desc);
    d.update_output_desc_y(desc);
    return d;
}

static int g_x1Num = 2; // x1 TensorList length; overridable via argv[1]

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto op1 = op::MultiAddRmsNormDynamicQuant("multiAddRmsNormDynamicQuant1");

    const int x1Num = g_x1Num; // x1 TensorList length (valid range 1~5), from argv[1] (default 2)
    const int64_t rows = 4;    // N: flattened leading dims
    const int64_t dim = 8;     // D: last / normalized dim (gamma & smooth_scale = [D])
    vector<int64_t> xShape = {rows, dim};
    vector<int64_t> dShape = {dim};
    vector<int64_t> sShape = {rows};
    int idx = 0;

    // x1: dynamic input tensor list -> multi-add
    op1.create_dynamic_input_x1(x1Num);
    for (int i = 0; i < x1Num; ++i) {
        auto d = MkHalf("x1_" + std::to_string(i), idx++, xShape, input);
        graph.AddOp(d);
        inputs.push_back(d);
        op1.set_dynamic_input_x1(i, d);
    }

    // x2 (required)
    auto x2 = MkHalf("x2", idx++, xShape, input);
    graph.AddOp(x2);
    inputs.push_back(x2);
    op1.set_input_x2(x2);

    // gamma (required, [D])
    auto gamma = MkHalf("gamma", idx++, dShape, input);
    graph.AddOp(gamma);
    inputs.push_back(gamma);
    op1.set_input_gamma(gamma);

    // smooth_scale1 / smooth_scale2 (optional; provide both -> dual-smooth, all outputs valid)
    auto ss1 = MkHalf("smooth_scale1", idx++, dShape, input);
    graph.AddOp(ss1);
    inputs.push_back(ss1);
    op1.set_input_smooth_scale1(ss1);

    auto ss2 = MkHalf("smooth_scale2", idx++, dShape, input);
    graph.AddOp(ss2);
    inputs.push_back(ss2);
    op1.set_input_smooth_scale2(ss2);

    op1.set_attr_epsilon(1e-6f);

    // outputs: y1/y2 int8, x/y same as input dtype (fp16), scale1/scale2 fp32 (per-row [N])
    op1.update_output_desc_y1(TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT8));
    op1.update_output_desc_y2(TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_INT8));
    op1.update_output_desc_x(TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT16));
    op1.update_output_desc_y(TensorDesc(ge::Shape(xShape), FORMAT_ND, DT_FLOAT16));
    op1.update_output_desc_scale1(TensorDesc(ge::Shape(sShape), FORMAT_ND, DT_FLOAT));
    op1.update_output_desc_scale2(TensorDesc(ge::Shape(sShape), FORMAT_ND, DT_FLOAT));

    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    if (argc > 1) {
        g_x1Num = atoi(argv[1]);
    }
    Graph graph("tc_ge_irrun_test");
    vector<ge::Tensor> input;

    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<Operator> inputs{};
    vector<Operator> outputs{};
    if (CreateOppInGraph(input, inputs, outputs, graph) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> options = {};
    Session* session = new Session(options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    session->AddGraph(graphId, graph, options);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    string dumpPath = "./dump";
    aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.length());

    vector<ge::Tensor> output;
    if (session->RunGraph(graphId, input, output) != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success, output num = %zu\n", GetTime().c_str(), output.size());

    delete session;
    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
