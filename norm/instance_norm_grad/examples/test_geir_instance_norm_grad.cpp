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
 * \file test_geir_instance_norm_grad.cpp
 * \brief GE graph construction sample for InstanceNormGrad (arch35 ND format).
 */
#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../op_graph/instance_norm_grad_proto.h"

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::string;
using std::vector;

string GetTime()
{
    time_t t;
    time(&t);
    char b[32];
    strftime(b, 32, "%Y-%m-%d %H:%M:%S", localtime(&t));
    return string(b);
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, DataType dt, int val)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto d : shapes)
        size *= d;
    if (dt == DT_FLOAT) {
        vector<float> v(size, (float)val);
        desc.SetShape(ge::Shape(shapes));
        tensor.SetTensorDesc(desc);
        tensor.SetData((uint8_t*)v.data(), size * sizeof(float));
    } else {
        vector<uint16_t> v(size, 0x3C00);
        desc.SetShape(ge::Shape(shapes)); // fp16 ~1.0
        tensor.SetTensorDesc(desc);
        tensor.SetData((uint8_t*)v.data(), size * sizeof(uint16_t));
    }
    return SUCCESS;
}

#define ADD_INPUT(idx, opInputName, inDtype, inShape)                                       \
    vector<int64_t> ph##idx##_shape = inShape;                                              \
    auto ph##idx = op::Data("ph" + std::string(#idx)).set_attr_index(0);                    \
    TensorDesc ph##idx##_desc = TensorDesc(ge::Shape(ph##idx##_shape), FORMAT_ND, inDtype); \
    ph##idx##_desc.SetPlacement(ge::kPlacementHost);                                        \
    ph##idx##_desc.SetOriginFormat(FORMAT_ND);                                              \
    ph##idx##_desc.SetOriginShape(ge::Shape(ph##idx##_shape));                              \
    Tensor tensor_ph##idx;                                                                  \
    ret = GenOnesData(ph##idx##_shape, tensor_ph##idx, ph##idx##_desc, inDtype, 1);         \
    ph##idx.update_input_desc_x(ph##idx##_desc);                                            \
    ph##idx.update_output_desc_y(ph##idx##_desc);                                           \
    input.push_back(tensor_ph##idx);                                                        \
    graph.AddOp(ph##idx);                                                                   \
    op_node.set_input_##opInputName(ph##idx);                                               \
    inputs.push_back(ph##idx)

#define ADD_OUTPUT(opOutputName, outDtype, outShape)                                       \
    TensorDesc opOutputName##_desc = TensorDesc(ge::Shape(outShape), FORMAT_ND, outDtype); \
    opOutputName##_desc.SetOriginFormat(FORMAT_ND);                                        \
    opOutputName##_desc.SetOriginShape(ge::Shape(outShape));                               \
    op_node.update_output_desc_##opOutputName(opOutputName##_desc);

int CreateGraph(DataType dt, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op_node = op::InstanceNormGrad("instance_norm_grad_op");
    // NDHWC 5D; variance/mean = [N,1,1,1,C]; gamma = [C].
    std::vector<int64_t> xShape = {2, 2, 2, 2, 16};
    std::vector<int64_t> rShape = {2, 1, 1, 1, 16};
    std::vector<int64_t> cShape = {16};
    ADD_INPUT(1, dy, dt, xShape);
    ADD_INPUT(2, x, dt, xShape);
    ADD_INPUT(3, variance, dt, rShape);
    ADD_INPUT(4, mean, dt, rShape);
    ADD_INPUT(5, gamma, dt, cShape);
    ADD_OUTPUT(pd_x, dt, xShape);
    ADD_OUTPUT(pd_gamma, dt, cShape);
    ADD_OUTPUT(pd_beta, dt, cShape);
    outputs.push_back(op_node);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    std::vector<ge::Tensor> input;
    std::map<AscendString, AscendString> gopt = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(gopt) != SUCCESS) {
        printf("%s - init failed\n", GetTime().c_str());
        return FAILED;
    }
    std::vector<Operator> inputs{}, outputs{};
    DataType dt = DT_FLOAT;
    if (CreateGraph(dt, input, inputs, outputs, graph) != SUCCESS) {
        printf("create failed\n");
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);
    std::map<AscendString, AscendString> bopt = {};
    ge::Session* session = new Session(bopt);
    uint32_t gid = 0;
    std::map<AscendString, AscendString> gropt = {};
    session->AddGraph(gid, graph, gropt);
    std::vector<ge::Tensor> output;
    Status ret = session->RunGraph(gid, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run graph success, outputs=%zu\n", GetTime().c_str(), output.size());
    for (size_t i = 0; i < output.size(); i++) {
        auto d = output[i].GetTensorDesc();
        printf("  output[%zu] dim=%zu\n", i, d.GetShape().GetDimNum());
    }
    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
