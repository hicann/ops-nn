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
 * \file test_geir_arg_max_grad.cpp
 * \brief GE graph construction sample for ArgMaxGrad (arch35 ND format).
 */
#include <iostream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <ctime>
#include <algorithm>
#include <memory>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../op_graph/arg_max_grad_proto.h"

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

// 数据只为构图打通, 不校验数值
// 输入 buffer 通过带 DeleteFunc 的 SetData 重载交由 Tensor 接管所有权,
// 保证内存存活到 RunGraph 之后(Tensor 析构时自动释放), 避免局部 vector 返回后悬垂。
int32_t GenData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, DataType dt)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto d : shapes)
        size *= d;
    desc.SetShape(ge::Shape(shapes));
    tensor.SetTensorDesc(desc);
    if (dt == DT_FLOAT) {
        std::unique_ptr<float[]> data(new (std::nothrow) float[static_cast<size_t>(size)]);
        if (data == nullptr) {
            return FAILED;
        }
        std::fill(data.get(), data.get() + size, 0.5f);
        if (tensor.SetData(reinterpret_cast<uint8_t*>(data.get()), size * sizeof(float),
                           [](uint8_t* p) { delete[] reinterpret_cast<float*>(p); }) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
        data.release();
    } else if (dt == DT_INT32) {
        std::unique_ptr<int32_t[]> data(new (std::nothrow) int32_t[static_cast<size_t>(size)]);
        if (data == nullptr) {
            return FAILED;
        }
        std::fill(data.get(), data.get() + size, 1);
        if (tensor.SetData(reinterpret_cast<uint8_t*>(data.get()), size * sizeof(int32_t),
                           [](uint8_t* p) { delete[] reinterpret_cast<int32_t*>(p); }) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
        data.release();
    } else {
        std::unique_ptr<uint16_t[]> data(new (std::nothrow) uint16_t[static_cast<size_t>(size)]);
        if (data == nullptr) {
            return FAILED;
        }
        std::fill(data.get(), data.get() + size, static_cast<uint16_t>(0x3800)); // fp16 ~0.5
        if (tensor.SetData(reinterpret_cast<uint8_t*>(data.get()), size * sizeof(uint16_t),
                           [](uint8_t* p) { delete[] reinterpret_cast<uint16_t*>(p); }) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
        data.release();
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
    if (GenData(ph##idx##_shape, tensor_ph##idx, ph##idx##_desc, inDtype) != SUCCESS) {     \
        printf("create failed: prepare input %s failed\n", #opInputName);                   \
        return FAILED;                                                                      \
    }                                                                                       \
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
    auto op_node = op::ArgMaxGrad("arg_max_grad_op");
    // var 为 (outer, D, inner); indices/updates 把 dimension 轴置 1
    const int64_t dimension = 1;
    std::vector<int64_t> varShape = {4, 16, 8};
    // indices/updates 把 dimension 轴 squeeze 掉(GE 侧 verify 要求 rank(var) == rank(updates) + 1)
    std::vector<int64_t> idxShape = {4, 8};
    ADD_INPUT(1, var, dt, varShape);
    ADD_INPUT(2, indices, DT_INT32, idxShape);
    ADD_INPUT(3, updates, dt, idxShape);
    op_node.set_attr_dimension(dimension);
    ADD_OUTPUT(y, dt, varShape);
    outputs.push_back(op_node);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    std::vector<ge::Tensor> input;
    std::map<AscendString, AscendString> gopt = {
        {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.jit_compile", "0"}};
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
