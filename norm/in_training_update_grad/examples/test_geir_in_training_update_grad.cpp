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
 * \file test_geir_in_training_update_grad.cpp
 * \brief GE graph construction sample for INTrainingUpdateGrad (the only call path is the GE graph).
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../op_graph/in_training_update_grad_proto.h"

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return 4;
    } else if (dt == ge::DT_FLOAT16) {
        return 2;
    }
    return 4;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t unitSize = GetDataTypeSize(data_type);
    uint32_t data_len = size * unitSize;
    // Byte-wise buffer; every element gets the full value pattern of its dtype width.
    uint8_t* pData = new (std::nothrow) uint8_t[data_len == 0 ? 1 : data_len];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate host input buffer failed\n", GetTime().c_str());
        return FAILED;
    }
    for (uint32_t i = 0; i < data_len; i += unitSize) {
        for (uint32_t j = 0; j < unitSize; ++j) {
            pData[i + j] = static_cast<uint8_t>(value);
        }
    }
    // Hand the buffer to the Tensor with a deleter: ownership transfers and the memory is
    // released by the Tensor itself, so the lifetime never depends on constructor copy semantics.
    input_tensor.SetTensorDesc(input_tensor_desc);
    if (input_tensor.SetData(pData, data_len, [](uint8_t* p) { delete[] p; }) != ge::GRAPH_SUCCESS) {
        delete[] pData;
        printf("%s - ERROR - [XIR]: Set tensor data failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}

// Add one Data placeholder and wire it to the given op input.
// Data 占位节点: format/origin 全用 FORMAT_ND + 6 维 (N, D, C1, H, W, C0) shape：
// - Ascend950 的 GE 在 CheckInternalFormat 阶段不允许用户图 Data 节点携带内部格式
//   NDC1HWC0 作 origin format(报 "Origin_format NDC1HWC0 is not supported")，故 Data 用 ND;
// - 算子 infershape 要求输入为 6 维(NDC1HWC0 排布)，故 shape 本身取 6 维。
// 算子节点输入 desc(见 update_input_desc_*)另行声明 format/origin=NDC1HWC0，以满足
// 引擎选择阶段算子 binary 的格式白名单(否则报 "origin format ND ... not supported")。
#define ADD_INPUT(idx, opInputName, inDtype, inShape)                                               \
    vector<int64_t> ph##idx##_shape = inShape;                                                      \
    auto ph##idx = op::Data("ph" + std::string(#idx)).set_attr_index(idx - 1);                      \
    TensorDesc ph##idx##_desc = TensorDesc(ge::Shape(ph##idx##_shape), FORMAT_ND, inDtype);         \
    ph##idx##_desc.SetPlacement(ge::kPlacementHost);                                                \
    ph##idx##_desc.SetFormat(FORMAT_ND);                                                            \
    ph##idx##_desc.SetOriginFormat(FORMAT_ND);                                                      \
    ph##idx##_desc.SetOriginShape(ge::Shape(ph##idx##_shape));                                      \
    Tensor tensor_ph##idx;                                                                          \
    ret = GenOnesData(ph##idx##_shape, tensor_ph##idx, ph##idx##_desc, inDtype, 2);                 \
    if (ret != SUCCESS) {                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());              \
        return FAILED;                                                                              \
    }                                                                                               \
    ph##idx.update_input_desc_x(ph##idx##_desc);                                                    \
    ph##idx.update_output_desc_y(ph##idx##_desc);                                                   \
    input.push_back(tensor_ph##idx);                                                                \
    graph.AddOp(ph##idx);                                                                           \
    TensorDesc ph##idx##_opdesc = TensorDesc(ge::Shape(ph##idx##_shape), FORMAT_NDC1HWC0, inDtype); \
    ph##idx##_opdesc.SetFormat(FORMAT_NDC1HWC0);                                                    \
    ph##idx##_opdesc.SetOriginFormat(FORMAT_NDC1HWC0);                                              \
    ph##idx##_opdesc.SetOriginShape(ge::Shape(ph##idx##_shape));                                    \
    op_node.update_input_desc_##opInputName(ph##idx##_opdesc);                                      \
    op_node.set_input_##opInputName(ph##idx);                                                       \
    inputs.push_back(ph##idx)

#define ADD_OUTPUT(opOutputName, outDtype, outShape)                                             \
    TensorDesc opOutputName##_desc = TensorDesc(ge::Shape(outShape), FORMAT_NDC1HWC0, outDtype); \
    opOutputName##_desc.SetOriginFormat(FORMAT_NDC1HWC0);                                        \
    opOutputName##_desc.SetOriginShape(ge::Shape(outShape));                                     \
    op_node.update_output_desc_##opOutputName(opOutputName##_desc);

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op_node = op::INTrainingUpdateGrad("in_training_update_grad_op");

    // 6 维 storage shape (N, D, C1, H, W, C0)，C0=16；variance/mean/outputs 的空间维为 1。
    std::vector<int64_t> spatialShape = {2, 1, 1, 2, 2, 16};
    std::vector<int64_t> reducedShape = {2, 1, 1, 1, 1, 16};

    ADD_INPUT(1, dy, inDtype, spatialShape);
    ADD_INPUT(2, x, inDtype, spatialShape);
    ADD_INPUT(3, variance, ge::DT_FLOAT, reducedShape);
    ADD_INPUT(4, mean, ge::DT_FLOAT, reducedShape);

    ADD_OUTPUT(res_gamma, ge::DT_FLOAT, reducedShape);
    ADD_OUTPUT(res_beta, ge::DT_FLOAT, reducedShape);

    outputs.push_back(op_node);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // dy/x are float16 or float32 (kept equal); this sample uses float16.
    DataType inDtype = DT_FLOAT16;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run graph success\n", GetTime().c_str());

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::cout << "Error message: " << std::string(error_msg.GetString()) << std::endl;
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}
