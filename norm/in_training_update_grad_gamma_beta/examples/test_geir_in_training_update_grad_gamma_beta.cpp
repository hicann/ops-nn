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
 * \file test_geir_in_training_update_grad_gamma_beta.cpp
 * \brief GE graph construction sample for INTrainingUpdateGradGammaBeta (the only call path is the GE graph).
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

#include "../op_graph/in_training_update_grad_gamma_beta_proto.h"

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::map;
using std::string;
using std::vector;

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
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[data_len == 0 ? 1 : data_len];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    // Tensor(desc, data, size) deep-copies the buffer, so it can be freed right after.
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

// Add one Data placeholder in NCHW format and wire it to the given op input.
// origin 与 storage 同为 NCHW(4D)直通:与算子 def 声明一致,GE 不插 TransData。
#define ADD_INPUT(idx, opInputName, inDtype, inShape)                                         \
    vector<int64_t> ph##idx##_shape = inShape;                                                \
    auto ph##idx = op::Data("ph" + std::string(#idx)).set_attr_index(idx);                    \
    TensorDesc ph##idx##_desc = TensorDesc(ge::Shape(ph##idx##_shape), FORMAT_NCHW, inDtype); \
    ph##idx##_desc.SetPlacement(ge::kPlacementHost);                                          \
    ph##idx##_desc.SetFormat(FORMAT_NCHW);                                                    \
    ph##idx##_desc.SetOriginFormat(FORMAT_NCHW);                                              \
    ph##idx##_desc.SetOriginShape(ge::Shape(ph##idx##_shape));                                \
    Tensor tensor_ph##idx;                                                                    \
    ret = GenOnesData(ph##idx##_shape, tensor_ph##idx, ph##idx##_desc, inDtype, 2);           \
    if (ret != SUCCESS) {                                                                     \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());        \
        return FAILED;                                                                        \
    }                                                                                         \
    ph##idx.update_input_desc_x(ph##idx##_desc);                                              \
    ph##idx.update_output_desc_y(ph##idx##_desc);                                             \
    input.push_back(tensor_ph##idx);                                                          \
    graph.AddOp(ph##idx);                                                                     \
    op_node.set_input_##opInputName(ph##idx);                                                 \
    inputs.push_back(ph##idx)

#define ADD_OUTPUT(opOutputName, outDtype, outShape)                                         \
    TensorDesc opOutputName##_desc = TensorDesc(ge::Shape(outShape), FORMAT_NCHW, outDtype); \
    opOutputName##_desc.SetOriginFormat(FORMAT_NCHW);                                        \
    opOutputName##_desc.SetOriginShape(ge::Shape(outShape));                                 \
    op_node.update_output_desc_##opOutputName(opOutputName##_desc);

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op_node = op::INTrainingUpdateGradGammaBeta("in_training_update_grad_gamma_beta_op");

    // NCHW (N, C, 1, 1): the InstanceNorm-backward partial sums; the op reduces over N.
    std::vector<int64_t> partialShape = {4, 64, 1, 1};
    std::vector<int64_t> reducedShape = {1, 64, 1, 1};

    ADD_INPUT(0, res_gamma, inDtype, partialShape);
    ADD_INPUT(1, res_beta, inDtype, partialShape);

    ADD_OUTPUT(pd_gamma, ge::DT_FLOAT, reducedShape);
    ADD_OUTPUT(pd_beta, ge::DT_FLOAT, reducedShape);

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

    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::string file_path = "./dump";
    ret = aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Dump graph failed\n", GetTime().c_str());
    }

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
