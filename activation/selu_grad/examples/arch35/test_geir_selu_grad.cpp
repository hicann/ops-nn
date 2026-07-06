/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
#include "../op_graph/selu_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT_GRAD(intputIndex, intputName, intputDtype, inputShape)                                            \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesDataFloat32(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                     \
                             placeholder##intputIndex##_desc, 1);                                                   \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_gradients(placeholder##intputIndex##_desc);                          \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    selu_grad_op.set_input_##intputName(placeholder##intputIndex);                                                  \
    inputs.push_back(placeholder##intputIndex);

#define ADD_INPUT_OUT(intputIndex, intputName, intputDtype, inputShape)                                             \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesDataFloat32(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                     \
                             placeholder##intputIndex##_desc, 2);                                                   \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_outputs(placeholder##intputIndex##_desc);                            \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    selu_grad_op.set_input_##intputName(placeholder##intputIndex);                                                  \
    inputs.push_back(placeholder##intputIndex);

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT32)
        return 4;
    if (dt == ge::DT_INT64)
        return 8;
    if (dt == ge::DT_INT8)
        return 1;
    return 4;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * 4;
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr)
        return FAILED;
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto selu_grad_op = op::SeluGrad("selu_grad");
    std::vector<int64_t> shape = {4, 4};
    ADD_INPUT_GRAD(1, gradients, inDtype, shape);
    ADD_INPUT_OUT(2, outputs, inDtype, shape);

    outputs.push_back(selu_grad_op);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
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

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_selu_grad_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    delete session;
    session = nullptr;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
