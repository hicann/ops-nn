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
 * \file test_geir_scatter_nd_min.cpp
 * \brief Graph-mode (GE IR) sample for ScatterNdMin.
 *
 * Modeled on activation/relu_grad/examples/test_geir_relu_backward.cpp.
 * ScatterNdMin computes the element-wise minimum between an input tensor
 * and sparse updates at specified indices.
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

// Protobuf-based operator header for ScatterNdMin
#include "../op_graph/scatter_nd_min_proto.h"

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT8)
        return 1;
    if (dt == ge::DT_HIFLOAT8)
        return 1;
    if (dt == ge::DT_FLOAT8_E5M2)
        return 1;
    if (dt == ge::DT_FLOAT8_E4M3FN)
        return 1;
    return 1;
}

// Generate a float tensor filled with a given constant value.
int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
    delete[] pData;
    return SUCCESS;
}

// Generate an int32 tensor with specified values.
int32_t GenInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                     const vector<int32_t>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    int32_t* pData = new (std::nothrow) int32_t[size];
    for (size_t i = 0; i < size; ++i) {
        pData[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(int32_t));
    delete[] pData;
    return SUCCESS;
}

// Generate a float tensor with specific element values.
int32_t GenFloatValues(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                       const vector<float>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        pData[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    // ScatterNdMin: var_out = scatter_nd_min(var, indices, updates)
    auto scatterNdMin = op::ScatterNdMin("scatter_nd_min");

    // -------------------- Input 0: var --------------------
    // var shape: [8], initial values: [1,2,3,4,5,6,7,8]
    std::vector<int64_t> varShape = {8};
    auto varData = op::Data("var").set_attr_index(0);
    TensorDesc varDesc = TensorDesc(ge::Shape(varShape), FORMAT_ND, inDtype);
    varDesc.SetPlacement(ge::kPlacementHost);
    varDesc.SetFormat(FORMAT_ND);

    Tensor tensor_var;
    vector<float> varVals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    ret = GenFloatValues(varShape, tensor_var, varDesc, varVals);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate var data failed\n", GetTime().c_str());
        return FAILED;
    }
    varData.update_input_desc_x(varDesc);
    varData.update_output_desc_y(varDesc);
    input.push_back(tensor_var);
    graph.AddOp(varData);

    // -------------------- Input 1: indices --------------------
    // indices shape: [4,1], values: [[2],[4],[1],[7]]
    std::vector<int64_t> idxShape = {4, 1};
    auto indicesData = op::Data("indices").set_attr_index(1);
    TensorDesc indicesDesc = TensorDesc(ge::Shape(idxShape), FORMAT_ND, DT_INT32);
    indicesDesc.SetPlacement(ge::kPlacementHost);
    indicesDesc.SetFormat(FORMAT_ND);

    Tensor tensor_indices;
    vector<int32_t> idxVals = {2, 4, 1, 7};
    ret = GenInt32Data(idxShape, tensor_indices, indicesDesc, idxVals);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate indices data failed\n", GetTime().c_str());
        return FAILED;
    }
    indicesData.update_input_desc_x(indicesDesc);
    indicesData.update_output_desc_y(indicesDesc);
    input.push_back(tensor_indices);
    graph.AddOp(indicesData);

    // -------------------- Input 2: updates --------------------
    // updates shape: [4], values: [6,7,8,9]
    std::vector<int64_t> updShape = {4};
    auto updatesData = op::Data("updates").set_attr_index(2);
    TensorDesc updatesDesc = TensorDesc(ge::Shape(updShape), FORMAT_ND, inDtype);
    updatesDesc.SetPlacement(ge::kPlacementHost);
    updatesDesc.SetFormat(FORMAT_ND);

    Tensor tensor_updates;
    vector<float> updVals = {6.0f, 7.0f, 8.0f, 9.0f};
    ret = GenFloatValues(updShape, tensor_updates, updatesDesc, updVals);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate updates data failed\n", GetTime().c_str());
        return FAILED;
    }
    updatesData.update_input_desc_x(updatesDesc);
    updatesData.update_output_desc_y(updatesDesc);
    input.push_back(tensor_updates);
    graph.AddOp(updatesData);

    // -------------------- Connect to ScatterNdMin --------------------
    scatterNdMin.set_input_var(varData);
    scatterNdMin.set_input_indices(indicesData);
    scatterNdMin.set_input_updates(updatesData);

    // Optional attribute: use_locking (default false)
    scatterNdMin.set_attr_use_locking(false);

    // Output tensor (updated var) has same shape & dtype as var
    TensorDesc outDesc = TensorDesc(ge::Shape(varShape), FORMAT_ND, inDtype);
    scatterNdMin.update_output_desc_var(outDesc);

    inputs.push_back(varData);
    inputs.push_back(indicesData);
    inputs.push_back(updatesData);
    outputs.push_back(scatterNdMin);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test_scatter_nd_min";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype : " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_scatter_nd_min_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, shape size = " << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file.c_str(), data_size, output_data_i);
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::cout << "Error message: " << error_msg.GetString() << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::cout << "Warning message: " << warning_msg.GetString() << std::endl;

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
