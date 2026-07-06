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
 * \file test_geir_ascend_anti_quant.cpp
 * \brief Graph-mode (GE IR) sample for AscendAntiQuant.
 *
 * Modeled on activation/relu_grad/examples/test_geir_relu_backward.cpp.
 * AscendAntiQuant has no aclnn API; this sample shows how to construct
 * the op via GE IR with the required scalar `scale` / `offset` attrs and
 * the optional `dtype` / `sqrt_mode` attrs.
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

#include "../op_graph/ascend_anti_quant_proto.h"

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

// Fill an int8 tensor with a fixed value (works for any 1-byte dtype).
int32_t GenInt8Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, int8_t value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    int8_t* pData = new (std::nothrow) int8_t[size];
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, DataType outDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto antiQuant1 = op::AscendAntiQuant("antiQuant1");

    // Single input x (per-tensor anti-quant, scale/offset are scalar attrs).
    std::vector<int64_t> xShape = {4, 8};
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);

    Tensor tensor_placeholder1;
    ret = GenInt8Data(xShape, tensor_placeholder1, placeholder1_desc, 4);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);

    antiQuant1.set_input_x(placeholder1);

    // y = cast<outDtype>((x + offset) * scale), sqrt_mode = false
    antiQuant1.set_attr_scale(0.5f);
    antiQuant1.set_attr_offset(1.0f);
    antiQuant1.set_attr_dtype(static_cast<int64_t>(outDtype));
    antiQuant1.set_attr_sqrt_mode(false);

    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, outDtype);
    antiQuant1.update_output_desc_y(yDesc);

    inputs.push_back(placeholder1);
    outputs.push_back(antiQuant1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test_ascend_anti_quant";
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

    DataType inDtype = DT_INT8;
    DataType outDtype = DT_FLOAT16;
    ret = CreateOppInGraph(inDtype, outDtype, input, inputs, outputs, graph);
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
        string output_file = "./tc_ge_irrun_ascend_anti_quant_output_" + std::to_string(i) + ".bin";
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
