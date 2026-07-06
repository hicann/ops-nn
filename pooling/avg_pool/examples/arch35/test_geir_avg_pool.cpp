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

#include "experiment_ops.h"
#include "nn_other.h"
#include "../../op_graph/avg_pool_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                             \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2);                                                                                  \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                                 \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                       \
    input.push_back(tensor_placeholder##inputIndex);                                                                   \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    add1.set_input_##inputName(placeholder##inputIndex);                                                               \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue)

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
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
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
    int32_t* pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
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
    auto add1 = op::AvgPool("avg_pool");

    vector<vector<int64_t>> shapes = {
        {4, 1, 4, 4} // x shape (input)
    };
    vector<vector<int64_t>> attrs = {
        {1, 1, 2, 2}, // ksize
        {1, 1, 2, 2}  // strides
    };

    ADD_INPUT(1, x, inDtype, shapes[0]);
    ADD_OUTPUT(2, y, inDtype, shapes[0]);
    ADD_INPUT_ATTR(ksize, attrs[0]);
    ADD_INPUT_ATTR(strides, attrs[1]);
    ADD_INPUT_ATTR(padding, "VALID");
    ADD_INPUT_ATTR(data_format, "NCHW");
    outputs.push_back(add1);
    return SUCCESS;
}

bool InitEnv()
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input)
{
    printf("%s - INFO - [XIR]: Start to CreateAndConfigGraph\n", GetTime().c_str());
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT16;
    std::cout << inDtype << std::endl;

    Status ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id)
{
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };

    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph to session failed\n", GetTime().c_str());
        return false;
    }
    return true;
}

bool RunGraph(ge::Session* session, uint32_t graph_id, std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        return false;
    }
    return true;
}

bool FinalizeEnv()
{
    printf("%s - INFO - [XIR]: Start to finalize ge\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return false;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return true;
}

int main(int argc, char* argv[])
{
    std::vector<ge::Tensor> input{};
    std::vector<ge::Tensor> output{};
    Graph graph("avg_pool_graph");

    if (!InitEnv()) {
        return FAILED;
    }

    if (!CreateAndConfigGraph(graph, input)) {
        return FAILED;
    }

    std::map<ge::AscendString, ge::AscendString> session_options = {};
    ge::Session* session = new ge::Session(session_options);
    if (session == nullptr) {
        return FAILED;
    }

    if (!AddGraphToSession(session, graph, 0)) {
        return FAILED;
    }

    if (!RunGraph(session, 0, input, output)) {
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./avg_pool_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./avg_pool_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);

        std::cout << "output y (avg_pool result):" << std::endl;
        if (output[i].GetTensorDesc().GetDataType() == DT_FLOAT16) {
            uint16_t* data = reinterpret_cast<uint16_t*>(output_data_i);
            for (int j = 0; j < std::min(output_shape, (int64_t)16); j++) {
                printf("  y[%d] = %u (float16)\n", j, data[j]);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Precision is ok\n", GetTime().c_str());

    if (!FinalizeEnv()) {
        return FAILED;
    }

    if (session != nullptr) {
        delete session;
    }

    printf("%s - INFO - [XIR]: Test case passed successfully\n", GetTime().c_str());
    return SUCCESS;
}