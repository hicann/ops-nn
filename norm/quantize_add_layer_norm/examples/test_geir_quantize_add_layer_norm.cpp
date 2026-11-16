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
 * \file test_geir_quantize_add_layer_norm.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <ctime>
#include <limits>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <new>
#include "assert.h"
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../op_graph/quantize_add_layer_norm_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                \
    vector<int64_t> placeholder##inputIndex##Shape = inputShape;                                                \
    auto placeholder##inputIndex = op::Data("placeholder" + std::to_string(inputIndex)).set_attr_index(0);      \
    TensorDesc placeholder##inputIndex##Desc = TensorDesc(ge::Shape(placeholder##inputIndex##Shape), FORMAT_ND, \
                                                          inputDtype);                                          \
    placeholder##inputIndex##Desc.SetPlacement(ge::kPlacementHost);                                             \
    placeholder##inputIndex##Desc.SetFormat(FORMAT_ND);                                                         \
    Tensor tensorPlaceholder##inputIndex;                                                                       \
    ret = GenOnesDataFloat32(placeholder##inputIndex##Shape, tensorPlaceholder##inputIndex,                     \
                             placeholder##inputIndex##Desc, 2);                                                 \
    if (ret != SUCCESS) {                                                                                       \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
        return FAILED;                                                                                          \
    }                                                                                                           \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##Desc);                                 \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##Desc);                                \
    input.push_back(tensorPlaceholder##inputIndex);                                                             \
    if (graph.AddOp(placeholder##inputIndex) != ge::GRAPH_SUCCESS) {                                            \
        printf("%s - ERROR - [XIR]: Add input op to graph failed\n", GetTime().c_str());                        \
        return FAILED;                                                                                          \
    }                                                                                                           \
    add1.set_input_##inputName(placeholder##inputIndex);                                                        \
    inputs.push_back(placeholder##inputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue);

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep = time(nullptr);
    std::tm localTime{};
    char tmp[64] = {};
    if (timep == static_cast<time_t>(-1) || localtime_r(&timep, &localTime) == nullptr ||
        strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", &localTime) == 0) {
        return "unknown-time";
    }
    return tmp;
}

uint32_t GetDataTypeSize(DataType dataType)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dataType == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dataType == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dataType == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dataType == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dataType == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dataType == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dataType == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dataType == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dataType == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dataType == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

bool GetElementCount(const vector<int64_t>& shapes, size_t& elementCount)
{
    elementCount = 1;
    for (int64_t dim : shapes) {
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / elementCount) {
            return false;
        }
        elementCount *= static_cast<size_t>(dim);
    }
    return true;
}

int32_t GenOnesDataFloat32(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, float value)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    if (!GetElementCount(shapes, size) || size > std::numeric_limits<uint32_t>::max() / sizeof(float)) {
        return FAILED;
    }
    uint32_t dataLen = static_cast<uint32_t>(size * sizeof(float));
    std::vector<float> data(size, value);
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t*>(data.data()), dataLen);
    return SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    if (inputData == nullptr && dataSize != 0) {
        return FAILED;
    }
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    int closeStatus = fclose(fp);
    return (written == dataSize && closeStatus == 0) ? SUCCESS : FAILED;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::QuantizeAddLayerNorm("add1");
    std::vector<int64_t> x1Shape = {4, 1, 8};
    std::vector<int64_t> x2Shape = {4, 1, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasShape = {8};
    std::vector<int64_t> scalesShape = {8};
    std::vector<int64_t> pointsShape = {8};
    ADD_INPUT(1, x1, inDtype, x1Shape);
    ADD_INPUT(2, x2, inDtype, x2Shape);
    ADD_INPUT(3, gamma, inDtype, gammaShape);
    ADD_INPUT(4, beta, inDtype, betaShape);
    ADD_INPUT(5, bias, inDtype, biasShape);
    ADD_INPUT(6, scales, inDtype, scalesShape);
    ADD_INPUT(7, zero_points, inDtype, pointsShape);
    ADD_INPUT_ATTR(dtype, DT_INT8);
    ADD_INPUT_ATTR(additional_output, true);

    if (graph.AddOp(add1) != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: Add QuantizeAddLayerNorm op to graph failed\n", GetTime().c_str());
        return FAILED;
    }
    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

int RunExample()
{
    const char* graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;
    Status ret = SUCCESS;

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;

    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options{};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session session(build_options);
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options{};
    uint32_t graph_id = 0;
    ret = session.AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add compute graph to ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    auto dumpStatus = aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    if (dumpStatus != ge::GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: Dump graph failed, status=%u\n", GetTime().c_str(),
               static_cast<uint32_t>(dumpStatus));
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session.RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write input data failed: %s\n", GetTime().c_str(), input_file.c_str());
            return FAILED;
        }
    }

    int output_num = output.size();
    if (output_num != 2 || output[0].GetTensorDesc().GetDataType() != DT_INT8 ||
        output[1].GetTensorDesc().GetDataType() != DT_FLOAT || output[0].GetData() == nullptr ||
        output[1].GetData() == nullptr) {
        printf("%s - ERROR - [XIR]: Graph returned invalid y/x outputs\n", GetTime().c_str());
        return FAILED;
    }
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write output data failed: %s\n", GetTime().c_str(), output_file.c_str());
            return FAILED;
        }
        DataType outputDtype = output[i].GetTensorDesc().GetDataType();
        if (outputDtype == DT_INT8) {
            int8_t* resultData = reinterpret_cast<int8_t*>(output_data_i);
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%" PRId64 "] is: %d\n", j, static_cast<int>(resultData[j]));
            }
        } else if (outputDtype == DT_FLOAT) {
            float* resultData = reinterpret_cast<float*>(output_data_i);
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%" PRId64 "] is: %f\n", j, resultData[j]);
            }
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    const char* errorMessage = error_msg.GetString();
    std::string error_str(errorMessage == nullptr ? "" : errorMessage);
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    const char* warningMessage = warning_msg.GetString();
    std::string warning_str(warningMessage == nullptr ? "" : warningMessage);
    std::cout << "Warning message: " << warning_str << std::endl;
    return SUCCESS;
}

int main()
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status initStatus = ge::GEInitialize(globalOptions);
    if (initStatus != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    int runStatus = RunExample();
    printf("%s - INFO - [XIR]: Start to finalize ge\n", GetTime().c_str());
    Status finalizeStatus = ge::GEFinalize();
    if (finalizeStatus != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    return runStatus;
}
