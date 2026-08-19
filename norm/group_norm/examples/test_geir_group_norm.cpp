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
 * \file test_geir_group_norm.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <limits>
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

#include "../op_graph/group_norm_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenTensorData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                          \
                        placeholder##intputIndex##_desc, intputDtype, 2.0F);                                        \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue);

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenTensorData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                          \
                        placeholder##intputIndex##_desc, intputDtype, 2.0F);                                        \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    add1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                                           \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc);

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
    // 返回GE数据类型对应的单元素字节数。
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
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

bool GetElementCount(const vector<int64_t>& shapes, size_t& elementCount)
{
    // 计算元素总数并检查维度乘积溢出。
    elementCount = 1;
    for (const int64_t dim : shapes) {
        if (dim < 0 || (dim != 0 && elementCount > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))) {
            return false;
        }
        elementCount *= static_cast<size_t>(dim);
    }
    return true;
}

int32_t GenTensorData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc,
                      DataType dataType, float value)
{
    // 按数据类型生成固定值输入张量。
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t elementCount = 0;
    if (!GetElementCount(shapes, elementCount)) {
        return FAILED;
    }

    inputTensor = Tensor(inputTensorDesc);
    if (dataType == ge::DT_FLOAT) {
        const vector<float> data(elementCount, value);
        return inputTensor.SetData(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float)) ==
                       ge::GRAPH_SUCCESS ?
                   SUCCESS :
                   FAILED;
    }
    if (dataType == ge::DT_FLOAT16) {
        static_assert(sizeof(_Float16) == 2, "_Float16 must use IEEE 754 binary16 storage");
        const vector<_Float16> data(elementCount, static_cast<_Float16>(value));
        return inputTensor.SetData(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(_Float16)) ==
                       ge::GRAPH_SUCCESS ?
                   SUCCESS :
                   FAILED;
    }
    return FAILED;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, const uint8_t* inputData)
{
    // 将张量数据完整写入二进制文件。
    if ((dataSize != 0 && inputData == nullptr) || dataSize > std::numeric_limits<size_t>::max()) {
        return FAILED;
    }
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    const size_t expectedSize = static_cast<size_t>(dataSize);
    const size_t writtenSize = expectedSize == 0 ? 0 : fwrite(inputData, sizeof(uint8_t), expectedSize, fp);
    const int closeResult = fclose(fp);
    return writtenSize == expectedSize && closeResult == 0 ? SUCCESS : FAILED;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 构造GroupNorm节点并设置输入shape。
    auto add1 = op::GroupNorm("add1");
    std::vector<int64_t> xShape = {36, 48, 24, 1};
    std::vector<int64_t> gammaShape = {48};
    std::vector<int64_t> betaShape = {48};

    ADD_INPUT(1, x, inDtype, xShape);
    ADD_INPUT(2, gamma, inDtype, gammaShape);
    ADD_INPUT(3, beta, inDtype, betaShape);
    // 显式设置GroupNorm的分组和数值属性。
    ADD_INPUT_ATTR(num_groups, 6);
    ADD_INPUT_ATTR(data_format, "NCHW");
    ADD_INPUT_ATTR(eps, 1e-4F);
    ADD_INPUT_ATTR(is_training, true);

    outputs.push_back(add1);
    return SUCCESS;
}

int main()
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    // 初始化GE运行环境。
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

    std::cout << inDtype << std::endl;

    // 构造单算子图并绑定图输入输出。
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
    // 将计算图加入Session并执行。
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    // 保存输入数据以便复核。
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write input data failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    // 保存并打印输出数据。
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write output data failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    // 输出GE诊断信息并释放Session。
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
