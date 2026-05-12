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
 * \file test_geir_max_pool_v2.cpp
 * \brief MaxPoolV2 GE IR test example
 */

#include <ctime>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdint.h>
#include <string>
#include <string.h>
#include <vector>

#include "assert.h"

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"

#include "nn_other.h"
#include "../op_graph/max_pool_v2_proto.h"

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
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    } else if (dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_DOUBLE) {
        return eightByte;
    }
    return oneByte;
}

template <typename T>
int32_t GenTensorData(
    const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, const vector<T>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto dim : shapes) {
        size *= dim;
    }
    if (size != values.size()) {
        printf(
            "%s - ERROR - [XIR]: GenTensorData size mismatch, expected %zu, got %zu\n", GetTime().c_str(), size,
            values.size());
        return FAILED;
    }
    auto* data = new (std::nothrow) T[size];
    if (data == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(data), size * sizeof(T));
    return SUCCESS;
}

template <typename T>
int32_t GenOnesData(
   vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type, const vector<T>& values)
{
   input_tensor_desc.SetRealDimCnt(shapes.size());
   size_t size = 1;
   for (uint32_t i = 0; i < shapes.size(); i++) {
       size *= shapes[i];
   }
    auto* data = new (std::nothrow) T[size];
    if (data == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(data), size * sizeof(T));
    return SUCCESS;
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, uint8_t* input_data)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: Failed to open file %s\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputValues)                            \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                          \
    auto placeholder##inputIndex =                                                                         \
        op::Data("placeholder" + inputIndex).set_attr_index((inputIndex) - 1);                             \
    TensorDesc placeholder##inputIndex##_desc =                                                            \
        TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, inputDtype);                    \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                       \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                   \
    Tensor tensor_placeholder##inputIndex;                                                                 \
    ret = GenTensorData(                                                                                    \
        placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
        inputValues);                                                                                       \
    if (ret != SUCCESS) {                                                                                   \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                      \
        return FAILED;                                                                                      \
    }                                                                                                       \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                          \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                         \
    input.push_back(tensor_placeholder##inputIndex);                                                       \
    graph.AddOp(placeholder##inputIndex);                                                                  \
    max_pool_v2.set_input_##inputName(placeholder##inputIndex);                                           \
    inputs.push_back(placeholder##inputIndex)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape, inputValues)                     \
   vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                          \
   auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                 \
   TensorDesc placeholder##intputIndex##_desc =                                                            \
       TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);                    \
   placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                       \
   placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                   \
   Tensor tensor_placeholder##intputIndex;                                                                 \
   ret = GenOnesData(                                                                                      \
       placeholder##intputIndex##_shape, tensor_placeholder##intputIndex, placeholder##intputIndex##_desc, \
       intputDtype, inputValues);                                                                          \
   if (ret != SUCCESS) {                                                                                   \
       printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                      \
       return FAILED;                                                                                      \
   }                                                                                                       \
   placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                             \
   placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                         \
   graph.AddOp(placeholder##intputIndex);                                                                  \
   max_pool_v2.set_input_##intputName(placeholder##intputIndex);                                    \
   max_pool_v2.update_input_desc_##intputName(placeholder##intputIndex##_desc);                     \
   inputs.push_back(placeholder##intputIndex);

#define ADD_INPUT_ATTR(attrName, attrValue) max_pool_v2.set_attr_##attrName(attrValue)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                        \
    TensorDesc outputName##outputIndex##_desc_ = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    max_pool_v2.update_output_desc_##outputName(outputName##outputIndex##_desc_)

int CreateOppInGraph(
    std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto max_pool_v2 = op::MaxPoolV2("max_pool_v2");

    std::vector<int64_t> x_shape = {1, 1, 1, 1};
    std::vector<int64_t> ksize_shape = {4};
    std::vector<int64_t> strides_shape = {4};
    std::vector<int64_t> y_shape = {1};

    std::vector<uint16_t> x_data = { 1 };
    std::vector<int32_t> ksize_data = {1, 1, 1, 1};
    std::vector<int32_t> strides_data = {1, 6, 36, 1};

    ADD_INPUT(1, x, DT_FLOAT16, x_shape, x_data);
    ADD_CONST_INPUT(2, ksize, DT_INT32, ksize_shape, ksize_data);
    ADD_CONST_INPUT(3, strides, DT_INT32, strides_shape, strides_data);

    max_pool_v2.set_attr_padding("VALID");
    ADD_INPUT_ATTR(data_format, "NHWC");

    ADD_OUTPUT(1, y, DT_FLOAT16, y_shape);

    outputs.push_back(max_pool_v2);

    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graph_name = "tc_ge_irrun_test_max_pool_v2";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
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

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            printf("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

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