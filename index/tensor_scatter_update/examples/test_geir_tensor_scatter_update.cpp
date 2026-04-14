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
 * \file test_geir_tensor_scatter_update.cpp
 * \brief
 */

#include <ctime>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
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

#include "nn_other.h"
#include "../op_graph/tensor_scatter_update_proto.h"

#define FAILED -1
#define SUCCESS 0

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputData)                                    \
    do {                                                                                                       \
        vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                          \
        auto placeholder##inputIndex =                                                                          \
            op::Data("placeholder" + std::to_string(inputIndex)).set_attr_index((inputIndex) - 1);            \
        TensorDesc placeholder##inputIndex##_desc =                                                            \
            TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, inputDtype);                    \
        placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                       \
        placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                   \
        Tensor tensor_placeholder##inputIndex;                                                                 \
        ret = GenTensorData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                  \
            placeholder##inputIndex##_desc, inputData);                                                        \
        if (ret != SUCCESS) {                                                                                  \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                    \
            return FAILED;                                                                                     \
        }                                                                                                      \
        placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                          \
        placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                         \
        input.push_back(tensor_placeholder##inputIndex);                                                       \
        graph.AddOp(placeholder##inputIndex);                                                                  \
        tensor_scatter_update.set_input_##inputName(placeholder##inputIndex);                                  \
        inputs.push_back(placeholder##inputIndex);                                                             \
    } while (0)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                          \
    do {                                                                                                       \
        TensorDesc outputName##outputIndex##_desc =                                                            \
            TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);                                        \
        outputName##outputIndex##_desc.SetPlacement(ge::kPlacementHost);                                       \
        outputName##outputIndex##_desc.SetFormat(FORMAT_ND);                                                   \
        tensor_scatter_update.update_output_desc_##outputName(outputName##outputIndex##_desc);                \
    } while (0)

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
    uint32_t one_byte = 1;
    uint32_t two_byte = 2;
    uint32_t four_byte = 4;
    uint32_t eight_byte = 8;
    uint32_t sixteen_byte = 16;

    if (dt == ge::DT_FLOAT) {
        return four_byte;
    } else if (dt == ge::DT_FLOAT16) {
        return two_byte;
    } else if (dt == ge::DT_DOUBLE) {
        return eight_byte;
    } else if (dt == ge::DT_BF16) {
        return two_byte;
    } else if (dt == ge::DT_INT16) {
        return two_byte;
    } else if (dt == ge::DT_UINT16) {
        return two_byte;
    } else if (dt == ge::DT_INT32) {
        return four_byte;
    } else if (dt == ge::DT_UINT32) {
        return four_byte;
    } else if (dt == ge::DT_INT64) {
        return eight_byte;
    } else if (dt == ge::DT_UINT64) {
        return eight_byte;
    } else if (dt == ge::DT_INT8) {
        return one_byte;
    } else if (dt == ge::DT_UINT8) {
        return one_byte;
    } else if (dt == ge::DT_BOOL) {
        return one_byte;
    } else if (dt == ge::DT_COMPLEX64) {
        return eight_byte;
    } else if (dt == ge::DT_COMPLEX128) {
        return sixteen_byte;
    }
    return 0;
}

void PrintComplexTensorData(const std::complex<double> *result_data, int64_t output_shape)
{
    for (int64_t j = 0; j < output_shape; j++) {
        LOG_PRINT("result[%ld] is: (%lf, %lf)\n", j, result_data[j].real(), result_data[j].imag());
    }
}

template <typename T>
int32_t GenTensorData(const vector<int64_t> &shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    const vector<T> &values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    if (size != values.size()) {
        return FAILED;
    }

    uint32_t data_len = size * sizeof(T);
    T *p_data = new (std::nothrow) T[size];
    if (p_data == nullptr) {
        delete[] p_data;
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        p_data[i] = values[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(p_data), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(const string &bin_file, uint64_t data_size, uint8_t *input_data)
{
    FILE *fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

void SaveInputOutput(std::vector<ge::Tensor> &input, std::vector<ge::Tensor> &output)
{
    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t *input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        DataType output_dtype = output[i].GetTensorDesc().GetDataType();
        uint32_t data_size = output_shape * GetDataTypeSize(output_dtype);
        WriteDataToFile(output_file, data_size, output_data_i);
        PrintComplexTensorData(reinterpret_cast<std::complex<double> *>(output_data_i), output_shape);
    }
}

int CreateOppInGraph(DataType input_dtype, DataType indices_dtype, std::vector<ge::Tensor> &input,
    std::vector<Operator> &inputs, std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto tensor_scatter_update = op::TensorScatterUpdate("tensor_scatter_update");

    std::vector<int64_t> x_shape = {3, 5};
    std::vector<int64_t> indices_shape = {2, 1};
    std::vector<int64_t> updates_shape = {2, 5};

    std::vector<std::complex<double>> x_data = {
        {0.0, 0.5},   {1.0, -0.5},  {2.0, 1.5},   {3.0, -1.5},  {4.0, 2.5},
        {5.0, -2.5},  {6.0, 3.5},   {7.0, -3.5},  {8.0, 4.5},   {9.0, -4.5},
        {10.0, 5.5},  {11.0, -5.5}, {12.0, 6.5},  {13.0, -6.5}, {14.0, 7.5}};
    std::vector<int64_t> indices_data = {0, 2};
    std::vector<std::complex<double>> updates_data = {
        {100.0, 10.0}, {101.0, 11.0}, {102.0, 12.0}, {103.0, 13.0}, {104.0, 14.0},
        {200.0, 20.0}, {201.0, 21.0}, {202.0, 22.0}, {203.0, 23.0}, {204.0, 24.0}};

    ADD_INPUT(1, x, input_dtype, x_shape, x_data);
    ADD_INPUT(2, indices, indices_dtype, indices_shape, indices_data);
    ADD_INPUT(3, updates, input_dtype, updates_shape, updates_data);
    ADD_OUTPUT(1, y, input_dtype, x_shape);

    outputs.push_back(tensor_scatter_update);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed.ret = %d\n", GetTime().c_str(), ret);
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType input_dtype = DT_COMPLEX128;
    DataType indices_dtype = DT_INT64;

    ret = CreateOppInGraph(input_dtype, indices_dtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Session add ir compute graph failed\n", GetTime().c_str());
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

    SaveInputOutput(input, output);

    delete session;

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
