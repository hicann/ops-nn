
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_embedding_bag.cpp
 * \brief
 */

#include <ctime>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <map>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include "assert.h"

#include "graph.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/embedding_bag_proto.h"

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
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return 2;
    }
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return 4;
    }
    if (dt == ge::DT_DOUBLE || dt == ge::DT_INT64 || dt == ge::DT_UINT64 || dt == ge::DT_COMPLEX64) {
        return 8;
    }
    if (dt == ge::DT_COMPLEX128) {
        return 16;
    }
    return 1;
}

template <typename T>
int32_t GenTensorData(const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                      const vector<T>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto dim : shapes) {
        size *= dim;
    }
    if (size != values.size()) {
        printf("%s - ERROR - [XIR]: Input data size %zu does not match shape size %zu\n", GetTime().c_str(),
               values.size(), size);
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

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, uint8_t* input_data)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

void PrintTensorData(DataType dtype, uint8_t* output_data, int64_t output_shape_size)
{
    if (dtype == ge::DT_FLOAT) {
        auto* result_data = reinterpret_cast<float*>(output_data);
        for (int64_t j = 0; j < output_shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result_data[j]);
        }
    } else if (dtype == ge::DT_INT32) {
        auto* result_data = reinterpret_cast<int32_t*>(output_data);
        for (int64_t j = 0; j < output_shape_size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, result_data[j]);
        }
    } else if (dtype == ge::DT_INT64) {
        auto* result_data = reinterpret_cast<int64_t*>(output_data);
        for (int64_t j = 0; j < output_shape_size; j++) {
            LOG_PRINT("result[%ld] is: %ld\n", j, result_data[j]);
        }
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        auto* result_data = reinterpret_cast<uint16_t*>(output_data);
        for (int64_t j = 0; j < output_shape_size; j++) {
            LOG_PRINT("result[%ld] is: 0x%04x\n", j, result_data[j]);
        }
    }
}

#define ADD_INPUT(input_index, input_name, input_dtype, input_shape, input_data)                                    \
    do {                                                                                                            \
        auto placeholder = op::Data("placeholder" + std::to_string(input_index)).set_attr_index((input_index) - 1); \
        TensorDesc desc(ge::Shape(input_shape), FORMAT_ND, input_dtype);                                            \
        desc.SetPlacement(ge::kPlacementHost);                                                                      \
        desc.SetFormat(FORMAT_ND);                                                                                  \
        Tensor tensor;                                                                                              \
        ret = GenTensorData(input_shape, tensor, desc, input_data);                                                 \
        if (ret != SUCCESS) {                                                                                       \
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                          \
            return FAILED;                                                                                          \
        }                                                                                                           \
        placeholder.update_input_desc_x(desc);                                                                      \
        placeholder.update_output_desc_y(desc);                                                                     \
        input.push_back(tensor);                                                                                    \
        graph.AddOp(placeholder);                                                                                   \
        add1.set_input_##input_name(placeholder);                                                                   \
        inputs.push_back(placeholder);                                                                              \
    } while (0)

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::EmbeddingBag("add1");
    std::vector<int64_t> weightShape = {4, 2};
    std::vector<int64_t> indicesShape = {4};
    std::vector<int64_t> offsetsShape = {2};
    std::vector<int64_t> perSampleWeightsShape = {4};
    std::vector<int64_t> yShape = {2, 2};

    std::vector<float> weightData(weightShape[0] * weightShape[1], 1.0f);
    std::vector<int32_t> indicesData = {0, 1, 2, 3};
    std::vector<int32_t> offsetsData = {0, 2};
    std::vector<float> perSampleWeightsData(perSampleWeightsShape[0], 1.0f);

    ADD_INPUT(1, weight, inDtype, weightShape, weightData);
    ADD_INPUT(2, indices, DT_INT32, indicesShape, indicesData);
    ADD_INPUT(3, offsets, DT_INT32, offsetsShape, offsetsData);
    ADD_INPUT(4, per_sample_weights, inDtype, perSampleWeightsShape, perSampleWeightsData);
    add1.set_attr_mode("sum");
    add1.set_attr_scale_grad_by_freq(false);
    add1.set_attr_sparse(false);
    add1.set_attr_include_last_offset(false);
    add1.set_attr_padding_idx(-1);
    add1.update_output_desc_y(TensorDesc(ge::Shape(yShape), FORMAT_ND, inDtype));
    add1.update_output_desc_offset2bag(TensorDesc(ge::Shape(indicesShape), FORMAT_ND, DT_INT32));
    add1.update_output_desc_bag_size(TensorDesc(ge::Shape(offsetsShape), FORMAT_ND, DT_INT32));
    add1.update_output_desc_max_indices(TensorDesc(ge::Shape(offsetsShape), FORMAT_ND, DT_INT32));

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
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
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };
    uint32_t graph_id = 0;
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

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
        PrintTensorData(output[i].GetTensorDesc().GetDataType(), output_data_i, output_shape);
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
