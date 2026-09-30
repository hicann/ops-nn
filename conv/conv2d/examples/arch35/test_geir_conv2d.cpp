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
 * \file test_geir_conv2d.cpp
 * \brief Conv2D GE IR example for arch35.
 */

#include <cstdio>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/conv2d_proto.h"

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

    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    } else if (dt == ge::DT_INT64 || dt == ge::DT_UINT64) {
        return eightByte;
    }
    return oneByte;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = static_cast<uint32_t>(size) * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + std::to_string(inputIndex))                                \
                                       .set_attr_index((inputIndex) - 1);                                              \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_NCHW,    \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_NCHW);                                                             \
    placeholder##inputIndex##_desc.SetOriginFormat(FORMAT_NCHW);                                                       \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2);                                                                                  \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate inputTensors data failed\n", GetTime().c_str());                          \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    inputTensors.push_back(tensor_placeholder##inputIndex);                                                            \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    tmpOp.set_input_##inputName(placeholder##inputIndex);                                                              \
    tmpOp.update_input_desc_##inputName(placeholder##inputIndex##_desc);                                               \
    inputOps.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                             \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), ge::FORMAT_NCHW, outputDtype); \
    tmpOp.update_output_desc_##outputName(outputName##outputIndex##_desc)

int CreateConv2DInGraph(std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputOps,
                        std::vector<Operator>& outputOps, Graph& graph)
{
    Status ret = SUCCESS;
    auto tmpOp = op::Conv2D("conv2d");
    std::vector<int64_t> xShape = {1, 32, 16, 16};
    std::vector<int64_t> filterShape = {64, 32, 3, 3};
    std::vector<int64_t> yShape = {1, 64, 16, 16};
    tmpOp.set_attr_strides({1, 1, 1, 1});
    tmpOp.set_attr_pads({1, 1, 1, 1});
    tmpOp.set_attr_dilations({1, 1, 1, 1});
    tmpOp.set_attr_groups(1);
    tmpOp.set_attr_data_format("NCHW");
    tmpOp.set_attr_offset_x(0);
    ADD_INPUT(1, x, ge::DT_FLOAT16, xShape);
    ADD_INPUT(2, filter, ge::DT_FLOAT16, filterShape);
    ADD_OUTPUT(1, y, ge::DT_FLOAT16, yShape);
    outputOps.push_back(tmpOp);
    if (!inputOps.empty() && !outputOps.empty()) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test_conv2d";
    Graph graph(graph_name);

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputOps{};
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> outputOps{};
    ret = CreateConv2DInGraph(inputTensors, inputOps, outputOps, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return FAILED;
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
    std::string file_path = "./conv2d_dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, inputTensors, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = static_cast<int>(inputTensors.size());
    for (int i = 0; i < input_num; i++) {
        string input_file = "./tc_ge_irrun_test_conv2d_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = inputTensors[i].GetData();
        int64_t input_shape = inputTensors[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(input_shape) *
                             GetDataTypeSize(inputTensors[i].GetTensorDesc().GetDataType());
        WriteDataToFile(input_file, data_size, input_data_i);
    }

    int output_num = static_cast<int>(output.size());
    for (int i = 0; i < output_num; i++) {
        string output_file = "./tc_ge_irrun_test_conv2d_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(output_shape) *
                             GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
    }

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
