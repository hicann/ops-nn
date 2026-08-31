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
#include <cstdio>
#include <ctime>
#include <new>
#include <map>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/batch_norm_grad_ext2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, inputFormat)                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                   \
    auto placeholder##inputIndex = op::Data(std::string("placeholder") + std::to_string(inputIndex))                \
                                       .set_attr_index(0);                                                          \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), inputFormat, \
                                                           inputDtype);                                             \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                \
    placeholder##inputIndex##_desc.SetFormat(inputFormat);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                          \
    ret = GenOnesDataFloat32(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                       \
                             placeholder##inputIndex##_desc, 2);                                                    \
    if (ret != SUCCESS) {                                                                                           \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                    \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                   \
    input.push_back(tensor_placeholder##inputIndex);                                                                \
    graph.AddOp(placeholder##inputIndex);                                                                           \
    batchNorm_1.set_input_##inputName(placeholder##inputIndex);                                                     \
    inputs.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape, outputFormat)                             \
    TensorDesc outputName##outputIndex##_desc_ = TensorDesc(ge::Shape(outputShape), outputFormat, outputDtype); \
    batchNorm_1.update_output_desc_##outputName(outputName##outputIndex##_desc_);

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
    if (dt == ge::DT_FLOAT) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16) {
        return 2;
    }
    return 1;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto dim : shapes) {
        size *= static_cast<size_t>(dim);
    }
    auto* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto batchNorm_1 = op::BatchNormGradExt2("batchNorm_1");
    std::vector<int64_t> xShape = {2, 4, 5, 3};
    std::vector<int64_t> channelShape = {3};
    ADD_INPUT(1, y_backprop, inDtype, xShape, FORMAT_NHWC);
    ADD_INPUT(2, x, inDtype, xShape, FORMAT_NHWC);
    ADD_INPUT(3, scale, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_INPUT(4, reserve_space_1, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_INPUT(5, reserve_space_2, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_OUTPUT(1, x_backprop, inDtype, xShape, FORMAT_NHWC);
    ADD_OUTPUT(2, scale_backprop, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_OUTPUT(3, offset_backprop, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_OUTPUT(4, reserve_space_3, DT_FLOAT, channelShape, FORMAT_ND);
    ADD_OUTPUT(5, reserve_space_4, DT_FLOAT, channelShape, FORMAT_ND);
    batchNorm_1.set_attr_epsilon(1e-4f);
    batchNorm_1.set_attr_data_format("NHWC");
    batchNorm_1.set_attr_is_training(true);
    outputs.push_back(batchNorm_1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    std::cout << argv[1] << std::endl;
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    LOG_PRINT("init ge\n");
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("GEInitialize failed\n");
        return FAILED;
    }

    DataType inDtype = DT_FLOAT;
    LOG_PRINT("build graph\n");
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("CreateOppInGraph failed\n");
        GEFinalize();
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    LOG_PRINT("create session\n");
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("create session failed\n");
        GEFinalize();
        return FAILED;
    }
    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    LOG_PRINT("add graph\n");
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        LOG_PRINT("AddGraph failed\n");
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        LOG_PRINT("GE error: %s\n", error_msg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }

    LOG_PRINT("run graph\n");
    aclgrphDumpGraph(graph, "./dump", 5);
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("RunGraph failed\n");
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        LOG_PRINT("GE error: %s\n", error_msg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
    }

    delete session;
    GEFinalize();
    return SUCCESS;
}
