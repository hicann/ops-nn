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
 * \file test_geir_cross_entropy_sum_exp_and_index_logit.cpp
 * \brief geir(graph mode) test for CrossEntropySumExpAndIndexLogit
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
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

#include "../op_graph/cross_entropy_sum_exp_and_index_logit_proto.h"

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
    uint32_t dataTypeSize = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dataTypeSize = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dataTypeSize = twoByte;
    } else if (dt == ge::DT_BF16) {
        dataTypeSize = twoByte;
    } else if (dt == ge::DT_INT16) {
        dataTypeSize = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dataTypeSize = twoByte;
    } else if (dt == ge::DT_INT32) {
        dataTypeSize = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dataTypeSize = fourByte;
    } else if (dt == ge::DT_INT64) {
        dataTypeSize = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dataTypeSize = eightByte;
    } else if (dt == ge::DT_INT8) {
        dataTypeSize = oneByte;
    }
    return dataTypeSize;
}

// 生成 float32 输入数据（本示例输入仅使用 DT_FLOAT，data_type 与 host_data 保持一致）
int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                     const vector<float>& host_data)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = host_data[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

// 生成 int32 输入数据
int32_t GenInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc,
                     const vector<int32_t>& host_data)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(ge::DT_INT32);
    int32_t* pData = new (std::nothrow) int32_t[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = host_data[i];
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto op1 = op::CrossEntropySumExpAndIndexLogit("CrossEntropySumExpAndIndexLogit1");

    // 示例使用二维 shape：N=4, V_local=16（FLOAT32 时 V_local 需为 8 的倍数）
    const int64_t N = 4;
    const int64_t vLocal = 16;
    int64_t vocabStartIndex = 0;
    int64_t vocabEndIndex = vLocal; // target 均落在 [0, 16) 内，target_mask 全 0

    std::vector<int64_t> logitsShape = {N, vLocal};
    std::vector<int64_t> targetShape = {N};

    // vocab_parallel_logits: float32, shape [N, V_local]
    vector<float> logitsHostData(N * vLocal, 0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < vLocal; j++) {
            logitsHostData[i * vLocal + j] = static_cast<float>(i) * 1.0f + static_cast<float>(j) * 0.1f;
        }
    }
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(logitsShape), FORMAT_ND, inDtype);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenFloatData(logitsShape, tensor_placeholder1, placeholder1_desc, inDtype, logitsHostData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate vocab_parallel_logits data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    op1.set_input_vocab_parallel_logits(placeholder1);
    inputs.push_back(placeholder1);

    // target: int32, shape [N]，全局 vocab 索引
    vector<int32_t> targetHostData = {2, 5, 10, 15};
    auto placeholder2 = op::Data("placeholder2").set_attr_index(1);
    TensorDesc placeholder2_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, ge::DT_INT32);
    placeholder2_desc.SetPlacement(ge::kPlacementHost);
    placeholder2_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder2;
    ret = GenInt32Data(targetShape, tensor_placeholder2, placeholder2_desc, targetHostData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate target data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder2.update_input_desc_x(placeholder2_desc);
    placeholder2.update_output_desc_y(placeholder2_desc);
    input.push_back(tensor_placeholder2);
    graph.AddOp(placeholder2);
    op1.set_input_target(placeholder2);
    inputs.push_back(placeholder2);

    // global_logits_max: float32, shape [N]
    vector<float> maxHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    auto placeholder3 = op::Data("placeholder3").set_attr_index(2);
    TensorDesc placeholder3_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, inDtype);
    placeholder3_desc.SetPlacement(ge::kPlacementHost);
    placeholder3_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder3;
    ret = GenFloatData(targetShape, tensor_placeholder3, placeholder3_desc, inDtype, maxHostData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate global_logits_max data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder3.update_input_desc_x(placeholder3_desc);
    placeholder3.update_output_desc_y(placeholder3_desc);
    input.push_back(tensor_placeholder3);
    graph.AddOp(placeholder3);
    op1.set_input_global_logits_max(placeholder3);
    inputs.push_back(placeholder3);

    // 属性：当前 rank vocab 分片范围 [vocab_start_index, vocab_end_index)
    op1.set_attr_vocab_start_index(vocabStartIndex);
    op1.set_attr_vocab_end_index(vocabEndIndex);

    // 输出 desc：predicted_logits/sum_exp_logits/target_offset/target_mask 与 target 同 shape，
    // exp_logits 与 vocab_parallel_logits 同 shape
    TensorDesc predicted_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, ge::DT_FLOAT);
    op1.update_output_desc_predicted_logits(predicted_desc);
    TensorDesc sum_exp_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, ge::DT_FLOAT);
    op1.update_output_desc_sum_exp_logits(sum_exp_desc);
    TensorDesc exp_logits_desc = TensorDesc(ge::Shape(logitsShape), FORMAT_ND, ge::DT_FLOAT);
    op1.update_output_desc_exp_logits(exp_logits_desc);
    TensorDesc target_offset_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, ge::DT_INT32);
    op1.update_output_desc_target_offset(target_offset_desc);
    TensorDesc target_mask_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, ge::DT_INT32);
    op1.update_output_desc_target_mask(target_mask_desc);

    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "e2e_verify_geir_cross_entropy_sum_exp_and_index_logit";
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
    const char* binaryConfigPath = std::getenv("BINARY_CONFIG_PATH");
    if (binaryConfigPath != nullptr && strlen(binaryConfigPath) > 0) {
        build_options["ge.binary_config_path"] = binaryConfigPath;
        printf("%s - INFO - [XIR]: Use BINARY_CONFIG_PATH=%s\n", GetTime().c_str(), binaryConfigPath);
    } else {
        printf("%s - WARN - [XIR]: BINARY_CONFIG_PATH not set, GE will use default search path\n", GetTime().c_str());
    }
    printf("%s - INFO - [XIR]: Start to create session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create session success\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Add graph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::string error_str(error_msg.GetString());
        std::cout << "Error message: " << error_str << std::endl;
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        DataType dt = output[i].GetTensorDesc().GetDataType();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint8_t* output_data_i = output[i].GetData();
        std::cout << "output " << i << " shape size = " << output_shape << ", dtype = " << dt << std::endl;
        if (dt == ge::DT_INT32) {
            int32_t* resultData = reinterpret_cast<int32_t*>(output_data_i);
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
            }
        } else {
            float* resultData = reinterpret_cast<float*>(output_data_i);
            for (int64_t j = 0; j < output_shape; j++) {
                LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
            }
        }
    }

    printf("%s - INFO - [XIR]: GE IR pathway verification PASSED\n", GetTime().c_str());

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
