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
 * \file test_geir_poisson_nll_loss.cpp
 * \brief geir(graph mode) test for poisson_nll_loss, aligned with ascend910b IR
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
#include "../op_graph/poisson_nll_loss_proto.h"

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
    if (dt == ge::DT_FLOAT) {
        return 4;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return 1;
}

int32_t GenData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto poissonNllLoss1 = op::PoissonNllLoss("poissonNllLoss1");

    std::vector<int64_t> inputShape = {4, 2};
    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc placeholder1_desc = TensorDesc(ge::Shape(inputShape), FORMAT_ND, inDtype);
    placeholder1_desc.SetPlacement(ge::kPlacementHost);
    placeholder1_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder1;
    ret = GenData(inputShape, tensor_placeholder1, placeholder1_desc, inDtype, 0.5);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder1.update_input_desc_x(placeholder1_desc);
    placeholder1.update_output_desc_y(placeholder1_desc);
    input.push_back(tensor_placeholder1);
    graph.AddOp(placeholder1);
    poissonNllLoss1.set_input_input_x(placeholder1);
    inputs.push_back(placeholder1);

    std::vector<int64_t> targetShape = {4, 2};
    auto placeholder2 = op::Data("placeholder2").set_attr_index(1);
    TensorDesc placeholder2_desc = TensorDesc(ge::Shape(targetShape), FORMAT_ND, inDtype);
    placeholder2_desc.SetPlacement(ge::kPlacementHost);
    placeholder2_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder2;
    ret = GenData(targetShape, tensor_placeholder2, placeholder2_desc, inDtype, 1.0);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder2.update_input_desc_x(placeholder2_desc);
    placeholder2.update_output_desc_y(placeholder2_desc);
    input.push_back(tensor_placeholder2);
    graph.AddOp(placeholder2);
    poissonNllLoss1.set_input_target(placeholder2);
    inputs.push_back(placeholder2);

    // attrs: log_input / full / eps / reduction (defaults align with ascend910b)
    poissonNllLoss1.set_attr_log_input(true);
    poissonNllLoss1.set_attr_full(false);
    poissonNllLoss1.set_attr_eps(1e-8f);
    poissonNllLoss1.set_attr_reduction("mean");

    // reduction=mean -> scalar output
    std::vector<int64_t> lossShape = {1};
    TensorDesc loss_desc = TensorDesc(ge::Shape(lossShape), FORMAT_ND, inDtype);
    poissonNllLoss1.update_output_desc_loss(loss_desc);

    outputs.push_back(poissonNllLoss1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

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
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        float* resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
