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
 * \file test_geir_apply_adagrad_d.cpp
 * \brief GE IR runner example for ApplyAdagradD (experimental/optim).
 *
 * Runs a simple shape={1} test with DT_FLOAT to verify the operator end-to-end.
 * Expected result (update_slots=true):
 *   accum_out = accum + grad*grad = 1.0 + 2.0*2.0 = 5.0
 *   var_out   = var - lr*grad/sqrt(accum_out) = 1.0 - 0.1*2.0/sqrt(5.0) ≈ 0.9107
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
#include "../op_graph/apply_adagrad_d_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

// ADD_INPUT: dynamic input, filled with float value `fillVal`
#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, fillVal)                                         \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                 \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND, \
                                                           inputDtype);                                           \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                              \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                        \
    ret = GenOnesDataFloat32(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                     \
                             placeholder##inputIndex##_desc, static_cast<float>(fillVal));                        \
    if (ret != SUCCESS) {                                                                                         \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                            \
        return FAILED;                                                                                            \
    }                                                                                                             \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                  \
    placeholder##inputIndex.update_output_desc_y(placeholder##inputIndex##_desc);                                 \
    input.push_back(tensor_placeholder##inputIndex);                                                              \
    graph.AddOp(placeholder##inputIndex);                                                                         \
    applyAdagradD1.set_input_##inputName(placeholder##inputIndex);                                                \
    inputs.push_back(placeholder##inputIndex)

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_BF16)
        return 2;
    if (dt == ge::DT_INT32)
        return 4;
    if (dt == ge::DT_INT64)
        return 8;
    return 1;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = static_cast<uint32_t>(size) * 4;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i)
        pData[i] = value;
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
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
    auto applyAdagradD1 = op::ApplyAdagradD("applyAdagradD1");
    applyAdagradD1.SetAttr("update_slots", true);

    std::vector<int64_t> varShape = {4, 4}; // var, accum, grad: same shape
    std::vector<int64_t> lrShape = {1};     // lr: scalar

    // var=1.0, accum=1.0, lr=0.1, grad=2.0
    ADD_INPUT(1, var, inDtype, varShape, 1.0f);
    ADD_INPUT(2, accum, inDtype, varShape, 1.0f);
    ADD_INPUT(3, lr, inDtype, lrShape, 0.1f);
    ADD_INPUT(4, grad, inDtype, varShape, 2.0f);

    outputs.push_back(applyAdagradD1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_apply_adagrad_d";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize GE\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // Default dtype: DT_FLOAT; pass "fp16" as argv[1] for float16
    DataType inDtype = DT_FLOAT;
    if (argc > 1 && std::string(argv[1]) == "fp16") {
        inDtype = DT_FLOAT16;
    } else if (argc > 1 && std::string(argv[1]) == "bf16") {
        inDtype = DT_BF16;
    }
    printf("Using dtype: %d\n", static_cast<int>(inDtype));

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: CreateOppInGraph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options{};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options{};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    std::string dump_path = "./dump";
    aclgrphDumpGraph(graph, dump_path.c_str(), dump_path.length());

    printf("%s - INFO - [XIR]: Running graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    int output_num = static_cast<int>(output.size());
    const char* output_names[] = {"var_out", "accum_out"};
    for (int i = 0; i < output_num; i++) {
        string output_file = string("./apply_adagrad_d_output_") + output_names[i] + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(output_shape) *
                             GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
        printf("\n[Output %d: %s] shape_size=%ld\n", i, output_names[i], output_shape);
        float* resultData = reinterpret_cast<float*>(output_data_i);
        for (int64_t j = 0; j < output_shape && j < 8; j++) {
            LOG_PRINT("  result[%ld] = %f\n", j, resultData[j]);
        }
    }
    // Expected (DT_FLOAT, var=1.0, accum=1.0, lr=0.1, grad=2.0, update_slots=true):
    //   accum_out = 1.0 + 2.0^2 = 5.0
    //   var_out   = 1.0 - 0.1*2.0/sqrt(5.0) = 1.0 - 0.08944 ≈ 0.9106

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    if (!error_str.empty())
        std::cout << "Error: " << error_str << std::endl;

    delete session;
    ret = ge::GEFinalize();
    return (ret == SUCCESS) ? SUCCESS : FAILED;
}
