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
#include "../../op_graph/inplace_apply_adagrad_da_proto.h"

#define FAILED -1
#define SUCCESS 0

// 数据类型字节大小
constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t FLOAT16_BYTES = sizeof(uint16_t);
constexpr uint32_t INT32_BYTES = sizeof(int32_t);
constexpr uint32_t INT64_BYTES = sizeof(int64_t);
// var, gradient_accumulator, gradient_squared_accumulator, grad (4 个 tensor 输入)
constexpr int32_t TENSOR_INPUT_COUNT = 4;
// lr, l1, l2 (3 个标量 Tensor 输入，索引 4~6)
constexpr int32_t SCALAR_INPUT_START = 4;
constexpr int32_t SCALAR_INPUT_END = 7;
// 时间字符串缓冲区大小
constexpr int32_t TIME_STR_BUF_SIZE = 64;

using namespace ge;
using std::map;
using std::string;
using std::vector;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[TIME_STR_BUF_SIZE];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT) {
        return FLOAT_BYTES;
    } else if (dt == ge::DT_FLOAT16) {
        return FLOAT16_BYTES;
    } else if (dt == ge::DT_INT32) {
        return INT32_BYTES;
    } else if (dt == ge::DT_INT64) {
        return INT64_BYTES;
    }
    return FLOAT_BYTES;
}

int32_t GenData(vector<int64_t> shapes, Tensor &inputTensor, TensorDesc &desc, float value)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t dataLen = size * FLOAT_BYTES;
    float *pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    inputTensor = Tensor(desc, (uint8_t *)pData, dataLen);
    delete[] pData;
    return SUCCESS;
}

int32_t GenScalarData(Tensor &inputTensor, TensorDesc &desc, float value)
{
    desc.SetRealDimCnt(1);
    uint32_t dataLen = FLOAT_BYTES;
    float *pData = new (std::nothrow) float[1];
    if (pData == nullptr) {
        return FAILED;
    }
    *pData = value;
    inputTensor = Tensor(desc, (uint8_t *)pData, dataLen);
    delete[] pData;
    return SUCCESS;
}

int32_t GenIntScalarData(Tensor &inputTensor, TensorDesc &desc, int32_t value)
{
    desc.SetRealDimCnt(1);
    uint32_t dataLen = INT32_BYTES;
    int32_t *pData = new (std::nothrow) int32_t[1];
    if (pData == nullptr) {
        return FAILED;
    }
    *pData = value;
    inputTensor = Tensor(desc, (uint8_t *)pData, dataLen);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string binFile, uint64_t dataSize, uint8_t *inputData)
{
    FILE *fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType varDtype, DataType gsDtype, std::vector<ge::Tensor> &input,
                     std::vector<Operator> &inputs, std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto dadOp = op::InplaceApplyAdagradDA("inplace_apply_adagrad_da");

    // var, gradient_accumulator, gradient_squared_accumulator, grad (4 tensors, same shape)
    std::vector<int64_t> tensorShape = {4, 4};
    for (int32_t i = 0; i < TENSOR_INPUT_COUNT; i++) {
        auto dataOp = op::Data("placeholder" + std::to_string(i));
        TensorDesc desc = TensorDesc(ge::Shape(tensorShape), FORMAT_ND, varDtype);
        desc.SetPlacement(ge::kPlacementHost);
        Tensor tensor;
        ret = GenData(tensorShape, tensor, desc, 1.0f);
        if (ret != SUCCESS) { return FAILED; }
        dataOp.update_input_desc_x(desc);
        dataOp.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(dataOp);
        if (i == 0) { dadOp.set_input_var(dataOp); }
        else if (i == 1) { dadOp.set_input_gradient_accumulator(dataOp); }
        else if (i == 2) { dadOp.set_input_gradient_squared_accumulator(dataOp); }
        else if (i == 3) { dadOp.set_input_grad(dataOp); }
        inputs.push_back(dataOp);
    }

    // lr, l1, l2 (3 scalar tensors, same dtype as var)
    for (int32_t i = SCALAR_INPUT_START; i < SCALAR_INPUT_END; i++) {
        auto dataOp = op::Data("placeholder" + std::to_string(i));
        TensorDesc desc = TensorDesc(ge::Shape({1}), FORMAT_ND, varDtype);
        desc.SetPlacement(ge::kPlacementHost);
        Tensor tensor;
        float val = (i == 4) ? 0.01f : ((i == 5) ? 0.0f : 0.01f);
        ret = GenScalarData(tensor, desc, val);
        if (ret != SUCCESS) { return FAILED; }
        dataOp.update_input_desc_x(desc);
        dataOp.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(dataOp);
        if (i == 4) { dadOp.set_input_lr(dataOp); }
        else if (i == 5) { dadOp.set_input_l1(dataOp); }
        else if (i == 6) { dadOp.set_input_l2(dataOp); }
        inputs.push_back(dataOp);
    }

    // global_step (int32 scalar)
    auto gsOp = op::Data("placeholder7");
    TensorDesc gsDesc = TensorDesc(ge::Shape({1}), FORMAT_ND, gsDtype);
    gsDesc.SetPlacement(ge::kPlacementHost);
    Tensor gsTensor;
    ret = GenIntScalarData(gsTensor, gsDesc, 100);
    if (ret != SUCCESS) { return FAILED; }
    gsOp.update_input_desc_x(gsDesc);
    gsOp.update_output_desc_y(gsDesc);
    input.push_back(gsTensor);
    graph.AddOp(gsOp);
    dadOp.set_input_global_step(gsOp);
    inputs.push_back(gsOp);

    // use_locking attribute
    dadOp.set_attr_use_locking(false);

    outputs.push_back(dadOp);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}, {"ge.runFlag", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType varDtype = DT_FLOAT;
    DataType gsDtype = DT_INT32;

    ret = CreateOppInGraph(varDtype, gsDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session *session = new Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::string filePath = "./dump";
    aclgrphDumpGraph(graph, filePath.c_str(), filePath.length());

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    int outputNum = output.size();
    for (int i = 0; i < outputNum; i++) {
        std::cout << "output " << i << " dtype: " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string outputFile = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *outputData = output[i].GetData();
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t dataSize = outputShape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(outputFile, dataSize, outputData);
    }

    delete session;
    session = nullptr;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
