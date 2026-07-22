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
#include "../../op_graph/inplace_apply_adagrad_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

// 数据类型字节大小
constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t FLOAT16_BYTES = sizeof(uint16_t);
constexpr uint32_t INT32_BYTES = sizeof(int32_t);
constexpr uint32_t INT64_BYTES = sizeof(int64_t);
// 时间字符串缓冲区大小
constexpr int32_t TIME_STR_BUF_SIZE = 64;
// var + accum 两个 tensor 输入
constexpr int32_t TENSOR_INPUT_COUNT = 2;

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
    // 算子仅支持 FLOAT，examples 测试数据也用 FLOAT
    if (dt == ge::DT_FLOAT) {
        return FLOAT_BYTES;
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

int CreateOppInGraph(DataType varDtype, std::vector<ge::Tensor> &input,
                     std::vector<Operator> &inputs, std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;
    auto v2DOp = op::InplaceApplyAdagradV2("inplace_apply_adagrad_v2");

    // 输入顺序对齐 CANNDEV ApplyAdagradV2D：var, accum, lr, grad
    // var, accum (2 tensors, same shape, same dtype)
    std::vector<int64_t> tensorShape = {8};
    float tensorValues[] = {1.0f, 0.5f}; // var, accum
    for (int32_t i = 0; i < TENSOR_INPUT_COUNT; i++) {
        auto dataOp = op::Data("placeholder" + std::to_string(i));
        TensorDesc desc = TensorDesc(ge::Shape(tensorShape), FORMAT_ND, varDtype);
        desc.SetPlacement(ge::kPlacementHost);
        Tensor tensor;
        ret = GenData(tensorShape, tensor, desc, tensorValues[i]);
        if (ret != SUCCESS) { return FAILED; }
        dataOp.update_input_desc_x(desc);
        dataOp.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(dataOp);
        if (i == 0) { v2DOp.set_input_var(dataOp); }
        else if (i == 1) { v2DOp.set_input_accum(dataOp); }
        inputs.push_back(dataOp);
    }

    // lr (1 scalar tensor, dtype must be FLOAT per proto)
    {
        auto dataOp = op::Data("placeholder2");
        TensorDesc desc = TensorDesc(ge::Shape({1}), FORMAT_ND, DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        Tensor tensor;
        ret = GenScalarData(tensor, desc, 0.01f); // lr
        if (ret != SUCCESS) { return FAILED; }
        dataOp.update_input_desc_x(desc);
        dataOp.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(dataOp);
        v2DOp.set_input_lr(dataOp);
        inputs.push_back(dataOp);
    }

    // grad (1 tensor, same shape/dtype as var)
    {
        auto dataOp = op::Data("placeholder3");
        TensorDesc desc = TensorDesc(ge::Shape(tensorShape), FORMAT_ND, varDtype);
        desc.SetPlacement(ge::kPlacementHost);
        Tensor tensor;
        ret = GenData(tensorShape, tensor, desc, 0.2f); // grad
        if (ret != SUCCESS) { return FAILED; }
        dataOp.update_input_desc_x(desc);
        dataOp.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(dataOp);
        v2DOp.set_input_grad(dataOp);
        inputs.push_back(dataOp);
    }

    // epsilon attribute (REQUIRED_ATTR, Float)
    v2DOp.set_attr_epsilon(1e-10f);
    // update_slots attribute (default true)
    v2DOp.set_attr_update_slots(true);
    // use_locking attribute (default false)
    v2DOp.set_attr_use_locking(false);

    outputs.push_back(v2DOp);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType varDtype = DT_FLOAT;

    ret = CreateOppInGraph(varDtype, input, inputs, outputs, graph);
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
