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
 * \file test_geir_swiglu_group_quant.cpp
 * \brief
 */

#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/swiglu_group_quant_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr uint32_t UINT8_BYTE_SIZE = 1;
constexpr uint32_t FP16_BYTE_SIZE = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;
constexpr int64_t DST_TYPE_FP8_E4M3FN = static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN);

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return FP16_BYTE_SIZE;
    }
    if (dt == ge::DT_FLOAT8_E4M3FN || dt == ge::DT_FLOAT8_E5M2 || dt == ge::DT_FLOAT8_E8M0) {
        return UINT8_BYTE_SIZE;
    }
    return FP32_BYTE_SIZE;
}

int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& tensorDesc, DataType dataType)
{
    tensorDesc.SetRealDimCnt(shape.size());
    size_t elementNum = static_cast<size_t>(GetShapeSize(shape));
    size_t dataLen = elementNum * GetDataTypeSize(dataType);
    uint8_t* data = new (std::nothrow) uint8_t[dataLen];
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc input data failed\n", GetTime().c_str());
        return FAILED;
    }

    for (size_t i = 0; i < dataLen; ++i) {
        data[i] = static_cast<uint8_t>(i % 23);
    }
    tensor = Tensor(tensorDesc, data, dataLen);
    delete[] data;
    return SUCCESS;
}

struct SwigluGroupQuantCase {
    const char* name;
    int64_t quantMode;
    int64_t blockSize;
    bool roundScale;
    vector<int64_t> yScaleShape;
    DataType yScaleDtype;
};

void PrintTensorInfo(const char* caseName, const vector<Tensor>& tensors, const char* prefix)
{
    for (size_t i = 0; i < tensors.size(); ++i) {
        const TensorDesc& desc = tensors[i].GetTensorDesc();
        std::cout << caseName << " " << prefix << " " << i << " dtype: " << desc.GetDataType() << std::endl;
        std::cout << caseName << " " << prefix << " " << i << " shape size = " << desc.GetShape().GetShapeSize()
                  << std::endl;
    }
}

int CreateSwigluGroupQuantGraph(const SwigluGroupQuantCase& testCase, Graph& graph, vector<Tensor>& input,
                                vector<Operator>& inputs, vector<Operator>& outputs)
{
    Status ret = SUCCESS;
    auto swigluGroupQuant = op::SwigluGroupQuant("swiglu_group_quant");

    vector<int64_t> xShape = {2, 256};
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, ge::DT_FLOAT16);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);

    Tensor xTensor;
    ret = GenInputData(xShape, xTensor, xDesc, ge::DT_FLOAT16);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }

    auto xData = op::Data("x").set_attr_index(0);
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);
    swigluGroupQuant.set_input_x(xData);
    input.push_back(xTensor);
    inputs.push_back(xData);

    vector<int64_t> yShape = {2, 128};
    vector<int64_t> yOriginShape = {2, 128};
    TensorDesc yDesc = TensorDesc(ge::Shape(yShape), FORMAT_ND, ge::DT_FLOAT8_E4M3FN);
    TensorDesc yScaleDesc = TensorDesc(ge::Shape(testCase.yScaleShape), FORMAT_ND, testCase.yScaleDtype);
    TensorDesc yOriginDesc = TensorDesc(ge::Shape(yOriginShape), FORMAT_ND, ge::DT_FLOAT16);
    swigluGroupQuant.update_output_desc_y(yDesc);
    swigluGroupQuant.update_output_desc_y_scale(yScaleDesc);
    swigluGroupQuant.update_output_desc_y_origin(yOriginDesc);

    swigluGroupQuant.set_attr_dst_type(DST_TYPE_FP8_E4M3FN);
    swigluGroupQuant.set_attr_quant_mode(testCase.quantMode);
    swigluGroupQuant.set_attr_block_size(testCase.blockSize);
    swigluGroupQuant.set_attr_round_scale(testCase.roundScale);
    swigluGroupQuant.set_attr_clamp_limit(-1.0f);
    swigluGroupQuant.set_attr_dst_type_max(448.0f);
    swigluGroupQuant.set_attr_output_origin(false);

    outputs.push_back(swigluGroupQuant);
    return SUCCESS;
}

int RunSwigluGroupQuantCase(const SwigluGroupQuantCase& testCase)
{
    printf("%s - INFO - [XIR]: Run %s: quant_mode=%ld\n", GetTime().c_str(), testCase.name, testCase.quantMode);

    Graph graph(testCase.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    Status ret = CreateSwigluGroupQuantGraph(testCase, graph, input, inputs, outputs);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }

    vector<Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run %s graph failed\n", GetTime().c_str(), testCase.name);
        delete session;
        return FAILED;
    }

    PrintTensorInfo(testCase.name, input, "input");
    PrintTensorInfo(testCase.name, output, "output");
    printf("%s - INFO - [XIR]: Run %s graph success\n", GetTime().c_str(), testCase.name);
    delete session;
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<SwigluGroupQuantCase> testCases = {
        {"block_fp8", 0, 0, false, {2, 1}, ge::DT_FLOAT},
        {"mx_fp8", 1, 0, true, {2, 2, 2}, ge::DT_FLOAT8_E8M0},
    };

    for (const auto& testCase : testCases) {
        ret = RunSwigluGroupQuantCase(testCase);
        if (ret != SUCCESS) {
            (void)ge::GEFinalize();
            return FAILED;
        }
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return SUCCESS;
}
