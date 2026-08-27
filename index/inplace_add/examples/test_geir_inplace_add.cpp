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
 * \file test_geir_inplace_add.cpp
 * \brief GE IR example for InplaceAdd.
 */

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <new>
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

#include "../op_graph/inplace_add_proto.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr int32_t RET_FAILED = -1;
constexpr int32_t RET_SUCCESS = 0;
constexpr uint16_t kFloat16Two = 0x4000U;
constexpr uint16_t kFloat16Four = 0x4400U;
constexpr uint16_t kBfloat16Two = 0x4000U;
constexpr uint16_t kBfloat16Four = 0x4080U;

struct TestScenario {
    const char* name;
    DataType dtype{DT_UNDEFINED};
};

const vector<TestScenario> kTestScenarios = {{"float", DT_FLOAT}, {"float16", DT_FLOAT16}, {"bfloat16", DT_BF16},
                                             {"int8", DT_INT8},   {"uint8", DT_UINT8},     {"int32", DT_INT32}};
std::vector<std::unique_ptr<uint8_t[]>> g_inputHolder;

bool GetTestScenario(int argc, char* argv[], TestScenario& scenario)
{
    const string key = argc > 1 ? argv[1] : "0";
    for (size_t i = 0; i < kTestScenarios.size(); ++i) {
        if (key == kTestScenarios[i].name || key == std::to_string(i)) {
            scenario = kTestScenarios[i];
            return true;
        }
    }
    return false;
}

string GetTime()
{
    time_t timep{};
    if (time(&timep) == static_cast<time_t>(-1)) {
        return "unknown-time";
    }
    std::tm timeInfo{};
    if (localtime_r(&timep, &timeInfo) == nullptr) {
        return "unknown-time";
    }
    char tmp[64]{};
    if (strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", &timeInfo) == 0) {
        return "unknown-time";
    }
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
            return sizeof(float);
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return sizeof(uint16_t);
        case ge::DT_INT8:
            return sizeof(int8_t);
        case ge::DT_UINT8:
            return sizeof(uint8_t);
        case ge::DT_INT32:
            return sizeof(int32_t);
        case ge::DT_INT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

bool GetShapeSize(const vector<int64_t>& shape, size_t& size)
{
    size = 1;
    for (auto dim : shape) {
        if (dim < 0 || (size != 0 && static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max() / size)) {
            return false;
        }
        size *= static_cast<size_t>(dim);
    }
    return true;
}

template <typename T>
int32_t GenScalarData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, T value)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 0;
    if (!GetShapeSize(shape, size) || size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return RET_FAILED;
    }
    size_t dataSize = size * sizeof(T);
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[dataSize]);
    if (data == nullptr) {
        return RET_FAILED;
    }
    auto typedData = reinterpret_cast<T*>(data.get());
    for (size_t i = 0; i < size; ++i) {
        typedData[i] = value;
    }
    tensor = Tensor(desc, data.get(), dataSize);
    g_inputHolder.emplace_back(std::move(data));
    return RET_SUCCESS;
}

int32_t GenNumericData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return GenScalarData<float>(shape, tensor, desc, 2.0f);
        case ge::DT_FLOAT16:
            return GenScalarData<uint16_t>(shape, tensor, desc, kFloat16Two);
        case ge::DT_BF16:
            return GenScalarData<uint16_t>(shape, tensor, desc, kBfloat16Two);
        case ge::DT_INT8:
            return GenScalarData<int8_t>(shape, tensor, desc, static_cast<int8_t>(2));
        case ge::DT_UINT8:
            return GenScalarData<uint8_t>(shape, tensor, desc, static_cast<uint8_t>(2));
        case ge::DT_INT32:
            return GenScalarData<int32_t>(shape, tensor, desc, static_cast<int32_t>(2));
        case ge::DT_INT64:
            return GenScalarData<int64_t>(shape, tensor, desc, static_cast<int64_t>(2));
        default:
            return RET_FAILED;
    }
}

int32_t GenInt32Indices(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 0;
    if (!GetShapeSize(shape, size) || size > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        return RET_FAILED;
    }
    size_t dataSize = size * sizeof(int32_t);
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[dataSize]);
    if (data == nullptr) {
        return RET_FAILED;
    }
    auto indicesData = reinterpret_cast<int32_t*>(data.get());
    for (size_t i = 0; i < size; ++i) {
        indicesData[i] = static_cast<int32_t>(i);
    }
    tensor = Tensor(desc, data.get(), dataSize);
    g_inputHolder.emplace_back(std::move(data));
    return RET_SUCCESS;
}

int32_t WriteDataToFile(const string& binFile, uint64_t dataSize, uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "wb");
    if (fp == nullptr) {
        return RET_FAILED;
    }
    const size_t written = fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    const int closeResult = fclose(fp);
    return written == dataSize && closeResult == 0 ? RET_SUCCESS : RET_FAILED;
}

template <typename T>
bool CheckScalarOutput(const Tensor& output, const vector<int64_t>& expectedShape, T originalValue, T updatedValue)
{
    const auto* values = reinterpret_cast<const T*>(output.GetData());
    for (size_t row = 0; row < static_cast<size_t>(expectedShape[0]); ++row) {
        const T expected = row < 2 ? updatedValue : originalValue;
        for (size_t column = 0; column < static_cast<size_t>(expectedShape[1]); ++column) {
            const size_t offset = row * static_cast<size_t>(expectedShape[1]) + column;
            if (values[offset] != expected) {
                return false;
            }
        }
    }
    return true;
}

bool CheckOutput(const Tensor& output, DataType dtype)
{
    const vector<int64_t> expectedShape = {4, 6};
    size_t expectedElementCount = 0;
    const uint32_t dtypeSize = GetDataTypeSize(dtype);
    if (!GetShapeSize(expectedShape, expectedElementCount) || dtypeSize == 0 ||
        expectedElementCount > std::numeric_limits<size_t>::max() / dtypeSize) {
        return false;
    }
    const size_t expectedDataSize = expectedElementCount * dtypeSize;
    if (output.GetTensorDesc().GetDataType() != dtype || output.GetTensorDesc().GetShape().GetDims() != expectedShape ||
        output.GetSize() != expectedDataSize || output.GetData() == nullptr) {
        return false;
    }

    switch (dtype) {
        case ge::DT_FLOAT:
            return CheckScalarOutput<float>(output, expectedShape, 2.0F, 4.0F);
        case ge::DT_FLOAT16:
            return CheckScalarOutput<uint16_t>(output, expectedShape, kFloat16Two, kFloat16Four);
        case ge::DT_BF16:
            return CheckScalarOutput<uint16_t>(output, expectedShape, kBfloat16Two, kBfloat16Four);
        case ge::DT_INT8:
            return CheckScalarOutput<int8_t>(output, expectedShape, static_cast<int8_t>(2), static_cast<int8_t>(4));
        case ge::DT_UINT8:
            return CheckScalarOutput<uint8_t>(output, expectedShape, static_cast<uint8_t>(2), static_cast<uint8_t>(4));
        case ge::DT_INT32:
            return CheckScalarOutput<int32_t>(output, expectedShape, static_cast<int32_t>(2), static_cast<int32_t>(4));
        default:
            return false;
    }
}

template <typename DataOp>
int32_t AddDataInput(Graph& graph, vector<Tensor>& input, vector<Operator>& inputs, const string& name, uint32_t index,
                     DataType dtype, const vector<int64_t>& shape, bool isIndices, Tensor& tensor, DataOp& dataOp)
{
    dataOp = op::Data(name.c_str()).set_attr_index(index);
    TensorDesc desc(ge::Shape(shape), FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(FORMAT_ND);

    int32_t ret = RET_SUCCESS;
    if (isIndices) {
        ret = GenInt32Indices(shape, tensor, desc);
    } else {
        ret = GenNumericData(shape, tensor, desc, dtype);
    }
    if (ret != RET_SUCCESS) {
        return ret;
    }

    dataOp.update_input_desc_x(desc);
    dataOp.update_output_desc_y(desc);
    input.push_back(tensor);
    graph.AddOp(dataOp);
    inputs.push_back(dataOp);
    return RET_SUCCESS;
}

int32_t CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph,
                         const TestScenario& scenario)
{
    auto inplaceAdd = op::InplaceAdd("inplaceAdd1");
    // Run every scenario through the public GE graph and verify the result on supported products.
    DataType xDtype = scenario.dtype;
    vector<int64_t> xShape = {4, 6};
    vector<int64_t> indicesShape = {2};
    vector<int64_t> vShape = {2, 6};

    Tensor xTensor;
    Tensor indicesTensor;
    Tensor vTensor;
    op::Data xData = op::Data("data_x");
    op::Data indicesData = op::Data("data_indices");
    op::Data vData = op::Data("data_v");

    if (AddDataInput(graph, input, inputs, "data_x", 0, xDtype, xShape, false, xTensor, xData) != RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddDataInput(graph, input, inputs, "data_indices", 1, DT_INT32, indicesShape, true, indicesTensor,
                     indicesData) != RET_SUCCESS) {
        return RET_FAILED;
    }
    if (AddDataInput(graph, input, inputs, "data_v", 2, xDtype, vShape, false, vTensor, vData) != RET_SUCCESS) {
        return RET_FAILED;
    }

    inplaceAdd.set_input_x(xData);
    inplaceAdd.set_input_indices(indicesData);
    inplaceAdd.set_input_v(vData);
    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, xDtype);
    inplaceAdd.update_output_desc_y(yDesc);
    outputs.push_back(inplaceAdd);
    return RET_SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    TestScenario scenario = kTestScenarios.front();
    if (!GetTestScenario(argc, argv, scenario)) {
        printf("FAIL - unsupported scenario '%s'; use 0..5 or float/float16/bfloat16/int8/uint8/int32\n",
               argc > 1 ? argv[1] : "");
        return RET_FAILED;
    }
    Graph graph("tc_ge_irrun_test");
    vector<Tensor> input;

    printf("%s - INFO - [XIR]: Start %s scenario and initialize ge\n", GetTime().c_str(), scenario.name);
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return RET_FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    if (CreateOppInGraph(input, inputs, outputs, graph, scenario) != RET_SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        GEFinalize();
        return RET_FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> buildOptions;
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return RET_FAILED;
    }

    uint32_t graphId = 0;
    map<AscendString, AscendString> graphOptions;
    if (session->AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return RET_FAILED;
    }

    vector<Tensor> output;
    if (session->RunGraph(graphId, input, output) != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return RET_FAILED;
    }
    if (output.size() != 1 || !CheckOutput(output[0], scenario.dtype)) {
        printf("%s - ERROR - [XIR]: Output verification failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return RET_FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    for (size_t i = 0; i < output.size(); ++i) {
        string outputFile = "./tc_ge_irrun_test_" + string(scenario.name) + "_npu_output_" + std::to_string(i) + ".bin";
        size_t outputShapeSize = 0;
        if (!GetShapeSize(output[i].GetTensorDesc().GetShape().GetDims(), outputShapeSize)) {
            delete session;
            ge::GEFinalize();
            return RET_FAILED;
        }
        uint32_t dtypeSize = GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (dtypeSize == 0 || outputShapeSize > std::numeric_limits<uint64_t>::max() / dtypeSize) {
            delete session;
            ge::GEFinalize();
            return RET_FAILED;
        }
        uint64_t dataSize = static_cast<uint64_t>(outputShapeSize) * dtypeSize;
        if (WriteDataToFile(outputFile, dataSize, output[i].GetData()) != RET_SUCCESS) {
            delete session;
            ge::GEFinalize();
            return RET_FAILED;
        }
    }

    delete session;
    if (ge::GEFinalize() != ge::SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return RET_FAILED;
    }
    printf("PASS - InplaceAdd GEIR %s scenario\n", scenario.name);
    return RET_SUCCESS;
}
