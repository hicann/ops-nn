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
 * \file test_geir_swiglu_group_quant_with_dual_axis.cpp
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
#include "../op_graph/swiglu_group_quant_with_dual_axis_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr int64_t T = 64;
constexpr int64_t H = 64;
constexpr int64_t DIM = 2 * H;
constexpr int64_t GROUP_NUM = 2;
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
    if (dt == ge::DT_INT64) {
        return 8;
    }
    if (dt == ge::DT_FLOAT) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    return 1;
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

    if (dataType == ge::DT_INT64) {
        // group_index is a cumsum vector; the last endpoint must equal T.
        int64_t* values = reinterpret_cast<int64_t*>(data);
        for (size_t i = 0; i < elementNum; ++i) {
            values[i] = T * static_cast<int64_t>(i + 1) / static_cast<int64_t>(elementNum);
        }
    } else {
        for (size_t i = 0; i < dataLen; ++i) {
            data[i] = static_cast<uint8_t>(i % 23);
        }
    }
    tensor = Tensor(tensorDesc, data, dataLen);
    delete[] data;
    return SUCCESS;
}

struct SwigluGroupQuantWithDualAxisCase {
    const char* name;
    bool useGroup;
    bool useWeight;
    bool outputOrigin;
    float clampLimit;
    float alpha;
    float bias;
};

int CreateSwigluGroupQuantWithDualAxisGraph(const SwigluGroupQuantWithDualAxisCase& testCase, Graph& graph,
                                            vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs)
{
    int32_t ret = SUCCESS;
    auto swigluGroupQuantWithDualAxis = op::SwigluGroupQuantWithDualAxis("swiglu_group_quant_with_dual_axis");

    int64_t inputIndex = 0;

    vector<int64_t> xShape = {T, DIM};
    auto xData = op::Data("x").set_attr_index(inputIndex++);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), FORMAT_ND, ge::DT_FLOAT16);
    xDesc.SetPlacement(ge::kPlacementHost);
    Tensor xTensor;
    ret = GenInputData(xShape, xTensor, xDesc, ge::DT_FLOAT16);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    graph.AddOp(xData);
    swigluGroupQuantWithDualAxis.set_input_x(xData);
    input.push_back(xTensor);
    inputs.push_back(xData);

    // weight is only legal together with group_index.
    if (testCase.useWeight) {
        vector<int64_t> weightShape = {T};
        auto weightData = op::Data("weight").set_attr_index(inputIndex++);
        TensorDesc weightDesc = TensorDesc(ge::Shape(weightShape), FORMAT_ND, ge::DT_FLOAT16);
        weightDesc.SetPlacement(ge::kPlacementHost);
        Tensor weightTensor;
        ret = GenInputData(weightShape, weightTensor, weightDesc, ge::DT_FLOAT16);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: Generate weight data failed\n", GetTime().c_str());
            return FAILED;
        }
        weightData.update_input_desc_x(weightDesc);
        graph.AddOp(weightData);
        swigluGroupQuantWithDualAxis.set_input_weight(weightData);
        input.push_back(weightTensor);
        inputs.push_back(weightData);
    }

    if (testCase.useGroup) {
        vector<int64_t> groupIndexShape = {GROUP_NUM};
        auto groupIndexData = op::Data("group_index").set_attr_index(inputIndex++);
        TensorDesc groupIndexDesc = TensorDesc(ge::Shape(groupIndexShape), FORMAT_ND, ge::DT_INT64);
        groupIndexDesc.SetPlacement(ge::kPlacementHost);
        Tensor groupIndexTensor;
        ret = GenInputData(groupIndexShape, groupIndexTensor, groupIndexDesc, ge::DT_INT64);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: Generate group_index data failed\n", GetTime().c_str());
            return FAILED;
        }
        groupIndexData.update_input_desc_x(groupIndexDesc);
        graph.AddOp(groupIndexData);
        swigluGroupQuantWithDualAxis.set_input_group_index(groupIndexData);
        input.push_back(groupIndexTensor);
        inputs.push_back(groupIndexData);
    }

    vector<int64_t> yShape = {T, H};
    TensorDesc yDesc = TensorDesc(ge::Shape(yShape), FORMAT_ND, ge::DT_FLOAT8_E4M3FN);
    swigluGroupQuantWithDualAxis.update_output_desc_y1(yDesc);
    swigluGroupQuantWithDualAxis.update_output_desc_y2(yDesc);

    vector<int64_t> scale1Shape = {T, (H / 32 + 1) / 2, 2};
    TensorDesc scale1Desc = TensorDesc(ge::Shape(scale1Shape), FORMAT_ND, ge::DT_FLOAT8_E8M0);
    swigluGroupQuantWithDualAxis.update_output_desc_mxscale1(scale1Desc);

    int64_t scale2Rows = testCase.useGroup ? (T / 64 + GROUP_NUM) : (T / 64 + (T % 64 != 0 ? 1 : 0));
    vector<int64_t> scale2Shape = {scale2Rows, H, 2};
    TensorDesc scale2Desc = TensorDesc(ge::Shape(scale2Shape), FORMAT_ND, ge::DT_FLOAT8_E8M0);
    swigluGroupQuantWithDualAxis.update_output_desc_mxscale2(scale2Desc);

    TensorDesc originDesc = TensorDesc(ge::Shape(yShape), FORMAT_ND, ge::DT_FLOAT16);
    swigluGroupQuantWithDualAxis.update_output_desc_y_origin(originDesc);

    swigluGroupQuantWithDualAxis.set_attr_dst_type(DST_TYPE_FP8_E4M3FN);
    swigluGroupQuantWithDualAxis.set_attr_quant_mode(1);
    swigluGroupQuantWithDualAxis.set_attr_clamp_limit(testCase.clampLimit);
    swigluGroupQuantWithDualAxis.set_attr_output_origin(testCase.outputOrigin);
    swigluGroupQuantWithDualAxis.set_attr_alpha(testCase.alpha);
    swigluGroupQuantWithDualAxis.set_attr_bias(testCase.bias);

    outputs.push_back(swigluGroupQuantWithDualAxis);
    return SUCCESS;
}

int RunSwigluGroupQuantWithDualAxisCase(const SwigluGroupQuantWithDualAxisCase& testCase)
{
    printf("%s - INFO - [XIR]: Run %s\n", GetTime().c_str(), testCase.name);

    Graph graph(testCase.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    int32_t ret = CreateSwigluGroupQuantWithDualAxisGraph(testCase, graph, input, inputs, outputs);
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

    vector<SwigluGroupQuantWithDualAxisCase> testCases = {
        {"dual_axis_non_group", false, false, true, 7.0f, 1.702f, 1.0f},
        {"dual_axis_group_weight", true, true, true, 7.0f, 1.702f, 1.0f},
    };

    for (const auto& testCase : testCases) {
        ret = RunSwigluGroupQuantWithDualAxisCase(testCase);
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
