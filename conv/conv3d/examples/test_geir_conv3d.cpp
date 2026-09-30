/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_geir_conv3d.cpp
 * @brief Conv3D 算子 GE IR 图模式调用示例
 *
 * 构图并运行 Conv3D 单算子子图（x: NDHWC float32，filter: DHWCN float32，y: NDHWC float32）：
 *   - 输入：x[n,d,h,w,c]、filter[kd,kh,kw,c,co]
 *   - 输出：y[n,od,oh,ow,co]
 *
 * 目标平台：Ascend950（arch35）。
 */

#include <iostream>
#include <stdint.h>
#include <ctime>
#include <cstdio>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"

#include "../op_graph/conv3d_proto.h"

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
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        return 4;
    }
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return 2;
    }
    if (dt == ge::DT_INT64) {
        return 8;
    }
    return 1;
}

// 生成 [0, 1) 区间的伪随机浮点数据
int32_t GenFloatData(const vector<int64_t>& shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc, DataType dataType)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t dataLen = size * GetDataTypeSize(dataType);
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate memory for input data failed, size=%zu\n", GetTime().c_str(), size);
        return FAILED;
    }
    uint32_t seed = 12345U;
    for (size_t i = 0; i < size; ++i) {
        seed = seed * 1103515245U + 12345U;
        pData[i] = static_cast<float>((seed >> 16) & 0x7FFF) / 32768.0f - 0.5f;
    }
    inputTensor = Tensor(inputTensorDesc, reinterpret_cast<uint8_t*>(pData), dataLen);
    delete[] pData;
    return SUCCESS;
}

static int32_t AddDataInput(const string& name, uint32_t index, const vector<int64_t>& shape, ge::Format format,
                            DataType dtype, Graph& graph, vector<ge::Tensor>& input, vector<Operator>& inputs)
{
    auto data = op::Data(name.c_str()).set_attr_index(index);
    TensorDesc desc = TensorDesc(ge::Shape(shape), format, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(format);
    Tensor tensor;
    if (GenFloatData(shape, tensor, desc, dtype) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed for %s\n", GetTime().c_str(), name.c_str());
        return FAILED;
    }
    data.update_input_desc_x(desc);
    input.push_back(tensor);
    graph.AddOp(data);
    inputs.push_back(data);
    return SUCCESS;
}

int CreateConv3DInGraph(Graph& graph, vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs)
{
    // x: NDHWC [n=1, d=4, h=8, w=8, c=16]，filter: DHWCN [kd=3, kh=3, kw=3, c=16, co=32]
    vector<int64_t> xShape = {1, 4, 8, 8, 16};
    vector<int64_t> wShape = {3, 3, 3, 16, 32};
    // NDHWC strides [n,d,h,w,c] = [1,2,2,2,1]: od = (4+2-2-1)/2+1 = 2, oh = (8+2-2-1)/2+1 = 4, ow = 4
    vector<int64_t> yShape = {1, 2, 4, 4, 32};

    auto conv3d = op::Conv3D("conv3d_1");

    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), ge::FORMAT_NDHWC, ge::DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(ge::FORMAT_NDHWC);
    xDesc.SetOriginFormat(ge::FORMAT_NDHWC);
    TensorDesc wDesc = TensorDesc(ge::Shape(wShape), ge::FORMAT_DHWCN, ge::DT_FLOAT);
    wDesc.SetPlacement(ge::kPlacementHost);
    wDesc.SetFormat(ge::FORMAT_DHWCN);
    wDesc.SetOriginFormat(ge::FORMAT_DHWCN);

    if (AddDataInput("placeholder0", 0, xShape, ge::FORMAT_NDHWC, ge::DT_FLOAT, graph, input, inputs) != SUCCESS) {
        return FAILED;
    }
    conv3d.set_input_x(inputs[0]);
    conv3d.update_input_desc_x(xDesc);
    if (AddDataInput("placeholder1", 1, wShape, ge::FORMAT_DHWCN, ge::DT_FLOAT, graph, input, inputs) != SUCCESS) {
        return FAILED;
    }
    conv3d.set_input_filter(inputs[1]);
    conv3d.update_input_desc_filter(wDesc);

    conv3d.set_attr_strides({1, 2, 2, 2, 1});
    conv3d.set_attr_pads({1, 1, 1, 1, 1, 1});
    conv3d.set_attr_dilations({1, 1, 1, 1, 1});
    conv3d.set_attr_groups(1);
    conv3d.set_attr_data_format("NDHWC");
    conv3d.set_attr_offset_x(0);

    TensorDesc yDesc = TensorDesc(ge::Shape(yShape), ge::FORMAT_NDHWC, ge::DT_FLOAT);
    conv3d.update_output_desc_y(yDesc);

    graph.AddOp(conv3d);
    outputs.push_back(conv3d);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graphName = "test_geir_conv3d";
    Graph graph(graphName);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs{};
    vector<Operator> outputs{};

    ret = CreateConv3DInGraph(graph, input, inputs, outputs);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: CreateConv3DInGraph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success, outputs=%zu\n", GetTime().c_str(), output.size());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t outputShapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "output " << i << " shape size = " << outputShapeSize
                  << ", dtype = " << output[i].GetTensorDesc().GetDataType() << std::endl;
    }

    printf("%s - INFO - [XIR]: GE IR pathway verification PASSED\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}
