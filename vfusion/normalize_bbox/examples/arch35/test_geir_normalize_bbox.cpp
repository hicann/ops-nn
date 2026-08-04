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
#include "nn_detect_ops.h"

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    if (dt == ge::DT_INT32)
        return 4;
    return 4;
}

int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& desc, DataType dtype)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t byteSize = GetDataTypeSize(dtype);
    uint32_t data_len = size * byteSize;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr)
        return FAILED;
    if (dtype == ge::DT_FLOAT) {
        float* p = (float*)pData;
        for (size_t i = 0; i < size; ++i)
            p[i] = static_cast<float>((i * 10 + 5) % 1000);
    } else if (dtype == ge::DT_FLOAT16) {
        for (size_t i = 0; i < size; ++i) {
            float val = static_cast<float>((i * 10 + 5) % 1000);
            uint32_t bits;
            memcpy(&bits, &val, sizeof(uint32_t));
            uint16_t sign = (bits >> 16) & 0x8000;
            uint16_t exponent = (bits >> 23) & 0xff;
            uint32_t mantissa = bits & 0x7fffff;
            if (exponent >= 113 && exponent <= 142) {
                uint16_t fp16Bits = sign | ((exponent - 112) << 10) | (mantissa >> 13);
                memcpy(pData + i * 2, &fp16Bits, 2);
            } else {
                uint16_t zero = 0;
                memcpy(pData + i * 2, &zero, 2);
            }
        }
    }
    input_tensor = Tensor(desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenInt32Data(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& desc)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t data_len = size * 4;
    int32_t* pData = new (std::nothrow) int32_t[size];
    if (pData == nullptr)
        return FAILED;
    for (size_t i = 0; i < size; ++i) {
        pData[i] = (i % 3 == 0) ? 100 : (i % 3 == 1) ? 200 : 0;
    }
    input_tensor = Tensor(desc, (uint8_t*)pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_normalize_bbox";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    DataType boxesDtype = DT_FLOAT;
    bool reversedBox = false;
    if (argc > 1) {
        string arg = argv[1];
        if (arg == "fp16")
            boxesDtype = DT_FLOAT16;
        if (arg == "fp32")
            boxesDtype = DT_FLOAT;
    }
    printf("%s - INFO - [XIR]: boxes dtype=%d reversedBox=%d\n", GetTime().c_str(), boxesDtype, reversedBox);

    // boxes: (1, 2, 4) float
    vector<int64_t> boxesShape = {1, 2, 4};
    auto boxesData = op::Data("placeholder_boxes").set_attr_index(0);
    TensorDesc boxesDesc = TensorDesc(ge::Shape(boxesShape), FORMAT_ND, boxesDtype);
    boxesDesc.SetPlacement(ge::kPlacementHost);
    boxesDesc.SetFormat(FORMAT_ND);
    Tensor tensor_boxes;
    ret = GenFloatData(boxesShape, tensor_boxes, boxesDesc, boxesDtype);
    if (ret != SUCCESS) {
        printf("GenFloatData boxes failed\n");
        return FAILED;
    }
    boxesData.update_input_desc_x(boxesDesc);
    boxesData.update_output_desc_y(boxesDesc);
    input.push_back(tensor_boxes);
    graph.AddOp(boxesData);

    // shape_hw: (1, 3) int32 [h=100, w=200, 0]
    vector<int64_t> shapeHwShape = {1, 3};
    auto shapeHwData = op::Data("placeholder_shape_hw").set_attr_index(1);
    TensorDesc shapeHwDesc = TensorDesc(ge::Shape(shapeHwShape), FORMAT_ND, DT_INT32);
    shapeHwDesc.SetPlacement(ge::kPlacementHost);
    shapeHwDesc.SetFormat(FORMAT_ND);
    Tensor tensor_shape_hw;
    ret = GenInt32Data(shapeHwShape, tensor_shape_hw, shapeHwDesc);
    if (ret != SUCCESS) {
        printf("GenInt32Data shape_hw failed\n");
        return FAILED;
    }
    shapeHwData.update_input_desc_x(shapeHwDesc);
    shapeHwData.update_output_desc_y(shapeHwDesc);
    input.push_back(tensor_shape_hw);
    graph.AddOp(shapeHwData);

    // NormalizeBBox op
    auto nbbox_op = op::NormalizeBBox("normalize_bbox");
    nbbox_op.set_input_boxes(boxesData);
    nbbox_op.set_input_shape_hw(shapeHwData);
    nbbox_op.set_attr_reversed_box(reversedBox);

    std::vector<Operator> inputs = {boxesData, shapeHwData};
    std::vector<Operator> outputs = {nbbox_op};
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Create session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("Create session failed\n");
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed ret=%u\n", GetTime().c_str(), ret);
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: RunGraph start\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed ret=%u\n", GetTime().c_str(), ret);
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        printf("%s - ERROR - [XIR]: error: %s\n", GetTime().c_str(), error_msg.GetString());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    // verify output
    int output_num = output.size();
    printf("output_num=%d\n", output_num);
    for (int i = 0; i < output_num; i++) {
        DataType outDtype = output[i].GetTensorDesc().GetDataType();
        int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%d] dtype=%d shapeSize=%ld\n", i, outDtype, outSize);
        uint8_t* outData = output[i].GetData();
        if (outData == nullptr) {
            printf("output[%d] GetData() returned nullptr\n", i);
            continue;
        }
        if (outDtype == ge::DT_FLOAT) {
            float* p = (float*)outData;
            // boxes[i] = (i*10+5)%1000, divisor = [100, 200, 100, 200] (h=100, w=200)
            float divisors[4] = {100.0f, 200.0f, 100.0f, 200.0f};
            for (int64_t j = 0; j < outSize; j++) {
                float expect = static_cast<float>((j * 10 + 5) % 1000) / divisors[j % 4];
                printf("  result[%ld] = %f (expect %f)\n", j, p[j], expect);
            }
        }
    }

    delete session;
    session = nullptr;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("GEFinalize failed\n");
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return SUCCESS;
}
