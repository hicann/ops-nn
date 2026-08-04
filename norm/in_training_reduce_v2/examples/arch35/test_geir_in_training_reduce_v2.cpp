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
#include "../../op_graph/in_training_reduce_v2_proto.h"

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
    return string(tmp);
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16)
        return 2;
    return 4;
}

void GenFloat16Data(uint8_t* pData, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        float val = static_cast<float>((i * 10 + 5) % 1000) / 1000.0f;
        uint32_t bits;
        memcpy(&bits, &val, sizeof(uint32_t));
        uint16_t sign = (bits >> 16) & 0x8000;
        uint16_t exponent = (bits >> 23) & 0xff;
        uint32_t mantissa = bits & 0x7fffff;
        uint16_t fp16Bits;
        if (exponent == 0) {
            fp16Bits = sign;
        } else if (exponent >= 113 && exponent <= 142) {
            fp16Bits = sign | ((exponent - 112) << 10) | (mantissa >> 13);
        } else if (exponent < 113) {
            if (exponent >= 103) {
                uint32_t shift = 113 - exponent;
                uint32_t shifted = (mantissa | 0x800000) >> shift;
                fp16Bits = sign | static_cast<uint16_t>(shifted);
                if ((shifted >> 4) & 1)
                    fp16Bits++;
            } else {
                fp16Bits = sign;
            }
        } else {
            fp16Bits = sign | 0x7c00;
        }
        memcpy(pData + i * 2, &fp16Bits, 2);
    }
}

int32_t GenFloatData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& desc, DataType dtype)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++)
        size *= shapes[i];
    uint32_t byteSize = GetDataTypeSize(dtype);
    uint32_t data_len = size * byteSize;
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr)
        return FAILED;
    if (dtype == ge::DT_FLOAT) {
        float* p = (float*)pData;
        for (size_t i = 0; i < size; ++i)
            p[i] = static_cast<float>((i * 10 + 5) % 1000) / 1000.0f;
    } else if (dtype == ge::DT_FLOAT16) {
        GenFloat16Data(pData, size);
    }
    input_tensor = Tensor(desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_in_training_reduce_v2";
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

    DataType xDtype = DT_FLOAT;
    Format xFormat = FORMAT_NCHW;
    vector<int64_t> xShape = {2, 3, 8, 8};

    if (argc > 1) {
        string arg = argv[1];
        if (arg == "fp16") {
            xDtype = DT_FLOAT16;
        }
        if (arg == "fp32") {
            xDtype = DT_FLOAT;
        }
    }
    if (argc > 2) {
        string fmt = argv[2];
        if (fmt == "NCDHW") {
            xFormat = FORMAT_NCDHW;
            xShape = {2, 3, 4, 4, 8};
        }
        if (fmt == "ND") {
            xFormat = FORMAT_ND;
            xShape = {2, 3, 64};
        }
        if (fmt == "NCHW") {
            xFormat = FORMAT_NCHW;
            xShape = {2, 3, 8, 8};
        }
    }
    printf("%s - INFO - [XIR]: x dtype=%d format=%d shape=[", GetTime().c_str(), xDtype, xFormat);
    for (size_t i = 0; i < xShape.size(); i++)
        printf("%ld,", xShape[i]);
    printf("]\n");

    auto xData = op::Data("placeholder_x").set_attr_index(0);
    TensorDesc xDesc = TensorDesc(ge::Shape(xShape), xFormat, xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(xFormat);
    Tensor tensor_x;
    ret = GenFloatData(xShape, tensor_x, xDesc, xDtype);
    if (ret != SUCCESS) {
        printf("GenFloatData x failed\n");
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(tensor_x);
    graph.AddOp(xData);

    auto reduce_op = op::INTrainingReduceV2("in_training_reduce_v2");
    reduce_op.set_input_x(xData);

    std::vector<Operator> inputs = {xData};
    std::vector<Operator> outputs = {reduce_op};
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

    int output_num = output.size();
    printf("output_num=%d\n", output_num);
    for (int i = 0; i < output_num; i++) {
        DataType outDtype = output[i].GetTensorDesc().GetDataType();
        int64_t outSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%d] dtype=%d shapeSize=%ld\n", i, outDtype, outSize);
        if (outDtype == ge::DT_FLOAT) {
            float* p = (float*)output[i].GetData();
            int64_t printCnt = outSize < 6 ? outSize : 6;
            for (int64_t j = 0; j < printCnt; j++)
                printf("  result[%ld] = %f\n", j, p[j]);
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
