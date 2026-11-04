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
 * \file test_geir_mat_mul_v2.cpp
 * \brief MatMulV2 GE IR example (fp16 inputs, fp16 output).
 */

#include <cstdio>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <map>
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

#include "../op_graph/mat_mul_v2_proto.h"

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
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32 || dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16 || dt == ge::DT_INT16 || dt == ge::DT_UINT16) {
        return twoByte;
    }
    return 1;
}

uint16_t FloatToHalfBits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

float HalfBitsToFloat(uint16_t bits)
{
    uint32_t sign = static_cast<uint32_t>((bits >> 15) & 0x1) << 31;
    int32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;
    uint32_t fBits;
    if (exp == 0) {
        fBits = sign;
    } else if (exp == 31) {
        fBits = sign | 0x7F800000 | (mant << 13);
    } else {
        fBits = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    float value;
    memcpy(&value, &fBits, sizeof(value));
    return value;
}

// 按dtype编码数据:float32直接写float值,float16/bfloat16写float值转换后的位模式
int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = static_cast<uint32_t>(size) * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        return FAILED;
    }
    if (data_type == ge::DT_FLOAT) {
        float* pFloat = reinterpret_cast<float*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pFloat[i] = value;
        }
    } else if (data_type == ge::DT_FLOAT16 || data_type == ge::DT_BF16) {
        uint16_t bits = FloatToHalfBits(value);
        uint16_t* pHalf = reinterpret_cast<uint16_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pHalf[i] = bits;
        }
    } else {
        memset(pData, 0, data_len);
    }
    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    delete[] pData;
    return SUCCESS;
}

// 全量输出与golden常数比对;非有限值(NaN/Inf)直接计为失败,防止NaN恒不过阈值导致漏报
int32_t CompareWithGolden(const uint8_t* outData, int64_t outSize, ge::DataType outDtype, float golden)
{
    float relTol = 1e-6f; // 用例结果为精确常量,容差收紧至最低限度
    float absTol = 1e-3f; // 覆盖fp16输出在130/128量级的最低表示精度
    double maxDiff = 0.0;
    double maxRel = 0.0;
    int64_t badCount = 0;
    for (int64_t i = 0; i < outSize; ++i) {
        float actual = (outDtype == ge::DT_FLOAT16) ? HalfBitsToFloat(reinterpret_cast<const uint16_t*>(outData)[i]) :
                                                      reinterpret_cast<const float*>(outData)[i];
        if (!std::isfinite(actual)) {
            ++badCount;
            continue;
        }
        double diff = fabs(static_cast<double>(actual) - static_cast<double>(golden));
        double rel = diff / (fabs(static_cast<double>(golden)) + 1e-6);
        maxDiff = std::max(maxDiff, diff);
        maxRel = std::max(maxRel, rel);
        if (diff > absTol + relTol * fabs(static_cast<double>(golden))) {
            ++badCount;
        }
    }
    printf("[XIR]: golden compare (expect %g): max abs diff = %g, max rel diff = %g, mismatch = %ld / %ld -> %s\n",
           static_cast<double>(golden), maxDiff, maxRel, badCount, outSize, badCount == 0 ? "PASS" : "FAIL");
    return badCount == 0 ? SUCCESS : FAILED;
}

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape)                                                       \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                      \
    auto placeholder##inputIndex = op::Data("placeholder" + std::to_string(inputIndex))                                \
                                       .set_attr_index((inputIndex) - 1);                                              \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_ND,      \
                                                           inputDtype);                                                \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                   \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_ND);                                                               \
    placeholder##inputIndex##_desc.SetOriginFormat(FORMAT_ND);                                                         \
    Tensor tensor_placeholder##inputIndex;                                                                             \
    ret = GenOnesData(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex, placeholder##inputIndex##_desc, \
                      inputDtype, 2.0f);                                                                               \
    if (ret != SUCCESS) {                                                                                              \
        printf("%s - ERROR - [XIR]: Generate inputTensors data failed\n", GetTime().c_str());                          \
        return FAILED;                                                                                                 \
    }                                                                                                                  \
    inputTensors.push_back(tensor_placeholder##inputIndex);                                                            \
    graph.AddOp(placeholder##inputIndex);                                                                              \
    tmpOp.set_input_##inputName(placeholder##inputIndex);                                                              \
    tmpOp.update_input_desc_##inputName(placeholder##inputIndex##_desc);                                               \
    inputOps.push_back(placeholder##inputIndex);

#define ADD_OUTPUT(outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    tmpOp.update_output_desc_##outputName(outputName##_desc)

int32_t CreateMatMulV2InGraph(std::vector<ge::Tensor>& inputTensors, std::vector<Operator>& inputOps,
                              std::vector<Operator>& outputOps, Graph& graph)
{
    Status ret = SUCCESS;
    auto tmpOp = op::MatMulV2("mat_mul_v2");
    std::vector<int64_t> x1Shape = {16, 32};
    std::vector<int64_t> x2Shape = {32, 48};
    std::vector<int64_t> biasShape = {48};
    std::vector<int64_t> yShape = {16, 48};
    tmpOp.set_attr_transpose_x1(false);
    tmpOp.set_attr_transpose_x2(false);
    tmpOp.set_attr_offset_x(0);
    ADD_INPUT(1, x1, ge::DT_FLOAT16, x1Shape);
    ADD_INPUT(2, x2, ge::DT_FLOAT16, x2Shape);
    ADD_INPUT(3, bias, ge::DT_FLOAT16, biasShape);
    // y跟随x1为fp16(MatMul不支持16进32出);期望y[0][0] = 2.0 * 2.0 * 32 + 2.0 = 130.0
    ADD_OUTPUT(y, ge::DT_FLOAT16, yShape);
    outputOps.push_back(tmpOp);
    if (!inputOps.empty() && !outputOps.empty()) {
        graph.SetInputs(inputOps).SetOutputs(outputOps);
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test_mat_mul_v2";
    Graph graph(graph_name);

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputOps{};
    std::vector<ge::Tensor> inputTensors;
    std::vector<Operator> outputOps{};
    ret = CreateMatMulV2InGraph(inputTensors, inputOps, outputOps, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./mat_mul_v2_dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, inputTensors, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int32_t compareRet = SUCCESS;
    int output_num = static_cast<int>(output.size());
    for (int i = 0; i < output_num; i++) {
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        if (CompareWithGolden(output_data_i, output_shape, output[i].GetTensorDesc().GetDataType(), 130.0f) !=
            SUCCESS) {
            compareRet = FAILED;
        }
    }

    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return compareRet;
}
