/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_masked_softmax_with_rel_pos_bias.cpp
 * \brief
 */
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include "cassert"

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
#include "../op_graph/masked_softmax_with_rel_pos_bias_proto.h"

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::map;
using std::string;
using std::vector;

const int B = 1;
const int W = 1;
const int N = 1;
const int S1 = 2;
const int S2 = 16;

#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape)                                                 \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + (intputIndex)).set_attr_index(0);                      \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    masked_softmax_with_rel_pos_bias_op.set_input_##intputName(placeholder##intputIndex);                           \
    inputs.push_back(placeholder##intputIndex)

#define ADD_INPUT_ATTR(attrName, attrValue) masked_softmax_with_rel_pos_bias_op.set_attr_##attrName(attrValue)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape)                                           \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                                         \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = GenOnesData(placeholder##intputIndex##_shape, tensor_placeholder##intputIndex,                            \
                      placeholder##intputIndex##_desc, intputDtype, 2);                                             \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);                                     \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);                                 \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    masked_softmax_with_rel_pos_bias_op.set_input_##intputName(placeholder##intputIndex);                           \
    masked_softmax_with_rel_pos_bias_op.update_input_desc_##intputName(placeholder##intputIndex##_desc);            \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    masked_softmax_with_rel_pos_bias_op.update_output_desc_##outputName(outputName##outputIndex##_desc);

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
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t byteSizeFloat32 = 4;
    uint32_t data_len = size * byteSizeFloat32;
    float* pData = new (std::nothrow) float[size];

    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t*)pData, data_len);
    return SUCCESS;
}

int32_t GenOnesData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type,
                    int value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    int32_t* pData = new (std::nothrow) int32_t[data_len];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp;
    fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, bool useRank4, bool hasMask, std::vector<ge::Tensor>& input,
                     std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto masked_softmax_with_rel_pos_bias_op = op::MaskedSoftmaxWithRelPosBias(
        "test_geir_masked_softmax_with_rel_pos_bias");

    // shape定义：4维(B*W, N, S1, S2)或5维(B, W, N, S1, S2)
    std::vector<int64_t> x_shape = useRank4 ? std::vector<int64_t>{B * W, N, S1, S2} :
                                              std::vector<int64_t>{B, W, N, S1, S2};
    std::vector<int64_t> atten_mask_shape = {W, S1, S2};
    std::vector<int64_t> relative_pos_bias_shape = {N, S1, S2};
    std::vector<int64_t> y_shape = x_shape;

    // 添加输入（顺序严格匹配 proto.h）；atten_mask为可选输入，hasMask为false时不连接（走mask=null路径）
    ADD_INPUT(1, x, inDtype, x_shape);
    if (hasMask) {
        ADD_INPUT(2, atten_mask, inDtype, atten_mask_shape);
    }
    ADD_INPUT(3, relative_pos_bias, inDtype, relative_pos_bias_shape);

    // 添加输出（顺序严格匹配 proto.h）
    ADD_OUTPUT(1, y, inDtype, y_shape);

    // 添加属性（顺序严格匹配 proto.h）
    ADD_INPUT_ATTR(scale_value, 1.0);
    ADD_INPUT_ATTR(inner_precision_mode, 0);

    outputs.push_back(masked_softmax_with_rel_pos_bias_op);
    // 添加完毕
    return SUCCESS;
}

// 单场景执行：GE 初始化→建图→AddGraph→RunGraph→清理。
// GE 同一进程内多次 AddGraph 同一算子会触发 tiling 模板重复注册（堆损坏），
// 因此多场景测试由 main 以子进程方式逐场景调用本函数（每场景独立进程）。
static int RunSingleScenario(uint32_t scenarioIdx)
{
    const std::vector<DataType> testDtypes = {DT_FLOAT, DT_FLOAT16, DT_BF16};
    const uint32_t dtypeIdx = scenarioIdx / 4;        // 场景号 → dtype 下标（3 dtype）
    const bool useRank4 = (scenarioIdx % 4 / 2) == 0; // → 4维/5维
    const bool hasMask = (scenarioIdx % 2) == 1;      // → mask 有/无
    const DataType inDtype = testDtypes[dtypeIdx];

    printf("%s - INFO - [XIR]: === scenario %u: dtype=%d, rank=%d, hasMask=%d ===\n", GetTime().c_str(), scenarioIdx,
           static_cast<int>(inDtype), useRank4 ? 4 : 5, hasMask ? 1 : 0);

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    // 内层作用域：Graph/Session/Tensor 等 GE 对象须在 GEFinalize 之前全部析构，
    // 否则栈对象在 GEFinalize 之后析构会触发 double free
    {
        Graph graph("tc_ge_irrun_test");
        std::vector<ge::Tensor> input;
        std::vector<Operator> inputs{};
        std::vector<Operator> outputs{};
        ret = CreateOppInGraph(inDtype, useRank4, hasMask, input, inputs, outputs, graph);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: scenario %u create graph failed\n", GetTime().c_str(), scenarioIdx);
            GEFinalize();
            return FAILED;
        }
        if (!inputs.empty() && !outputs.empty()) {
            graph.SetInputs(inputs).SetOutputs(outputs);
        }
        if (scenarioIdx == 0) {
            printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
            std::string file_path = "./dump";
            aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
        }

        std::map<AscendString, AscendString> build_options = {};
        ge::Session* session = new Session(build_options);
        if (session == nullptr) {
            printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
            GEFinalize();
            return FAILED;
        }
        std::map<AscendString, AscendString> graph_options = {};
        uint32_t graph_id = 0;
        ret = session->AddGraph(graph_id, graph, graph_options);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: scenario %u add graph failed\n", GetTime().c_str(), scenarioIdx);
            delete session;
            GEFinalize();
            return FAILED;
        }

        printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
        std::vector<ge::Tensor> output;
        ret = session->RunGraph(graph_id, input, output);
        if (ret != SUCCESS) {
            printf("%s - ERROR - [XIR]: scenario %u run graph failed\n", GetTime().c_str(), scenarioIdx);
            delete session;
            GEFinalize();
            return FAILED;
        }
        printf("%s - INFO - [XIR]: scenario %u run success, output size=%ld\n", GetTime().c_str(), scenarioIdx,
               output.empty() ? 0 : output[0].GetTensorDesc().GetShape().GetShapeSize());

        delete session;
    }
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: scenario %u finalize success\n", GetTime().c_str(), scenarioIdx);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // 多场景测试矩阵：3 dtype × 4维/5维 × mask有/无，共12个场景；
    // 带 <scenarioIdx> 参数时执行单个场景（供父进程子调用），否则顺序调度全部场景
    const uint32_t totalScenarios = 12;
    if (argc > 2) {
        uint32_t scenarioIdx = static_cast<uint32_t>(atoi(argv[2]));
        if (scenarioIdx >= totalScenarios) {
            printf("invalid scenario index %u, expect 0~%u\n", scenarioIdx, totalScenarios - 1);
            return FAILED;
        }
        return RunSingleScenario(scenarioIdx);
    }

    uint32_t failedCount = 0;
    for (uint32_t scenarioIdx = 0; scenarioIdx < totalScenarios; scenarioIdx++) {
        char cmd[512] = {0};
        snprintf(cmd, sizeof(cmd), "%s self %u", argv[0], scenarioIdx);
        int status = system(cmd);
        if (status != 0) {
            printf("%s - ERROR - [XIR]: scenario %u finished with error, status=%d\n", GetTime().c_str(), scenarioIdx,
                   status);
            failedCount++;
        }
    }
    printf("%s - INFO - [XIR]: GEIR multi-scenario summary: total=%u, failed=%u\n", GetTime().c_str(), totalScenarios,
           failedCount);
    return failedCount == 0 ? SUCCESS : FAILED;
}
