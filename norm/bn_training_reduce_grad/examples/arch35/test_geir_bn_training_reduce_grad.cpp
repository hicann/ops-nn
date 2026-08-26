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
 * @file test_geir_bn_training_reduce_grad.cpp
 * @brief BNTrainingReduceGrad 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：sqrtVar = sqrt(batch_variance + epsilon)；num = N*R（R 为 grads 后导维展平长度）
 *           multiplier = (diff_scale * (-1/num)) / sqrtVar
 *           addend     = (batch_mean / sqrtVar) * (diff_scale * (1/num)) + diff_offset * (-1/num)
 *           mulScale   = scale / sqrtVar
 *           y          = ((grads + multiplier * x) + addend) * mulScale
 *
 * 本示例构造两张仅含 op::BNTrainingReduceGrad 节点的计算图并在 ascend950 上执行。
 * 两个 case 均取 multiplier*x 与 addend 首项相消的常量数据，期望值为：
 *   y = (grads + diff_offset * (-1/num)) * scale / sqrt(batch_variance + epsilon)
 *   case1：grads/x{2,3,4,5}=1.0，diff_scale=2.0，diff_offset=0.5，scale=2.0，
 *          batch_mean=1.0，batch_variance=1.0，epsilon=1e-5，num=40
 *          => y = (1 - 0.5/40) * 2/sqrt(1+1e-5) = 1.975/sqrt(1.00001)
 *   case2：grads=1.0/x=2.0 {4,64,7,7}，diff_scale=1.0，diff_offset=0.25，scale=1.0，
 *          batch_mean=2.0，batch_variance=0.0，epsilon=1e-3，num=196
 *          => y = (1 - 0.25/196) / sqrt(1e-3)
 * 校验图模式全链路：proto 注册 / OpDef(ascend950) / infershape+inferDataType / tiling / kernel。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/bn_training_reduce_grad_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

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

int32_t GenConstDataFloat32(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, float value,
                            vector<float*>& allocBufs)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    uint32_t dataLen = size * sizeof(float);
    float* data = new (std::nothrow) float[size];
    if (data == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: alloc input data (%zu floats) failed\n", GetTime().c_str(), size);
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        data[i] = value;
    }
    tensor = Tensor(desc, (uint8_t*)data, dataLen);
    allocBufs.push_back(data); // 登记，RunGraph 完成后统一 delete[]
    return SUCCESS;
}

// 构造并运行一张仅含 BNTrainingReduceGrad 节点的图，校验单路输出 y。
int32_t RunGraphCase(const char* graphName, const vector<int64_t>& mainShape, int64_t c, float gradsValue, float xValue,
                     float diffScaleValue, float diffOffsetValue, float scaleValue, float batchMeanValue,
                     float batchVarValue, float epsilon, float tol)
{
    Graph graph(graphName);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    vector<float*> allocBufs; // 输入常量数据缓冲，RunGraph 完成后统一释放
    auto freeBufs = [&allocBufs]() {
        for (float* p : allocBufs) {
            delete[] p;
        }
        allocBufs.clear();
    };
    Status ret = SUCCESS;

    auto iniOp = op::BNTrainingReduceGrad("bn_training_reduce_grad");
    iniOp.set_attr_epsilon(epsilon);
    vector<int64_t> statShape = {c};

    vector<const vector<int64_t>*> inShapes = {&mainShape, &mainShape, &statShape, &statShape,
                                               &statShape, &statShape, &statShape};
    float inValues[7] = {gradsValue, xValue,         diffScaleValue, diffOffsetValue,
                         scaleValue, batchMeanValue, batchVarValue};
    for (int i = 0; i < 7; i++) {
        auto data = op::Data("placeholder" + std::to_string(i)).set_attr_index(0);
        TensorDesc desc = TensorDesc(ge::Shape(*inShapes[i]), FORMAT_ND, DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        Tensor tensor;
        ret = GenConstDataFloat32(*inShapes[i], tensor, desc, inValues[i], allocBufs);
        if (ret != SUCCESS) {
            LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
            freeBufs();
            return FAILED;
        }
        data.update_input_desc_x(desc);
        input.push_back(tensor);
        graph.AddOp(data);
        switch (i) {
            case 0:
                iniOp.set_input_grads(data);
                break;
            case 1:
                iniOp.set_input_x(data);
                break;
            case 2:
                iniOp.set_input_diff_scale(data);
                break;
            case 3:
                iniOp.set_input_diff_offset(data);
                break;
            case 4:
                iniOp.set_input_scale(data);
                break;
            case 5:
                iniOp.set_input_batch_mean(data);
                break;
            default:
                iniOp.set_input_batch_variance(data);
                break;
        }
        inputs.push_back(data);
    }
    // y：声明输出 desc
    TensorDesc yDesc = TensorDesc(ge::Shape(mainShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_y(yDesc);
    outputs.push_back(iniOp);

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO - [XIR]: AddGraph + RunGraph (%s)\n", GetTime().c_str(), graphName);
    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        freeBufs();
        return FAILED;
    }
    std::map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        freeBufs();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    delete session;
    freeBufs(); // 输入数据已被 RunGraph 消费，Tensor 生命周期结束，统一释放
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: RunGraph success, outputs=%zu\n", GetTime().c_str(), output.size());

    // 期望值：multiplier*x 与 addend 首项相消 => y = (grads + diff_offset*(-1/num)) * scale/sqrt(var+eps)
    int64_t num = mainShape[0];
    for (size_t i = 2; i < mainShape.size(); i++) {
        num *= mainShape[i];
    }
    double negNumRecip = -1.0 / static_cast<double>(num);
    double expectY = (static_cast<double>(gradsValue) + static_cast<double>(diffOffsetValue) * negNumRecip) *
                     static_cast<double>(scaleValue) /
                     std::sqrt(static_cast<double>(batchVarValue) + static_cast<double>(epsilon));

    int failCnt = 0;
    if (output.size() != 1) {
        LOG_PRINT("[CHECK][%s] output num %zu != 1\n", graphName, output.size());
        failCnt++;
    } else {
        float* yData = reinterpret_cast<float*>(output[0].GetData());
        int64_t ySize = output[0].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t i = 0; i < ySize; i++) {
            if (fabsf(yData[i] - static_cast<float>(expectY)) > tol) {
                LOG_PRINT("y[%ld] = %f, expect %f\n", i, yData[i], expectY);
                failCnt++;
            }
        }
        LOG_PRINT("[CHECK][%s] y[0]=%f expect=%f\n", graphName, yData[0], expectY);
    }
    LOG_PRINT("[CHECK][%s] %s\n", graphName, failCnt == 0 ? "PASS" : "FAIL");
    return failCnt == 0 ? SUCCESS : FAILED;
}

int main()
{
    LOG_PRINT("%s - INFO - [XIR]: GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    int failCnt = 0;
    // case1：小 shape，num=40，y = (1-0.5/40)*2/sqrt(1+1e-5) ≈ 1.974990
    failCnt += (RunGraphCase("bn_training_reduce_grad_geir_case1", {2, 3, 4, 5}, 3, 1.0f, 1.0f, 2.0f, 0.5f, 2.0f, 1.0f,
                             1.0f, 1e-5f, 1e-4f) != SUCCESS);
    // case2：多 channel、R 非对齐，num=196，y = (1-0.25/196)/sqrt(1e-3) ≈ 31.5824
    failCnt += (RunGraphCase("bn_training_reduce_grad_geir_case2", {4, 64, 7, 7}, 64, 1.0f, 2.0f, 1.0f, 0.25f, 1.0f,
                             2.0f, 0.0f, 1e-3f, 1e-3f) != SUCCESS);

    LOG_PRINT("[CHECK] total fail: %d, %s\n", failCnt, failCnt == 0 ? "PASS" : "FAIL");
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
