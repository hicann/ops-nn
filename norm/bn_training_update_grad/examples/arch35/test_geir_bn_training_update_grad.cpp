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
 * @file test_geir_bn_training_update_grad.cpp
 * @brief BNTrainingUpdateGrad 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：rstd = 1/sqrt(batch_variance+epsilon)；
 *           diff_scale[c] = sum_{n,r} grads*(x-batch_mean[c])*rstd；
 *           diff_offset[c] = sum_{n,r} grads
 *           其中归约轴为 N 维与后导维 R（R 为 grads 后导维展平长度）
 *
 * 本示例构造两张仅含 op::BNTrainingUpdateGrad 节点的计算图并在 ascend950 上执行：
 *   case1：grads{2,3,4,5} 全 1.0；x{2,3,4,5} 全 2.0；batch_mean{3}=1.0、batch_variance{3}=1.0，
 *          epsilon=1e-5 => rstd=1/sqrt(1+1e-5)；diff_offset = 40；diff_scale = 40*rstd ≈ 39.9998
 *   case2：grads{4,64,7,7} 全 0.5；x{4,64,7,7} 全 2.0；batch_mean{64}=1.5、batch_variance{64}=0.25，
 *          epsilon=1e-3 => rstd=1/sqrt(0.251)；diff_offset = 98；diff_scale = 49*rstd ≈ 97.805
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

#include "../../op_graph/bn_training_update_grad_proto.h"

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

// 构造并运行一张仅含 BNTrainingUpdateGrad 节点的图，校验两路输出。
int32_t RunGraphCase(const char* graphName, const vector<int64_t>& gradsShape, int64_t c, float gradsValue,
                     float xValue, float meanValue, float varValue, float epsilon, float expectDiffScale,
                     float expectDiffOffset)
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

    auto iniOp = op::BNTrainingUpdateGrad("bn_training_update_grad");
    iniOp.set_attr_epsilon(epsilon);
    vector<int64_t> statShape = {c};

    vector<const vector<int64_t>*> inShapes = {&gradsShape, &gradsShape, &statShape, &statShape};
    float inValues[4] = {gradsValue, xValue, meanValue, varValue};
    for (int i = 0; i < 4; i++) {
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
                iniOp.set_input_batch_mean(data);
                break;
            default:
                iniOp.set_input_batch_variance(data);
                break;
        }
        inputs.push_back(data);
    }
    // diff_scale / diff_offset：声明输出 desc
    TensorDesc dsDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_diff_scale(dsDesc);
    TensorDesc doDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_diff_offset(doDesc);
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

    int failCnt = 0;
    if (output.size() != 2) {
        LOG_PRINT("[CHECK][%s] output num %zu != 2\n", graphName, output.size());
        failCnt++;
    } else {
        const float expectVals[2] = {expectDiffScale, expectDiffOffset};
        const char* outNames[2] = {"diff_scale", "diff_offset"};
        for (int k = 0; k < 2; k++) {
            float* statData = reinterpret_cast<float*>(output[k].GetData());
            int64_t statSize = output[k].GetTensorDesc().GetShape().GetShapeSize();
            for (int64_t i = 0; i < statSize; i++) {
                if (fabsf(statData[i] - expectVals[k]) > 1e-3f) {
                    LOG_PRINT("%s[%ld] = %f, expect %f\n", outNames[k], i, statData[i], expectVals[k]);
                    failCnt++;
                }
            }
        }
        float* dsData = reinterpret_cast<float*>(output[0].GetData());
        float* doData = reinterpret_cast<float*>(output[1].GetData());
        LOG_PRINT("[CHECK][%s] diff_scale[0]=%f diff_offset[0]=%f\n", graphName, dsData[0], doData[0]);
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
    // case1：小 shape，rstd=1/sqrt(1+1e-5)，diff_offset=40、diff_scale=40*rstd
    failCnt += (RunGraphCase("bn_training_update_grad_geir_case1", {2, 3, 4, 5}, 3, 1.0f, 2.0f, 1.0f, 1.0f, 1e-5f,
                             40.0f / sqrtf(1.0f + 1e-5f), 40.0f) != SUCCESS);
    // case2：多 channel、R 非对齐，rstd=1/sqrt(0.251)，diff_offset=98、diff_scale=49*rstd
    failCnt += (RunGraphCase("bn_training_update_grad_geir_case2", {4, 64, 7, 7}, 64, 0.5f, 2.0f, 1.5f, 0.25f, 1e-3f,
                             49.0f / sqrtf(0.25f + 1e-3f), 98.0f) != SUCCESS);

    LOG_PRINT("[CHECK] total fail: %d, %s\n", failCnt, failCnt == 0 ? "PASS" : "FAIL");
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
