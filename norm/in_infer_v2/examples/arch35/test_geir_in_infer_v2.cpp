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
 * @file test_geir_in_infer_v2.cpp
 * @brief INInferV2 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：y = (x - mean) * (gamma / sqrt(variance + epsilon)) + beta
 *           batch_mean = mean；batch_variance = variance（透传）
 *
 * 本示例构造两张仅含 op::INInferV2 节点的计算图并在 ascend950 上执行：
 *   gb  ：x{2,3,4,5} 全 1.0；gamma/beta/mean/variance{2,3} 分别全 2.0 / 0.5 / 0.25 / 1.0
 *         => y = 0.75 * (2/sqrt(1+1e-5)) + 0.5 ≈ 2.0；batch_mean = 0.25；batch_variance = 1.0
 *   nogb：gamma/beta 不接边（optional 输入缺席）=> y = 0.75/sqrt(1+1e-5) ≈ 0.75
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

#include "../../op_graph/in_infer_v2_proto.h"

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

// 构造并运行一张仅含 INInferV2 节点的图，校验输出。
// hasGammaBeta=false 时不接 gamma/beta（optional 输入缺席），y 退化为 (x-mean)/sqrt(var+eps)。
int32_t RunGraphCase(const char* graphName, bool hasGammaBeta)
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

    // 算子节点：x shape {2,3,4,5}；gamma/beta/mean/variance shape {2,3}（N*C 标量）
    auto iniOp = op::INInferV2("in_infer_v2");
    vector<int64_t> xShape = {2, 3, 4, 5};
    vector<int64_t> statShape = {2, 3};
    vector<int64_t> yShape = {2, 3, 4, 5};

    vector<vector<int64_t>*> inShapes = {&xShape, &statShape, &statShape, &statShape, &statShape};
    float inValues[5] = {1.0f, 2.0f, 0.5f, 0.25f, 1.0f}; // x / gamma / beta / mean / variance
    for (int i = 0; i < 5; i++) {
        if (!hasGammaBeta && (i == 1 || i == 2)) {
            continue; // nogb：gamma/beta 不建 Data、不接边
        }
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
                iniOp.set_input_x(data);
                break;
            case 1:
                iniOp.set_input_gamma(data);
                break;
            case 2:
                iniOp.set_input_beta(data);
                break;
            case 3:
                iniOp.set_input_mean(data);
                break;
            default:
                iniOp.set_input_variance(data);
                break;
        }
        inputs.push_back(data);
    }
    // y / batch_mean / batch_variance：声明输出 desc
    TensorDesc yDesc = TensorDesc(ge::Shape(yShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_y(yDesc);
    TensorDesc bmDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_batch_mean(bmDesc);
    TensorDesc bvDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_batch_variance(bvDesc);
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

    // 校验：gb 时 y = 0.75*(2/sqrt(1+eps))+0.5 ≈ 2.0；nogb 时 y = 0.75/sqrt(1+eps) ≈ 0.75
    // batch_mean == 0.25；batch_variance == 1.0
    const float expectY = hasGammaBeta ? 2.0f : 0.75f;
    int failCnt = 0;
    if (output.size() != 3) {
        LOG_PRINT("[CHECK][%s] output num %zu != 3\n", graphName, output.size());
        failCnt++;
    } else {
        float* yData = reinterpret_cast<float*>(output[0].GetData());
        int64_t ySize = output[0].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t i = 0; i < ySize; i++) {
            if (fabsf(yData[i] - expectY) > 1e-4f) {
                LOG_PRINT("y[%ld] = %f, expect %f\n", i, yData[i], expectY);
                failCnt++;
            }
        }
        float* bmData = reinterpret_cast<float*>(output[1].GetData());
        int64_t bmSize = output[1].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t i = 0; i < bmSize; i++) {
            if (fabsf(bmData[i] - 0.25f) > 1e-6f) {
                LOG_PRINT("batch_mean[%ld] = %f, expect 0.25\n", i, bmData[i]);
                failCnt++;
            }
        }
        float* bvData = reinterpret_cast<float*>(output[2].GetData());
        int64_t bvSize = output[2].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t i = 0; i < bvSize; i++) {
            if (fabsf(bvData[i] - 1.0f) > 1e-6f) {
                LOG_PRINT("batch_variance[%ld] = %f, expect 1.0\n", i, bvData[i]);
                failCnt++;
            }
        }
        LOG_PRINT("[CHECK][%s] y[0]=%f batch_mean[0]=%f batch_variance[0]=%f\n", graphName, yData[0], bmData[0],
                  bvData[0]);
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
    failCnt += (RunGraphCase("in_infer_v2_geir_gb", true) != SUCCESS);    // 全输入
    failCnt += (RunGraphCase("in_infer_v2_geir_nogb", false) != SUCCESS); // gamma/beta 缺席

    LOG_PRINT("[CHECK] total fail: %d, %s\n", failCnt, failCnt == 0 ? "PASS" : "FAIL");
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
