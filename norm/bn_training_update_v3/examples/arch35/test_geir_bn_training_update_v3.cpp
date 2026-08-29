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
 * @file test_geir_bn_training_update_v3.cpp
 * @brief BNTrainingUpdateV3 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：save_mean = sum/num；save_variance = square_sum/num - save_mean^2；
 *           batch_mean = save_mean；reserve_1 = save_mean；reserve_2 = save_variance；
 *           batch_variance = save_variance * num/(num-1)（无偏估计，num==1 时为 0）；
 *           y = scale/sqrt(save_variance+epsilon) * x + (offset - scale*save_mean/sqrt(save_variance+epsilon))
 *           其中 num = N*R（R 为 x 后导维展平长度）
 *
 * 本示例构造两张仅含 op::BNTrainingUpdateV3 节点的计算图并在 ascend950 上执行：
 *   case1：x{2,3,4,5} 全 1.0；sum/square_sum/scale/offset{3} = 40/80/2.0/0.5，epsilon=1e-5
 *          => save_mean = 1.0；save_variance = 2-1 = 1.0；batch_variance = 1.0*40/39 ≈ 1.025641；
 *             y = 2/sqrt(1+1e-5)*1 + (0.5-2*1/sqrt(1+1e-5)) ≈ 0.5
 *   case2：x{4,64,7,7} 全 2.0；sum=4*49*2=392、square_sum=784、scale=1.0、offset=0.25，
 *          epsilon=1e-3 => save_mean = 2.0；save_variance = 4-4 = 0.0；batch_variance = 0.0；
 *             y = 1/sqrt(1e-3)*2 + (0.25-2/sqrt(1e-3)) = 0.25
 *   case3：NHWC（origin format NHWC，C=末维=3，num=numel/C=40），x{2,4,5,3} 全 1.0；
 *          sum=square_sum=40、scale=2.0、offset=0.5，epsilon=1e-5
 *          => save_mean = 1.0；save_variance = 0.0；batch_variance = 0.0；y ≈ 0.5
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

#include "../../op_graph/bn_training_update_v3_proto.h"

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

// 构造并运行一张仅含 BNTrainingUpdateV3 节点的图，校验五路输出。
// xFmt：x/y 的 origin format（ND 或 NHWC；NHWC 时 C=xShape 末维，统计量仍 [C] ND）
int32_t RunGraphCase(const char* graphName, const vector<int64_t>& xShape, int64_t c, float xValue, float sumValue,
                     float squareSumValue, float scaleValue, float offsetValue, float epsilon, float expectY,
                     float expectMean, float expectVar, float expectBatchVar, ge::Format xFmt = ge::FORMAT_ND)
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

    auto iniOp = op::BNTrainingUpdateV3("bn_training_update_v3");
    iniOp.set_attr_epsilon(epsilon);
    vector<int64_t> statShape = {c};

    vector<const vector<int64_t>*> inShapes = {&xShape, &statShape, &statShape, &statShape, &statShape};
    float inValues[5] = {xValue, sumValue, squareSumValue, scaleValue, offsetValue};
    for (int i = 0; i < 5; i++) {
        auto data = op::Data("placeholder" + std::to_string(i)).set_attr_index(0);
        ge::Format inFmt = (i == 0) ? xFmt : ge::FORMAT_ND;
        TensorDesc desc = TensorDesc(ge::Shape(*inShapes[i]), inFmt, DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(inFmt);
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
                iniOp.set_input_sum(data);
                break;
            case 2:
                iniOp.set_input_square_sum(data);
                break;
            case 3:
                iniOp.set_input_scale(data);
                break;
            default:
                iniOp.set_input_offset(data);
                break;
        }
        inputs.push_back(data);
    }
    // y / batch_mean / batch_variance / reserve_1 / reserve_2：声明输出 desc（y 格式随 x）
    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), xFmt, DT_FLOAT);
    iniOp.update_output_desc_y(yDesc);
    TensorDesc bmDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_batch_mean(bmDesc);
    TensorDesc bvDesc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_batch_variance(bvDesc);
    TensorDesc r1Desc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_reserve_1(r1Desc);
    TensorDesc r2Desc = TensorDesc(ge::Shape(statShape), FORMAT_ND, DT_FLOAT);
    iniOp.update_output_desc_reserve_2(r2Desc);
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
    if (output.size() != 5) {
        LOG_PRINT("[CHECK][%s] output num %zu != 5\n", graphName, output.size());
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
        // 四路统计量输出：[batch_mean, batch_variance, reserve_1, reserve_2]
        // 期望值：           [save_mean,  save_var*scaler, save_mean, save_var]
        const float expectVals[4] = {expectMean, expectBatchVar, expectMean, expectVar};
        const char* outNames[4] = {"batch_mean", "batch_variance", "reserve_1", "reserve_2"};
        for (int k = 0; k < 4; k++) {
            float* statData = reinterpret_cast<float*>(output[k + 1].GetData());
            int64_t statSize = output[k + 1].GetTensorDesc().GetShape().GetShapeSize();
            for (int64_t i = 0; i < statSize; i++) {
                if (fabsf(statData[i] - expectVals[k]) > 1e-5f) {
                    LOG_PRINT("%s[%ld] = %f, expect %f\n", outNames[k], i, statData[i], expectVals[k]);
                    failCnt++;
                }
            }
        }
        float* bmData = reinterpret_cast<float*>(output[1].GetData());
        float* bvData = reinterpret_cast<float*>(output[2].GetData());
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
    // case1：小 shape，save_mean=1.0、save_var=1.0、batch_variance=40/39、y≈0.5
    failCnt += (RunGraphCase("bn_training_update_v3_geir_case1", {2, 3, 4, 5}, 3, 1.0f, 40.0f, 80.0f, 2.0f, 0.5f, 1e-5f,
                             0.5f, 1.0f, 1.0f, 40.0f / 39.0f) != SUCCESS);
    // case2：多 channel、R 非对齐，save_mean=2.0、save_var=0.0、batch_variance=0.0、y=0.25
    failCnt += (RunGraphCase("bn_training_update_v3_geir_case2", {4, 64, 7, 7}, 64, 2.0f, 392.0f, 784.0f, 1.0f, 0.25f,
                             1e-3f, 0.25f, 2.0f, 0.0f, 0.0f) != SUCCESS);
    // case3：NHWC（C=末维=3，num=numel/C=40），x 全 1.0：save_mean=1.0、save_var=0.0、
    // batch_variance=0.0、multiplier=scale/sqrt(eps)、addend=offset-mult*1.0 → y=0.5
    failCnt += (RunGraphCase("bn_training_update_v3_geir_case3", {2, 4, 5, 3}, 3, 1.0f, 40.0f, 40.0f, 2.0f, 0.5f, 1e-5f,
                             0.5f, 1.0f, 0.0f, 0.0f, ge::FORMAT_NHWC) != SUCCESS);

    LOG_PRINT("[CHECK] total fail: %d, %s\n", failCnt, failCnt == 0 ? "PASS" : "FAIL");
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
