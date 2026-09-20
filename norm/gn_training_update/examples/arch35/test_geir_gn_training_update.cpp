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
 * @file test_geir_gn_training_update.cpp
 * @brief GNTrainingUpdate 图模式（GE IR）构图调用示例（ascend950 真机）
 *
 * 算子功能：mean = sum/M；variance = square_sum/M - mean^2（M=(C/G)*H*W）；
 *           y = (x - mean)/sqrt(variance+epsilon) * scale + offset；
 *           batch_mean = mean；batch_variance = variance（有偏，未加 epsilon）。
 *
 * 本示例构造三张仅含 op::GNTrainingUpdate 节点的计算图并在 ascend950 上执行：
 *   case1（NCHW）：x{2,4,2,3} 全 1.0，G=2，M=12；sum=square_sum=12（[N,G,1,1,1]），
 *          scale=2.0、offset=0.5、epsilon=1e-5 => mean=1.0、variance=0.0、y=0.5
 *   case2（NCHW）：x{4,64,7,7} 全 2.0，G=32，M=98；sum=196、square_sum=392，
 *          scale=1.0、offset=0.25、epsilon=1e-3 => mean=2.0、variance=0.0、y=0.25
 *   case3（NHWC）：x{2,4,5,3} 全 1.0，G=3，统计量 [N,1,1,G,1]，M=20；sum=square_sum=20，
 *          scale=2.0、offset=0.5、epsilon=1e-5 => mean=1.0、variance=0.0、y=0.5
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

#include "../../op_graph/gn_training_update_proto.h"

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
    allocBufs.push_back(data);
    return SUCCESS;
}

// 构造并运行一张仅含 GNTrainingUpdate 节点的图，校验三路输出。
// xFmt：x/y 的 origin format（ND 或 NHWC）；统计量/仿射为 5D ND（NCHW 时 [N,G,1,1,1]/[1,G,1,1,1]，
// NHWC 时 [N,1,1,G,1]/[1,1,1,G,1]）。
int32_t RunGraphCase(const char* graphName, const vector<int64_t>& xShape, const vector<int64_t>& statShape,
                     const vector<int64_t>& affineShape, int64_t numGroups, float xValue, float sumValue,
                     float squareSumValue, float scaleValue, float offsetValue, float epsilon, float expectY,
                     float expectMean, float expectVar, ge::Format xFmt = ge::FORMAT_ND)
{
    Graph graph(graphName);
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    vector<float*> allocBufs;
    auto freeBufs = [&allocBufs]() {
        for (float* p : allocBufs) {
            delete[] p;
        }
        allocBufs.clear();
    };
    Status ret = SUCCESS;

    auto iniOp = op::GNTrainingUpdate("gn_training_update");
    iniOp.set_attr_num_groups(numGroups);
    iniOp.set_attr_epsilon(epsilon);

    vector<const vector<int64_t>*> inShapes = {&xShape, &statShape, &statShape, &affineShape, &affineShape};
    float inValues[5] = {xValue, sumValue, squareSumValue, scaleValue, offsetValue};
    for (int i = 0; i < 5; i++) {
        auto data = op::Data("placeholder" + std::to_string(i)).set_attr_index(0);
        ge::Format inFmt = (i == 0) ? xFmt : ge::FORMAT_ND;
        TensorDesc desc = TensorDesc(ge::Shape(*inShapes[i]), inFmt, DT_FLOAT);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(inFmt);
        desc.SetOriginFormat(inFmt); // tiling 读取 OriginFormat：统计量/仿射必须为 ND
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
    // y / batch_mean / batch_variance：声明输出 desc（y 格式随 x）
    TensorDesc yDesc = TensorDesc(ge::Shape(xShape), xFmt, DT_FLOAT);
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
    freeBufs();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: RunGraph success, outputs=%zu\n", GetTime().c_str(), output.size());

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
        const float expectVals[2] = {expectMean, expectVar};
        const char* outNames[2] = {"batch_mean", "batch_variance"};
        for (int k = 0; k < 2; k++) {
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
    // case1（NCHW）：小 shape，mean=1.0、variance=0.0、y=0.5
    failCnt += (RunGraphCase("gn_training_update_geir_case1", {2, 4, 2, 3}, {2, 2, 1, 1, 1}, {1, 2, 1, 1, 1}, 2, 1.0f,
                             12.0f, 12.0f, 2.0f, 0.5f, 1e-5f, 0.5f, 1.0f, 0.0f) != SUCCESS);
    // case2（NCHW）：多 channel、G=32，mean=2.0、variance=0.0、y=0.25
    failCnt += (RunGraphCase("gn_training_update_geir_case2", {4, 64, 7, 7}, {4, 32, 1, 1, 1}, {1, 32, 1, 1, 1}, 32,
                             2.0f, 196.0f, 392.0f, 1.0f, 0.25f, 1e-3f, 0.25f, 2.0f, 0.0f) != SUCCESS);
    // case3（NHWC）：C=末维=3、G=3，统计量 [N,1,1,G,1]，mean=1.0、variance=0.0、y=0.5
    failCnt += (RunGraphCase("gn_training_update_geir_case3", {2, 4, 5, 3}, {2, 1, 1, 3, 1}, {1, 1, 1, 3, 1}, 3, 1.0f,
                             20.0f, 20.0f, 2.0f, 0.5f, 1e-5f, 0.5f, 1.0f, 0.0f, ge::FORMAT_NHWC) != SUCCESS);

    LOG_PRINT("[CHECK] total fail: %d, %s\n", failCnt, failCnt == 0 ? "PASS" : "FAIL");
    ge::GEFinalize();
    return failCnt == 0 ? SUCCESS : FAILED;
}
