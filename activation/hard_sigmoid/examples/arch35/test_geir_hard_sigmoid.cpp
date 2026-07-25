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
 * \file test_geir_hard_sigmoid.cpp
 * \brief 通过GE IR构图方式调用HardSigmoid算子的样例，运行结果与CPU golden比对。
 */

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "../../op_graph/hard_sigmoid_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::vector;

namespace {
constexpr float kAlpha = 1.0f / 6.0f;
constexpr float kBeta = 0.5f;
constexpr float kTolerance = 1e-4f;

// 取值覆盖y=0截断区(x<=-3)、线性区与y=1截断区(x>=3)
const vector<float> kInputData = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
const vector<int64_t> kInputShape = {4, 2};

float HardSigmoidGolden(float x) { return std::fmax(0.0f, std::fmin(1.0f, kAlpha * x + kBeta)); }

// 构造Data -> HardSigmoid单节点图
Status BuildGraph(Graph& graph, vector<ge::Tensor>& inputTensors)
{
    TensorDesc xDesc(ge::Shape(kInputShape), FORMAT_ND, DT_FLOAT);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetRealDimCnt(kInputShape.size());

    auto data = op::Data("x").set_attr_index(0);
    data.update_input_desc_x(xDesc);
    data.update_output_desc_y(xDesc);

    auto hardSigmoid = op::HardSigmoid("hard_sigmoid");
    hardSigmoid.set_attr_alpha(kAlpha);
    hardSigmoid.set_attr_beta(kBeta);
    hardSigmoid.set_input_input_x(data);

    graph.AddOp(data);
    graph.SetInputs({data}).SetOutputs({hardSigmoid});

    inputTensors.emplace_back(xDesc, reinterpret_cast<const uint8_t*>(kInputData.data()),
                              kInputData.size() * sizeof(float));
    return SUCCESS;
}

// 与CPU golden比对，返回不符合预期的个数
int VerifyOutput(const ge::Tensor& output)
{
    const auto* result = reinterpret_cast<const float*>(output.GetData());
    const int64_t count = output.GetTensorDesc().GetShape().GetShapeSize();
    const size_t expectedBytes = kInputData.size() * sizeof(float);
    if (result == nullptr || count < 0 || static_cast<size_t>(count) != kInputData.size() ||
        output.GetSize() < expectedBytes) {
        printf("[ERROR] Invalid output buffer: count=%ld, bytes=%zu, expected_count=%zu, expected_bytes=%zu\n", count,
               output.GetSize(), kInputData.size(), expectedBytes);
        return 1;
    }
    int failures = 0;

    printf("%-10s %-12s %-12s %s\n", "x", "npu", "golden", "check");
    for (int64_t i = 0; i < count; i++) {
        float golden = HardSigmoidGolden(kInputData[i]);
        bool matched = std::fabs(result[i] - golden) <= kTolerance;
        printf("%-10.4f %-12.6f %-12.6f %s\n", kInputData[i], result[i], golden, matched ? "OK" : "MISMATCH");
        if (!matched) {
            failures++;
        }
    }
    return failures;
}
} // namespace

int main()
{
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("[ERROR] GEInitialize failed: %s\n", ge::GEGetErrorMsgV2().GetString());
        return FAILED;
    }

    Graph graph("hard_sigmoid_geir");
    vector<ge::Tensor> inputTensors;
    if (BuildGraph(graph, inputTensors) != SUCCESS) {
        printf("[ERROR] Build graph failed\n");
        (void)ge::GEFinalize();
        return FAILED;
    }

    std::map<AscendString, AscendString> sessionOptions = {};
    std::unique_ptr<Session> session(new (std::nothrow) Session(sessionOptions));
    if (session == nullptr) {
        printf("[ERROR] Create GE session failed\n");
        (void)ge::GEFinalize();
        return FAILED;
    }

    int failures = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    const uint32_t graphId = 0;
    // AddGraph只注册图，算子编译在RunGraph阶段触发
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("[ERROR] AddGraph failed: %s\n", ge::GEGetErrorMsgV2().GetString());
        failures = 1;
    } else {
        vector<ge::Tensor> outputTensors;
        if (session->RunGraph(graphId, inputTensors, outputTensors) != SUCCESS) {
            printf("[ERROR] RunGraph failed: %s\n", ge::GEGetErrorMsgV2().GetString());
            failures = 1;
        } else if (outputTensors.empty()) {
            printf("[ERROR] RunGraph produced no output\n");
            failures = 1;
        } else {
            failures = VerifyOutput(outputTensors[0]);
        }
    }

    session.reset();
    if (ge::GEFinalize() != SUCCESS) {
        printf("[WARN] GEFinalize failed: %s\n", ge::GEGetErrorMsgV2().GetString());
    }

    printf("%s\n", failures == 0 ? "[INFO] test_geir_hard_sigmoid PASSED" : "[ERROR] test_geir_hard_sigmoid FAILED");
    return failures == 0 ? SUCCESS : FAILED;
}
