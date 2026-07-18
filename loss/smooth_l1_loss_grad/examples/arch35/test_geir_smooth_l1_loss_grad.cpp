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
 * SmoothL1LossGrad GE IR 调用示例
 *
 * 演示通过 GE IR 图模式调用 SmoothL1LossGrad 算子：
 * 1. ge::GEInitialize() 初始化（graphRunMode=1，deviceId=0）
 * 2. 构建 Data(predict/label/dout) -> SmoothL1LossGrad(sigma) 计算图
 * 3. ge::Session AddGraph + RunGraph 执行（经 TBE 动态编译加载自定义 kernel）
 * 4. 读取输出张量并与 CPU golden 逐元素比对（双万分之一 fp32）
 * 5. ge::GEFinalize() 资源释放
 *
 * 算子语义：gradient = clamp((predict - label) / sigma, -1, 1) * dout（逐元素，reduction='none'）
 *   GEIR 通路：predict/label/dout 为输入 Tensor，sigma 为编译期 Float 属性（默认 1.0），gradient 为输出 Tensor。
 *
 * 编译：参见 CMakeLists.txt   运行：参见 run.sh
 *   ⚠ GE IR 执行触发 TBE 动态编译，需系统 Python 3.9（见 run.sh 环境设置）。
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/smooth_l1_loss_grad_proto.h"

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

// CPU golden：gradient = clamp((predict - label) / sigma, -1, 1) * dout（独立于 kernel，double 直算）
static void ComputeGolden(const float* predict, const float* label, const float* dout, float* gradient, int64_t n,
                          double sigma)
{
    for (int64_t i = 0; i < n; ++i) {
        double diff = static_cast<double>(predict[i]) - static_cast<double>(label[i]);
        double c = std::min(std::max(diff, -sigma), sigma);
        gradient[i] = static_cast<float>((c / sigma) * static_cast<double>(dout[i]));
    }
}

// 构建 float32 输入数据：覆盖 diff>σ / |diff|≤σ / diff<−σ / diff=0
static void MakeInputs(int64_t n, std::vector<float>& hPredict, std::vector<float>& hLabel, std::vector<float>& hDout)
{
    const std::vector<float> segP = {3.f, -2.f, 0.5f, 1.f, -0.5f, 5.f, -5.f, 2.f};
    const std::vector<float> segL = {0.f, 0.f, 0.f, 0.5f, 0.5f, 0.f, 0.f, 2.f};
    const std::vector<float> segD = {1.f, 1.f, 1.f, 2.f, 2.f, 0.5f, 0.5f, 3.f};
    hPredict.resize(static_cast<size_t>(n));
    hLabel.resize(static_cast<size_t>(n));
    hDout.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        hPredict[static_cast<size_t>(i)] = segP[static_cast<size_t>(i % (int64_t)segP.size())];
        hLabel[static_cast<size_t>(i)] = segL[static_cast<size_t>(i % (int64_t)segL.size())];
        hDout[static_cast<size_t>(i)] = segD[static_cast<size_t>(i % (int64_t)segD.size())];
    }
}

int CreateSmoothL1LossGradGraph(DataType inDtype, float sigma, const std::vector<int64_t>& tensorShape,
                                std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                                std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    int64_t n = 1;
    for (auto d : tensorShape)
        n *= d;

    // ---- 输入 predict / label / dout（Data 占位）：host 侧构造 float32 数据
    TensorDesc inDesc(ge::Shape(tensorShape), FORMAT_ND, inDtype);
    inDesc.SetPlacement(ge::kPlacementHost);
    inDesc.SetFormat(FORMAT_ND);
    inDesc.SetRealDimCnt(tensorShape.size());

    std::vector<float> hPredict, hLabel, hDout;
    MakeInputs(n, hPredict, hLabel, hDout);
    uint32_t dataLen = static_cast<uint32_t>(n) * sizeof(float);

    auto placeholderPredict = op::Data("predict").set_attr_index(0);
    placeholderPredict.update_input_desc_x(inDesc);
    placeholderPredict.update_output_desc_y(inDesc);
    Tensor predictTensor(inDesc, reinterpret_cast<uint8_t*>(hPredict.data()), dataLen);
    input.push_back(predictTensor);
    graph.AddOp(placeholderPredict);
    inputs.push_back(placeholderPredict);

    auto placeholderLabel = op::Data("label").set_attr_index(1);
    placeholderLabel.update_input_desc_x(inDesc);
    placeholderLabel.update_output_desc_y(inDesc);
    Tensor labelTensor(inDesc, reinterpret_cast<uint8_t*>(hLabel.data()), dataLen);
    input.push_back(labelTensor);
    graph.AddOp(placeholderLabel);
    inputs.push_back(placeholderLabel);

    auto placeholderDout = op::Data("dout").set_attr_index(2);
    placeholderDout.update_input_desc_x(inDesc);
    placeholderDout.update_output_desc_y(inDesc);
    Tensor doutTensor(inDesc, reinterpret_cast<uint8_t*>(hDout.data()), dataLen);
    input.push_back(doutTensor);
    graph.AddOp(placeholderDout);
    inputs.push_back(placeholderDout);

    // ---- SmoothL1LossGrad 节点：set_input_predict/label/dout + set_attr_sigma
    auto grad = op::SmoothL1LossGrad("smooth_l1_loss_grad_0");
    grad.set_input_predict(placeholderPredict);
    grad.set_input_label(placeholderLabel);
    grad.set_input_dout(placeholderDout);
    grad.set_attr_sigma(sigma);
    grad.update_input_desc_predict(inDesc);
    grad.update_input_desc_label(inDesc);
    grad.update_input_desc_dout(inDesc);

    TensorDesc outDesc(ge::Shape(tensorShape), FORMAT_ND, inDtype);
    outDesc.SetFormat(FORMAT_ND);
    grad.update_output_desc_gradient(outDesc);

    graph.AddOp(grad);
    outputs.push_back(grad);
    (void)ret;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_smooth_l1_loss_grad";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    const float sigma = 1.0f;
    const std::vector<int64_t> tensorShape = {4, 8};
    int64_t numElements = 1;
    for (auto d : tensorShape)
        numElements *= d;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    DataType inDtype = DT_FLOAT;

    ret = CreateSmoothL1LossGradGraph(inDtype, sigma, tensorShape, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create operator in graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::cout << "Error message: " << error_msg.GetString() << std::endl;
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    // ---- 结果验证：GE IR 输出 vs CPU golden（fp32 双万分之一）
    bool passed = true;
    if (output.empty()) {
        printf("%s - ERROR - [XIR]: No output tensor produced\n", GetTime().c_str());
        passed = false;
    } else {
        std::vector<float> hPredict, hLabel, hDout;
        MakeInputs(numElements, hPredict, hLabel, hDout);
        std::vector<float> hGolden(static_cast<size_t>(numElements));
        ComputeGolden(hPredict.data(), hLabel.data(), hDout.data(), hGolden.data(), numElements, sigma);

        const float* outData = reinterpret_cast<const float*>(output[0].GetData());
        int64_t outNum = output[0].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "=== SmoothL1LossGrad GE IR Test"
                  << " (gradient = clamp((predict-label)/sigma, -1, 1) * dout) ===" << std::endl;
        std::cout << "dtype: float32, shape: [4, 8], sigma=" << sigma << std::endl;
        std::cout << "output shape size = " << outNum << std::endl;

        const float rtol = 1e-4f;
        const float atol = 1e-4f;
        int64_t cmpNum = std::min<int64_t>(outNum, numElements);
        for (int64_t i = 0; i < cmpNum; ++i) {
            float err = std::abs(outData[i] - hGolden[static_cast<size_t>(i)]);
            float tol = atol + rtol * std::abs(hGolden[static_cast<size_t>(i)]);
            bool ok = (err <= tol);
            if (!ok)
                passed = false;
            if (i < 8) { // 打印首 8 个（一个完整段模式）
                std::cout << "  [" << i << "] predict=" << hPredict[static_cast<size_t>(i)]
                          << " label=" << hLabel[static_cast<size_t>(i)] << " dout=" << hDout[static_cast<size_t>(i)]
                          << " -> gradient=" << outData[i] << " (expected " << hGolden[static_cast<size_t>(i)]
                          << ", err " << err << (ok ? " OK" : " FAIL") << ")" << std::endl;
            }
        }
    }

    std::cout << std::endl;
    if (passed) {
        printf("%s - INFO - [XIR]: GE IR pathway verification PASSED\n", GetTime().c_str());
        std::cout << "[PASS] GE IR results match golden reference." << std::endl;
    } else {
        std::cout << "[FAIL] GE IR results do not match golden reference." << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to finalize ge\n", GetTime().c_str());
    Status finRet = ge::GEFinalize();
    if (finRet != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
    }
    delete session;
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());

    return passed ? SUCCESS : FAILED;
}
