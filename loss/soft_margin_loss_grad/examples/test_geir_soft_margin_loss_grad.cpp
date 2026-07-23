/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// soft_margin_loss_grad GE IR 图模式调用示例
// 构造 SoftMarginLossGrad 单算子图：self、target、grad_output 三输入 + reduction 属性，输出 out。
// ============================================================================

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

#include "../op_graph/soft_margin_loss_grad_proto.h"

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
    if (dt == ge::DT_FLOAT)
        return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        return 2;
    return 4;
}

// 生成填充常值的 FP32 输入数据
int32_t GenData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto d : shapes)
        size *= d;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i)
        pData[i] = value;
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
    delete[] pData;
    return SUCCESS;
}

int CreateGraph(Graph& graph, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                std::vector<Operator>& outputs)
{
    Status ret = SUCCESS;
    std::vector<int64_t> shape = {2, 3};
    DataType dt = DT_FLOAT;

    auto op = op::SoftMarginLossGrad("smlg");
    op.set_attr_reduction("none");

    // predict (x)
    auto predictData = op::Data("predict").set_attr_index(0);
    TensorDesc predictDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor predictT;
    ret = GenData(shape, predictT, predictDesc, 0.5f);
    predictData.update_input_desc_x(predictDesc);
    predictData.update_output_desc_y(predictDesc);
    op.set_input_predict(predictData);
    input.push_back(predictT);
    inputs.push_back(predictData);

    // label (y, ±1)
    auto labelData = op::Data("label").set_attr_index(1);
    TensorDesc labelDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor labelT;
    ret = GenData(shape, labelT, labelDesc, 1.0f);
    labelData.update_input_desc_x(labelDesc);
    labelData.update_output_desc_y(labelDesc);
    op.set_input_label(labelData);
    input.push_back(labelT);
    inputs.push_back(labelData);

    // dout
    auto doutData = op::Data("dout").set_attr_index(2);
    TensorDesc doutDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor doutT;
    ret = GenData(shape, doutT, doutDesc, 1.0f);
    doutData.update_input_desc_x(doutDesc);
    doutData.update_output_desc_y(doutDesc);
    op.set_input_dout(doutData);
    input.push_back(doutT);
    inputs.push_back(doutData);

    TensorDesc gradientDesc(ge::Shape(shape), FORMAT_ND, dt);
    op.update_output_desc_gradient(gradientDesc);
    outputs.push_back(op);
    (void)ret;
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("soft_margin_loss_grad_geir_test");
    std::vector<ge::Tensor> input;

    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO: GEInitialize success\n", GetTime().c_str());

    std::vector<Operator> inputs{}, outputs{};
    if (CreateGraph(graph, input, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR: create graph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty())
        graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR: create session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graph_id = 0;
    std::map<AscendString, AscendString> graph_options;
    session->AddGraph(graph_id, graph, graph_options);
    aclgrphDumpGraph(graph, "./dump", strlen("./dump"));

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO: Run graph success, outputs=%zu\n", GetTime().c_str(), output.size());

    delete session;
    GEFinalize();
    printf("%s - INFO: GE IR example done\n", GetTime().c_str());
    return SUCCESS;
}
