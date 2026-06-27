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
    if (dt == ge::DT_FLOAT) return 4;
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) return 2;
    return 4;
}

// 生成填充常值的 FP32 输入数据
int32_t GenData(vector<int64_t> shapes, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (auto d : shapes) size *= d;
    float* pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) pData[i] = value;
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

    // self (x)
    auto selfData = op::Data("self").set_attr_index(0);
    TensorDesc selfDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor selfT; ret = GenData(shape, selfT, selfDesc, 0.5f);
    selfData.update_input_desc_x(selfDesc); selfData.update_output_desc_y(selfDesc);
    op.set_input_self(selfData); input.push_back(selfT);
    inputs.push_back(selfData);

    // target (y, ±1)
    auto targetData = op::Data("target").set_attr_index(1);
    TensorDesc targetDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor targetT; ret = GenData(shape, targetT, targetDesc, 1.0f);
    targetData.update_input_desc_x(targetDesc); targetData.update_output_desc_y(targetDesc);
    op.set_input_target(targetData); input.push_back(targetT);
    inputs.push_back(targetData);

    // grad_output
    auto gradData = op::Data("grad_output").set_attr_index(2);
    TensorDesc gradDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor gradT; ret = GenData(shape, gradT, gradDesc, 1.0f);
    gradData.update_input_desc_x(gradDesc); gradData.update_output_desc_y(gradDesc);
    op.set_input_grad_output(gradData); input.push_back(gradT);
    inputs.push_back(gradData);

    TensorDesc outDesc(ge::Shape(shape), FORMAT_ND, dt);
    op.update_output_desc_out(outDesc);
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
    if (ret != SUCCESS) { printf("%s - ERROR: GEInitialize failed\n", GetTime().c_str()); return FAILED; }
    printf("%s - INFO: GEInitialize success\n", GetTime().c_str());

    std::vector<Operator> inputs{}, outputs{};
    if (CreateGraph(graph, input, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR: create graph failed\n", GetTime().c_str()); GEFinalize(); return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    if (session == nullptr) { printf("%s - ERROR: create session failed\n", GetTime().c_str()); return FAILED; }

    uint32_t graph_id = 0;
    std::map<AscendString, AscendString> graph_options;
    session->AddGraph(graph_id, graph, graph_options);
    aclgrphDumpGraph(graph, "./dump", strlen("./dump"));

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO: Run graph failed\n", GetTime().c_str()); delete session; GEFinalize(); return FAILED;
    }
    printf("%s - INFO: Run graph success, outputs=%zu\n", GetTime().c_str(), output.size());

    delete session;
    GEFinalize();
    printf("%s - INFO: GE IR example done\n", GetTime().c_str());
    return SUCCESS;
}
