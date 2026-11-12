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
// soft_margin_loss GE IR 图模式调用示例
// 构造 SoftMarginLoss 单算子图：input_x、input_y 两输入 + reduction 属性，输出 output_z。
// ============================================================================

#include <ctime>
#include <iostream>
#include <map>
#include <new>
#include <string.h>
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

#include "../../op_graph/soft_margin_loss_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

// 生成填充常值的 FP32 输入数据
int32_t GenData(const std::vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (auto d : shape) {
        size *= static_cast<size_t>(d);
    }
    float* pData = new (std::nothrow) float[size];
    CHECK_RET(pData != nullptr, return FAILED);
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), size * sizeof(float));
    delete[] pData;
    return SUCCESS;
}

int CreateGraph(Graph& graph, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                std::vector<Operator>& outputs)
{
    const std::vector<int64_t> shape = {2, 3};
    const DataType dt = DT_FLOAT;

    auto op = op::SoftMarginLoss("sml");
    op.set_attr_reduction("none");

    // input_x
    auto inputXData = op::Data("input_x").set_attr_index(0);
    TensorDesc inputXDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor inputXT;
    CHECK_RET(GenData(shape, inputXT, inputXDesc, 0.5f) == SUCCESS, return FAILED);
    inputXData.update_input_desc_x(inputXDesc);
    inputXData.update_output_desc_y(inputXDesc);
    op.set_input_input_x(inputXData);
    CHECK_RET(graph.AddOp(inputXData) == ge::GRAPH_SUCCESS, return FAILED);
    input.push_back(inputXT);
    inputs.push_back(inputXData);

    // input_y (±1)
    auto inputYData = op::Data("input_y").set_attr_index(1);
    TensorDesc inputYDesc(ge::Shape(shape), FORMAT_ND, dt);
    Tensor inputYT;
    CHECK_RET(GenData(shape, inputYT, inputYDesc, 1.0f) == SUCCESS, return FAILED);
    inputYData.update_input_desc_x(inputYDesc);
    inputYData.update_output_desc_y(inputYDesc);
    op.set_input_input_y(inputYData);
    CHECK_RET(graph.AddOp(inputYData) == ge::GRAPH_SUCCESS, return FAILED);
    input.push_back(inputYT);
    inputs.push_back(inputYData);

    TensorDesc outputZDesc(ge::Shape(shape), FORMAT_ND, dt);
    op.update_output_desc_output_z(outputZDesc);
    CHECK_RET(graph.AddOp(op) == ge::GRAPH_SUCCESS, return FAILED);
    outputs.push_back(op);
    return SUCCESS;
}

int main()
{
    Graph graph("soft_margin_loss_geir_test");
    std::vector<ge::Tensor> input;
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR: GEInitialize failed\n", GetTime().c_str()); return FAILED);
    LOG_PRINT("%s - INFO: GEInitialize success\n", GetTime().c_str());

    ret = CreateGraph(graph, input, inputs, outputs);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR: create graph failed\n", GetTime().c_str()); GEFinalize();
              return FAILED);
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions;
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    CHECK_RET(session != nullptr, LOG_PRINT("%s - ERROR: create session failed\n", GetTime().c_str()); GEFinalize();
              return FAILED);

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions;
    ret = session->AddGraph(graphId, graph, graphOptions);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR: add graph failed\n", GetTime().c_str()); delete session;
              GEFinalize(); return FAILED);

    const char* dumpPath = "./dump";
    ret = aclgrphDumpGraph(graph, dumpPath, static_cast<uint32_t>(strlen(dumpPath)));
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - WARNING: dump graph failed\n", GetTime().c_str()));

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR: run graph failed\n", GetTime().c_str()); delete session;
              GEFinalize(); return FAILED);
    LOG_PRINT("%s - INFO: Run graph success, outputs=%zu\n", GetTime().c_str(), output.size());

    delete session;
    ret = GEFinalize();
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR: GEFinalize failed\n", GetTime().c_str()); return FAILED);
    LOG_PRINT("%s - INFO: GE IR example done\n", GetTime().c_str());
    return SUCCESS;
}
