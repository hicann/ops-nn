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
 * \file test_geir_cosine_embedding_loss.cpp
 * \brief CosineEmbeddingLoss arch35 GEIR example.
 */

#include <ctime>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "../../op_graph/cosine_embedding_loss_proto.h"

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

int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int GenTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, Tensor& tensor,
              TensorDesc& tensorDesc)
{
    tensorDesc.SetRealDimCnt(shape.size());
    auto* data = new (std::nothrow) float[hostData.size()];
    CHECK_RET(data != nullptr, return FAILED);
    for (size_t i = 0; i < hostData.size(); ++i) {
        data[i] = hostData[i];
    }
    tensor = Tensor(tensorDesc, reinterpret_cast<uint8_t*>(data), hostData.size() * sizeof(float));
    return SUCCESS;
}

int CreateGraph(std::vector<Tensor>& graphInputs, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                Graph& graph)
{
    auto cosineEmbeddingLoss = op::CosineEmbeddingLoss("cosineEmbeddingLoss");
    const std::vector<int64_t> inputShape = {2, 3};
    const std::vector<int64_t> targetShape = {2};
    const std::vector<int64_t> outputShape = {2};

    auto x1 = op::Data("x1").set_attr_index(0);
    TensorDesc x1Desc = TensorDesc(Shape(inputShape), FORMAT_ND, DT_FLOAT);
    x1Desc.SetPlacement(kPlacementHost);
    Tensor x1Tensor;
    CHECK_RET(GenTensor({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, inputShape, x1Tensor, x1Desc) == SUCCESS, return FAILED);
    x1.update_input_desc_x(x1Desc);
    x1.update_output_desc_y(x1Desc);
    graphInputs.push_back(x1Tensor);
    graph.AddOp(x1);
    cosineEmbeddingLoss.set_input_x1(x1);
    inputs.push_back(x1);

    auto x2 = op::Data("x2").set_attr_index(1);
    TensorDesc x2Desc = TensorDesc(Shape(inputShape), FORMAT_ND, DT_FLOAT);
    x2Desc.SetPlacement(kPlacementHost);
    Tensor x2Tensor;
    CHECK_RET(GenTensor({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, inputShape, x2Tensor, x2Desc) == SUCCESS, return FAILED);
    x2.update_input_desc_x(x2Desc);
    x2.update_output_desc_y(x2Desc);
    graphInputs.push_back(x2Tensor);
    graph.AddOp(x2);
    cosineEmbeddingLoss.set_input_x2(x2);
    inputs.push_back(x2);

    auto target = op::Data("target").set_attr_index(2);
    TensorDesc targetDesc = TensorDesc(Shape(targetShape), FORMAT_ND, DT_FLOAT);
    targetDesc.SetPlacement(kPlacementHost);
    Tensor targetTensor;
    CHECK_RET(GenTensor({1.0f, -1.0f}, targetShape, targetTensor, targetDesc) == SUCCESS, return FAILED);
    target.update_input_desc_x(targetDesc);
    target.update_output_desc_y(targetDesc);
    graphInputs.push_back(targetTensor);
    graph.AddOp(target);
    cosineEmbeddingLoss.set_input_target(target);
    inputs.push_back(target);

    cosineEmbeddingLoss.set_attr_margin(0.2f);
    cosineEmbeddingLoss.set_attr_reduction("none");
    TensorDesc yDesc = TensorDesc(Shape(outputShape), FORMAT_ND, DT_FLOAT);
    cosineEmbeddingLoss.update_output_desc_y(yDesc);
    outputs.push_back(cosineEmbeddingLoss);
    return SUCCESS;
}

int main()
{
    Graph graph("cosine_embedding_loss_arch35_example");
    std::vector<Tensor> graphInputs;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    LOG_PRINT("%s - INFO - [XIR]: Start to initialize GE\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = GEInitialize(globalOptions);
    CHECK_RET(ret == SUCCESS, LOG_PRINT("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str()); return FAILED);

    ret = CreateGraph(graphInputs, inputs, outputs, graph);
    CHECK_RET(ret == SUCCESS, GEFinalize(); return FAILED);
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions;
    Session* session = new (std::nothrow) Session(buildOptions);
    CHECK_RET(session != nullptr, GEFinalize(); return FAILED);

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions;
    ret = session->AddGraph(graphId, graph, graphOptions);
    CHECK_RET(ret == SUCCESS, delete session; GEFinalize(); return FAILED);

    std::vector<Tensor> output;
    ret = session->RunGraph(graphId, graphInputs, output);
    CHECK_RET(ret == SUCCESS, delete session; GEFinalize(); return FAILED);
    LOG_PRINT("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    for (size_t i = 0; i < output.size(); ++i) {
        auto* result = reinterpret_cast<float*>(output[i].GetData());
        int64_t size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        for (int64_t j = 0; j < size; ++j) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
    }

    delete session;
    ret = GEFinalize();
    CHECK_RET(ret == SUCCESS, return FAILED);
    return SUCCESS;
}
