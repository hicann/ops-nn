/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_apply_came_part1.cpp
 * \brief ApplyCamePart1 GE IR graph-mode verification.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "graph/error_codes.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/apply_came_part1_proto.h"
#include "../op_graph/apply_came_part1_graph_infer.cpp"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr float kEps = 0.125F;
constexpr float kTolerance = 1.0e-5F;

bool SameShape(const ge::TensorDesc& desc, const std::vector<int64_t>& expected)
{
    return desc.GetShape().GetDims() == expected;
}

bool VerifyTensor(const ge::Tensor& tensor, const std::vector<int64_t>& expectedShape,
                  const std::vector<float>& expected, const char* name)
{
    const ge::TensorDesc desc = tensor.GetTensorDesc();
    if (!SameShape(desc, expectedShape)) {
        std::cerr << name << " shape mismatch" << std::endl;
        return false;
    }
    if (desc.GetDataType() != ge::DT_FLOAT || tensor.GetData() == nullptr ||
        tensor.GetSize() != expected.size() * sizeof(float)) {
        std::cerr << name << " dtype or byte-size mismatch" << std::endl;
        return false;
    }

    const float* actual = reinterpret_cast<const float*>(tensor.GetData());
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > kTolerance) {
            std::cerr << name << " value mismatch at " << i << ": actual=" << actual[i] << ", expected=" << expected[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

int BuildGraph(ge::Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Operator>& inputs,
               std::vector<ge::Operator>& outputs, std::vector<float>& expectedRow, std::vector<float>& expectedCol,
               std::vector<float>& expectedAll)
{
    const std::vector<int64_t> gradShape = {4, 8};
    std::vector<float> grad(32);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = static_cast<float>(static_cast<int>(i % 11U) - 5) * 0.25F;
    }
    expectedRow.assign(gradShape[0], 0.0F);
    expectedCol.assign(gradShape[1], 0.0F);
    expectedAll.assign(1, 0.0F);
    for (size_t i = 0; i < grad.size(); ++i) {
        const float value = grad[i] * grad[i] + kEps;
        expectedRow[i / static_cast<size_t>(gradShape[1])] += value;
        expectedCol[i % static_cast<size_t>(gradShape[1])] += value;
        expectedAll[0] += value;
    }

    ge::TensorDesc gradDesc(ge::Shape(gradShape), ge::FORMAT_ND, ge::DT_FLOAT);
    gradDesc.SetPlacement(ge::kPlacementHost);
    gradDesc.SetRealDimCnt(gradShape.size());
    ge::op::Data gradData("grad");
    gradData.set_attr_index(0);
    gradData.update_input_desc_x(gradDesc);
    gradData.update_output_desc_y(gradDesc);
    graph.AddOp(gradData);
    input.emplace_back(gradDesc, reinterpret_cast<uint8_t*>(grad.data()), grad.size() * sizeof(float));
    inputs.push_back(gradData);

    const std::vector<int64_t> epsShape = {1};
    std::vector<float> eps = {kEps};
    ge::TensorDesc epsDesc(ge::Shape(epsShape), ge::FORMAT_ND, ge::DT_FLOAT);
    epsDesc.SetPlacement(ge::kPlacementHost);
    epsDesc.SetRealDimCnt(epsShape.size());
    ge::op::Data epsData("eps");
    epsData.set_attr_index(1);
    epsData.update_input_desc_x(epsDesc);
    epsData.update_output_desc_y(epsDesc);
    graph.AddOp(epsData);
    input.emplace_back(epsDesc, reinterpret_cast<uint8_t*>(eps.data()), eps.size() * sizeof(float));
    inputs.push_back(epsData);

    ge::op::ApplyCamePart1 op("apply_came_part1");
    op.set_input_grad(gradData);
    op.set_input_eps(epsData);
    op.update_input_desc_grad(gradDesc);
    op.update_input_desc_eps(epsDesc);
    op.update_output_desc_sum_grad_r(ge::TensorDesc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_sum_grad_c(ge::TensorDesc(ge::Shape({8}), ge::FORMAT_ND, ge::DT_FLOAT));
    op.update_output_desc_sum_grad_rc(ge::TensorDesc(ge::Shape(std::vector<int64_t>{}), ge::FORMAT_ND, ge::DT_FLOAT));
    graph.AddOp(op);
    outputs.push_back(op);
    return kSuccess;
}

int Run()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    int result = kFailed;
    {
        ge::Graph graph("apply_came_part1_geir");
        std::vector<ge::Tensor> input;
        std::vector<ge::Operator> inputs;
        std::vector<ge::Operator> outputs;
        std::vector<float> expectedRow;
        std::vector<float> expectedCol;
        std::vector<float> expectedAll;
        if (BuildGraph(graph, input, inputs, outputs, expectedRow, expectedCol, expectedAll) != kSuccess) {
            std::cerr << "BuildGraph failed" << std::endl;
        } else {
            graph.SetInputs(inputs).SetOutputs(outputs);
            std::map<ge::AscendString, ge::AscendString> options;
            ge::Session session(options);
            const uint32_t graphId = 0;
            if (session.AddGraph(graphId, graph, options) != ge::SUCCESS) {
                std::cerr << "Session::AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
            } else {
                std::vector<ge::Tensor> output;
                if (session.RunGraph(graphId, input, output) != ge::SUCCESS) {
                    std::cerr << "Session::RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
                } else if (output.size() != 3U) {
                    std::cerr << "Expected three outputs, got " << output.size() << std::endl;
                } else if (VerifyTensor(output[0], {4}, expectedRow, "sum_grad_r") &&
                           VerifyTensor(output[1], {8}, expectedCol, "sum_grad_c") &&
                           VerifyTensor(output[2], {}, expectedAll, "sum_grad_rc")) {
                    result = kSuccess;
                }
            }
        }
    }

    if (ge::GEFinalize() != ge::SUCCESS) {
        std::cerr << "GEFinalize failed" << std::endl;
        result = kFailed;
    }
    return result;
}
} // namespace

int main()
{
    const int result = Run();
    std::cout << (result == kSuccess ? "ApplyCamePart1 GEIR verification PASS" :
                                       "ApplyCamePart1 GEIR verification FAIL")
              << std::endl;
    return result;
}
