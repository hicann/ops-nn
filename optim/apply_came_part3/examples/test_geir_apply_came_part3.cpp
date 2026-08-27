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
 * \file test_geir_apply_came_part3.cpp
 * \brief ApplyCamePart3 GE IR graph-mode verification.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/error_codes.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/apply_came_part3_proto.h"
#include "../op_graph/apply_came_part3_graph_infer.cpp"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr int64_t kRows = 4;
constexpr int64_t kCols = 8;
constexpr float kEps = 0.125F;
constexpr float kBeta1 = 0.25F;
constexpr float kClipThreshold = 1.0F;
constexpr float kSumSquareU = 1.0F;
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

ge::TensorDesc MakeDesc(const std::vector<int64_t>& shape, ge::DataType dtype)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

void AddInput(ge::Graph& graph, const char* name, uint32_t index, const ge::TensorDesc& desc,
              const std::vector<uint8_t>& bytes, std::vector<ge::Tensor>& input, std::vector<ge::Operator>& inputs)
{
    ge::op::Data data(name);
    data.set_attr_index(index);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    graph.AddOp(data);
    input.emplace_back(desc, const_cast<uint8_t*>(bytes.data()), bytes.size());
    inputs.push_back(data);
}

template <typename T>
void StoreBytes(const std::vector<T>& values, std::vector<std::vector<uint8_t>>& storage)
{
    const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
    storage.emplace_back(begin, begin + values.size() * sizeof(values[0]));
}

void ComputeExpected(const std::vector<float>& u, const std::vector<float>& m, std::vector<float>& expectedM,
                     std::vector<float>& expectedRow, std::vector<float>& expectedCol, std::vector<float>& expectedAll)
{
    expectedM.resize(u.size());
    expectedRow.assign(static_cast<size_t>(kRows), 0.0F);
    expectedCol.assign(static_cast<size_t>(kCols), 0.0F);
    expectedAll.assign(1, 0.0F);
    const float scale = std::max(1.0F, kSumSquareU / (static_cast<float>(kRows * kCols) * kClipThreshold));
    for (size_t i = 0; i < u.size(); ++i) {
        expectedM[i] = (1.0F - kBeta1) * (u[i] / scale) + kBeta1 * m[i];
        const float delta = u[i] / scale - expectedM[i];
        const float value = delta * delta + kEps;
        expectedRow[i / static_cast<size_t>(kCols)] += value;
        expectedCol[i % static_cast<size_t>(kCols)] += value;
        expectedAll[0] += value;
    }
}

void AddApplyCamePart3(ge::Graph& graph, std::vector<ge::Operator>& inputs, std::vector<ge::Operator>& outputs)
{
    const ge::TensorDesc matrixDesc = MakeDesc({kRows, kCols}, ge::DT_FLOAT);
    const ge::TensorDesc scalarDesc = MakeDesc({1}, ge::DT_FLOAT);
    const ge::TensorDesc globalShapeDesc = MakeDesc({2}, ge::DT_INT64);

    ge::op::ApplyCamePart3 op("apply_came_part3");
    op.set_input_u(inputs[0]);
    op.set_input_m(inputs[1]);
    op.set_input_eps(inputs[2]);
    op.set_input_beta1(inputs[3]);
    op.set_input_clip_threshold(inputs[4]);
    op.set_input_sum_square_u(inputs[5]);
    op.set_input_global_shape(inputs[6]);
    op.set_attr_use_first_moment(true);
    op.update_input_desc_u(matrixDesc);
    op.update_input_desc_m(matrixDesc);
    op.update_input_desc_eps(scalarDesc);
    op.update_input_desc_beta1(scalarDesc);
    op.update_input_desc_clip_threshold(scalarDesc);
    op.update_input_desc_sum_square_u(scalarDesc);
    op.update_input_desc_global_shape(globalShapeDesc);
    op.update_output_desc_m(matrixDesc);
    op.update_output_desc_sum_u_r(MakeDesc({kRows}, ge::DT_FLOAT));
    op.update_output_desc_sum_u_c(MakeDesc({kCols}, ge::DT_FLOAT));
    op.update_output_desc_sum_u_rc(MakeDesc({1}, ge::DT_FLOAT));
    graph.AddOp(op);
    outputs.push_back(op);
}

int BuildGraph(ge::Graph& graph, std::vector<ge::Tensor>& input, std::vector<ge::Operator>& inputs,
               std::vector<ge::Operator>& outputs, std::vector<float>& expectedM, std::vector<float>& expectedRow,
               std::vector<float>& expectedCol, std::vector<float>& expectedAll,
               std::vector<std::vector<uint8_t>>& storage)
{
    std::vector<float> u(static_cast<size_t>(kRows * kCols));
    std::vector<float> m(static_cast<size_t>(kRows * kCols));
    for (size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<float>(static_cast<int>(i % 9U) - 4) * 0.125F;
        m[i] = static_cast<float>(static_cast<int>(i % 7U) - 3) * 0.0625F;
    }
    const std::vector<float> eps = {kEps};
    const std::vector<float> beta1 = {kBeta1};
    const std::vector<float> clipThreshold = {kClipThreshold};
    const std::vector<float> sumSquareU = {kSumSquareU};
    const std::vector<int64_t> globalShape = {kRows, kCols};
    ComputeExpected(u, m, expectedM, expectedRow, expectedCol, expectedAll);

    storage.reserve(7);
    StoreBytes(u, storage);
    StoreBytes(m, storage);
    StoreBytes(eps, storage);
    StoreBytes(beta1, storage);
    StoreBytes(clipThreshold, storage);
    StoreBytes(sumSquareU, storage);
    StoreBytes(globalShape, storage);
    const ge::TensorDesc matrixDesc = MakeDesc({kRows, kCols}, ge::DT_FLOAT);
    const ge::TensorDesc scalarDesc = MakeDesc({1}, ge::DT_FLOAT);
    const ge::TensorDesc globalShapeDesc = MakeDesc({2}, ge::DT_INT64);
    AddInput(graph, "u", 0, matrixDesc, storage[0], input, inputs);
    AddInput(graph, "m", 1, matrixDesc, storage[1], input, inputs);
    AddInput(graph, "eps", 2, scalarDesc, storage[2], input, inputs);
    AddInput(graph, "beta1", 3, scalarDesc, storage[3], input, inputs);
    AddInput(graph, "clip_threshold", 4, scalarDesc, storage[4], input, inputs);
    AddInput(graph, "sum_square_u", 5, scalarDesc, storage[5], input, inputs);
    AddInput(graph, "global_shape", 6, globalShapeDesc, storage[6], input, inputs);
    AddApplyCamePart3(graph, inputs, outputs);
    return kSuccess;
}

int ExecuteGraph(ge::Graph& graph, const std::vector<ge::Tensor>& input, const std::vector<ge::Operator>& inputs,
                 const std::vector<ge::Operator>& outputs, const std::vector<float>& expectedM,
                 const std::vector<float>& expectedRow, const std::vector<float>& expectedCol,
                 const std::vector<float>& expectedAll)
{
    graph.SetInputs(inputs).SetOutputs(outputs);
    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    const uint32_t graphId = 0;
    if (session.AddGraph(graphId, graph, options) != ge::SUCCESS) {
        std::cerr << "Session::AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    std::vector<ge::Tensor> output;
    if (session.RunGraph(graphId, input, output) != ge::SUCCESS) {
        std::cerr << "Session::RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    if (output.size() != 4U) {
        std::cerr << "Expected four outputs, got " << output.size() << std::endl;
        return kFailed;
    }
    return VerifyTensor(output[0], {kRows, kCols}, expectedM, "m") &&
                   VerifyTensor(output[1], {kRows}, expectedRow, "sum_u_r") &&
                   VerifyTensor(output[2], {kCols}, expectedCol, "sum_u_c") &&
                   VerifyTensor(output[3], {1}, expectedAll, "sum_u_rc") ?
               kSuccess :
               kFailed;
}

int Run()
{
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    int result = kFailed;
    {
        ge::Graph graph("apply_came_part3_geir");
        std::vector<ge::Tensor> input;
        std::vector<ge::Operator> inputs;
        std::vector<ge::Operator> outputs;
        std::vector<float> expectedM;
        std::vector<float> expectedRow;
        std::vector<float> expectedCol;
        std::vector<float> expectedAll;
        std::vector<std::vector<uint8_t>> storage;
        if (BuildGraph(graph, input, inputs, outputs, expectedM, expectedRow, expectedCol, expectedAll, storage) !=
            kSuccess) {
            std::cerr << "BuildGraph failed" << std::endl;
        } else {
            result = ExecuteGraph(graph, input, inputs, outputs, expectedM, expectedRow, expectedCol, expectedAll);
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
    std::cout << (result == kSuccess ? "ApplyCamePart3 GEIR verification PASS" :
                                       "ApplyCamePart3 GEIR verification FAIL")
              << std::endl;
    return result;
}
