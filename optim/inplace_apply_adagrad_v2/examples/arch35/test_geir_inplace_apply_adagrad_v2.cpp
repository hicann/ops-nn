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
 * @file test_geir_inplace_apply_adagrad_v2.cpp
 * @brief Static GE IR verification for InplaceApplyAdagradV2 on Ascend950.
 *
 * Mutable var and accum inputs are represented by Variable nodes and initialized
 * by Assign nodes. The optimizer depends on both Assign nodes, so the two outputs
 * are checked after exactly one deterministic state update.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "ge_api.h"
#include "graph.h"
#include "state_ops.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/inplace_apply_adagrad_v2_proto.h"

namespace {

constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr size_t kOutputCount = 2U;
constexpr float kEpsilon = 0.125F;
constexpr float kAtol = 1.0e-4F;
constexpr float kRtol = 1.0e-4F;

using ShapeVector = std::vector<int64_t>;

struct StaticInputs {
    ShapeVector shape;
    std::vector<float> var;
    std::vector<float> accum;
    float lr;
    std::vector<float> grad;
};

struct ExpectedOutputs {
    std::array<std::vector<float>, kOutputCount> values;
};

std::string ShapeToString(const ShapeVector& shape)
{
    std::string result = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0U) {
            result += ",";
        }
        result += std::to_string(shape[index]);
    }
    result += "]";
    return result;
}

int64_t ElementCount(const ShapeVector& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

ge::TensorDesc MakeDesc(const ShapeVector& shape)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

ge::Tensor MakeFloatTensor(const ShapeVector& shape, const std::vector<float>& data)
{
    const ge::TensorDesc desc = MakeDesc(shape);
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(float));
}

ge::Tensor MakeFloatScalar(float value)
{
    const ge::TensorDesc desc = MakeDesc({1});
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(&value), sizeof(value));
}

ge::Operator AddData(const std::string& name, uint32_t index, const ge::TensorDesc& desc, ge::Graph& graph,
                     std::vector<ge::Operator>& graphInputs)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    graph.AddOp(data);
    graphInputs.push_back(data);
    return data;
}

ge::Operator AddVariable(const std::string& name, uint32_t index, ge::Operator& initializer, const ge::TensorDesc& desc,
                         ge::Graph& graph)
{
    auto variable = ge::op::Variable(name.c_str())
                        .set_input_x(initializer)
                        .set_attr_index(index)
                        .set_attr_value(ge::Tensor(desc))
                        .set_attr_shared_name(name.c_str());
    variable.update_input_desc_x(desc);
    variable.update_output_desc_y(desc);
    graph.AddOp(variable);
    return variable;
}

ge::Operator AddInitializerAssign(const std::string& name, ge::Operator& variable, ge::Operator& value,
                                  const ge::TensorDesc& desc, ge::Graph& graph)
{
    auto assign = ge::op::Assign(name.c_str())
                      .set_input_ref(variable)
                      .set_input_value(value)
                      .set_attr_validate_shape(true)
                      .set_attr_use_locking(false);
    assign.update_input_desc_ref(desc);
    assign.update_input_desc_value(desc);
    assign.update_output_desc_ref(desc);
    graph.AddOp(assign);
    return assign;
}

void BuildGraph(const ShapeVector& shape, ge::Graph& graph, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs)
{
    const ge::TensorDesc tensorDesc = MakeDesc(shape);
    const ge::TensorDesc scalarDesc = MakeDesc({1});

    uint32_t index = 0U;
    auto varData = AddData("inplace_apply_adagrad_v2_var_data", index++, tensorDesc, graph, graphInputs);
    auto accumData = AddData("inplace_apply_adagrad_v2_accum_data", index++, tensorDesc, graph, graphInputs);
    auto lr = AddData("inplace_apply_adagrad_v2_lr", index++, scalarDesc, graph, graphInputs);
    auto grad = AddData("inplace_apply_adagrad_v2_grad", index, tensorDesc, graph, graphInputs);

    auto var = AddVariable("inplace_apply_adagrad_v2_var", 0U, varData, tensorDesc, graph);
    auto accum = AddVariable("inplace_apply_adagrad_v2_accum", 1U, accumData, tensorDesc, graph);
    auto assignVar = AddInitializerAssign("inplace_apply_adagrad_v2_assign_var", var, varData, tensorDesc, graph);
    auto assignAccum = AddInitializerAssign("inplace_apply_adagrad_v2_assign_accum", accum, accumData, tensorDesc,
                                            graph);

    auto op = ge::op::InplaceApplyAdagradV2("inplace_apply_adagrad_v2_static");
    op.set_input_var(var);
    op.set_input_accum(accum);
    op.set_input_lr(lr);
    op.set_input_grad(grad);
    op.set_attr_epsilon(kEpsilon);
    op.set_attr_update_slots(true);
    op.set_attr_use_locking(false);
    op.update_input_desc_var(tensorDesc);
    op.update_input_desc_accum(tensorDesc);
    op.update_input_desc_lr(scalarDesc);
    op.update_input_desc_grad(tensorDesc);
    op.update_output_desc_var(tensorDesc);
    op.update_output_desc_accum(tensorDesc);
    op.AddControlInput(assignVar);
    op.AddControlInput(assignAccum);
    graph.AddOp(op);

    graphOutputs.push_back(op);
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);
}

StaticInputs MakeInputs()
{
    return {{2, 4},
            {1.0F, 1.5F, 2.0F, -1.0F, -0.5F, 0.0F, 3.0F, 4.0F},
            {0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 1.75F, 2.0F, 2.25F},
            0.1F,
            {0.25F, -0.5F, 0.75F, -1.0F, 1.25F, -1.5F, 1.75F, -2.0F}};
}

std::vector<ge::Tensor> MakeFeeds(const StaticInputs& inputs)
{
    return {MakeFloatTensor(inputs.shape, inputs.var), MakeFloatTensor(inputs.shape, inputs.accum),
            MakeFloatScalar(inputs.lr), MakeFloatTensor(inputs.shape, inputs.grad)};
}

ExpectedOutputs ComputeGolden(const StaticInputs& inputs)
{
    ExpectedOutputs expected;
    for (std::vector<float>& output : expected.values) {
        output.resize(inputs.var.size());
    }

    for (size_t index = 0; index < inputs.var.size(); ++index) {
        const float accum = inputs.accum[index] + inputs.grad[index] * inputs.grad[index];
        const float var = inputs.var[index] - inputs.lr * inputs.grad[index] / (std::sqrt(accum) + kEpsilon);
        expected.values[0][index] = var;
        expected.values[1][index] = accum;
    }
    return expected;
}

bool VerifyOutputs(const std::vector<ge::Tensor>& outputs, const ShapeVector& expectedShape,
                   const ExpectedOutputs& expected)
{
    if (outputs.size() != kOutputCount) {
        std::cerr << "Expected " << kOutputCount << " outputs, got " << outputs.size() << std::endl;
        return false;
    }

    const int64_t count = ElementCount(expectedShape);
    const size_t expectedBytes = static_cast<size_t>(count) * sizeof(float);
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        const ge::Tensor& output = outputs[outputIndex];
        const ge::TensorDesc desc = output.GetTensorDesc();
        if (desc.GetShape().GetDims() != expectedShape) {
            std::cerr << "Output " << outputIndex << " shape mismatch: actual "
                      << ShapeToString(desc.GetShape().GetDims()) << ", expected " << ShapeToString(expectedShape)
                      << std::endl;
            return false;
        }
        if (desc.GetDataType() != ge::DT_FLOAT) {
            std::cerr << "Output " << outputIndex << " dtype mismatch: actual " << desc.GetDataType() << ", expected "
                      << ge::DT_FLOAT << std::endl;
            return false;
        }

        const auto* data = reinterpret_cast<const float*>(output.GetData());
        if (data == nullptr || output.GetSize() != expectedBytes) {
            std::cerr << "Output " << outputIndex << " buffer mismatch: actual bytes " << output.GetSize()
                      << ", expected bytes " << expectedBytes << std::endl;
            return false;
        }
        for (int64_t elementIndex = 0; elementIndex < count; ++elementIndex) {
            const float expectedValue = expected.values[outputIndex][static_cast<size_t>(elementIndex)];
            const float actualValue = data[elementIndex];
            const float tolerance = kAtol + kRtol * std::fabs(expectedValue);
            if (!std::isfinite(actualValue) || std::fabs(actualValue - expectedValue) > tolerance) {
                std::cerr << "Output " << outputIndex << " value mismatch at element " << elementIndex << ": actual "
                          << actualValue << ", expected " << expectedValue << ", tolerance " << tolerance << std::endl;
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main()
{
    const StaticInputs inputs = MakeInputs();
    const std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"},
                                                                        {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "ERROR: GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }

    ge::Graph graph("inplace_apply_adagrad_v2_static_graph");
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    BuildGraph(inputs.shape, graph, graphInputs, graphOutputs);

    bool verified = false;
    {
        const std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        const std::map<ge::AscendString, ge::AscendString> graphOptions;
        if (session.AddGraph(0U, graph, graphOptions) != ge::SUCCESS) {
            std::cerr << "ERROR: AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        } else {
            const std::vector<ge::Tensor> feeds = MakeFeeds(inputs);
            std::vector<ge::Tensor> outputs;
            if (session.RunGraph(0U, feeds, outputs) != ge::SUCCESS) {
                std::cerr << "ERROR: RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
            } else {
                verified = VerifyOutputs(outputs, inputs.shape, ComputeGolden(inputs));
            }
        }
    }

    const ge::Status finalizeStatus = ge::GEFinalize();
    if (finalizeStatus != ge::SUCCESS) {
        std::cerr << "ERROR: GEFinalize failed" << std::endl;
        return kFailed;
    }
    if (!verified) {
        std::cerr << "ERROR: InplaceApplyAdagradV2 static output verification failed" << std::endl;
        return kFailed;
    }

    std::cout << "Shape, dtype and values PASSED for " << ShapeToString(inputs.shape) << std::endl;
    std::cout << "InplaceApplyAdagradV2 static GEIR verification PASSED" << std::endl;
    return kSuccess;
}
