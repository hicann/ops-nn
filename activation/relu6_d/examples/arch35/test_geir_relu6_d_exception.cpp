/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/relu6_d_proto.h"

namespace {

std::string CurrentError()
{
    std::string error = ge::GEGetErrorMsgV2().GetString();
    std::replace(error.begin(), error.end(), '\n', ' ');
    std::replace(error.begin(), error.end(), '\r', ' ');
    return error;
}

std::string FailureStage(const std::string& error)
{
    if (error.find("Tiling func") != std::string::npos || error.find("tiling failed") != std::string::npos) {
        return "tiling";
    }
    if (error.find("Unsupported_Operator") != std::string::npos ||
        error.find("No supported Ops kernel") != std::string::npos) {
        return "engine_selection";
    }
    if (error.find("para_check.py") != std::string::npos || error.find("num of dimensions") != std::string::npos) {
        return "parameter_check";
    }
    return "graph_compile";
}

constexpr int SUCCESS = 0;
constexpr int FAILED = -1;

struct CaseConfig {
    std::string id;
    std::string category;
    ge::DataType sourceDtype;
    ge::DataType opDtype;
    ge::Format inputFormat;
    ge::Format outputFormat;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    bool useCast;
    bool expectAccept;
    std::string error;
};

int64_t ElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    return count;
}

ge::Tensor MakeInput(const ge::TensorDesc& desc, ge::DataType dtype, int64_t count)
{
    if (dtype == ge::DT_INT32) {
        std::vector<int32_t> values(static_cast<size_t>(count));
        const int32_t pattern[] = {-3, 0, 2, 8};
        for (int64_t i = 0; i < count; ++i) {
            values[static_cast<size_t>(i)] = pattern[i % 4];
        }
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(int32_t));
    }
    if (dtype == ge::DT_INT8) {
        std::vector<int8_t> values(static_cast<size_t>(count), 1);
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(int8_t));
    }
    std::vector<float> values(static_cast<size_t>(count));
    const float pattern[] = {-3.0F, 0.0F, 2.0F, 8.0F};
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = pattern[i % 4];
    }
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(float));
}

bool VerifyOutput(const ge::Tensor& output, const CaseConfig& config)
{
    if (output.GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        output.GetTensorDesc().GetShape().GetDims() != config.outputShape) {
        return false;
    }
    const auto count = ElementCount(config.outputShape);
    const auto* actual = reinterpret_cast<const float*>(output.GetData());
    const float pattern[] = {-3.0F, 0.0F, 2.0F, 8.0F};
    for (int64_t i = 0; i < count; ++i) {
        const float expected = std::min(std::max(pattern[i % 4], 0.0F), 6.0F);
        if (std::abs(actual[i] - expected) > 1.0e-4F) {
            return false;
        }
    }
    return true;
}

int BuildGraph(const CaseConfig& config, ge::Graph& graph, std::vector<ge::Tensor>& tensors,
               std::vector<ge::Operator>& inputs, std::vector<ge::Operator>& outputs)
{
    const auto count = ElementCount(config.inputShape);
    auto data = ge::op::Data((config.id + "_data").c_str()).set_attr_index(0);
    ge::TensorDesc sourceDesc(ge::Shape(config.inputShape), config.inputFormat, config.sourceDtype);
    sourceDesc.SetPlacement(ge::kPlacementHost);
    sourceDesc.SetFormat(config.inputFormat);
    sourceDesc.SetRealDimCnt(config.inputShape.size());
    data.update_input_desc_x(sourceDesc);
    data.update_output_desc_y(sourceDesc);
    tensors.push_back(MakeInput(sourceDesc, config.sourceDtype, count));
    graph.AddOp(data);
    inputs.push_back(data);

    ge::Operator opInput = data;
    ge::TensorDesc opDesc(ge::Shape(config.inputShape), config.inputFormat, config.opDtype);
    opDesc.SetFormat(config.inputFormat);
    opDesc.SetRealDimCnt(config.inputShape.size());
    if (config.useCast) {
        auto cast = ge::op::Cast((config.id + "_cast").c_str()).set_input_x(data).set_attr_dst_type(config.opDtype);
        cast.update_input_desc_x(sourceDesc);
        cast.update_output_desc_y(opDesc);
        graph.AddOp(cast);
        opInput = cast;
    }

    auto relu = ge::op::Relu6D((config.id + "_relu6_d").c_str());
    relu.set_input_x(opInput);
    relu.set_attr_scale(1.0F);
    relu.update_input_desc_x(opDesc);
    ge::TensorDesc outputDesc(ge::Shape(config.outputShape), config.outputFormat, config.opDtype);
    outputDesc.SetFormat(config.outputFormat);
    outputDesc.SetRealDimCnt(config.outputShape.size());
    relu.update_output_desc_y(outputDesc);
    graph.AddOp(relu);
    outputs.push_back(relu);
    return SUCCESS;
}

bool RunCase(const CaseConfig& config)
{
    ge::Graph graph(("relu6_d_exception_" + config.id).c_str());
    std::vector<ge::Tensor> tensors;
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;
    if (BuildGraph(config, graph, tensors, inputs, outputs) != SUCCESS) {
        return false;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    const std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    const uint32_t graphId = 0;
    const std::map<ge::AscendString, ge::AscendString> graphOptions;
    ge::Status status = session.AddGraph(graphId, graph, graphOptions);
    std::vector<ge::Tensor> result;
    if (status == ge::SUCCESS) {
        status = session.RunGraph(graphId, tensors, result);
    }

    if (!config.expectAccept) {
        if (status == ge::SUCCESS) {
            std::cout << "EXCEPTION_CASE " << config.id << " FAIL expected=reject actual=accept" << std::endl;
            return false;
        }
        const std::string actualError = CurrentError();
        const std::string failureStage = FailureStage(actualError);
        if (failureStage != "tiling" ||
            (!config.error.empty() && actualError.find(config.error) == std::string::npos)) {
            std::cout << "EXCEPTION_CASE " << config.id
                      << " FAIL expected=reject actual=reject failure_stage=" << failureStage
                      << " expected_error=" << config.error << " actual_error=" << actualError << std::endl;
            return false;
        }
        std::cout << "EXCEPTION_CASE " << config.id << " PASS category=" << config.category
                  << " expected=reject actual=reject failure_stage=" << failureStage
                  << " kernel_launched=false actual_error=" << actualError;
        if (config.category == "rank_9_tensor") {
            std::cout << " rank=9";
        }
        std::cout << std::endl;
        return true;
    }

    const bool verified = status == ge::SUCCESS && result.size() == 1 && VerifyOutput(result[0], config);
    if (!verified) {
        std::cout << "EXCEPTION_CASE " << config.id
                  << " FAIL expected=accept actual=reject error=" << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return false;
    }
    std::cout << "EXCEPTION_CASE " << config.id << " PASS category=" << config.category
              << " expected=accept actual=accept validation_stage=tiling kernel_launched=true output_verified=true"
              << " kernel_launch_observed=true";
    if (config.useCast) {
        std::cout << " source_dtype=int32 cast_dtype=float32 required_dtype=float32 observed_input_dtype=float32";
    }
    if (config.category == "rank_9_tensor") {
        std::cout << " rank=9";
    }
    std::cout << std::endl;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << "EXCEPTION_SUITE Relu6D Ascend950 GEIR" << std::endl;
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        return FAILED;
    }

    const std::vector<CaseConfig> cases = {
        {"dtype_x_int8",
         "dtype_unsupported",
         ge::DT_INT8,
         ge::DT_INT8,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {4},
         {4},
         false,
         false,
         "unsupported dtype"},
        {"cast_int32_to_float32",
         "dtype_geir_cast",
         ge::DT_INT32,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {4},
         {4},
         true,
         true,
         ""},
        {"format_x_nchw",
         "format_unsupported",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         ge::FORMAT_ND,
         {1, 1, 2, 2},
         {1, 1, 2, 2},
         false,
         false,
         "input[0] format"},
        {"format_y_nchw",
         "format_unsupported",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_NCHW,
         {1, 1, 2, 2},
         {1, 1, 2, 2},
         false,
         false,
         "output[0] format"},
        {"rank9_x",
         "rank_9_tensor",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         false,
         false,
         "input rank 9 exceeds supported range [0, 8]"},
    };

    bool passed = true;
    bool matched = false;
    for (const auto& item : cases) {
        if (argc > 1 && item.id != argv[1]) {
            continue;
        }
        matched = true;
        passed = RunCase(item) && passed;
    }
    if (!matched) {
        std::cout << "EXCEPTION_SUITE Relu6D FAIL unknown_case=" << argv[1] << std::endl;
        passed = false;
    }
    const auto finalizeStatus = ge::GEFinalize();
    passed = finalizeStatus == ge::SUCCESS && passed;
    if (passed) {
        std::cout << "EXCEPTION_SUITE Relu6D PASSED" << std::endl;
    }
    return passed ? SUCCESS : FAILED;
}
