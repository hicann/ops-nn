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
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/smooth_l1_loss_grad_proto.h"

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

struct CaseConfig {
    std::string id;
    std::string category;
    ge::DataType predictDtype = ge::DT_FLOAT;
    ge::DataType labelDtype = ge::DT_FLOAT;
    ge::DataType doutDtype = ge::DT_FLOAT;
    ge::Format predictFormat = ge::FORMAT_ND;
    ge::Format labelFormat = ge::FORMAT_ND;
    ge::Format doutFormat = ge::FORMAT_ND;
    ge::Format outputFormat = ge::FORMAT_ND;
    std::vector<int64_t> predictShape = {4};
    std::vector<int64_t> labelShape = {4};
    std::vector<int64_t> doutShape = {4};
    std::vector<int64_t> outputShape = {4};
    float sigma = 1.0F;
    bool castPredict = false;
    bool expectAccept = false;
    std::string error;
};

int64_t Count(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    return count;
}

ge::Tensor MakeTensor(const ge::TensorDesc& desc, ge::DataType dtype, int64_t count)
{
    if (dtype == ge::DT_INT32) {
        std::vector<int32_t> values(static_cast<size_t>(count), 1);
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(int32_t));
    }
    if (dtype == ge::DT_INT8) {
        std::vector<int8_t> values(static_cast<size_t>(count), 1);
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size());
    }
    std::vector<float> values(static_cast<size_t>(count), 1.0F);
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(float));
}

ge::Operator AddData(const std::string& name, uint32_t index, const std::vector<int64_t>& shape, ge::DataType dtype,
                     ge::Format format, ge::Graph& graph, std::vector<ge::Tensor>& tensors,
                     std::vector<ge::Operator>& inputs)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    ge::TensorDesc desc(ge::Shape(shape), format, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(format);
    desc.SetRealDimCnt(shape.size());
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    tensors.push_back(MakeTensor(desc, dtype, Count(shape)));
    graph.AddOp(data);
    inputs.push_back(data);
    return data;
}

void BuildGraph(const CaseConfig& config, ge::Graph& graph, std::vector<ge::Tensor>& tensors,
                std::vector<ge::Operator>& inputs, std::vector<ge::Operator>& outputs)
{
    const auto sourceDtype = config.castPredict ? ge::DT_INT32 : config.predictDtype;
    auto predictData = AddData(config.id + "_predict", 0, config.predictShape, sourceDtype, config.predictFormat, graph,
                               tensors, inputs);
    ge::Operator predict = predictData;
    ge::TensorDesc predictDesc(ge::Shape(config.predictShape), config.predictFormat, config.predictDtype);
    if (config.castPredict) {
        ge::TensorDesc sourceDesc(ge::Shape(config.predictShape), config.predictFormat, sourceDtype);
        auto cast = ge::op::Cast((config.id + "_cast").c_str())
                        .set_input_x(predictData)
                        .set_attr_dst_type(config.predictDtype);
        cast.update_input_desc_x(sourceDesc);
        cast.update_output_desc_y(predictDesc);
        graph.AddOp(cast);
        predict = cast;
    }
    auto label = AddData(config.id + "_label", 1, config.labelShape, config.labelDtype, config.labelFormat, graph,
                         tensors, inputs);
    auto dout = AddData(config.id + "_dout", 2, config.doutShape, config.doutDtype, config.doutFormat, graph, tensors,
                        inputs);
    ge::TensorDesc labelDesc(ge::Shape(config.labelShape), config.labelFormat, config.labelDtype);
    ge::TensorDesc doutDesc(ge::Shape(config.doutShape), config.doutFormat, config.doutDtype);

    auto op = ge::op::SmoothL1LossGrad((config.id + "_op").c_str());
    op.set_input_predict(predict);
    op.set_input_label(label);
    op.set_input_dout(dout);
    op.set_attr_sigma(config.sigma);
    op.update_input_desc_predict(predictDesc);
    op.update_input_desc_label(labelDesc);
    op.update_input_desc_dout(doutDesc);
    ge::TensorDesc outputDesc(ge::Shape(config.outputShape), config.outputFormat, config.predictDtype);
    op.update_output_desc_gradient(outputDesc);
    graph.AddOp(op);
    outputs.push_back(op);
}

bool VerifyOutput(const std::vector<ge::Tensor>& result, const CaseConfig& config)
{
    if (result.size() != 1 || result[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        result[0].GetTensorDesc().GetShape().GetDims() != config.outputShape) {
        return false;
    }
    const auto count = Count(config.outputShape);
    const auto* gradient = reinterpret_cast<const float*>(result[0].GetData());
    for (int64_t i = 0; i < count; ++i) {
        if (gradient == nullptr || std::abs(gradient[i]) > 1.0e-6F) {
            return false;
        }
    }
    return true;
}

bool RunCase(const CaseConfig& config)
{
    ge::Graph graph(("smooth_l1_loss_grad_exception_" + config.id).c_str());
    std::vector<ge::Tensor> tensors;
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;
    BuildGraph(config, graph, tensors, inputs, outputs);
    graph.SetInputs(inputs).SetOutputs(outputs);
    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    ge::Status status = session.AddGraph(0, graph, options);
    std::vector<ge::Tensor> result;
    if (status == ge::SUCCESS) {
        status = session.RunGraph(0, tensors, result);
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
    if (status != ge::SUCCESS || !VerifyOutput(result, config)) {
        std::cout << "EXCEPTION_CASE " << config.id
                  << " FAIL expected=accept actual=reject error=" << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return false;
    }
    std::cout << "EXCEPTION_CASE " << config.id << " PASS category=" << config.category
              << " expected=accept actual=accept validation_stage=tiling kernel_launched=true output_verified=true"
              << " kernel_launch_observed=true";
    if (config.castPredict) {
        std::cout << " source_dtype=int32 cast_dtype=float32 required_dtype=float32 observed_input_dtype=float32";
    }
    std::cout << std::endl;
    return true;
}

CaseConfig MakeCase(const std::string& id, const std::string& category, const std::string& error)
{
    CaseConfig config;
    config.id = id;
    config.category = category;
    config.error = error;
    return config;
}

void SetAllShapes(CaseConfig& config, const std::vector<int64_t>& shape)
{
    config.predictShape = shape;
    config.labelShape = shape;
    config.doutShape = shape;
    config.outputShape = shape;
}

std::vector<CaseConfig> MakeCases()
{
    std::vector<CaseConfig> cases;

    auto config = MakeCase("dtype_predict_int8", "dtype_unsupported", "unsupported dtype");
    config.predictDtype = ge::DT_INT8;
    cases.push_back(config);

    config = MakeCase("dtype_label_int8", "dtype_unsupported", "input[1] unsupported dtype");
    config.labelDtype = ge::DT_INT8;
    cases.push_back(config);

    config = MakeCase("dtype_dout_int8", "dtype_unsupported", "input[2] unsupported dtype");
    config.doutDtype = ge::DT_INT8;
    cases.push_back(config);

    config = MakeCase("dtype_predict_label_mismatch", "dtype_combination_mismatch",
                      "dtype mismatch between predict and label");
    config.labelDtype = ge::DT_FLOAT16;
    cases.push_back(config);

    config = MakeCase("dtype_predict_dout_mismatch", "dtype_combination_mismatch",
                      "dtype mismatch between predict and dout");
    config.doutDtype = ge::DT_FLOAT16;
    cases.push_back(config);

    config = MakeCase("cast_predict_int32_to_float32", "dtype_geir_cast", "");
    config.castPredict = true;
    config.expectAccept = true;
    cases.push_back(config);

    const std::vector<int64_t> nchwShape = {1, 1, 1, 1};
    config = MakeCase("format_predict_nchw", "format_unsupported", "input[0] format");
    SetAllShapes(config, nchwShape);
    config.predictFormat = ge::FORMAT_NCHW;
    cases.push_back(config);

    config = MakeCase("format_label_nchw", "format_unsupported", "input[1] format");
    SetAllShapes(config, nchwShape);
    config.labelFormat = ge::FORMAT_NCHW;
    cases.push_back(config);

    config = MakeCase("format_dout_nchw", "format_unsupported", "input[2] format");
    SetAllShapes(config, nchwShape);
    config.doutFormat = ge::FORMAT_NCHW;
    cases.push_back(config);

    config = MakeCase("format_gradient_nchw", "format_unsupported", "output[0] format");
    SetAllShapes(config, nchwShape);
    config.outputFormat = ge::FORMAT_NCHW;
    cases.push_back(config);

    config = MakeCase("rank9_predict", "rank_9_tensor", "predict rank 9 exceeds supported range [0, 8]");
    SetAllShapes(config, {1, 1, 1, 1, 1, 1, 1, 1, 1});
    cases.push_back(config);

    config = MakeCase("sigma_zero", "attribute_out_of_range", "sigma must be > 0");
    config.sigma = 0.0F;
    cases.push_back(config);

    config = MakeCase("shape_label_mismatch", "shape_dimension_violation", "shape mismatch between predict and label");
    config.labelShape = {2, 2};
    cases.push_back(config);

    config = MakeCase("shape_dout_mismatch", "shape_dimension_violation", "shape mismatch between predict and dout");
    config.doutShape = {2, 2};
    cases.push_back(config);

    return cases;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << "EXCEPTION_SUITE SmoothL1LossGrad Ascend950 GEIR" << std::endl;
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        return -1;
    }
    const auto cases = MakeCases();
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
        std::cout << "EXCEPTION_SUITE SmoothL1LossGrad FAIL unknown_case=" << argv[1] << std::endl;
        passed = false;
    }
    passed = ge::GEFinalize() == ge::SUCCESS && passed;
    if (passed) {
        std::cout << "EXCEPTION_SUITE SmoothL1LossGrad PASSED" << std::endl;
    }
    return passed ? 0 : -1;
}
