/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "graph/operator_reg.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/non_zero_with_value_shape_proto.h"

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr int64_t kUnknownDim = -1;
constexpr uint32_t kDeviceId = 0;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

template <typename T>
ge::Tensor BuildTensor(const std::vector<int64_t>& shape, ge::DataType dtype, const std::vector<T>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

bool HasUnknownDim(const std::vector<int64_t>& shape)
{
    for (const auto dim : shape) {
        if (dim == kUnknownDim) {
            return true;
        }
    }
    return false;
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    auto valueData = ge::op::Data("value").set_attr_index(0);
    ge::Tensor valueTensor = BuildTensor<float>({4}, ge::DT_FLOAT, {1.0F, 2.0F, 3.0F, 4.0F});
    ge::TensorDesc valueDesc = valueTensor.GetTensorDesc();
    valueData.update_input_desc_x(valueDesc);
    valueData.update_output_desc_y(valueDesc);
    graph.AddOp(valueData);

    auto indexData = ge::op::Data("index").set_attr_index(1);
    ge::Tensor indexTensor = BuildTensor<int32_t>({6}, ge::DT_INT32, {0, 1, 2, 0, 1, 2});
    ge::TensorDesc indexDesc = indexTensor.GetTensorDesc();
    indexData.update_input_desc_x(indexDesc);
    indexData.update_output_desc_y(indexDesc);
    graph.AddOp(indexData);

    auto countData = ge::op::Data("count").set_attr_index(2);
    ge::Tensor countTensor = BuildTensor<int32_t>({1}, ge::DT_INT32, {3});
    ge::TensorDesc countDesc = countTensor.GetTensorDesc();
    countData.update_input_desc_x(countDesc);
    countData.update_output_desc_y(countDesc);
    graph.AddOp(countData);

    auto nonZeroWithValueShape = ge::op::NonZeroWithValueShape("non_zero_with_value_shape");
    nonZeroWithValueShape.set_input_value(valueData);
    nonZeroWithValueShape.set_input_index(indexData);
    nonZeroWithValueShape.set_input_count(countData);

    inputs.push_back(valueTensor);
    inputs.push_back(indexTensor);
    inputs.push_back(countTensor);
    graphInputs.push_back(valueData);
    graphInputs.push_back(indexData);
    graphInputs.push_back(countData);
    graphOutputs.push_back(nonZeroWithValueShape);
    return kSuccess;
}

int CheckShape(const ge::Tensor& tensor, const std::vector<int64_t>& expectedShape)
{
    std::vector<int64_t> shape = tensor.GetTensorDesc().GetShape().GetDims();
    while (shape.size() > expectedShape.size() && shape.back() == 0) {
        shape.pop_back();
    }
    if (shape != expectedShape && !HasUnknownDim(shape)) {
        std::cerr << "Unexpected output shape, got [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cerr << shape[i] << (i + 1 == shape.size() ? "" : ", ");
        }
        std::cerr << "], expected [";
        for (size_t i = 0; i < expectedShape.size(); ++i) {
            std::cerr << expectedShape[i] << (i + 1 == expectedShape.size() ? "" : ", ");
        }
        std::cerr << "]" << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 2U) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }
    if (CheckShape(outputs[0], {3}) != kSuccess) {
        return kFailed;
    }
    return CheckShape(outputs[1], {2, 3});
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start NonZeroWithValueShape GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("non_zero_with_value_shape_graph");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    if (CreateGraph(inputs, graphInputs, graphOutputs, graph) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed" << std::endl;
        ge::GEFinalize();
        return kFailed;
    }

    const int ret = CheckOutput(outputs);
    ge::GEFinalize();
    if (ret != kSuccess) {
        return kFailed;
    }
    std::cout << GetTime() << " - INFO - NonZeroWithValueShape GEIR example success" << std::endl;
    return kSuccess;
}
