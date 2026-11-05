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

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/map_index_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
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

ge::op::Data CreateDataOp(const std::string& name, int32_t index, const ge::Tensor& tensor)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    ge::TensorDesc desc = tensor.GetTensorDesc();
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    return data;
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    // x: the queried shape, a 1D tensor of length rank.
    ge::Tensor xTensor = BuildTensor<int32_t>({2}, ge::DT_INT32, {3, 5});
    // data_seq: the mapped table, length must be a multiple of x length (rank) and at most 256 * rank.
    // Reshaped to [N, rank] = [[3, 5], [3, 7]], so x matches row 0.
    ge::Tensor dataSeqTensor = BuildTensor<int32_t>({4}, ge::DT_INT32, {3, 5, 3, 7});
    // level_index: optional secondary index, length equals data_seq length / x length (N).
    ge::Tensor levelIndexTensor = BuildTensor<int32_t>({2}, ge::DT_INT32, {0, 1});

    auto x = CreateDataOp("x", 0, xTensor);
    auto dataSeq = CreateDataOp("data_seq", 1, dataSeqTensor);
    auto levelIndex = CreateDataOp("level_index", 2, levelIndexTensor);
    graph.AddOp(x);
    graph.AddOp(dataSeq);
    graph.AddOp(levelIndex);

    auto op = ge::op::MapIndex("map_index");
    op.set_input_x(x);
    op.set_input_data_seq(dataSeq);
    op.set_input_level_index(levelIndex);
    op.set_attr_transpose(false);
    // y is a 0-dim scalar of int32, inferred by the registered InferShape of MapIndex.
    op.update_output_desc_y(ge::TensorDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32));

    inputs = {xTensor, dataSeqTensor, levelIndexTensor};
    graphInputs = {x, dataSeq, levelIndex};
    graphOutputs = {op};
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto& tensor = outputs[0];
    if (tensor.GetTensorDesc().GetShape().GetDims().size() != 0) {
        std::cerr << "Unexpected output shape: not a scalar" << std::endl;
        return kFailed;
    }

    const auto* data = reinterpret_cast<const int32_t*>(tensor.GetData());
    constexpr int32_t expected = 0; // x matches the first row of data_seq.
    std::cout << "map_index output[0] = " << data[0] << std::endl;
    if (data[0] != expected) {
        std::cerr << "Unexpected map_index result: " << data[0] << ", expected " << expected << std::endl;
        return kFailed;
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start MapIndex GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("map_index_graph");
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
    std::cout << GetTime() << " - INFO - MapIndex GEIR example success" << std::endl;
    return kSuccess;
}
