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
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "ge_api.h"
#include "ge_error_codes.h"
#include "graph.h"
#include "ops_proto_nn.h"
#include "securec.h"
#include "tensor.h"
#include "types.h"

// Data is not included in ops_proto_nn.h.
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

namespace {
bool CheckOutputs(const std::vector<ge::Tensor>& outputs)
{
    constexpr float expected[2][2] = {{12.0F, 24.0F}, {12.0F, 48.0F}};
    const char* names[] = {"sum", "square_sum"};
    const std::vector<int64_t> outputShape = {2};
    if (outputs.size() != 2) {
        std::printf("Expected two outputs, got %zu\n", outputs.size());
        return false;
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto desc = outputs[i].GetTensorDesc();
        if (desc.GetDataType() != ge::DT_FLOAT || desc.GetShape().GetDims() != outputShape ||
            outputs[i].GetData() == nullptr || outputs[i].GetSize() != sizeof(expected[i])) {
            std::printf("Unexpected descriptor or buffer size for %s\n", names[i]);
            return false;
        }
        float values[2];
        if (memcpy_s(values, sizeof(values), outputs[i].GetData(), sizeof(values)) != EOK) {
            return false;
        }
        for (size_t c = 0; c < 2; ++c) {
            std::printf("%s[%zu] = %g\n", names[i], c, values[c]);
            // These small integer sums are exactly representable in float32.
            if (values[c] != expected[i][c]) {
                std::printf("Expected %g\n", expected[i][c]);
                return false;
            }
        }
    }
    return true;
}

int RunExample()
{
    const std::vector<int64_t> xShape = {1, 2, 3, 4};
    // NCHW: channel 0 contains twelve ones; channel 1 contains twelve twos.
    const std::vector<float> xData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    ge::TensorDesc xDesc(ge::Shape(xShape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    xDesc.SetRealDimCnt(xShape.size());
    xDesc.SetPlacement(ge::kPlacementHost);
    ge::Tensor input;
    if (input.SetTensorDesc(xDesc) != ge::GRAPH_SUCCESS ||
        input.SetData(reinterpret_cast<const uint8_t*>(xData.data()), xData.size() * sizeof(float)) !=
            ge::GRAPH_SUCCESS) {
        std::printf("Prepare input tensor failed\n");
        return 1;
    }

    ge::Graph graph("bn_training_reduce_example");
    auto data = ge::op::Data("x").set_attr_index(0);
    auto reduce = ge::op::BNTrainingReduce("bn_training_reduce");
    reduce.set_input_x(data);
    const ge::TensorDesc outDesc(ge::Shape(std::vector<int64_t>{2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
    if (data.update_input_desc_x(xDesc) != ge::GRAPH_SUCCESS || data.update_output_desc_y(xDesc) != ge::GRAPH_SUCCESS ||
        reduce.update_input_desc_x(xDesc) != ge::GRAPH_SUCCESS ||
        reduce.update_output_desc_sum(outDesc) != ge::GRAPH_SUCCESS ||
        reduce.update_output_desc_square_sum(outDesc) != ge::GRAPH_SUCCESS || graph.AddOp(data) != ge::GRAPH_SUCCESS ||
        graph.AddOp(reduce) != ge::GRAPH_SUCCESS) {
        std::printf("Prepare graph failed\n");
        return 1;
    }
    const std::vector<ge::Operator> inputs = {data};
    const std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs = {{reduce, {0U, 1U}}};
    graph.SetInputs(inputs).SetOutputs(outputs);

    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    const uint32_t graphId = 0;
    auto status = session.AddGraph(graphId, graph, options);
    if (status != ge::SUCCESS) {
        std::printf("AddGraph failed: %u\n", status);
        return 1;
    }
    std::vector<ge::Tensor> results;
    status = session.RunGraph(graphId, {input}, results);
    if (status != ge::SUCCESS) {
        std::printf("RunGraph failed: %u\n", status);
        return 1;
    }
    const bool matches = CheckOutputs(results);
    status = session.RemoveGraph(graphId);
    if (status != ge::SUCCESS) {
        std::printf("RemoveGraph failed: %u\n", status);
        return 1;
    }
    return matches ? 0 : 1;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        std::printf("GEInitialize failed: %u\n", status);
        return 1;
    }
    const int result = RunExample();
    status = ge::GEFinalize();
    if (status != ge::SUCCESS) {
        std::printf("GEFinalize failed: %u\n", status);
        return 1;
    }
    return result;
}
