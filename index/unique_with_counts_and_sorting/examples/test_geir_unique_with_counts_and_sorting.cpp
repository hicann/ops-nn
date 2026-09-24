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
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "ge/ge_api.h"
#include "../op_graph/unique_with_counts_and_sorting_proto.h"

// Avoid including array_ops.h, which may contain an older, unguarded copy of the Unique prototype.
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge

namespace {
constexpr uint32_t VALUES_GRAPH_ID = 0;
constexpr uint32_t COUNTS_GRAPH_ID = 1;
constexpr size_t VALUES_OUTPUT_COUNT = 1;
constexpr size_t ALL_OUTPUT_COUNT = 3;

template <typename T>
bool CheckTensor(const ge::Tensor& tensor, ge::DataType dtype, const std::vector<T>& expected)
{
    if (tensor.GetTensorDesc().GetDataType() != dtype ||
        tensor.GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>{static_cast<int64_t>(expected.size())} ||
        tensor.GetData() == nullptr || tensor.GetSize() < expected.size() * sizeof(T)) {
        std::cerr << "Unexpected output dtype, shape or buffer size" << std::endl;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        T actual;
        std::memcpy(&actual, tensor.GetData() + i * sizeof(T), sizeof(T));
        if (actual != expected[i]) {
            std::cerr << "Unexpected output at index " << i << ": " << actual << ", expected " << expected[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

bool RunCase(ge::Session& session, uint32_t graphId, bool withInverseAndCounts)
{
    const std::vector<float> values = {3.0F, 1.0F, 3.0F, 2.0F};
    ge::TensorDesc desc(ge::Shape({static_cast<int64_t>(values.size())}), ge::FORMAT_ND, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(1);
    ge::Tensor input(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
    auto data = ge::op::Data("x").set_attr_index(0);
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    auto unique = ge::op::UniqueWithCountsAndSorting("unique")
                      .set_input_x(data)
                      .set_attr_sorted(true)
                      .set_attr_return_inverse(withInverseAndCounts)
                      .set_attr_return_counts(withInverseAndCounts)
                      .set_attr_out_idx(ge::DT_INT64);
    ge::Graph graph(withInverseAndCounts ? "unique_with_inverse_counts" : "unique_values_only");
    std::vector<ge::Operator> graphInputs{data};
    std::vector<std::pair<ge::Operator, std::vector<size_t>>> graphOutputs{
        {unique, withInverseAndCounts ? std::vector<size_t>{0, 1, 2} : std::vector<size_t>{0}}};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);
    if (session.AddGraph(graphId, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed for graph " << graphId << std::endl;
        return false;
    }
    std::vector<ge::Tensor> inputs{input};
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graphId, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed for graph " << graphId << std::endl;
        return false;
    }
    const size_t expectedOutputs = withInverseAndCounts ? ALL_OUTPUT_COUNT : VALUES_OUTPUT_COUNT;
    if (outputs.size() != expectedOutputs || !CheckTensor<float>(outputs[0], ge::DT_FLOAT, {1.0F, 2.0F, 3.0F})) {
        return false;
    }
    if (withInverseAndCounts && (!CheckTensor<int64_t>(outputs[1], ge::DT_INT64, {2, 0, 2, 1}) ||
                                 !CheckTensor<int64_t>(outputs[2], ge::DT_INT64, {1, 1, 2}))) {
        return false;
    }
    std::cout << (withInverseAndCounts ? "values + inverse + counts" : "values only") << ": PASS" << std::endl;
    return true;
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> options{{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return 1;
    }
    bool passed = false;
    {
        // Destroy the session before GEFinalize.
        std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        const bool valuesPassed = RunCase(session, VALUES_GRAPH_ID, false);
        const bool countsPassed = RunCase(session, COUNTS_GRAPH_ID, true);
        passed = valuesPassed && countsPassed;
    }
    if (ge::GEFinalize() != ge::SUCCESS) {
        std::cerr << "GEFinalize failed" << std::endl;
        return 1;
    }
    return passed ? 0 : 1;
}
