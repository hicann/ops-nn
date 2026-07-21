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

#include "../op_graph/index_to_addr_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr int64_t kExpectedOutputElementCount = 16;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

ge::Tensor BuildInt64Tensor(const std::vector<int64_t>& shape, const std::vector<int64_t>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT64);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int64_t));
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    auto baseAddrData = ge::op::Data("base_addr").set_attr_index(0);
    const std::vector<int64_t> baseAddrShape = {2};
    const std::vector<int64_t> baseAddrValues = {20, 40};
    ge::Tensor baseAddrTensor = BuildInt64Tensor(baseAddrShape, baseAddrValues);
    ge::TensorDesc baseAddrDesc = baseAddrTensor.GetTensorDesc();
    baseAddrData.update_input_desc_x(baseAddrDesc);
    baseAddrData.update_output_desc_y(baseAddrDesc);
    graph.AddOp(baseAddrData);

    auto indexData = ge::op::Data("x").set_attr_index(1);
    const std::vector<int64_t> indexShape = {2};
    const std::vector<int64_t> indexValues = {0, 3};
    ge::Tensor indexTensor = BuildInt64Tensor(indexShape, indexValues);
    ge::TensorDesc indexDesc = indexTensor.GetTensorDesc();
    indexData.update_input_desc_x(indexDesc);
    indexData.update_output_desc_y(indexDesc);
    graph.AddOp(indexData);

    auto indexToAddr = ge::op::IndexToAddr("index_to_addr");
    indexToAddr.set_input_base_addr(baseAddrData);
    indexToAddr.set_input_x(indexData);
    indexToAddr.set_attr_ori_shape({16, 16});
    indexToAddr.set_attr_block_size({4, 4});
    indexToAddr.set_attr_ori_storage_mode("Matrix");
    indexToAddr.set_attr_block_storage_mode("Matrix");
    indexToAddr.set_attr_rank_id(0);
    indexToAddr.set_attr_dtype(ge::DT_FLOAT);

    inputs.push_back(baseAddrTensor);
    inputs.push_back(indexTensor);
    graphInputs.push_back(baseAddrData);
    graphInputs.push_back(indexData);
    graphOutputs.push_back(indexToAddr);
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto& tensor = outputs[0];
    const int64_t elemCount = tensor.GetTensorDesc().GetShape().GetShapeSize();
    if (elemCount != kExpectedOutputElementCount) {
        std::cerr << "Unexpected output element count: " << elemCount << std::endl;
        return kFailed;
    }

    const auto* data = reinterpret_cast<const int64_t*>(tensor.GetData());
    const int64_t expected[] = {0, 68, 88, 16, 0, 132, 152, 16, 0, 196, 216, 16, 0, 260, 280, 16};
    for (int64_t i = 0; i < elemCount; ++i) {
        std::cout << "index_to_addr output[" << i << "] = " << data[i] << std::endl;
        if (data[i] != expected[i]) {
            std::cerr << "Unexpected index_to_addr result at index " << i << ": " << data[i] << ", expected "
                      << expected[i] << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start IndexToAddr GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("index_to_addr_graph");
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
    std::cout << GetTime() << " - INFO - IndexToAddr GEIR example success" << std::endl;
    return kSuccess;
}
