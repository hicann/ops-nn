/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
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

#include "../op_graph/sorted_sparse_segment_mean_grad_proto.h"

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
    // x 为上游 SparseSegmentMean 的梯度，按 segment 存放，shape 为 [segmentNum, innerSize]
    ge::Tensor xTensor = BuildTensor<float>({2, 1}, ge::DT_FLOAT, {2.0F, 4.0F});
    // 已排序的下标，值域为 [0, output_dim0)
    ge::Tensor sortedIndicesTensor = BuildTensor<int32_t>({3}, ge::DT_INT32, {0, 1, 2});
    // 排序前的原始位置，长度与 sorted_indices 相同
    ge::Tensor locationTensor = BuildTensor<int32_t>({3}, ge::DT_INT32, {0, 1, 2});
    // 每个元素归属的 segment，segment 0 有 2 个元素，segment 1 有 1 个元素
    ge::Tensor segmentIdsTensor = BuildTensor<int32_t>({3}, ge::DT_INT32, {0, 0, 1});
    // 上游张量的第 0 维大小
    ge::Tensor outputDim0Tensor = BuildTensor<int32_t>({1}, ge::DT_INT32, {3});

    auto x = CreateDataOp("x", 0, xTensor);
    auto sortedIndices = CreateDataOp("sorted_indices", 1, sortedIndicesTensor);
    auto location = CreateDataOp("pre_location_indices", 2, locationTensor);
    auto segmentIds = CreateDataOp("segment_ids", 3, segmentIdsTensor);
    auto outputDim0 = CreateDataOp("output_dim0", 4, outputDim0Tensor);
    graph.AddOp(x);
    graph.AddOp(sortedIndices);
    graph.AddOp(location);
    graph.AddOp(segmentIds);
    graph.AddOp(outputDim0);

    auto op = ge::op::SortedSparseSegmentMeanGrad("sorted_sparse_segment_mean_grad");
    op.set_input_x(x);
    op.set_input_sorted_indices(sortedIndices);
    op.set_input_pre_location_indices(location);
    op.set_input_segment_ids(segmentIds);
    op.set_input_output_dim0(outputDim0);
    op.update_output_desc_y(ge::TensorDesc(ge::Shape({3, 1}), ge::FORMAT_ND, ge::DT_FLOAT));

    inputs = {xTensor, sortedIndicesTensor, locationTensor, segmentIdsTensor, outputDim0Tensor};
    graphInputs = {x, sortedIndices, location, segmentIds, outputDim0};
    graphOutputs = {op};
    return kSuccess;
}

bool CheckFloatArray(const float* data, const std::vector<float>& expected)
{
    constexpr float kTolerance = 1e-5F;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(data[i] - expected[i]) > kTolerance) {
            std::cerr << "Unexpected output at index " << i << ": " << data[i] << ", expected " << expected[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto* y = reinterpret_cast<const float*>(outputs[0].GetData());
    // segment 0 有 2 个元素权重为 1/2，segment 1 有 1 个元素权重为 1：
    // y[0] = x[0] * 1/2, y[1] = x[0] * 1/2, y[2] = x[1] * 1
    if (!CheckFloatArray(y, {1.0F, 1.0F, 4.0F})) {
        return kFailed;
    }
    return kSuccess;
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start SortedSparseSegmentMeanGrad GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("sorted_sparse_segment_mean_grad_graph");
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
    std::cout << GetTime() << " - INFO - SortedSparseSegmentMeanGrad GEIR example success" << std::endl;
    return kSuccess;
}
