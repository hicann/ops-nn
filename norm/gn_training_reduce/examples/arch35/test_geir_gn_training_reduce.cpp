/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_gn_training_reduce.cpp
 * \brief Minimal graph (GEIR) invocation example for GNTrainingReduce.
 */

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/gn_training_reduce_proto.h"

// 示例统一走工程专用日志接口（printf 只允许出现在该接口实现内部）。
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr uint32_t kDeviceId = 0U;
constexpr int64_t kNumGroups = 2;

ge::Tensor BuildInput(const std::vector<int64_t>& shape, const std::vector<float>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(static_cast<int64_t>(shape.size()));
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}
} // namespace

int main()
{
    const std::vector<int64_t> xShape = {1, 4, 2, 2};
    const std::vector<int64_t> outShape = {1, kNumGroups, 1, 1, 1};
    const ge::Tensor input = BuildInput(xShape, std::vector<float>(16, 1.0F));
    const ge::TensorDesc inDesc = input.GetTensorDesc();
    const ge::TensorDesc outDesc(ge::Shape(outShape), ge::FORMAT_ND, ge::DT_FLOAT);

    auto data = ge::op::Data("gn_training_reduce_data").set_attr_index(0);
    data.update_input_desc_x(inDesc);
    data.update_output_desc_y(inDesc);

    auto reduce = ge::op::GNTrainingReduce("gn_training_reduce");
    reduce.set_input_x(data).set_attr_num_groups(kNumGroups);
    reduce.update_input_desc_x(inDesc);
    reduce.update_output_desc_sum(outDesc);
    reduce.update_output_desc_square_sum(outDesc);

    ge::Graph graph("gn_training_reduce_graph");
    graph.AddOp(data);
    graph.AddOp(reduce);
    const std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs = {{reduce, {0U, 1U}}};
    graph.SetInputs({data}).SetOutputs(outputs);

    const std::map<ge::AscendString, ge::AscendString> options = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        LOG_PRINT("ERROR: GEInitialize failed.\n");
        return 1;
    }

    ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
    if (session.AddGraph(0, graph, std::map<ge::AscendString, ge::AscendString>{}) != ge::SUCCESS) {
        LOG_PRINT("ERROR: AddGraph failed.\n");
        ge::GEFinalize();
        return 1;
    }

    std::vector<ge::Tensor> runOutputs;
    const ge::Status status = session.RunGraph(0, {input}, runOutputs);
    ge::GEFinalize();
    if (status != ge::SUCCESS) {
        LOG_PRINT("ERROR: RunGraph failed.\n");
        return 1;
    }
    if (runOutputs.size() != 2U) {
        LOG_PRINT("ERROR: unexpected output count: %zu\n", runOutputs.size());
        return 1;
    }

    LOG_PRINT("GNTrainingReduce graph mode passed, outputs = %zu\n", runOutputs.size());
    return 0;
}
