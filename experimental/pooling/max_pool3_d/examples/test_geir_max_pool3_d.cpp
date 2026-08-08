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
#include <map>
#include <vector>

#include "ge_api.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"
#include "../op_graph/max_pool3_d_proto.h"

namespace {
constexpr int kSuccess = 0;
constexpr int kFailed = -1;

ge::Tensor MakeInput(const std::vector<int64_t>& shape)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_NDHWC, ge::DT_FLOAT);
    desc.SetPlacement(ge::kPlacementHost);
    std::vector<float> values(static_cast<size_t>(desc.GetShape().GetShapeSize()));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(i);
    }
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float));
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        return kFailed;
    }

    const std::vector<int64_t> xShape = {1, 4, 4, 4, 2};
    const std::vector<int64_t> yShape = {1, 2, 2, 2, 2};
    auto data = ge::op::Data("x").set_attr_index(0);
    ge::TensorDesc xDesc(ge::Shape(xShape), ge::FORMAT_NDHWC, ge::DT_FLOAT);
    data.update_input_desc_x(xDesc);

    auto pool = ge::op::MaxPool3D("max_pool3_d")
                    .set_input_x(data)
                    .set_attr_ksize({1, 2, 2, 2, 1})
                    .set_attr_strides({1, 2, 2, 2, 1})
                    .set_attr_padding("VALID")
                    .set_attr_pads({0, 0, 0, 0, 0, 0})
                    .set_attr_dilation({1, 1, 1, 1, 1})
                    .set_attr_ceil_mode(0)
                    .set_attr_data_format("NDHWC");
    pool.update_input_desc_x(xDesc);
    pool.update_output_desc_y(ge::TensorDesc(ge::Shape(yShape), ge::FORMAT_NDHWC, ge::DT_FLOAT));

    ge::Graph graph("max_pool3_d_example");
    graph.AddOp(data);
    graph.AddOp(pool);
    graph.SetInputs({data});
    graph.SetOutputs({pool});

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    constexpr uint32_t graphId = 0;
    if (session.AddGraph(graphId, graph) != ge::SUCCESS) {
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    std::vector<ge::Tensor> inputs = {MakeInput(xShape)};
    if (session.RunGraph(graphId, inputs, outputs) != ge::SUCCESS) {
        ge::GEFinalize();
        return kFailed;
    }
    if (outputs.size() != 1U || outputs[0].GetTensorDesc().GetShape().GetDims() != yShape) {
        ge::GEFinalize();
        return kFailed;
    }
    ge::GEFinalize();
    return kSuccess;
}
