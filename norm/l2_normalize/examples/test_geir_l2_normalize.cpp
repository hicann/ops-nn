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
 * \file test_geir_l2_normalize.cpp
 * \brief GE graph construction sample for L2Normalize (arch35 ND format).
 */
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>
#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph/operator_reg.h"
#include "../op_graph/l2_normalize_proto.h"

// Data is still owned by the duplicate-heavy legacy compatibility header in older CANN releases, so keep only its
// small graph-construction class test-local instead of including two overlapping operator domains.
#ifndef OPS_PROTO_DEF_DATA
#define OPS_PROTO_DEF_DATA
namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge
#endif // OPS_PROTO_DEF_DATA

#define FAILED -1
#define SUCCESS 0
using namespace ge;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    const time_t current = time(nullptr);
    if (current == static_cast<time_t>(-1)) {
        return "unknown-time";
    }
    std::tm localTime = {};
    if (localtime_r(&current, &localTime) == nullptr) {
        return "unknown-time";
    }
    char buffer[32] = {};
    if (strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &localTime) == 0) {
        return "unknown-time";
    }
    return string(buffer);
}

int32_t GenOnesData(const vector<int64_t>& shapes, Tensor& tensor, TensorDesc& desc, DataType dt, int val)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto d : shapes) {
        size *= d;
    }
    if (dt == DT_FLOAT) {
        vector<float> v(size, static_cast<float>(val));
        desc.SetShape(ge::Shape(shapes));
        tensor.SetTensorDesc(desc);
        if (tensor.SetData(reinterpret_cast<const uint8_t*>(v.data()), size * sizeof(float)) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
    } else {
        vector<uint16_t> v(size, 0x3C00);
        desc.SetShape(ge::Shape(shapes));
        tensor.SetTensorDesc(desc);
        if (tensor.SetData(reinterpret_cast<const uint8_t*>(v.data()), size * sizeof(uint16_t)) != ge::GRAPH_SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

int CreateGraph(DataType dt, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                std::vector<Operator>& outputs, Graph& graph)
{
    auto op_node = op::L2Normalize("l2_normalize_op");
    const std::vector<int64_t> shape = {4, 16}; // axis={1} reduces the last dim
    auto data = op::Data("ph1").set_attr_index(0);
    TensorDesc xDesc(ge::Shape(shape), FORMAT_ND, dt);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetOriginFormat(FORMAT_ND);
    xDesc.SetOriginShape(ge::Shape(shape));
    Tensor xTensor;
    if (GenOnesData(shape, xTensor, xDesc, dt, 1) != SUCCESS) {
        return FAILED;
    }
    data.update_input_desc_x(xDesc);
    data.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(data);
    op_node.set_input_x(data);
    inputs.push_back(data);

    op_node.set_attr_axis({1});
    op_node.set_attr_eps(0.0001f);
    TensorDesc yDesc(ge::Shape(shape), FORMAT_ND, dt);
    yDesc.SetOriginFormat(FORMAT_ND);
    yDesc.SetOriginShape(ge::Shape(shape));
    op_node.update_output_desc_y(yDesc);
    outputs.push_back(op_node);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    Graph graph("tc_ge_irrun_test");
    std::vector<ge::Tensor> input;
    std::map<AscendString, AscendString> gopt = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(gopt) != SUCCESS) {
        LOG_PRINT("%s - FAIL - GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    std::vector<Operator> inputs{}, outputs{};
    DataType dt = DT_FLOAT;
    if (CreateGraph(dt, input, inputs, outputs, graph) != SUCCESS) {
        LOG_PRINT("%s - FAIL - CreateGraph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);
    std::map<AscendString, AscendString> bopt = {};
    uint32_t gid = 0;
    std::map<AscendString, AscendString> gropt = {};
    std::vector<ge::Tensor> output;
    int32_t result = SUCCESS;
    {
        // Destroy the session before GEFinalize while retaining one cleanup path for every outcome.
        Session session(bopt);
        if (session.AddGraph(gid, graph, gropt) != SUCCESS) {
            LOG_PRINT("%s - FAIL - [XIR]: AddGraph failed\n", GetTime().c_str());
            result = FAILED;
        } else if (session.RunGraph(gid, input, output) != SUCCESS) {
            LOG_PRINT("%s - FAIL - [XIR]: RunGraph failed\n", GetTime().c_str());
            result = FAILED;
        } else {
            LOG_PRINT("%s - PASS - [XIR]: Session RunGraph succeeded, outputs=%zu\n", GetTime().c_str(), output.size());
            for (size_t i = 0; i < output.size(); i++) {
                LOG_PRINT("  output[%zu] dim=%zu\n", i, output[i].GetTensorDesc().GetShape().GetDimNum());
            }
        }
    }
    ge::GEFinalize();
    return result;
}
