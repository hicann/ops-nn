/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "graph.h"
#include "tensor.h"
#include "../../op_graph/sparse_apply_proximal_adagrad_proto.h"
#include "../../tests/fixtures/sparse_apply_proximal_adagrad_regression.h"

namespace {
using namespace sparse_apply_proximal_adagrad_regression;

constexpr uint32_t GRAPH_ID = 0;
constexpr size_t INPUT_COUNT = 7;
constexpr size_t OUTPUT_COUNT = 2;
constexpr std::array<const char*, INPUT_COUNT> PORTS = {"var", "accum", "lr", "l1", "l2", "grad", "indices"};
const std::vector<int64_t> STATE_SHAPE = {ROWS, COLS};

template <typename T>
ge::Tensor MakeTensor(const ge::TensorDesc& desc, const T* data, size_t bytes)
{
    const auto* begin = reinterpret_cast<const uint8_t*>(data);
    return ge::Tensor(desc, std::vector<uint8_t>(begin, begin + bytes));
}

bool CheckOutput(const Fixture& fixture, const char* port, const ge::Tensor& output, const StateBits& expected)
{
    const auto desc = output.GetTensorDesc();
    if (desc.GetDataType() != ge::DT_FLOAT || desc.GetShape().GetDims() != STATE_SHAPE ||
        output.GetSize() != sizeof(expected) || output.GetData() == nullptr) {
        std::fprintf(stderr, "%s: unexpected %s output metadata\n", fixture.name, port);
        return false;
    }

    StateBits actual{};
    std::memcpy(actual.data(), output.GetData(), sizeof(actual));
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s: %s[%zu] got 0x%08x, expected 0x%08x\n", fixture.name, port, i, actual[i],
                         expected[i]);
        }
    }
    return actual == expected;
}

bool RunCase(ge::Session& session, const std::vector<ge::TensorDesc>& desc, const Fixture& fixture)
{
    std::vector<ge::Tensor> inputs = {
        MakeTensor(desc[0], VAR_BITS.data(), sizeof(VAR_BITS)),
        MakeTensor(desc[1], fixture.accum.data(), sizeof(fixture.accum)),
        MakeTensor(desc[2], &LR, sizeof(LR)),
        MakeTensor(desc[3], &ZERO, sizeof(ZERO)),
        MakeTensor(desc[4], &ZERO, sizeof(ZERO)),
        MakeTensor(desc[5], fixture.grad.data(), sizeof(fixture.grad)),
        MakeTensor(desc[6], &INDEX, sizeof(INDEX)),
    };
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(GRAPH_ID, inputs, outputs) != ge::SUCCESS || outputs.size() != OUTPUT_COUNT) {
        std::fprintf(stderr, "%s: RunGraph failed or did not produce both outputs\n", fixture.name);
        return false;
    }
    const bool varOk = CheckOutput(fixture, PORTS[0], outputs[0], fixture.expectedVar);
    const bool accumOk = CheckOutput(fixture, PORTS[1], outputs[1], fixture.expectedAccum);
    std::printf("%s: var bits %s, accum bits %s\n", fixture.name, varOk ? "PASS" : "FAIL", accumOk ? "PASS" : "FAIL");
    return varOk && accumOk;
}

bool RunRegression()
{
    const ge::TensorDesc state(ge::Shape(STATE_SHAPE), ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc scalar(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc grad(ge::Shape({1, COLS}), ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc indices(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
    std::vector<ge::TensorDesc> desc = {state, state, scalar, scalar, scalar, grad, indices};

    ge::Graph graph("sparse_apply_proximal_adagrad_regression");
    auto op = ge::op::SparseApplyProximalAdagrad("sparse_update").set_attr_use_locking(true);
    std::vector<ge::Operator> source;
    for (size_t i = 0; i < INPUT_COUNT; ++i) {
        auto data = ge::op::Data(PORTS[i]).set_attr_index(static_cast<int64_t>(i));
        desc[i].SetPlacement(ge::kPlacementHost);
        data.update_input_desc_x(desc[i]);
        graph.AddOp(data);
        op.SetInput(PORTS[i], data);
        source.push_back(data);
    }
    op.update_output_desc_var(state);
    op.update_output_desc_accum(state);
    graph.SetInputs(source).SetOutputs({op});

    ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
    if (session.AddGraph(GRAPH_ID, graph) != ge::SUCCESS) {
        std::fprintf(stderr, "AddGraph failed\n");
        return false;
    }
    const bool zeroOk = RunCase(session, desc, ZERO_GRADIENT);
    const bool nonzeroOk = RunCase(session, desc, NONZERO_GRADIENT);
    return zeroOk && nonzeroOk;
}
} // namespace

int main()
{
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::fprintf(stderr, "GEInitialize failed\n");
        return 1;
    }
    const bool passed = RunRegression();
    const bool finalized = ge::GEFinalize() == ge::SUCCESS;
    return passed && finalized ? 0 : 1;
}
