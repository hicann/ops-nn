/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../op_graph/dynamic_augru_proto.h"
#include "ge_api.h"
#include "graph/graph.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

namespace {
using Dims = std::vector<int64_t>;
using Case = std::array<int64_t, 4>;
const std::array<const char*, 8> names = {"x",          "weight_input", "weight_hidden", "weight_att",
                                          "bias_input", "bias_hidden",  "seq_length",    "init_h"};

std::vector<Dims> Shapes(const Case& shape)
{
    const auto [time, batch, inputSize, hidden] = shape;
    return {{time, batch, inputSize},
            {inputSize, 3 * hidden},
            {hidden, 3 * hidden},
            {time, batch},
            {3 * hidden},
            {3 * hidden},
            {batch},
            {1, batch, hidden}};
}

ge::DataType Type(size_t index, ge::DataType state)
{
    return index < 4 ? ge::DT_FLOAT16 : (index == 6 ? ge::DT_INT32 : state);
}

ge::Tensor Input(const Case& shape, size_t index, ge::DataType state)
{
    auto dims = Shapes(shape)[index];
    size_t count = 1;
    for (auto dim : dims) {
        count *= dim;
    }
    auto dtype = Type(index, state);
    const size_t width = dtype == ge::DT_FLOAT16 ? 2 : 4;
    std::vector<uint8_t> bytes(count * width, 0);
    for (size_t elementIndex = 0; elementIndex < count; ++elementIndex) {
        if (index == 6) {
            int32_t length = static_cast<int32_t>(std::min<int64_t>(shape[0], elementIndex + 1));
            std::memcpy(bytes.data() + elementIndex * width, &length, width);
        } else if (index == 3 || index == 7) {
            if (dtype == ge::DT_FLOAT16) {
                uint16_t bits = index == 3 ? 0x3800 : 0x3000; // 0.5 or 0.125
                std::memcpy(bytes.data() + elementIndex * width, &bits, width);
            } else {
                float value = 0.125F;
                std::memcpy(bytes.data() + elementIndex * width, &value, width);
            }
        }
    }
    return ge::Tensor(ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, dtype), bytes.data(), bytes.size());
}

float ReadHalf(const uint8_t* data)
{
    uint16_t bits;
    std::memcpy(&bits, data, sizeof(bits));
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    float result = exponent == 0  ? std::ldexp(static_cast<float>(fraction), -24) :
                   exponent == 31 ? (fraction ? NAN : INFINITY) :
                                    std::ldexp(static_cast<float>(1024 + fraction), exponent - 25);
    return bits & 0x8000 ? -result : result;
}

bool Check(const Case& shape, ge::DataType state, const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 7) {
        return false;
    }
    const size_t width = state == ge::DT_FLOAT16 ? 2 : 4;
    for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
        const auto& value = outputs[outputIndex];
        if (value.GetTensorDesc().GetShape().GetDims() != Dims{shape[0], shape[1], shape[3]} ||
            value.GetTensorDesc().GetDataType() != state ||
            value.GetSize() != static_cast<size_t>(shape[0] * shape[1] * shape[3]) * width) {
            return false;
        }
        for (int64_t time = 0; time < shape[0]; ++time) {
            for (int64_t batch = 0; batch < shape[1]; ++batch) {
                // Zero weights and biases give z=r=0.5, candidate=hidden_new=0.
                // Attention=0.5 gives z_att=0.25; state=0.125*(0.25)^active_steps.
                float expected = 0.0F;
                if (outputIndex < 2) {
                    const auto activeSteps = std::min<int64_t>(time + 1, std::min<int64_t>(shape[0], batch + 1));
                    expected = std::ldexp(0.125F, -2 * static_cast<int>(activeSteps));
                } else if (outputIndex == 2 || outputIndex == 4) {
                    expected = 0.5F;
                } else if (outputIndex == 3) {
                    expected = 0.25F;
                }
                for (int64_t hidden = 0; hidden < shape[3]; ++hidden) {
                    size_t index = (time * shape[1] + batch) * shape[3] + hidden;
                    float actual;
                    if (width == 2) {
                        actual = ReadHalf(value.GetData() + index * width);
                    } else {
                        std::memcpy(&actual, value.GetData() + index * width, width);
                    }
                    if (!std::isfinite(actual) ||
                        std::abs(actual - expected) >
                            (width == 2 ? 1e-3F + 1e-2F * std::abs(expected) : 5e-4F + 5e-3F * std::abs(expected))) {
                        std::cerr << "Output mismatch " << outputIndex << " index " << index << " actual=" << actual
                                  << " expected=" << expected << " dtype=" << state << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool Run(const char* scenario, int declaration, ge::DataType state, const std::vector<Case>& cases)
{
    ge::Graph graph(scenario);
    ge::op::DynamicAUGRU op(scenario);
    std::vector<ge::Operator> inputs;
    auto shapes = Shapes(cases.front());
    for (size_t inputIndex = 0; inputIndex < shapes.size(); ++inputIndex) {
        auto declared = declaration == -2 ? Dims{-2} : shapes[inputIndex];
        if (declaration == -1) {
            std::fill(declared.begin(), declared.end(), -1);
        }
        ge::TensorDesc desc(ge::Shape(declared), ge::FORMAT_ND, Type(inputIndex, state));
        desc.SetOriginShape(ge::Shape(declared));
        ge::op::Data data(names[inputIndex]);
        data.set_attr_index(inputIndex);
        data.update_input_desc_x(desc);
        data.update_output_desc_y(desc);
        inputs.push_back(data);
        op.UpdateInputDesc(names[inputIndex], desc);
        op.SetInput(names[inputIndex], inputs.back());
    }
    for (size_t inputIndex = 0; inputIndex < 7; ++inputIndex) {
        Dims shape = declaration ? Dims{-1, -1, -1} : Dims{cases[0][0], cases[0][1], cases[0][3]};
        op.UpdateOutputDesc(inputIndex, ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, state));
    }
    graph.SetInputs(inputs).SetOutputs({op});
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.jit_compile", "0"}};
    ge::Session session(options);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << "\n";
        return false;
    }
    std::cout << "Scenario " << scenario << " declaration=" << declaration << " dtype=" << state << " AddGraph=1\n";
    for (const auto& shape : cases) {
        std::vector<ge::Tensor> tensors, outputs;
        for (size_t inputIndex = 0; inputIndex < 8; ++inputIndex) {
            tensors.push_back(Input(shape, inputIndex, state));
        }
        if (session.RunGraph(0, tensors, outputs) != ge::SUCCESS || !Check(shape, state, outputs)) {
            std::cerr << ge::GEGetErrorMsgV2().GetString() << "\n";
            return false;
        }
        std::cout << "Shape, dtype and values PASSED for [" << shape[0] << "," << shape[1] << "," << shape[2] << ","
                  << shape[3] << "]\n";
    }
    std::cout << "Scenario " << scenario << " summary: " << cases.size() << "/" << cases.size() << " passed\n";
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << "\n";
        return 1;
    }
    bool ok = true;
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        if (argc > 1 && std::string(argv[1]) == "fp32" && dtype != ge::DT_FLOAT) {
            continue;
        }
        // One example covers static shape, unknown dimensions and unknown rank.
        ok = Run("static", 0, dtype, {{2, 1, 3, 4}}) && ok;
        for (int mode : {-1, -2}) {
            ok = Run(mode == -1 ? "unknown_dim_minus_1" : "unknown_rank_minus_2", mode, dtype,
                     {{2, 1, 3, 4}, {3, 2, 5, 7}, {4, 3, 4, 8}}) &&
                 ok;
        }
    }
    ge::GEFinalize();
    if (ok) {
        std::cout << "DynamicAUGRU GEIR verification PASSED\n";
    }
    return ok ? 0 : 1;
}
