/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_deformable_conv2d.cpp
 * \brief DeformableConv2D (x/filter/offsets/bias -> y) GE IR E2E verification.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "graph/operator_reg.h"

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
} // namespace ge
#include "../../op_graph/deformable_conv2d_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

struct TestScenario {
    const char* name;
    std::vector<int64_t> xShape;      // NCHW
    std::vector<int64_t> filterShape; // NCHW (kn, kc, kh, kw)
    std::vector<int64_t> strides;     // NCHW order
    std::vector<int64_t> pads;        // t, b, l, r
    std::vector<int64_t> dilations;   // NCHW order
    int64_t groups;
    int64_t deformableGroups;
    bool withBias;
    bool expectSuccess;
    bool numericCheck;
    std::vector<int64_t> expectedShape; // NCHW
    bool modulatedTrue;
    bool breakOffsetsChannel;
};

static const std::vector<TestScenario> gScenarios = {
    {"s1_basic_nchw",
     {1, 4, 8, 8},
     {4, 4, 3, 3},
     {1, 1, 1, 1},
     {0, 0, 0, 0},
     {1, 1, 1, 1},
     1,
     1,
     true,
     true,
     true,
     {1, 4, 6, 6},
     true,
     false},

};

static int64_t Prod(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= (d < 0) ? 1 : d;
    }
    return n;
}

static float HalfToFloat(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h >> 15) & 1U;
    const uint32_t exp = static_cast<uint32_t>(h >> 10) & 0x1fU;
    const uint32_t frac = static_cast<uint32_t>(h) & 0x3ffU;
    float value = 0.0f;
    if (exp == 0) {
        value = static_cast<float>(frac) * (1.0f / 1024.0f / 1024.0f); // subnormal
    } else if (exp == 31) {
        value = (frac == 0) ? std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(1.0f + static_cast<float>(frac) / 1024.0f, static_cast<int>(exp) - 15);
    }
    return (sign != 0U) ? -value : value;
}

static uint16_t FloatToHalf(float f)
{
    const uint32_t x = *reinterpret_cast<const uint32_t*>(&f);
    const uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    uint32_t frac = x & 0x7fffffU;
    if (((x >> 23) & 0xffU) == 0xffU) {
        return static_cast<uint16_t>(sign | 0x7c00U); // inf/nan
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (exp <= 0) {
        return static_cast<uint16_t>(sign); // flush subnormals to zero
    }
    frac >>= 13;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | frac);
}

static ge::TensorDesc MakeHostDesc(const std::vector<int64_t>& shape, ge::Format format,
                                   ge::DataType dtype = ge::DT_FLOAT16)
{
    ge::TensorDesc desc = TensorDesc(ge::Shape(shape), format, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(format);
    desc.SetOriginFormat(format);
    return desc;
}

static ge::Tensor MakeHalfTensor(const std::vector<int64_t>& shape, ge::Format format, const std::vector<float>& data,
                                 ge::DataType dtype = ge::DT_FLOAT16)
{
    const size_t total = static_cast<size_t>(Prod(shape));
    std::vector<uint16_t> half(total);
    for (size_t i = 0; i < total; i++) {
        half[i] = FloatToHalf(data[i]);
    }
    ge::Tensor tensor;
    tensor.SetTensorDesc(MakeHostDesc(shape, format, dtype));
    tensor.SetData(reinterpret_cast<const uint8_t*>(half.data()), static_cast<int32_t>(total * sizeof(uint16_t)));
    return tensor;
}

static std::vector<float> MakeXData(const TestScenario& s)
{
    std::vector<float> data(static_cast<size_t>(Prod(s.xShape)));
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = 0.5f + 0.01f * static_cast<float>(i % 97);
    }
    return data;
}

static std::vector<float> MakeWData(const TestScenario& s)
{
    std::vector<float> data(static_cast<size_t>(Prod(s.filterShape)));
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = 0.05f + 0.001f * static_cast<float>(i % 31);
    }
    return data;
}

static std::vector<float> MakeOffsetsData(int64_t channel, int64_t spatial, int64_t maskBlockStart)
{
    // offsets layout: [dx | dy | mask], each block dg*kh*kw channels
    std::vector<float> data(static_cast<size_t>(channel * spatial), 0.0f);
    for (int64_t c = maskBlockStart; c < channel; c++) {
        for (int64_t p = 0; p < spatial; p++) {
            data[static_cast<size_t>(c * spatial + p)] = 1.0f;
        }
    }
    return data;
}

// CPU reference: plain conv2d (NCHW x, NCHW filter, groups) + optional bias.
static void CpuConv2d(const float* x, const float* w, const float* bias, const TestScenario& s, float* y)
{
    const int64_t in = s.xShape[0];
    const int64_t ic = s.xShape[1];
    const int64_t ih = s.xShape[2];
    const int64_t iw = s.xShape[3];
    const int64_t kn = s.filterShape[0];
    const int64_t kc = s.filterShape[1];
    const int64_t kh = s.filterShape[2];
    const int64_t kw = s.filterShape[3];
    const int64_t strh = s.strides[2];
    const int64_t strw = s.strides[3];
    const int64_t dilh = s.dilations[2];
    const int64_t dilw = s.dilations[3];
    const int64_t oh = (ih + s.pads[0] + s.pads[1] - dilh * (kh - 1) - 1) / strh + 1;
    const int64_t ow = (iw + s.pads[2] + s.pads[3] - dilw * (kw - 1) - 1) / strw + 1;
    const int64_t oc = kn;
    for (int64_t n = 0; n < in; n++) {
        for (int64_t ocIdx = 0; ocIdx < oc; ocIdx++) {
            const int64_t g = (kn / s.groups == 0) ? 0 : ocIdx / (kn / s.groups);
            for (int64_t ohIdx = 0; ohIdx < oh; ohIdx++) {
                for (int64_t owIdx = 0; owIdx < ow; owIdx++) {
                    float acc = 0.0f;
                    for (int64_t kcIdx = 0; kcIdx < kc; kcIdx++) {
                        const int64_t icIdx = g * kc + kcIdx;
                        for (int64_t khIdx = 0; khIdx < kh; khIdx++) {
                            for (int64_t kwIdx = 0; kwIdx < kw; kwIdx++) {
                                const int64_t ihIdx = ohIdx * strh - s.pads[0] + khIdx * dilh;
                                const int64_t iwIdx = owIdx * strw - s.pads[2] + kwIdx * dilw;
                                if (ihIdx < 0 || ihIdx >= ih || iwIdx < 0 || iwIdx >= iw) {
                                    continue;
                                }
                                const float xv = x[static_cast<size_t>(((n * ic + icIdx) * ih + ihIdx) * iw + iwIdx)];
                                const float
                                    wv = w[static_cast<size_t>((((ocIdx * kc) + kcIdx) * kh + khIdx) * kw + kwIdx)];
                                acc += xv * wv;
                            }
                        }
                    }
                    if (bias != nullptr) {
                        acc += bias[static_cast<size_t>(ocIdx)];
                    }
                    y[static_cast<size_t>(((n * oc + ocIdx) * oh + ohIdx) * ow + owIdx)] = acc;
                }
            }
        }
    }
}

static void AddDataInput(const char* name, uint32_t index, const std::vector<int64_t>& shape, ge::Format format,
                         const std::vector<float>& data, ge::Graph& graph, std::vector<ge::Tensor>& input,
                         std::vector<ge::Operator>& inputs, ge::DataType dtype = ge::DT_FLOAT16)
{
    auto dataOp = op::Data(name).set_attr_index(index);
    const ge::TensorDesc desc = MakeHostDesc(shape, format, dtype);
    dataOp.update_input_desc_x(desc);
    dataOp.update_output_desc_y(desc);
    const ge::Tensor tensor = MakeHalfTensor(shape, format, data, dtype);
    input.push_back(tensor);
    graph.AddOp(dataOp);
    inputs.push_back(dataOp);
}

static int BuildAndRunScenario(ge::Session* session, uint32_t graphId, const TestScenario& scenario,
                               std::vector<ge::Tensor>& output, const std::vector<float>& xData,
                               const std::vector<float>& wData, const std::vector<float>& offsetsData,
                               const std::vector<float>& biasData, int64_t offsetsChannel)
{
    const int64_t outH = (scenario.expectedShape.size() == 4) ? scenario.expectedShape[2] : 6;
    const int64_t outW = (scenario.expectedShape.size() == 4) ? scenario.expectedShape[3] : 6;
    const int64_t offsetsCh = scenario.breakOffsetsChannel ? offsetsChannel - 1 : offsetsChannel;
    const std::vector<int64_t> offsetsShape{1, offsetsCh, outH, outW};

    ge::Graph graph(scenario.name);
    std::vector<ge::Tensor> input;
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;

    auto dfm = op::DeformableConv2D(scenario.name);
    AddDataInput("x_input", 0, scenario.xShape, ge::FORMAT_NCHW, xData, graph, input, inputs);
    dfm.set_input_x(inputs[0]);
    dfm.update_input_desc_x(MakeHostDesc(scenario.xShape, ge::FORMAT_NCHW));
    AddDataInput("filter_input", 1, scenario.filterShape, ge::FORMAT_NCHW, wData, graph, input, inputs);
    dfm.set_input_filter(inputs[1]);
    dfm.update_input_desc_filter(MakeHostDesc(scenario.filterShape, ge::FORMAT_NCHW));
    AddDataInput("offsets_input", 2, offsetsShape, ge::FORMAT_NCHW, offsetsData, graph, input, inputs);
    dfm.set_input_offsets(inputs[2]);
    dfm.update_input_desc_offsets(MakeHostDesc(offsetsShape, ge::FORMAT_NCHW));
    if (scenario.withBias) {
        const std::vector<int64_t> biasShape{scenario.filterShape[0]};
        AddDataInput("bias_input", 3, biasShape, ge::FORMAT_ND, biasData, graph, input, inputs);
        dfm.set_input_bias(inputs[3]);
        dfm.update_input_desc_bias(MakeHostDesc(biasShape, ge::FORMAT_ND));
    }

    dfm.set_attr_strides(scenario.strides);
    dfm.set_attr_pads(scenario.pads);
    dfm.set_attr_dilations(scenario.dilations);
    dfm.set_attr_groups(scenario.groups);
    dfm.set_attr_data_format("NCHW");
    dfm.set_attr_deformable_groups(scenario.deformableGroups);
    dfm.set_attr_modulated(scenario.modulatedTrue);

    const std::vector<int64_t> yShape = (scenario.expectedShape.size() == 4) ? scenario.expectedShape :
                                                                               std::vector<int64_t>{1, 1, 1, 1};
    dfm.update_output_desc_y(MakeHostDesc(yShape, ge::FORMAT_NCHW));

    graph.AddOp(dfm);
    outputs.push_back(dfm);
    graph.SetInputs(inputs).SetOutputs(outputs);

    map<AscendString, AscendString> graphOptions = {};
    Status ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - Add graph failed (ret=%d)\n", scenario.name, ret);
        return scenario.expectSuccess ? FAILED : SUCCESS;
    }
    printf("%s - Start to run graph\n", scenario.name);
    output.clear();
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        printf("%s - Run graph failed (ret=%d)\n", scenario.name, ret);
        if (!scenario.expectSuccess) {
            return SUCCESS;
        }
        printf("%s - FAILED: Run graph failed (ret=%d)\n", scenario.name, ret);
        return FAILED;
    }
    if (!scenario.expectSuccess) {
        printf("%s - ERROR: Expected failure but graph succeeded!\n", scenario.name);
        return FAILED;
    }
    return SUCCESS;
}

static int VerifyScenario(const TestScenario& scenario, const std::vector<ge::Tensor>& output,
                          const std::vector<float>& xData, const std::vector<float>& wData,
                          const std::vector<float>& biasData)
{
    auto outShape = output[0].GetTensorDesc().GetShape().GetDims();
    printf("%s - output shape: [", scenario.name);
    for (size_t j = 0; j < outShape.size(); j++) {
        printf("%ld%s", outShape[j], (j < outShape.size() - 1 ? ", " : ""));
    }
    printf("], dtype: %d\n", output[0].GetTensorDesc().GetDataType());
    if (outShape != scenario.expectedShape) {
        printf("%s - ERROR: Shape mismatch, expected [%ld, %ld, %ld, %ld]\n", scenario.name, scenario.expectedShape[0],
               scenario.expectedShape[1], scenario.expectedShape[2], scenario.expectedShape[3]);
        return FAILED;
    }
    if (!scenario.numericCheck) {
        printf("%s - PASSED\n", scenario.name);
        return SUCCESS;
    }
    const int64_t yTotal = Prod(scenario.expectedShape);
    std::vector<float> ref(static_cast<size_t>(yTotal), 0.0f);
    CpuConv2d(xData.data(), wData.data(), scenario.withBias ? biasData.data() : nullptr, scenario, ref.data());
    const uint16_t* npuHalf = reinterpret_cast<const uint16_t*>(output[0].GetData());
    if (npuHalf == nullptr) {
        printf("%s - ERROR: null output data\n", scenario.name);
        return FAILED;
    }
    float maxDiff = 0.0f;
    float maxRelDiff = 0.0f;
    const float atol = 1e-3f;
    const float rtol = 1e-3f;
    int64_t worstIdx = 0;
    for (int64_t i = 0; i < yTotal; i++) {
        const float npu = HalfToFloat(npuHalf[static_cast<size_t>(i)]);
        const float refv = ref[static_cast<size_t>(i)];
        const float diff = std::fabs(refv - npu);
        const float relDiff = diff / std::fmax(std::fabs(refv), 1e-6f);
        if (diff > maxDiff) {
            maxDiff = diff;
            worstIdx = i;
        }
        if (relDiff > maxRelDiff) {
            maxRelDiff = relDiff;
        }
        if (diff > atol + rtol * std::fabs(refv)) {
            printf("%s - NUMERIC MISMATCH at [%ld]: cpu=%.6f npu=%.6f diff=%.6f relDiff=%.6f\n", scenario.name, i,
                   ref[static_cast<size_t>(i)], npu, diff, relDiff);
            return FAILED;
        }
    }
    printf("%s - NUMERIC PASS (maxDiff=%.6f maxRelDiff=%.6f atol=%.1e rtol=%.1e)\n", scenario.name, maxDiff, maxRelDiff,
           atol, rtol);
    printf("%s - PASSED\n", scenario.name);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    printf("INFO - [XIR]: DeformableConv2D graph interface infershape verification\n");

    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("ERROR - [XIR]: GEInitialize failed\n");
        return FAILED;
    }
    map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("ERROR - [XIR]: Create session failed\n");
        ge::GEFinalize();
        return FAILED;
    }

    int totalPass = 0;
    int totalFail = 0;
    uint32_t graphId = 0;
    for (const auto& scenario : gScenarios) {
        const int64_t blockCh = scenario.deformableGroups * scenario.filterShape[2] * scenario.filterShape[3];
        const int64_t offsetsChannel = (scenario.modulatedTrue ? 3 : 2) * blockCh;
        const int64_t maskBlockStart = scenario.modulatedTrue ? 2 * blockCh : offsetsChannel;
        const int64_t outH = (scenario.expectedShape.size() == 4) ? scenario.expectedShape[2] : 6;
        const int64_t outW = (scenario.expectedShape.size() == 4) ? scenario.expectedShape[3] : 6;
        const std::vector<float> xData = MakeXData(scenario);
        const std::vector<float> wData = MakeWData(scenario);
        const std::vector<float> offsetsData = MakeOffsetsData(offsetsChannel, outH * outW, maskBlockStart);
        const std::vector<float> biasData(static_cast<size_t>(scenario.filterShape[0]), 0.25f);
        std::vector<ge::Tensor> output;
        const int r = BuildAndRunScenario(session, graphId++, scenario, output, xData, wData, offsetsData, biasData,
                                          offsetsChannel);
        if (r == SUCCESS && scenario.expectSuccess) {
            if (VerifyScenario(scenario, output, xData, wData, biasData) == SUCCESS) {
                totalPass++;
            } else {
                totalFail++;
            }
        } else if (r == SUCCESS) {
            totalPass++;
        } else {
            totalFail++;
        }
    }

    printf("\nINFO - ========== Verification Summary ==========\n");
    printf("  PASSED: %d\n", totalPass);
    printf("  FAILED: %d\n", totalFail);
    printf("  ==============================================\n\n");

    delete session;
    ge::GEFinalize();
    return totalFail == 0 ? SUCCESS : FAILED;
}
